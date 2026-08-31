// GPU kernels in Metal Shading Language (MSL) — the device-side twin of math.rs + model.rs.
//
// Reading guide: one kernel = a function executed by thousands of threads at once.
//   [[thread_position_in_grid]]        = this thread's index within the whole dispatch
//   [[threadgroup_position_in_grid]]   = which threadgroup this is
//   threadgroup_barrier(...)           = rendezvous: every thread in the group must arrive
//
// Every kernel operates on "rows" (n_rows = number of tokens processed together):
//   decode  → n_rows = 1   (single token, mirroring the CPU path exactly)
//   prefill → n_rows = a whole prompt chunk (matrix-matrix — the source of the speedup)
// One code path serves both modes; only the dispatched grid size differs.
//
// Data-type convention: weights and the KV cache are half (f16) to halve memory
// traffic; activations stay float (f32); accumulation is always float.
//
// Every params struct must match its #[repr(C)] counterpart in gpu/metal.rs exactly.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

// ---------- embedding: token ids → one vector per row (step 1 of model.rs) ----------

struct EmbedParams {
    uint dim;
    uint n_rows;
};

kernel void embed(
    device const half *table [[buffer(0)]],
    device const uint *ids [[buffer(1)]],
    device float *x [[buffer(2)]],
    constant EmbedParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n_rows * p.dim) {
        uint row = gid / p.dim;
        uint i = gid % p.dim;
        x[gid] = (float)table[(ulong)ids[row] * p.dim + i];
    }
}

// ---------- GGUF quant dequantization (-b lowmem) ----------
// Block layouts follow ggml (llama.cpp, MIT — studied and reimplemented; the
// structs live in ggml-common.h, the reference walks in ggml-quants.c). Every
// value dequantizes to f32 in registers with the SAME expression shapes as the
// CPU reference dequant_row_ref, and the quant pipelines are built from the
// engine's PRECISE (fast-math-off) library so multiplies and subtracts stay
// un-fused IEEE f32 — the oracle gate demands bit-for-bit, not "close".
//
// LM_W_QTYPE selects the weight encoding at pipeline build (the switch folds):
//   0 = staged f16 (default; every existing pipeline)
//   1 = raw bf16 through the mmap view (supersedes the old LM_W_BF16 flag)
//   2 = Q8_0   34 B / 32 elems : f16 d, int8 qs[32]         → q*d
//   3 = Q4_0   18 B / 32 elems : f16 d, nibbles lo|hi       → (q-8)*d
//   4 = Q4_K  144 B / 256      : f16 d,dmin, 6-bit packed scales/mins ×8, nibbles
//   5 = Q6_K  210 B / 256      : ql nibbles + qh 2-bit highs, int8 scales ×16, f16 d
//   6 = Q5_K  176 B / 256      : Q4_K plus one high bit per element (qh[32])
constant uint LM_W_QTYPE_FC [[function_constant(25)]];
constant uint LM_W_QTYPE = is_function_constant_defined(LM_W_QTYPE_FC) ? LM_W_QTYPE_FC : 0;

// f16 scale at an arbitrary (unaligned) byte offset — 18/34/210-byte blocks
// put half the scales on odd addresses.
inline float lm_f16_at(device const uchar *p) {
    return (float)as_type<half>((ushort)(p[0] | (p[1] << 8)));
}

inline float lm_dequant_q8_0(device const uchar *row, uint col) {
    device const uchar *b = row + (col >> 5) * 34;
    float d = lm_f16_at(b);
    return (float)(char)b[2 + (col & 31)] * d;
}

inline float lm_dequant_q4_0(device const uchar *row, uint col) {
    device const uchar *b = row + (col >> 5) * 18;
    float d = lm_f16_at(b);
    uint j = col & 31;
    uchar byte = b[2 + (j & 15)];
    int q = (int)((j < 16) ? (byte & 0x0F) : (byte >> 4)) - 8;
    return (float)q * d;
}

// The 6-bit packed scale/min unpack — THE classic silent-rot spot. Mirrors
// ggml's get_scale_min_k4 exactly, including the j-th (not j+4-th) byte
// donating the top bits of the MIN in the second half.
inline void lm_scale_min_k4(uint j, device const uchar *q, thread uint &sc, thread uint &mn) {
    if (j < 4) {
        sc = q[j] & 63;
        mn = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

inline float lm_dequant_q4_K(device const uchar *row, uint col) {
    device const uchar *b = row + (col >> 8) * 144;
    float d = lm_f16_at(b);
    float dmin = lm_f16_at(b + 2);
    device const uchar *scales = b + 4;
    device const uchar *qs = b + 16;
    uint ib = col & 255;
    uint sc, mn;
    lm_scale_min_k4(ib >> 5, scales, sc, mn);
    float d1 = d * (float)sc;
    float m1 = dmin * (float)mn;
    // qs: per 64-element group, 32 bytes — low nibbles first 32, high next 32.
    uchar byte = qs[(ib >> 6) * 32 + (ib & 31)];
    uint q = (ib & 32) ? (byte >> 4) : (byte & 0xF);
    return d1 * (float)q - m1;
}

inline float lm_dequant_q5_K(device const uchar *row, uint col) {
    device const uchar *b = row + (col >> 8) * 176;
    float d = lm_f16_at(b);
    float dmin = lm_f16_at(b + 2);
    device const uchar *scales = b + 4;
    device const uchar *qh = b + 16;
    device const uchar *qs = b + 48;
    uint ib = col & 255;
    uint g = ib >> 6;             // 64-element group; nibbles like Q4_K
    uint hi_half = (ib >> 5) & 1; // low or high nibbles of the group
    uint l = ib & 31;
    uint sc, mn;
    lm_scale_min_k4(ib >> 5, scales, sc, mn);
    float d1 = d * (float)sc;
    float m1 = dmin * (float)mn;
    uchar byte = qs[g * 32 + l];
    uint nib = hi_half ? (byte >> 4) : (byte & 0xF);
    uint hi = ((qh[l] >> (2 * g + hi_half)) & 1) << 4; // the fifth bit
    return d1 * (float)(nib + hi) - m1;
}

inline float lm_dequant_q6_K(device const uchar *row, uint col) {
    device const uchar *b = row + (col >> 8) * 210;
    uint ib = col & 255;
    uint h = ib >> 7; // which 128-element half
    uint r = ib & 127;
    uint grp = r >> 5; // the reference's q1..q4 lanes
    uint l = r & 31;
    device const uchar *ql = b + h * 64;
    device const uchar *qh = b + 128 + h * 32;
    device const char *sc = (device const char *)(b + 192) + h * 8;
    float d = lm_f16_at(b + 208);
    uchar lowbyte = ql[l + (grp & 1) * 32];
    uint low = (grp < 2) ? (lowbyte & 0xF) : (lowbyte >> 4);
    uint hi2 = (qh[l] >> (2 * grp)) & 3;
    int q = (int)(low | (hi2 << 4)) - 32;
    return d * (float)sc[(l >> 4) + 2 * grp] * (float)q;
}

inline float lm_dequant(device const uchar *row, uint col) {
    switch (LM_W_QTYPE) {
        case 2: return lm_dequant_q8_0(row, col);
        case 3: return lm_dequant_q4_0(row, col);
        case 4: return lm_dequant_q4_K(row, col);
        case 5: return lm_dequant_q6_K(row, col);
        case 6: return lm_dequant_q5_K(row, col);
        default: return 0.0f;
    }
}

// The oracle gate's kernel: dequantize whole rows through the SAME inline
// functions the matvec/matmul paths use, so the bit-for-bit comparison against
// dequant_row_ref covers the production math.
struct LmDeqParams {
    uint cols;
    uint row_bytes;
    uint n_rows;
};

kernel void lm_dequant_oracle(
    device const uchar *src [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant LmDeqParams &p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) // x = column, y = row
{
    if (gid.x >= p.cols || gid.y >= p.n_rows) {
        return;
    }
    out[(ulong)gid.y * p.cols + gid.x] =
        lm_dequant(src + (ulong)gid.y * p.row_bytes, gid.x);
}


// ---------- matvec: y = W·x + bias (math.rs::matvec) — used during decode ----------
// One simdgroup (32 threads executing in lockstep) owns one row of W: adjacent
// threads read adjacent elements (coalesced), then combine with simd_sum, a
// hardware reduction that never touches memory.
//
// The dot products load half4/float4 vectors (8/16 bytes per instruction) — decode is
// memory-bandwidth-bound, and wide loads are what gets a small kernel near peak
// bandwidth. A scalar tail handles in_dim % 4 (zero for every supported model).

// The matvec family's weight read, switched by LM_W_QTYPE (folds at pipeline
// build). `w` is bound as half* but is really: staged f16 (0), raw bf16 over
// the mmap view (1 — values round through f16 so results are bit-identical to
// the staged path), or raw GGUF quant blocks (2..5 — dequant to f32 through
// the SAME lm_dequant_* the oracle gate verifies; those pipelines compile in
// the precise fast-math-off library so the gate's bit equality holds here too).
// Rows of quant blocks aren't element-addressable, so the row index comes in
// and each arm does its own row arithmetic.
inline ulong lm_row_bytes(uint in_dim) {
    switch (LM_W_QTYPE) {
        case 2: return (ulong)(in_dim / 32) * 34;   // Q8_0
        case 3: return (ulong)(in_dim / 32) * 18;   // Q4_0
        case 4: return (ulong)(in_dim / 256) * 144; // Q4_K
        case 5: return (ulong)(in_dim / 256) * 210; // Q6_K
        case 6: return (ulong)(in_dim / 256) * 176; // Q5_K
        default: return 0; // unused for element-addressable types
    }
}

inline float dot_wx(device const half *w, uint row, device const float *x, uint in_dim, uint lane) {
    float acc = 0.0f;
    if (LM_W_QTYPE >= 2) {
        device const uchar *row_base =
            (device const uchar *)w + (ulong)row * lm_row_bytes(in_dim);
        for (uint i = lane; i < in_dim; i += 32) {
            acc += lm_dequant(row_base, i) * x[i];
        }
        return acc;
    }
    device const half *w_row = w + (ulong)row * in_dim;
    if (LM_W_QTYPE == 1) {
        // ushort2 keeps 4-byte alignment for any 4-aligned tensor offset
        // (the host falls back to staging when a span is odder than that).
        device const ushort2 *w2 = (device const ushort2 *)w_row;
        device const float2 *x2 = (device const float2 *)x;
        uint n2 = in_dim / 2;
#pragma clang loop unroll_count(4)
        for (uint i = lane; i < n2; i += 32) {
            ushort2 wb = w2[i];
            half2 h = half2((half)as_type<float>((uint)wb.x << 16),
                            (half)as_type<float>((uint)wb.y << 16));
            acc += dot(float2(h), x2[i]);
        }
        for (uint i = n2 * 2 + lane; i < in_dim; i += 32) {
            device const ushort *wu = (device const ushort *)w_row;
            acc += (float)(half)as_type<float>((uint)wu[i] << 16) * x[i];
        }
        return acc;
    }
    device const half4 *w4 = (device const half4 *)w_row;
    device const float4 *x4 = (device const float4 *)x;
    uint n4 = in_dim / 4;
    // Partially unrolled so the scheduler can keep several loads in flight —
    // this loop is where decode spends its memory-bound time.
#pragma clang loop unroll_count(4)
    for (uint i = lane; i < n4; i += 32) {
        acc += dot(float4(w4[i]), x4[i]);
    }
    for (uint i = n4 * 4 + lane; i < in_dim; i += 32) {
        acc += (float)w_row[i] * x[i];
    }
    return acc;
}

struct MatvecParams {
    uint in_dim;
    uint out_dim;
};

kernel void matvec(
    device const half *w [[buffer(0)]],
    device const half *bias [[buffer(1)]], // models without biases get an all-zero buffer (free add, no branch)
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant MatvecParams &p [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint sg_per_tg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint row = tgid * sg_per_tg + sgid;
    if (row >= p.out_dim) {
        return;
    }
    float sum = simd_sum(dot_wx(w, row, x, p.in_dim, lane));
    if (lane == 0) {
        y[row] = sum + (float)bias[row];
    }
}

// matvec with the residual connection fused: y[row] += W·x + bias. Writing straight
// into the residual stream saves a separate add_inplace dispatch (and one buffer
// round-trip) for o_proj and down_proj during decode.
kernel void matvec_acc(
    device const half *w [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant MatvecParams &p [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint sg_per_tg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint row = tgid * sg_per_tg + sgid;
    if (row >= p.out_dim) {
        return;
    }
    float sum = simd_sum(dot_wx(w, row, x, p.in_dim, lane));
    if (lane == 0) {
        y[row] += sum + (float)bias[row];
    }
}

// The whole SwiGLU inner step in one dispatch: y[row] = silu(Wg·x) * (Wu·x).
// Replaces three dispatches (gate matvec, up matvec, silu_mul) during decode;
// the weight traffic is identical, the launches and the intermediate buffers go away.
kernel void matvec_swiglu(
    device const half *w_gate [[buffer(0)]],
    device const half *w_up [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant MatvecParams &p [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint sg_per_tg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint row = tgid * sg_per_tg + sgid;
    if (row >= p.out_dim) {
        return;
    }
    float g = simd_sum(dot_wx(w_gate, row, x, p.in_dim, lane));
    float u = simd_sum(dot_wx(w_up, row, x, p.in_dim, lane));
    if (lane == 0) {
        y[row] = (g / (1.0f + exp(-g))) * u; // silu(g) * u — gate/up have no bias in this family
    }
}

// q, k, v projections in one dispatch, with k and v written straight into their cache
// slot (kv_off elements in). Row layout: [0, q_dim) → q, then k, then v.
// Replaces three matvec dispatches during decode.
struct QkvParams {
    uint in_dim;
    uint q_dim;   // n_heads * head_dim
    uint kv_dim;  // n_kv_heads * head_dim
    uint kv_off;  // element offset of this token's cache slot: pos * kv_dim
};

kernel void matvec_qkv(
    device const half *w_q [[buffer(0)]],
    device const half *b_q [[buffer(1)]],
    device const half *w_k [[buffer(2)]],
    device const half *b_k [[buffer(3)]],
    device const half *w_v [[buffer(4)]],
    device const half *b_v [[buffer(5)]],
    device const float *x [[buffer(6)]],
    device float *q [[buffer(7)]],
    device half *k_cache [[buffer(8)]],
    device half *v_cache [[buffer(9)]],
    constant QkvParams &p [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint sg_per_tg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint row = tgid * sg_per_tg + sgid;
    if (row >= p.q_dim + 2 * p.kv_dim) {
        return;
    }
    device const half *w;
    device const half *bias;
    uint r;
    if (row < p.q_dim) {
        w = w_q; bias = b_q; r = row;
    } else if (row < p.q_dim + p.kv_dim) {
        r = row - p.q_dim;
        w = w_k; bias = b_k;
    } else {
        r = row - p.q_dim - p.kv_dim;
        w = w_v; bias = b_v;
    }
    float sum = simd_sum(dot_wx(w, r, x, p.in_dim, lane));
    if (lane == 0) {
        float val = sum + (float)bias[r];
        if (row < p.q_dim) {
            q[r] = val; // activations stay f32
        } else if (row < p.q_dim + p.kv_dim) {
            k_cache[p.kv_off + r] = (half)val; // the cache is f16
        } else {
            v_cache[p.kv_off + r] = (half)val;
        }
    }
}

// ---------- matmul: Y = X·Wᵀ + bias — used during prefill (many tokens at once) ----------
// Why batch prefill is fast: matvec re-reads all of W for *every* token, while matmul
// tiles the work and stages tiles in threadgroup memory (on-chip SRAM, far faster than
// device memory) — one tile of W is read from device memory once and reused by every
// token in the tile.
//
// The multiply itself runs on the GPU's simdgroup matrix hardware
// (simdgroup_multiply_accumulate on 8×8 blocks) instead of scalar FMAs: 128 threads =
// 4 simdgroups, each owning one 8-token × 8-output subtile of the 8×32 output tile.
// Both tiles are staged as f32 (weights converted from f16 on the way in), so the
// accumulation precision is unchanged — only the compute engine differs.

#define MM_TM 32  // tokens per tile
#define MM_TN 64  // outputs per tile
#define MM_TK 32  // k-dimension slice staged per iteration
#define MM_THREADS 128

// Both operands are staged into threadgroup memory PRE-SWIZZLED into contiguous
// 8x8 blocks (the W blocks additionally element-transposed), so every
// simdgroup_load below is a dense stride-8 read with no transpose flag — the
// layout llama.cpp's mul_mm kernel proved out on this hardware. Each simdgroup
// owns a 16-token x 32-output quadrant as a 2x4 block outer product (8 f32
// accumulators fed by half operands), and each K slice costs ONE threadgroup
// barrier. W rows are read once per 32-token tile.
//
// Staging layout: W tile sa = 8 k-major 4KB... (see index math at the stores);
// X tile sb likewise; Cs is the f32 writeback staging (adds bias + guards).

struct MatmulParams {
    uint in_dim;
    uint out_dim;
    uint n_rows;
};

// NOTE: matmul_pg below is this kernel plus a y_stride — a fix or improvement
// in one belongs in both, or lowmem numerics silently drift from metal's.
kernel void matmul(
    device const half *w [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant MatmulParams &p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]], // x = output tile, y = token tile
    uint2 tpos [[thread_position_in_threadgroup]]) // (MSL requires matching vector widths)
{
    uint tid = tpos.x;
    uint sgid = tid / 32;
    uint out0 = tgid.x * MM_TN;
    uint row0 = tgid.y * MM_TM;

    // One 8 KB shared block, aliased by phase — threadgroup footprint bounds how
    // many threadgroups a core can host, and 8 KB keeps 4 in flight:
    //   loop:      sa (W tile, half, 4 KB) | sb (X tile, half, 2 KB) | biasTile (2 KB)
    //   writeback: Cs (f32 staging, 8 KB) — partial tiles only; full tiles store
    //              straight to device from the accumulators.
    threadgroup float shared_[2048];
    threadgroup half *sa = (threadgroup half *)shared_;
    threadgroup half *sb = (threadgroup half *)(shared_ + 1024);
    threadgroup float *biasTile = shared_ + 1536;
    threadgroup float *Cs = shared_;

    // Bias rides in the accumulators from the start (one 8-row replicated tile,
    // loaded per block) — no separate bias pass, no staging on the way out.
    for (uint idx = tid; idx < 8 * MM_TN; idx += MM_THREADS) {
        uint o = idx % MM_TN;
        biasTile[idx] = (out0 + o < p.out_dim) ? (float)bias[out0 + o] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8; i++) {
        simdgroup_load(mc[i], biasTile + (sgid % 2) * 32 + (i % 4) * 8, MM_TN);
    }

    uint w_row = tid / 2;   // local W row 0..63
    uint w_strip = tid % 2; // 16-wide k strip
    uint x_row = tid / 4;   // local token 0..31
    uint x_blk = tid % 4;   // 8-wide k block

    for (uint k0 = 0; k0 < p.in_dim; k0 += MM_TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 16; i++) {
            uint gk = k0 + w_strip * 16 + i;
            uint go = out0 + w_row;
            half v = (go < p.out_dim && gk < p.in_dim) ? w[(ulong)go * p.in_dim + gk] : 0.0h;
            uint ib = 8 * (2 * w_strip + i / 8) + w_row / 8; // (k block, row block)
            sa[64 * ib + 8 * (i % 8) + w_row % 8] = v;       // in-block transpose: [k][row]
        }
        for (uint i = 0; i < 8; i++) {
            uint gk = k0 + x_blk * 8 + i;
            uint gr = row0 + x_row;
            half v = (gr < p.n_rows && gk < p.in_dim) ? (half)x[(ulong)gr * p.in_dim + gk] : 0.0h;
            uint ib = 4 * x_blk + x_row / 8;
            sb[64 * ib + 8 * (x_row % 8) + i] = v;           // row-major: [token][k]
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2x4 block outer product per simdgroup: sgid%2 picks the output half,
        // sgid/2 the token half; walk the 4 k blocks of the slice.
        threadgroup const half *lsma = sa + 4 * 64 * (sgid % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (sgid / 2);
        for (uint ik = 0; ik < MM_TK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (row0 + MM_TM <= p.n_rows && out0 + MM_TN <= p.out_dim) {
        // Full tile: store the accumulators straight to device memory.
        for (uint i = 0; i < 8; i++) {
            ulong gr = row0 + (sgid / 2) * 16 + (i / 4) * 8;
            ulong go = out0 + (sgid % 2) * 32 + (i % 4) * 8;
            simdgroup_store(mc[i], y + gr * p.out_dim + go, p.out_dim);
        }
    } else {
        // Edge tile: stage and copy with guards (bias already accumulated).
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 8; i++) {
            uint t0 = (sgid / 2) * 16 + (i / 4) * 8;
            uint o0 = (sgid % 2) * 32 + (i % 4) * 8;
            simdgroup_store(mc[i], Cs + t0 * MM_TN + o0, MM_TN);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < MM_TM * MM_TN; idx += MM_THREADS) {
            uint m = idx / MM_TN;
            uint n = idx % MM_TN;
            uint gr = row0 + m;
            uint go = out0 + n;
            if (gr < p.n_rows && go < p.out_dim) {
                y[(ulong)gr * p.out_dim + go] = Cs[m * MM_TN + n];
            }
        }
    }
}

// Paged-tensor matmul (-b lowmem): identical algorithm to `matmul` above (keep
// them in sync — that kernel is the source of truth), with one difference: the
// weight buffer holds only a ROW BLOCK of the full tensor (a pool page), so the
// output columns this dispatch produces land inside a wider row. `out_dim` is
// the block's row count (guards + W indexing), `y_stride` the full output width;
// the y/bias buffers are bound with a byte offset selecting the block's columns.
struct MatmulPagedParams {
    uint in_dim;
    uint out_dim;   // rows in this weight block = output columns produced here
    uint n_rows;
    uint y_stride;  // full out_dim of the logical tensor
};

kernel void matmul_pg(
    device const half *w [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant MatmulPagedParams &p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]])
{
    uint tid = tpos.x;
    uint sgid = tid / 32;
    uint out0 = tgid.x * MM_TN;
    uint row0 = tgid.y * MM_TM;

    threadgroup float shared_[2048];
    threadgroup half *sa = (threadgroup half *)shared_;
    threadgroup half *sb = (threadgroup half *)(shared_ + 1024);
    threadgroup float *biasTile = shared_ + 1536;
    threadgroup float *Cs = shared_;

    for (uint idx = tid; idx < 8 * MM_TN; idx += MM_THREADS) {
        uint o = idx % MM_TN;
        biasTile[idx] = (out0 + o < p.out_dim) ? (float)bias[out0 + o] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8; i++) {
        simdgroup_load(mc[i], biasTile + (sgid % 2) * 32 + (i % 4) * 8, MM_TN);
    }

    uint w_row = tid / 2;
    uint w_strip = tid % 2;
    uint x_row = tid / 4;
    uint x_blk = tid % 4;

    for (uint k0 = 0; k0 < p.in_dim; k0 += MM_TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 16; i++) {
            uint gk = k0 + w_strip * 16 + i;
            uint go = out0 + w_row;
            // Quant specializations dequant the tile on the way into the f16
            // staging (a GEMM re-reads each weight tile per 32-token slice, so
            // staged-once f16 wins over per-read dequant; prefill numerics are
            // therefore f16-rounded quant values — logged in the lane notes).
            half v = 0.0h;
            if (go < p.out_dim && gk < p.in_dim) {
                v = (LM_W_QTYPE >= 2)
                    ? (half)lm_dequant((device const uchar *)w + (ulong)go * lm_row_bytes(p.in_dim), gk)
                    : w[(ulong)go * p.in_dim + gk];
            }
            uint ib = 8 * (2 * w_strip + i / 8) + w_row / 8;
            sa[64 * ib + 8 * (i % 8) + w_row % 8] = v;
        }
        for (uint i = 0; i < 8; i++) {
            uint gk = k0 + x_blk * 8 + i;
            uint gr = row0 + x_row;
            half v = (gr < p.n_rows && gk < p.in_dim) ? (half)x[(ulong)gr * p.in_dim + gk] : 0.0h;
            uint ib = 4 * x_blk + x_row / 8;
            sb[64 * ib + 8 * (x_row % 8) + i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4 * 64 * (sgid % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (sgid / 2);
        for (uint ik = 0; ik < MM_TK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (row0 + MM_TM <= p.n_rows && out0 + MM_TN <= p.out_dim) {
        for (uint i = 0; i < 8; i++) {
            ulong gr = row0 + (sgid / 2) * 16 + (i / 4) * 8;
            ulong go = out0 + (sgid % 2) * 32 + (i % 4) * 8;
            simdgroup_store(mc[i], y + gr * p.y_stride + go, p.y_stride);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 8; i++) {
            uint t0 = (sgid / 2) * 16 + (i / 4) * 8;
            uint o0 = (sgid % 2) * 32 + (i % 4) * 8;
            simdgroup_store(mc[i], Cs + t0 * MM_TN + o0, MM_TN);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < MM_TM * MM_TN; idx += MM_THREADS) {
            uint m = idx / MM_TN;
            uint n = idx % MM_TN;
            uint gr = row0 + m;
            uint go = out0 + n;
            if (gr < p.n_rows && go < p.out_dim) {
                y[(ulong)gr * p.y_stride + go] = Cs[m * MM_TN + n];
            }
        }
    }
}

// ---------- tensor-ops matmul: the Metal 4 path (macOS 26) ----------
// mpp::tensor_ops::matmul2d drives the same tensor hardware llama.cpp's mul_mm
// uses on this OS — measured 6x faster than any hand-tiled simdgroup variant we
// tried on the MLP GEMMs. Operands are half (X converted by f32_to_f16 below),
// the destination accumulates in f32 device memory directly, and the op does its
// own edge checking against the real extents. Tensors are built in-kernel from
// plain buffer pointers, so the host side binds ordinary buffers.

// Token-rows per tensor-ops matmul tile; injected by the Rust side (metal.rs
// MM_TILE_ROWS) so kernel and dispatch stay in sync.
#ifndef MM_TROWS
#define MM_TROWS 32
#endif

kernel void matmul_t(
    device const half *w [[buffer(0)]],
    device const half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant MatmulParams &p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], // x = output tile (64), y = token tile (MM_TROWS)
    uint2 tpos [[thread_position_in_threadgroup]])
{
    (void)tpos;
    // Cast away const: the tensor-ops templates only accept non-const elements.
    auto tA = tensor((device half *)x, dextents<int32_t, 2>(p.in_dim, p.n_rows), array<int, 2>({1, (int)p.in_dim}));
    auto tB = tensor((device half *)w, dextents<int32_t, 2>(p.in_dim, p.out_dim), array<int, 2>({1, (int)p.in_dim}));
    auto tC = tensor(y, dextents<int32_t, 2>(p.out_dim, p.n_rows), array<int, 2>({1, (int)p.out_dim}));

    // A (tokens over k, NN) x B^T (W stored [out][k] = NT right operand) -> C.
    // f32 accumulation lives in the cooperative destination tensor (registers);
    // store() writes the f32 result with edge checking against C's real extents.
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        MM_TROWS, 64, static_cast<int>(dynamic_extent), false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;

    auto mA = tA.slice(0, (int)(tgid.y * MM_TROWS));
    auto mB = tB.slice(0, (int)(tgid.x * 64));
    auto mC = tC.slice((int)(tgid.x * 64), (int)(tgid.y * MM_TROWS));
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT);
    cT.store(mC);
}

// Tensor-ops matmul with a half destination — the prefill k/v projections, whose
// output IS the KV cache. Same operands as matmul_t (the rmsnorm half copy is
// already staged when the projections run); the f32 accumulator is staged through
// threadgroup memory so bias-add and the half rounding happen in one step, exactly
// like matmul_h's epilogue (a direct half store would double-round biased layers).
kernel void matmul_th(
    device const half *w [[buffer(0)]],
    device const half *x [[buffer(1)]],
    device half *y [[buffer(2)]],
    device const half *bias [[buffer(3)]],
    constant MatmulParams &p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]], // x = output tile (64), y = token tile (32)
    uint2 tpos [[thread_position_in_threadgroup]])
{
    uint tid = tpos.x;
    auto tA = tensor((device half *)x, dextents<int32_t, 2>(p.in_dim, p.n_rows), array<int, 2>({1, (int)p.in_dim}));
    auto tB = tensor((device half *)w, dextents<int32_t, 2>(p.in_dim, p.out_dim), array<int, 2>({1, (int)p.in_dim}));

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        MM_TROWS, 64, static_cast<int>(dynamic_extent), false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;

    auto mA = tA.slice(0, (int)(tgid.y * MM_TROWS));
    auto mB = tB.slice(0, (int)(tgid.x * 64));
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT);

    threadgroup float Cs[MM_TROWS * 64];
    auto tC = tensor((threadgroup float *)Cs, dextents<int32_t, 2>(64, MM_TROWS), array<int, 2>({1, 64}));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < MM_TROWS * 64; idx += 128) {
        uint m = idx / 64;
        uint n = idx % 64;
        uint gr = tgid.y * MM_TROWS + m;
        uint go = tgid.x * 64 + n;
        if (gr < p.n_rows && go < p.out_dim) {
            y[(ulong)gr * p.out_dim + go] = (half)(Cs[m * 64 + n] + (float)bias[go]);
        }
    }
}

// matmul_t with the bias folded into the store epilogue — Qwen's q projection.
// Value-identical to matmul_t followed by bias_add (the f32 accumulator is
// stored, then bias added in f32, same operations in the same precision), but
// one dispatch and one barrier cheaper per biased layer.
kernel void matmul_tb(
    device const half *w [[buffer(0)]],
    device const half *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    device const half *bias [[buffer(3)]],
    constant MatmulParams &p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]])
{
    uint tid = tpos.x;
    auto tA = tensor((device half *)x, dextents<int32_t, 2>(p.in_dim, p.n_rows), array<int, 2>({1, (int)p.in_dim}));
    auto tB = tensor((device half *)w, dextents<int32_t, 2>(p.in_dim, p.out_dim), array<int, 2>({1, (int)p.in_dim}));

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        MM_TROWS, 64, static_cast<int>(dynamic_extent), false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;

    auto mA = tA.slice(0, (int)(tgid.y * MM_TROWS));
    auto mB = tB.slice(0, (int)(tgid.x * 64));
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT);

    threadgroup float Cs[MM_TROWS * 64];
    auto tC = tensor((threadgroup float *)Cs, dextents<int32_t, 2>(64, MM_TROWS), array<int, 2>({1, 64}));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < MM_TROWS * 64; idx += 128) {
        uint m = idx / 64;
        uint n = idx % 64;
        uint gr = tgid.y * MM_TROWS + m;
        uint go = tgid.x * 64 + n;
        if (gr < p.n_rows && go < p.out_dim) {
            y[(ulong)gr * p.out_dim + go] = Cs[m * 64 + n] + (float)bias[go];
        }
    }
}

// f32 activations -> half staging for the tensor-ops matmul operands.
kernel void f32_to_f16(
    device const float *x [[buffer(0)]],
    device half *y [[buffer(1)]],
    constant uint &dim [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < dim) {
        y[gid] = (half)x[gid];
    }
}

// y[row][o] += bias[o] — only dispatched for layers that actually carry a bias
// (Qwen's q projection on the tensor-ops path; k/v go through matmul_h).
kernel void bias_add(
    device float *y [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    constant MatmulParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n_rows * p.out_dim) {
        y[gid] += (float)bias[gid % p.out_dim];
    }
}

// Same tiling, half output — prefill's k/v projections land straight in the cache.
kernel void matmul_h(
    device const half *w [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device half *y [[buffer(3)]],
    constant MatmulParams &p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]])
{
    uint tid = tpos.x;
    uint sgid = tid / 32;
    uint out0 = tgid.x * MM_TN;
    uint row0 = tgid.y * MM_TM;

    threadgroup half sa[MM_TN * MM_TK];
    threadgroup half sb[MM_TM * MM_TK];
    threadgroup float Cs[MM_TM][MM_TN];

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    uint w_row = tid / 2;
    uint w_strip = tid % 2;
    uint x_row = tid / 4;
    uint x_blk = tid % 4;

    for (uint k0 = 0; k0 < p.in_dim; k0 += MM_TK) {
        for (uint i = 0; i < 16; i++) {
            uint gk = k0 + w_strip * 16 + i;
            uint go = out0 + w_row;
            half v = (go < p.out_dim && gk < p.in_dim) ? w[(ulong)go * p.in_dim + gk] : 0.0h;
            uint ib = 8 * (2 * w_strip + i / 8) + w_row / 8;
            sa[64 * ib + 8 * (i % 8) + w_row % 8] = v;
        }
        for (uint i = 0; i < 8; i++) {
            uint gk = k0 + x_blk * 8 + i;
            uint gr = row0 + x_row;
            half v = (gr < p.n_rows && gk < p.in_dim) ? (half)x[(ulong)gr * p.in_dim + gk] : 0.0h;
            uint ib = 4 * x_blk + x_row / 8;
            sb[64 * ib + 8 * (x_row % 8) + i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4 * 64 * (sgid % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (sgid / 2);
        for (uint ik = 0; ik < MM_TK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, ulong2(0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, ulong2(0), false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (uint i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 8; i++) {
        uint t0 = (sgid / 2) * 16 + (i / 4) * 8;
        uint o0 = (sgid % 2) * 32 + (i % 4) * 8;
        simdgroup_store(mc[i], &Cs[t0][o0], MM_TN);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < MM_TM * MM_TN; idx += MM_THREADS) {
        uint m = idx / MM_TN;
        uint n = idx % MM_TN;
        uint gr = row0 + m;
        uint go = out0 + n;
        if (gr < p.n_rows && go < p.out_dim) {
            y[(ulong)gr * p.out_dim + go] = (half)(Cs[m][n] + (float)bias[go]);
        }
    }
}

// matvec writing f16 — the k/v projections on the rare non-fused decode path.
kernel void matvec_h(
    device const half *w [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device half *y [[buffer(3)]],
    constant MatvecParams &p [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint sg_per_tg [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    uint row = tgid * sg_per_tg + sgid;
    if (row >= p.out_dim) {
        return;
    }
    float sum = simd_sum(dot_wx(w, row, x, p.in_dim, lane));
    if (lane == 0) {
        y[row] = (half)(sum + (float)bias[row]);
    }
}

// ---------- rmsnorm (math.rs::rmsnorm) ----------
// One threadgroup per row (token). Reduction is simdgroup-first: each 32-thread
// simdgroup collapses its partial sum in registers (simd_sum, no memory), only the
// per-simdgroup results touch threadgroup memory — 2 barriers instead of log2(256).

#define NORM_TG 256

struct NormParams {
    uint dim;
    float eps;
};

kernel void rmsnorm(
    device const float *x [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant NormParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    device const float *xr = x + (ulong)row * p.dim;
    device float *yr = y + (ulong)row * p.dim;

    threadgroup float partial[NORM_TG / 32];

    float acc = 0.0f;
    for (uint i = tid; i < p.dim; i += NORM_TG) {
        acc += xr[i] * xr[i];
    }
    float sg = simd_sum(acc);
    if (tid % 32 == 0) {
        partial[tid / 32] = sg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint j = 0; j < NORM_TG / 32; j++) {
            total += partial[j];
        }
        partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(partial[0] / (float)p.dim + p.eps);
    for (uint i = tid; i < p.dim; i += NORM_TG) {
        yr[i] = xr[i] * scale * (float)weight[i];
    }
}

// rmsnorm variant for the prefill path: same math, but the normalized row is
// written twice — f32 for the residual pipeline, half for the tensor-ops
// matmul operands (saves a separate conversion dispatch per use).
kernel void rmsnorm_hf(
    device const float *x [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device float *y [[buffer(2)]],
    device half *y_h [[buffer(3)]],
    constant NormParams &p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    device const float *xr = x + (ulong)row * p.dim;
    device float *yr = y + (ulong)row * p.dim;
    device half *yh = y_h + (ulong)row * p.dim;

    threadgroup float partial[NORM_TG / 32];

    float acc = 0.0f;
    for (uint i = tid; i < p.dim; i += NORM_TG) {
        acc += xr[i] * xr[i];
    }
    float sg = simd_sum(acc);
    if (tid % 32 == 0) {
        partial[tid / 32] = sg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint j = 0; j < NORM_TG / 32; j++) {
            total += partial[j];
        }
        partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(partial[0] / (float)p.dim + p.eps);
    for (uint i = tid; i < p.dim; i += NORM_TG) {
        float v = xr[i] * scale * (float)weight[i];
        yr[i] = v;
        yh[i] = (half)v;
    }
}

// ---------- RoPE (model.rs::rope) ----------
// One thread rotates one dimension pair (i, i+half) of one head of one row —
// fully independent work. Row r sits at real position pos0 + r
// (prefill: pos0 = chunk start, decode: pos0 = pos).

struct RopeParams {
    uint head_dim;
    uint n_heads;
    uint pos0;
    float theta;
    uint n_rows;
};

kernel void rope(
    device float *x [[buffer(0)]],
    constant RopeParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_dim = p.head_dim / 2;
    uint per_row = p.n_heads * half_dim;
    if (gid >= p.n_rows * per_row) {
        return;
    }
    uint row = gid / per_row;
    uint h = (gid % per_row) / half_dim;
    uint i = gid % half_dim;

    float freq = pow(p.theta, -2.0f * (float)i / (float)p.head_dim);
    float angle = (float)(p.pos0 + row) * freq;
    float c;
    float s = sincos(angle, c); // one intrinsic for both — cheaper than separate sin/cos
    device float *head = x + (ulong)row * p.n_heads * p.head_dim + h * p.head_dim;
    float a = head[i];
    float b = head[i + half_dim];
    head[i] = a * c - b * s;
    head[i + half_dim] = a * s + b * c;
}

// Both prefill rotations in one launch: q (f32) and the freshly written k rows
// (f16, in the KV cache). The per-element math is identical to rope/rope_h —
// the fusion only removes a dispatch and a pipeline switch per layer, which the
// qwen-level-gap profile measured at ~11 ms per 512-token chunk for the pair.
struct RopeQkPrefillParams {
    uint head_dim;
    uint n_q_heads;
    uint n_kv_heads;
    uint pos0;
    float theta;
    uint n_rows;
};

kernel void rope_qk_prefill(
    device float *q [[buffer(0)]],
    device half *k [[buffer(1)]],
    constant RopeQkPrefillParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_dim = p.head_dim / 2;
    uint q_per_row = p.n_q_heads * half_dim;
    uint k_per_row = p.n_kv_heads * half_dim;
    uint q_total = p.n_rows * q_per_row;
    bool is_q = gid < q_total;
    uint idx = is_q ? gid : gid - q_total;
    uint per_row = is_q ? q_per_row : k_per_row;
    if (!is_q && idx >= p.n_rows * k_per_row) {
        return;
    }
    uint row = idx / per_row;
    uint h = (idx % per_row) / half_dim;
    uint i = idx % half_dim;

    float freq = pow(p.theta, -2.0f * (float)i / (float)p.head_dim);
    float angle = (float)(p.pos0 + row) * freq;
    float c;
    float s = sincos(angle, c);
    if (is_q) {
        device float *head = q + (ulong)row * p.n_q_heads * p.head_dim + h * p.head_dim;
        float a = head[i];
        float b = head[i + half_dim];
        head[i] = a * c - b * s;
        head[i + half_dim] = a * s + b * c;
    } else {
        device half *head = k + (ulong)row * p.n_kv_heads * p.head_dim + h * p.head_dim;
        float a = head[i];
        float b = head[i + half_dim];
        head[i] = (half)(a * c - b * s);
        head[i + half_dim] = (half)(a * s + b * c);
    }
}

// Same rotation on an f16 buffer — prefill's freshly written k rows in the KV cache.
kernel void rope_h(
    device half *x [[buffer(0)]],
    constant RopeParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_dim = p.head_dim / 2;
    uint per_row = p.n_heads * half_dim;
    if (gid >= p.n_rows * per_row) {
        return;
    }
    uint row = gid / per_row;
    uint h = (gid % per_row) / half_dim;
    uint i = gid % half_dim;

    float freq = pow(p.theta, -2.0f * (float)i / (float)p.head_dim);
    float angle = (float)(p.pos0 + row) * freq;
    float c;
    float s = sincos(angle, c); // one intrinsic for both — cheaper than separate sin/cos
    device half *head = x + (ulong)row * p.n_heads * p.head_dim + h * p.head_dim;
    float a = head[i];
    float b = head[i + half_dim];
    head[i] = (half)(a * c - b * s);
    head[i + half_dim] = (half)(a * s + b * c);
}

// Decode-only RoPE: rotate q (n_q_heads) and this position's new k row (n_kv_heads)
// in a single dispatch — the grid covers the q pairs first, then the k pairs.
struct RopeQkParams {
    uint head_dim;
    uint n_q_heads;
    uint n_kv_heads;
    uint pos;
    float theta;
};

kernel void rope_qk_decode(
    device float *q [[buffer(0)]],
    device half *k [[buffer(1)]], // already offset to this position's (f16) cache row
    constant RopeQkParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_dim = p.head_dim / 2;
    uint q_pairs = p.n_q_heads * half_dim;
    if (gid >= q_pairs + p.n_kv_heads * half_dim) {
        return;
    }
    bool is_q = gid < q_pairs;
    uint idx = is_q ? gid : gid - q_pairs;
    uint h = idx / half_dim;
    uint i = idx % half_dim;

    float freq = pow(p.theta, -2.0f * (float)i / (float)p.head_dim);
    float angle = (float)p.pos * freq;
    float c;
    float s = sincos(angle, c); // one intrinsic for both — cheaper than separate sin/cos
    if (is_q) {
        device float *head = q + h * p.head_dim;
        float a = head[i];
        float b = head[i + half_dim];
        head[i] = a * c - b * s;
        head[i + half_dim] = a * s + b * c;
    } else {
        device half *head = k + h * p.head_dim;
        float a = head[i];
        float b = head[i + half_dim];
        head[i] = (half)(a * c - b * s);
        head[i + half_dim] = (half)(a * s + b * c);
    }
}

// ---------- windowed attention (-b lowmem), as function constants ----------
// The lowmem backend's KV store is a SINK region (slots [0, LM_SINKPAD), holding
// pinned positions 0..LM_SINK) followed by a RING of LM_RING slots holding the
// last window of positions: slot(p) = LM_SINKPAD + (p - LM_SINK) % LM_RING.
// Each query attends its last LM_WINDOW positions plus the sinks (StreamingLLM
// shape — sinks prevent the well-documented post-window collapse). The three
// attention kernels below gain windowed variants through these constants; left
// undefined (every existing metal/hybrid pipeline) the branches fold away and
// the kernels compile exactly as before.
constant uint LM_SINK [[function_constant(20)]];    // pinned sink tokens S
constant uint LM_SINKPAD [[function_constant(21)]]; // sink region width, 128-aligned
constant uint LM_RING [[function_constant(22)]];    // ring slots R, 128-aligned
constant uint LM_WINDOW [[function_constant(23)]];  // window W
constant bool LM_WINDOWED = is_function_constant_defined(LM_RING);

// Absolute position held by buffer slot `s` once `f` tokens exist (frontier —
// every position < f is written), or UINT_MAX for a slot holding nothing.
// The ring slot's position is the LATEST p < f with (p - LM_SINK) % LM_RING
// matching; padding slots between LM_SINK and LM_SINKPAD are never written.
inline uint lm_slot_pos(uint s, uint f) {
    if (s < LM_SINKPAD) {
        return (s < LM_SINK && s < f) ? s : 0xFFFFFFFFu;
    }
    uint rel = s - LM_SINKPAD;
    if (f <= LM_SINK + rel) {
        return 0xFFFFFFFFu;
    }
    return LM_SINK + rel + ((f - 1 - LM_SINK - rel) / LM_RING) * LM_RING;
}

// May the query at position qp attend position t? (UINT_MAX fails t <= qp.)
inline bool lm_attend(uint t, uint qp) {
    return t <= qp && (t + LM_WINDOW > qp || t < LM_SINK);
}

// ---------- attention (model.rs::attention) ----------
// One threadgroup per (query head, query row): the same three phases as the CPU code —
// scores → softmax → weighted sum of V — with barriers making each phase's results
// visible before the next. The causal mask falls out of the loop bound: the row at
// position q_pos only iterates over 0..=q_pos. The windowed variant walks the
// bounded slot buffer instead and masks by reconstructed position.

#define ATTN_TG 256

struct AttnParams {
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint pos0;
    uint max_seq;
    uint n_rows; // chunk rows in this dispatch (used by the flash kernel's tile guards)
};

kernel void attention(
    device const float *q [[buffer(0)]],
    device const half *k_cache [[buffer(1)]],
    device const half *v_cache [[buffer(2)]],
    device float *scores [[buffer(3)]], // scratch: [n_rows × n_heads × max_seq]
    device float *out [[buffer(4)]],
    constant AttnParams &p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], // x = query head, y = query row within the chunk
    uint2 tpos [[thread_position_in_threadgroup]]) // (MSL requires matching vector widths)
{
    uint tid = tpos.x;
    uint head = tg.x;
    uint row = tg.y;
    uint q_pos = p.pos0 + row; // this query row's real position
    uint hd = p.head_dim;
    uint kvd = p.n_kv_heads * hd;
    uint kv_off = (head / (p.n_heads / p.n_kv_heads)) * hd; // GQA: which kv head this q head uses
    device const float *q_head = q + (ulong)row * p.n_heads * hd + head * hd;
    device float *sc = scores + ((ulong)row * p.n_heads + head) * p.max_seq;
    float scale = rsqrt((float)hd);

    // Reductions are simdgroup-first (simd_max/simd_sum in registers, then one tiny
    // combine) — a couple of barriers instead of a log2(256)-step tree.
    threadgroup float red[ATTN_TG / 32];

    // Phase 1: q·k score for every position 0..=q_pos (windowed: every slot of
    // the bounded store, masked by reconstructed position), tracking the max for
    // a stable exp. Wide loads when head_dim allows (every supported model).
    uint t_lim = LM_WINDOWED ? (LM_SINKPAD + LM_RING) : (q_pos + 1);
    uint frontier = p.pos0 + p.n_rows;
    bool vec4 = (hd % 4) == 0;
    float local_max = -INFINITY;
    for (uint t = tid; t < t_lim; t += ATTN_TG) {
        if (LM_WINDOWED && !lm_attend(lm_slot_pos(t, frontier), q_pos)) {
            sc[t] = -INFINITY;
            continue;
        }
        device const half *k_t = k_cache + (ulong)t * kvd + kv_off;
        float d = 0.0f;
        if (vec4) {
            device const half4 *k4 = (device const half4 *)k_t;
            device const float4 *q4 = (device const float4 *)q_head;
            for (uint i = 0; i < hd / 4; i++) {
                d += dot(q4[i], float4(k4[i]));
            }
        } else {
            for (uint i = 0; i < hd; i++) {
                d += q_head[i] * (float)k_t[i];
            }
        }
        d *= scale;
        sc[t] = d;
        local_max = max(local_max, d);
    }
    float sg_max = simd_max(local_max);
    if (tid % 32 == 0) {
        red[tid / 32] = sg_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (uint j = 0; j < ATTN_TG / 32; j++) {
            m = max(m, red[j]);
        }
        red[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float score_max = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup); // red[] is reused — wait until everyone read red[0]

    // Phase 2: exponentiate and sum (the softmax denominator). Masked slots
    // hold -inf and exponentiate to an exact 0 weight.
    float local_sum = 0.0f;
    for (uint t = tid; t < t_lim; t += ATTN_TG) {
        float e = exp(sc[t] - score_max);
        sc[t] = e;
        local_sum += e;
    }
    // sc[] lives in device memory and phase 3 reads it across threads — device-level barrier.
    threadgroup_barrier(mem_flags::mem_device);
    float sg_sum = simd_sum(local_sum);
    if (tid % 32 == 0) {
        red[tid / 32] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint j = 0; j < ATTN_TG / 32; j++) {
            total += red[j];
        }
        red[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float score_sum = red[0];

    // Phase 3: output = weighted average of v. Threads reshape as
    // (position lane × output dim) so every thread works — the position loop
    // splits ATTN_TG/head_dim ways instead of running serially per dim (this was
    // the long-prompt prefill bottleneck).
    threadgroup float acc_red[ATTN_TG];
    if (hd <= ATTN_TG && (ATTN_TG % hd) == 0) {
        uint pn = ATTN_TG / hd;
        uint pl = tid / hd;
        uint di = tid % hd;
        float acc = 0.0f;
        for (uint t = pl; t < t_lim; t += pn) {
            acc += sc[t] * (float)v_cache[(ulong)t * kvd + kv_off + di];
        }
        acc_red[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < hd) {
            float a = 0.0f;
            for (uint j = 0; j < pn; j++) {
                a += acc_red[j * hd + tid];
            }
            out[(ulong)row * p.n_heads * hd + head * hd + tid] = a / score_sum;
        }
    } else {
        // Fallback for exotic head sizes: one thread per output dimension.
        for (uint i = tid; i < hd; i += ATTN_TG) {
            float acc = 0.0f;
            for (uint t = 0; t < t_lim; t++) {
                acc += sc[t] * (float)v_cache[(ulong)t * kvd + kv_off + i];
            }
            out[(ulong)row * p.n_heads * hd + head * hd + i] = acc / score_sum;
        }
    }
}

// ---------- flash prefill attention (head_dim 64) ----------
// The kernel above materializes every row's scores to device memory and walks V
// serially per position lane — at prefill widths that is the measured bottleneck
// (6x vs llama.cpp's flash kernel on the same machine). This one is
// flash-attention-2 shaped: a threadgroup owns a 16-row query tile of ONE head,
// streams the K/V cache in 32-position tiles through threadgroup memory, runs
// QK^T and P·V on the simdgroup matrix units (f32 operands and accumulators —
// the same numerics class as the matmul tiles), and keeps the softmax ONLINE
// (running max/sum with tile-to-tile rescale), so scores never touch device
// memory and V is consumed by matrix hardware instead of a serial walk.
//
// Specialized to head_dim 64 (every target model); other head sizes take the
// kernel above via the Rust-side gate in enc_attention.

// One threadgroup owns FA_Q query rows of one head; the simdgroups repartition
// the work by PHASE instead of by row (the shape llama.cpp's fa kernel uses —
// MIT-licensed, studied for structure and reimplemented here): QK^T splits by
// score column, the softmax by row, P·V by output column. Q is tiny, the KV
// walk is wide, K/V load straight from the cache, and shared memory stays
// ~6 KB so several threadgroups share a core and hide the device-load latency.
// The previous 96-row staged shape amortized an explicit K/V copy instead;
// measured head-to-head it ties direct loads at high occupancy and loses at
// low, and per-phase splitting stops every simdgroup re-loading every K block.
#define FA_Q 8         // query rows per threadgroup
#define FA_C 64        // cached positions walked per iteration
#define FA_NSG 4       // simdgroups; must divide FA_C/8 and FA_HD/8 and FA_Q
#define FA_HD 64       // the specialized head_dim
#define FA_THREADS (FA_NSG * 32)
#define FA_S_F (FA_Q * FA_C)       // scores, f32
#define FA_P_F (FA_Q * FA_C / 2)   // exp weights, half
#define FA_O_F (FA_Q * FA_HD)      // output accumulator, f32
#define FA_QH_F (FA_Q * FA_HD / 2) // staged Q, half
#define FA_SH_TOTAL (FA_S_F + FA_P_F + FA_O_F + FA_QH_F + 2 * FA_Q)

kernel void attention_prefill_flash(
    device const float *q [[buffer(0)]],
    device const half *k_cache [[buffer(1)]],
    device const half *v_cache [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant AttnParams &p [[buffer(4)]],
    device half *out_h [[buffer(5)]], // half copy for o_proj's tensor-ops operand
    uint2 tg [[threadgroup_position_in_grid]], // x = query head, y = query row tile
    uint2 tpos [[thread_position_in_threadgroup]])
{
    uint tid = tpos.x;
    uint sgid = tid / 32;
    uint lane = tid % 32;
    uint head = tg.x;
    uint r0 = tg.y * FA_Q;
// The KV row stride as a compile-time immediate when the host injects it
// (llama.cpp does the same via function constants) — the loads in the hot
// loop then need no runtime stride math.
#ifdef FA_KVD
    const uint kvd = FA_KVD;
#else
    uint kvd = p.n_kv_heads * FA_HD;
#endif
    uint kv_off = (head / (p.n_heads / p.n_kv_heads)) * FA_HD;
    float scale = rsqrt((float)FA_HD);

    threadgroup float SH[FA_SH_TOTAL];
    threadgroup float *S = SH;
    threadgroup half *P = (threadgroup half *)(SH + FA_S_F);
    threadgroup float *O = SH + FA_S_F + FA_P_F;
    threadgroup half *Qh = (threadgroup half *)(SH + FA_S_F + FA_P_F + FA_O_F);
    threadgroup float *m_i = SH + FA_S_F + FA_P_F + FA_O_F + FA_QH_F;
    threadgroup float *l_i = m_i + FA_Q;

    // A tile's masked tail may read up to FA_C-1 rows past position max_seq-1;
    // the cache allocations carry that much slack (zero-filled — 0 x NaN would
    // be NaN even under a masked-out weight), and the mask keeps those values
    // out of every output.
    device const half *kb = k_cache + kv_off;
    device const half *vb = v_cache + kv_off;

    for (uint idx = tid; idx < FA_Q * FA_HD; idx += FA_THREADS) {
        uint r = idx / FA_HD;
        uint gr = r0 + r;
        Qh[idx] = (gr < p.n_rows) ? (half)q[(ulong)gr * p.n_heads * FA_HD + head * FA_HD + (idx % FA_HD)] : 0.0h;
        O[idx] = 0.0f;
    }
    if (tid < FA_Q) {
        m_i[tid] = -INFINITY;
        l_i[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every simdgroup parks the same 8-row Q block in registers — it is tiny,
    // and it saves a shared-memory read per QK^T iteration.
    simdgroup_half8x8 qm[FA_HD / 8];
    for (uint kk = 0; kk < FA_HD / 8; kk++) {
        simdgroup_load(qm[kk], Qh + kk * 8, FA_HD);
    }

    // With 8-row tiles the causal loop bound is tight per tile: positions past
    // the tile's last query row are never visited at all. The windowed variant
    // instead walks the WHOLE bounded slot store (sinks + ring) — that width is
    // a constant, which is exactly what makes windowed prefill cost flat.
    uint t_hi = p.pos0 + min(r0 + FA_Q, p.n_rows);
    uint t_walk = LM_WINDOWED ? (LM_SINKPAD + LM_RING) : t_hi;
    uint frontier = p.pos0 + p.n_rows;

    for (uint t0 = 0; t0 < t_walk; t0 += FA_C) {
        // Phase 1 — S = Q·K^T: simdgroup sgid owns FA_C/8/FA_NSG score columns.
        // K blocks load in pairs so the compiler batches the device reads ahead
        // of the two MMAs (the issue pattern llama.cpp's fa kernel relies on).
        for (uint c = sgid * (FA_C / 8 / FA_NSG); c < (sgid + 1) * (FA_C / 8 / FA_NSG); c++) {
            simdgroup_float8x8 sc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#pragma unroll(4)
            for (uint kk = 0; kk < FA_HD / 8; kk += 2) {
                simdgroup_half8x8 B0;
                simdgroup_half8x8 B1;
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_load(B0, kb + (ulong)(t0 + c * 8) * kvd + kk * 8, kvd, ulong2(0), true);
                simdgroup_load(B1, kb + (ulong)(t0 + c * 8) * kvd + kk * 8 + 8, kvd, ulong2(0), true);
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(sc, qm[kk], B0, sc);
                simdgroup_multiply_accumulate(sc, qm[kk + 1], B1, sc);
            }
            simdgroup_store(sc, S + c * 8, FA_C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2 — online softmax: simdgroup sgid owns FA_Q/FA_NSG rows; the
        // rescale of O's rows rides along, so P·V below only accumulates. Deep
        // inside the causal region every position of the tile is valid for every
        // row, so the mask arithmetic drops out of the hot path entirely (the
        // guarded path computes identical values — the guards are all true).
        // Windowed pipelines always take the guarded path, with validity coming
        // from the slot's reconstructed position instead of the causal bound.
        bool tile_full = !LM_WINDOWED && (t0 + FA_C <= p.pos0 + r0) && (r0 + FA_Q <= p.n_rows);
        for (uint rr = sgid * (FA_Q / FA_NSG); rr < (sgid + 1) * (FA_Q / FA_NSG); rr++) {
            uint gr_s = r0 + rr;
            uint q_pos = p.pos0 + gr_s;
            bool row_live = gr_s < p.n_rows;
            float lmax = -INFINITY;
            if (tile_full) {
                for (uint j = lane; j < FA_C; j += 32) {
                    lmax = max(lmax, S[rr * FA_C + j] * scale);
                }
            } else {
                for (uint j = lane; j < FA_C; j += 32) {
                    uint t = t0 + j;
                    bool valid = LM_WINDOWED
                        ? (row_live && lm_attend(lm_slot_pos(t, frontier), q_pos))
                        : (row_live && t <= q_pos && t < t_hi);
                    if (valid) {
                        lmax = max(lmax, S[rr * FA_C + j] * scale);
                    }
                }
            }
            lmax = simd_max(lmax);
            float m_old = m_i[rr];
            float m_new = max(m_old, lmax);
            float corr = (m_old == -INFINITY) ? 0.0f : exp(m_old - m_new);
            float lsum = 0.0f;
            if (tile_full) {
                for (uint j = lane; j < FA_C; j += 32) {
                    float pv = exp(S[rr * FA_C + j] * scale - m_new);
                    P[rr * FA_C + j] = (half)pv;
                    lsum += pv;
                }
            } else {
                for (uint j = lane; j < FA_C; j += 32) {
                    uint t = t0 + j;
                    bool valid = LM_WINDOWED
                        ? (row_live && lm_attend(lm_slot_pos(t, frontier), q_pos))
                        : (row_live && t <= q_pos && t < t_hi);
                    // A row can meet a tile with no valid columns at all
                    // (padding or not-yet-written ring slots): m_new stays
                    // -inf there, and exp(-inf - -inf) would be NaN — the
                    // valid guard forces those weights to an exact 0.
                    float pv = valid ? exp(S[rr * FA_C + j] * scale - m_new) : 0.0f;
                    P[rr * FA_C + j] = (half)pv;
                    lsum += pv;
                }
            }
            lsum = simd_sum(lsum);
            if (lane == 0) {
                m_i[rr] = m_new;
                l_i[rr] = l_i[rr] * corr + lsum;
            }
            for (uint d = lane; d < FA_HD; d += 32) {
                O[rr * FA_HD + d] *= corr;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 3 — O += P·V: simdgroup sgid owns FA_HD/8/FA_NSG output columns,
        // V blocks straight from the cache.
        for (uint jb = sgid * (FA_HD / 8 / FA_NSG); jb < (sgid + 1) * (FA_HD / 8 / FA_NSG); jb++) {
            simdgroup_float8x8 acc;
            simdgroup_load(acc, O + jb * 8, FA_HD);
#pragma unroll(4)
            for (uint c = 0; c < FA_C / 8; c += 2) {
                simdgroup_half8x8 A0;
                simdgroup_half8x8 A1;
                simdgroup_half8x8 B0;
                simdgroup_half8x8 B1;
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_load(A0, P + c * 8, FA_C);
                simdgroup_load(B0, vb + (ulong)(t0 + c * 8) * kvd + jb * 8, kvd);
                simdgroup_load(A1, P + c * 8 + 8, FA_C);
                simdgroup_load(B1, vb + (ulong)(t0 + c * 8 + 8) * kvd + jb * 8, kvd);
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(acc, A0, B0, acc);
                simdgroup_multiply_accumulate(acc, A1, B1, acc);
            }
            simdgroup_store(acc, O + jb * 8, FA_HD);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write both copies.
    for (uint idx = tid; idx < FA_Q * FA_HD; idx += FA_THREADS) {
        uint r = idx / FA_HD;
        uint gr = r0 + r;
        if (gr < p.n_rows) {
            float v = O[idx] / l_i[r];
            uint d = idx % FA_HD;
            out[(ulong)gr * p.n_heads * FA_HD + head * FA_HD + d] = v;
            out_h[(ulong)gr * p.n_heads * FA_HD + head * FA_HD + d] = (half)v;
        }
    }
}

// ---------- decode attention: flash-decoding split over positions ----------
// The kernel above dispatches one threadgroup per (head, row). During decode that is
// n_heads threadgroups total (9 for SmolLM2) — most of the GPU idles, and the cost per
// token grows linearly with context. Here the cached positions are split into windows
// of ATTN_SPLIT: one threadgroup computes *partial* softmax-weighted sums per window,
// all in parallel, and a second tiny kernel merges the windows with the online-softmax
// rule. Scores never touch device memory.
//
// GQA twist: q heads sharing a kv head also share its K/V rows, so one threadgroup
// walks a kv head's window ONCE and scores every q head of the group against it —
// per-q-head walks would read each cached byte group-times over (7x for Qwen2.5's
// 14:2 heads), and at long context that KV traffic dominates decode. The group width
// is baked in per model via the GQA_CHUNK function constant; a group wider than
// MAX_GQA_CHUNK is covered by several chunks (grid x = kv heads × chunks).
//
// Requires head_dim <= DEC_TG (the Rust side falls back to the kernel above otherwise).

#define ATTN_SPLIT 128 // cached positions per window
#define DEC_TG 128     // threads per threadgroup: one per position in the window
// Upper bound on q heads per threadgroup — sizes the per-thread accumulator arrays.
#define MAX_GQA_CHUNK 8
// q heads actually processed per threadgroup: min(n_heads / n_kv_heads, MAX_GQA_CHUNK),
// set when the pipeline is built so the per-head loops below unroll flat.
constant uint GQA_CHUNK [[function_constant(0)]];
// Odd row stride for the phase-3 scratch keeps its column reads off bank conflicts.
#define ACC_STRIDE (GQA_CHUNK | 1u)

struct AttnDecParams {
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint pos;      // the query's position (context length = pos + 1)
    uint n_splits; // ceil((pos + 1) / ATTN_SPLIT)
};

// Per-head window results every thread holds after attn_dec_gqa_walk.
struct GqaPartial {
    float m[MAX_GQA_CHUNK]; // window max score
    float l[MAX_GQA_CHUNK]; // sum of exp(score - m) over the window
};

// Shared body of the two partial kernels: walk the K/V window [t0, t_end) of one
// kv head once, scoring q heads head_base..head_base+local_n against it. Leaves the
// exp-weighted V partials in acc_red (summed over position lanes by the caller) and
// returns m/l per head. Entries for g >= local_n are garbage — callers skip them.
// `qpos` is the query's position — only the windowed variant reads it, to mask
// slots by their reconstructed absolute position.
static GqaPartial attn_dec_gqa_walk(
    device const float *q_base, // first q row of the group (rows contiguous, stride hd)
    device const half *k_cache, // this sequence's cache (slot base already applied)
    device const half *v_cache,
    uint kvd, uint kv_off, uint hd, uint local_n, uint t0, uint t_end, uint qpos, uint tid,
    threadgroup float *q_s,     // [GQA_CHUNK × hd] staged q rows
    threadgroup float *es,      // [GQA_CHUNK × ATTN_SPLIT] exp(score - m)
    threadgroup float *acc_red, // [DEC_TG × ACC_STRIDE] phase-3 partial sums
    threadgroup float *red)     // [GQA_CHUNK × DEC_TG/32 + GQA_CHUNK] reduce scratch
{
    float scale = rsqrt((float)hd);
    threadgroup float *bcast = red + GQA_CHUNK * (DEC_TG / 32);

    // Stage the group's q rows: every thread re-reads them hd/4 times below.
    for (uint i = tid; i < local_n * hd; i += DEC_TG) {
        q_s[i] = q_base[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One thread = one position: read the K row once, score it against every q head.
    uint t = t0 + tid;
    float sc[MAX_GQA_CHUNK];
    for (uint g = 0; g < MAX_GQA_CHUNK; g++) {
        sc[g] = -INFINITY;
    }
    bool live = t < t_end;
    if (LM_WINDOWED && live) {
        live = lm_attend(lm_slot_pos(t, qpos + 1), qpos);
    }
    if (live) {
        // Wide loads: head_dim is a multiple of 4 in every supported model.
        device const half4 *k_t = (device const half4 *)(k_cache + (ulong)t * kvd + kv_off);
        float d[MAX_GQA_CHUNK] = {};
        for (uint i = 0; i < hd / 4; i++) {
            float4 k4 = float4(k_t[i]);
            for (uint g = 0; g < MAX_GQA_CHUNK; g++) {
                if (g < GQA_CHUNK) {
                    d[g] += dot(((threadgroup const float4 *)(q_s + g * hd))[i], k4);
                }
            }
        }
        for (uint g = 0; g < MAX_GQA_CHUNK; g++) {
            sc[g] = d[g] * scale;
        }
    }

    // Per-head max: simdgroup-first (see rmsnorm), then one thread per head folds
    // the simdgroup results and broadcasts through bcast.
    for (uint g = 0; g < MAX_GQA_CHUNK; g++) {
        if (g < GQA_CHUNK) {
            float sm = simd_max(sc[g]);
            if (tid % 32 == 0) {
                red[g * (DEC_TG / 32) + tid / 32] = sm;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < GQA_CHUNK) {
        float mm = -INFINITY;
        for (uint j = 0; j < DEC_TG / 32; j++) {
            mm = max(mm, red[tid * (DEC_TG / 32) + j]);
        }
        bcast[tid] = mm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    GqaPartial o;
    for (uint g = 0; g < MAX_GQA_CHUNK; g++) {
        if (g < GQA_CHUNK) {
            o.m[g] = bcast[g];
        }
    }

    // exp(score - m) per head (kept in es for phase 3), then the per-head sum.
    // The windowed variant guards on the score itself: a split whose every slot
    // is masked has m = -inf, and exp(-inf - -inf) would be NaN.
    for (uint g = 0; g < MAX_GQA_CHUNK; g++) {
        if (g < GQA_CHUNK) {
            float e = (LM_WINDOWED ? live : (t < t_end)) ? exp(sc[g] - o.m[g]) : 0.0f;
            es[g * ATTN_SPLIT + tid] = e;
            float ss = simd_sum(e);
            if (tid % 32 == 0) {
                red[g * (DEC_TG / 32) + tid / 32] = ss;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < GQA_CHUNK) {
        float total = 0.0f;
        for (uint j = 0; j < DEC_TG / 32; j++) {
            total += red[tid * (DEC_TG / 32) + j];
        }
        bcast[tid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < MAX_GQA_CHUNK; g++) {
        if (g < GQA_CHUNK) {
            o.l[g] = bcast[g];
        }
    }

    // Weighted V sum, threads reshaped as (position lane × output dim): each V element
    // is read once and weighted into every head of the group. tid = pl * hd + di, so
    // each output dim is covered by P position lanes.
    uint P = DEC_TG / hd;
    uint pl = tid / hd;
    uint di = tid % hd;
    float acc[MAX_GQA_CHUNK] = {};
    if (pl < P) {
        for (uint tt = t0 + pl; tt < t_end; tt += P) {
            float v = (float)v_cache[(ulong)tt * kvd + kv_off + di];
            for (uint g = 0; g < MAX_GQA_CHUNK; g++) {
                if (g < GQA_CHUNK) {
                    acc[g] += es[g * ATTN_SPLIT + tt - t0] * v;
                }
            }
        }
    }
    for (uint g = 0; g < MAX_GQA_CHUNK; g++) {
        if (g < GQA_CHUNK) {
            acc_red[tid * ACC_STRIDE + g] = acc[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return o;
}

// Partial per (head, window): [max, sum, acc[head_dim]] — acc is the exp-weighted
// V sum relative to this window's own max; the reduce step rescales.
kernel void attention_decode_partial(
    device const float *q [[buffer(0)]],
    device const half *k_cache [[buffer(1)]],
    device const half *v_cache [[buffer(2)]],
    device float *partials [[buffer(3)]],
    constant AttnDecParams &p [[buffer(4)]],
    threadgroup float *q_s [[threadgroup(0)]],
    threadgroup float *es [[threadgroup(1)]],
    threadgroup float *acc_red [[threadgroup(2)]],
    threadgroup float *red [[threadgroup(3)]],
    uint2 tg [[threadgroup_position_in_grid]], // x = kv head × group chunk, y = window
    uint2 tpos [[thread_position_in_threadgroup]])
{
    uint tid = tpos.x;
    uint hd = p.head_dim;
    uint group = p.n_heads / p.n_kv_heads;
    uint gchunks = (group + GQA_CHUNK - 1) / GQA_CHUNK;
    uint kvh = tg.x / gchunks;
    uint gc = tg.x % gchunks;
    uint head_base = kvh * group + gc * GQA_CHUNK;
    uint local_n = min(GQA_CHUNK, group - gc * GQA_CHUNK);
    uint t0 = tg.y * ATTN_SPLIT;
    uint lim = LM_WINDOWED ? (LM_SINKPAD + LM_RING) : (p.pos + 1);
    uint t_end = min(t0 + ATTN_SPLIT, lim); // exclusive

    GqaPartial o = attn_dec_gqa_walk(q + head_base * hd, k_cache, v_cache,
        p.n_kv_heads * hd, kvh * hd, hd, local_n, t0, t_end, p.pos, tid, q_s, es, acc_red, red);

    device float *out = partials + ((ulong)head_base * p.n_splits + tg.y) * (hd + 2);
    ulong head_stride = (ulong)p.n_splits * (hd + 2);
    uint P = DEC_TG / hd;
    for (uint g = 0; g < MAX_GQA_CHUNK; g++) {
        if (g < GQA_CHUNK && g < local_n) {
            if (tid == 0) {
                out[g * head_stride] = o.m[g];
                out[g * head_stride + 1] = o.l[g];
            }
            if (tid < hd) {
                float a = 0.0f;
                for (uint j = 0; j < P; j++) {
                    a += acc_red[(j * hd + tid) * ACC_STRIDE + g];
                }
                out[g * head_stride + 2 + tid] = a;
            }
        }
    }
}

// Merge the windows: rescale every partial to the global max and normalize.
// One threadgroup per head; the per-thread loops over n_splits are tiny.
kernel void attention_decode_reduce(
    device const float *partials [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant AttnDecParams &p [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]], // x = head
    uint2 tpos [[thread_position_in_threadgroup]])
{
    uint head = tg.x;
    uint di = tpos.x;
    uint hd = p.head_dim;
    device const float *ph = partials + (ulong)head * p.n_splits * (hd + 2);

    float m = -INFINITY;
    for (uint s = 0; s < p.n_splits; s++) {
        m = max(m, ph[s * (hd + 2)]);
    }
    float l = 0.0f;
    for (uint s = 0; s < p.n_splits; s++) {
        l += exp(ph[s * (hd + 2)] - m) * ph[s * (hd + 2) + 1];
    }
    if (di < hd) {
        float a = 0.0f;
        for (uint s = 0; s < p.n_splits; s++) {
            a += exp(ph[s * (hd + 2)] - m) * ph[s * (hd + 2) + 2 + di];
        }
        out[head * hd + di] = a / l;
    }
}

// ---------- batched decode: one step for several requests at once ----------
// Continuous batching's core. B rows, each with its own (token, position, KV slot);
// the KV cache is pooled per layer as [slot][max_seq][kv_dim] — plain static slots,
// no block tables: macOS commits the pages lazily, so an untouched slot tail costs
// no physical RAM, and the kernels keep fully linear, coalesced access. The
// weight-heavy kernels gain a batch grid dimension, so ONE read of the weights
// serves every active request — that is the entire point of batching decode.

struct RowMeta {
    uint pos;  // this row's position in its own sequence
    uint slot; // which pooled KV slot the row owns
};

struct QkvBatchParams {
    uint in_dim;
    uint q_dim;
    uint kv_dim;
    uint max_seq; // slot stride, in cache rows
};

kernel void matvec_qkv_batch(
    device const half *w_q [[buffer(0)]],
    device const half *b_q [[buffer(1)]],
    device const half *w_k [[buffer(2)]],
    device const half *b_k [[buffer(3)]],
    device const half *w_v [[buffer(4)]],
    device const half *b_v [[buffer(5)]],
    device const float *x [[buffer(6)]],
    device float *q [[buffer(7)]],
    device half *k_cache [[buffer(8)]],
    device half *v_cache [[buffer(9)]],
    constant QkvBatchParams &p [[buffer(10)]],
    device const RowMeta *meta [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]], // x = W-row tile, y = batch row
    uint2 tpos [[thread_position_in_threadgroup]])
{
    uint tid = tpos.x;
    uint lane = tid % 32;
    uint row = tgid.x * 4 + tid / 32; // 128 threads = 4 simdgroups per threadgroup
    if (row >= p.q_dim + 2 * p.kv_dim) {
        return;
    }
    uint b = tgid.y;
    device const float *xb = x + (ulong)b * p.in_dim;

    device const half *w;
    device const half *bias;
    uint r;
    if (row < p.q_dim) {
        w = w_q; bias = b_q; r = row;
    } else if (row < p.q_dim + p.kv_dim) {
        r = row - p.q_dim;
        w = w_k; bias = b_k;
    } else {
        r = row - p.q_dim - p.kv_dim;
        w = w_v; bias = b_v;
    }
    float sum = simd_sum(dot_wx(w, r, xb, p.in_dim, lane));
    if (lane == 0) {
        float val = sum + (float)bias[r];
        ulong kv_off = ((ulong)meta[b].slot * p.max_seq + meta[b].pos) * p.kv_dim;
        if (row < p.q_dim) {
            q[(ulong)b * p.q_dim + r] = val;
        } else if (row < p.q_dim + p.kv_dim) {
            k_cache[kv_off + r] = (half)val;
        } else {
            v_cache[kv_off + r] = (half)val;
        }
    }
}

// y[b] += W·x[b] + bias — batched o_proj/down_proj with the residual fused.
kernel void matvec_acc_batch(
    device const half *w [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant MatvecParams &p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]])
{
    uint tid = tpos.x;
    uint lane = tid % 32;
    uint row = tgid.x * 4 + tid / 32;
    if (row >= p.out_dim) {
        return;
    }
    uint b = tgid.y;
    float sum = simd_sum(dot_wx(w, row, x + (ulong)b * p.in_dim, p.in_dim, lane));
    if (lane == 0) {
        y[(ulong)b * p.out_dim + row] += sum + (float)bias[row];
    }
}

// y[b] = silu(Wg·x[b]) * (Wu·x[b]) — batched SwiGLU inner step.
kernel void matvec_swiglu_batch(
    device const half *w_gate [[buffer(0)]],
    device const half *w_up [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant MatvecParams &p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]])
{
    uint tid = tpos.x;
    uint lane = tid % 32;
    uint row = tgid.x * 4 + tid / 32;
    if (row >= p.out_dim) {
        return;
    }
    uint b = tgid.y;
    device const float *xb = x + (ulong)b * p.in_dim;
    float g = simd_sum(dot_wx(w_gate, row, xb, p.in_dim, lane));
    float u = simd_sum(dot_wx(w_up, row, xb, p.in_dim, lane));
    if (lane == 0) {
        y[(ulong)b * p.out_dim + row] = (g / (1.0f + exp(-g))) * u;
    }
}

// Batched RoPE on q and each row's freshly written k cache row.
struct RopeQkBatchParams {
    uint head_dim;
    uint n_q_heads;
    uint n_kv_heads;
    float theta;
    uint max_seq;
    uint kv_dim;
    uint n_rows;
};

kernel void rope_qk_batch(
    device float *q [[buffer(0)]],
    device half *k_cache [[buffer(1)]],
    device const RowMeta *meta [[buffer(2)]],
    constant RopeQkBatchParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_dim = p.head_dim / 2;
    uint q_pairs = p.n_q_heads * half_dim;
    uint per_b = q_pairs + p.n_kv_heads * half_dim;
    uint b = gid / per_b;
    if (b >= p.n_rows) {
        return;
    }
    uint idx = gid % per_b;
    bool is_q = idx < q_pairs;
    uint hidx = is_q ? idx : idx - q_pairs;
    uint h = hidx / half_dim;
    uint i = hidx % half_dim;

    float freq = pow(p.theta, -2.0f * (float)i / (float)p.head_dim);
    float angle = (float)meta[b].pos * freq;
    float c;
    float s = sincos(angle, c);
    if (is_q) {
        device float *head = q + (ulong)b * p.n_q_heads * p.head_dim + h * p.head_dim;
        float a0 = head[i];
        float b0 = head[i + half_dim];
        head[i] = a0 * c - b0 * s;
        head[i + half_dim] = a0 * s + b0 * c;
    } else {
        device half *head = k_cache
            + ((ulong)meta[b].slot * p.max_seq + meta[b].pos) * p.kv_dim + h * p.head_dim;
        float a0 = head[i];
        float b0 = head[i + half_dim];
        head[i] = (half)(a0 * c - b0 * s);
        head[i + half_dim] = (half)(a0 * s + b0 * c);
    }
}

// Batched flash-decoding attention: same two kernels as above with a batch grid
// dimension and per-row position/slot from meta.
struct AttnDecBatchParams {
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint max_seq;
    uint kv_dim;
    uint splits_max; // partials stride; rows with fewer windows leave the tail unused
};

kernel void attention_decode_partial_batch(
    device const float *q [[buffer(0)]],
    device const half *k_cache [[buffer(1)]],
    device const half *v_cache [[buffer(2)]],
    device float *partials [[buffer(3)]],
    device const RowMeta *meta [[buffer(4)]],
    constant AttnDecBatchParams &p [[buffer(5)]],
    threadgroup float *q_s [[threadgroup(0)]],
    threadgroup float *es [[threadgroup(1)]],
    threadgroup float *acc_red [[threadgroup(2)]],
    threadgroup float *red [[threadgroup(3)]],
    uint3 tg [[threadgroup_position_in_grid]], // x = kv head × group chunk, y = window, z = batch row
    uint3 tpos [[thread_position_in_threadgroup]])
{
    uint tid = tpos.x;
    uint b = tg.z;
    uint pos = meta[b].pos;
    uint t0 = tg.y * ATTN_SPLIT;
    if (t0 > pos) {
        return; // this row has fewer windows than the widest in the batch
    }
    uint hd = p.head_dim;
    uint group = p.n_heads / p.n_kv_heads;
    uint gchunks = (group + GQA_CHUNK - 1) / GQA_CHUNK;
    uint kvh = tg.x / gchunks;
    uint gc = tg.x % gchunks;
    uint head_base = kvh * group + gc * GQA_CHUNK;
    uint local_n = min(GQA_CHUNK, group - gc * GQA_CHUNK);
    uint t_end = min(t0 + ATTN_SPLIT, pos + 1);
    ulong base = (ulong)meta[b].slot * p.max_seq * p.kv_dim;

    GqaPartial o = attn_dec_gqa_walk(q + ((ulong)b * p.n_heads + head_base) * hd,
        k_cache + base, v_cache + base, p.kv_dim, kvh * hd, hd, local_n, t0, t_end, pos, tid,
        q_s, es, acc_red, red);

    device float *out = partials
        + (((ulong)b * p.n_heads + head_base) * p.splits_max + tg.y) * (hd + 2);
    ulong head_stride = (ulong)p.splits_max * (hd + 2);
    uint P = DEC_TG / hd;
    for (uint g = 0; g < MAX_GQA_CHUNK; g++) {
        if (g < GQA_CHUNK && g < local_n) {
            if (tid == 0) {
                out[g * head_stride] = o.m[g];
                out[g * head_stride + 1] = o.l[g];
            }
            if (tid < hd) {
                float a = 0.0f;
                for (uint j = 0; j < P; j++) {
                    a += acc_red[(j * hd + tid) * ACC_STRIDE + g];
                }
                out[g * head_stride + 2 + tid] = a;
            }
        }
    }
}

kernel void attention_decode_reduce_batch(
    device const float *partials [[buffer(0)]],
    device float *out [[buffer(1)]],
    device const RowMeta *meta [[buffer(2)]],
    constant AttnDecBatchParams &p [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], // x = head, y = batch row
    uint2 tpos [[thread_position_in_threadgroup]])
{
    uint head = tg.x;
    uint b = tg.y;
    uint di = tpos.x;
    uint hd = p.head_dim;
    uint n_splits = meta[b].pos / ATTN_SPLIT + 1;
    device const float *ph = partials
        + ((ulong)b * p.n_heads + head) * p.splits_max * (hd + 2);

    float m = -INFINITY;
    for (uint s = 0; s < n_splits; s++) {
        m = max(m, ph[s * (hd + 2)]);
    }
    float l = 0.0f;
    for (uint s = 0; s < n_splits; s++) {
        l += exp(ph[s * (hd + 2)] - m) * ph[s * (hd + 2) + 1];
    }
    if (di < hd) {
        float a = 0.0f;
        for (uint s = 0; s < n_splits; s++) {
            a += exp(ph[s * (hd + 2)] - m) * ph[s * (hd + 2) + 2 + di];
        }
        out[((ulong)b * p.n_heads + head) * hd + di] = a / l;
    }
}

// ---------- two tiny elementwise kernels ----------
// dim = total element count across all rows (rows are contiguous, so rows don't matter here).

struct ElemParams {
    uint dim;
};

// inner = silu(gate) * up, written back into gate (math.rs::silu + the SwiGLU line in model.rs)
kernel void silu_mul(
    device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    constant ElemParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.dim) {
        return;
    }
    float g = gate[gid];
    gate[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

// silu_mul variant for the prefill path: also emits the half copy that feeds
// down_proj's tensor-ops matmul.
kernel void silu_mul_hf(
    device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device half *gate_h [[buffer(2)]],
    constant ElemParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.dim) {
        return;
    }
    float g = gate[gid];
    float v = (g / (1.0f + exp(-g))) * up[gid];
    gate[gid] = v;
    gate_h[gid] = (half)v;
}

// -b lowmem stage-in: convert a weight page's raw bf16 bits — read STRAIGHT
// from the checkpoint's mmap through a no-copy buffer — into its f16 pool page.
// bf16 → f32 is an exact bit shift; f32 → f16 rounds to nearest even, the same
// two steps the CPU path takes, so the staged values are identical while the
// CPU's share of staging drops to zero (the OS pages the file in under the
// GPU's reads). p.x = element count, p.y = element offset after the 4-byte
// buffer-offset alignment. A value past f16 range flips the flag for the
// host's once-only warning.
kernel void bf16_to_f16_copy(
    device const ushort *src [[buffer(0)]],
    device half *dst [[buffer(1)]],
    constant uint2 &p [[buffer(2)]],
    device atomic_uint *clip [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.x) {
        return;
    }
    float v = as_type<float>((uint)src[p.y + gid] << 16);
    half h = (half)v;
    if (isinf(h) && !isinf(v)) {
        atomic_store_explicit(clip, 1u, memory_order_relaxed);
    }
    dst[gid] = h;
}

// x += y  (residual connection)
kernel void add_inplace(
    device float *x [[buffer(0)]],
    device const float *y [[buffer(1)]],
    constant ElemParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.dim) {
        x[gid] += y[gid];
    }
}
