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

// ---------- matvec: y = W·x + bias (math.rs::matvec) — used during decode ----------
// One simdgroup (32 threads executing in lockstep) owns one row of W: adjacent
// threads read adjacent elements (coalesced), then combine with simd_sum, a
// hardware reduction that never touches memory.
//
// The dot products load half4/float4 vectors (8/16 bytes per instruction) — decode is
// memory-bandwidth-bound, and wide loads are what gets a small kernel near peak
// bandwidth. A scalar tail handles in_dim % 4 (zero for every supported model).

inline float dot_wx(device const half *w_row, device const float *x, uint in_dim, uint lane) {
    device const half4 *w4 = (device const half4 *)w_row;
    device const float4 *x4 = (device const float4 *)x;
    uint n4 = in_dim / 4;
    float acc = 0.0f;
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
    float sum = simd_sum(dot_wx(w + (ulong)row * p.in_dim, x, p.in_dim, lane));
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
    float sum = simd_sum(dot_wx(w + (ulong)row * p.in_dim, x, p.in_dim, lane));
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
    float g = simd_sum(dot_wx(w_gate + (ulong)row * p.in_dim, x, p.in_dim, lane));
    float u = simd_sum(dot_wx(w_up + (ulong)row * p.in_dim, x, p.in_dim, lane));
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
    float sum = simd_sum(dot_wx(w + (ulong)r * p.in_dim, x, p.in_dim, lane));
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

#define MM_TM 8   // tokens per tile
#define MM_TN 32  // outputs per tile
#define MM_TK 32  // k-dimension slice staged per iteration
#define MM_THREADS 128

struct MatmulParams {
    uint in_dim;
    uint out_dim;
    uint n_rows;
};

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
    uint row0 = tgid.y * MM_TM;
    uint out0 = tgid.x * MM_TN;

    threadgroup float Xs[MM_TM][MM_TK];
    threadgroup float Ws[MM_TN][MM_TK];
    threadgroup float Cs[4][MM_TM][8];

    simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.in_dim; k0 += MM_TK) {
        // Cooperatively stage both tiles (out-of-bounds → zero, harmless).
        for (uint idx = tid; idx < MM_TM * MM_TK; idx += MM_THREADS) {
            uint lm = idx / MM_TK;
            uint lk = idx % MM_TK;
            uint gr = row0 + lm;
            uint gk = k0 + lk;
            Xs[lm][lk] = (gr < p.n_rows && gk < p.in_dim) ? x[(ulong)gr * p.in_dim + gk] : 0.0f;
        }
        for (uint idx = tid; idx < MM_TN * MM_TK; idx += MM_THREADS) {
            uint ln = idx / MM_TK;
            uint lk = idx % MM_TK;
            uint go = out0 + ln;
            uint gk = k0 + lk;
            Ws[ln][lk] = (go < p.out_dim && gk < p.in_dim) ? (float)w[(ulong)go * p.in_dim + gk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < MM_TK; kk += 8) {
            simdgroup_float8x8 A; // X block: [8 tokens][8 k]
            simdgroup_load(A, &Xs[0][kk], MM_TK);
            simdgroup_float8x8 B; // W block loaded transposed: [8 k][8 outputs]
            simdgroup_load(B, &Ws[sgid * 8][kk], MM_TK, ulong2(0), true);
            simdgroup_multiply_accumulate(C, A, B, C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // everyone done before tiles are overwritten
    }

    simdgroup_store(C, &Cs[sgid][0][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < MM_TM * MM_TN; idx += MM_THREADS) {
        uint m = idx / MM_TN;
        uint n = idx % MM_TN;
        uint gr = row0 + m;
        uint go = out0 + n;
        if (gr < p.n_rows && go < p.out_dim) {
            y[(ulong)gr * p.out_dim + go] = Cs[n / 8][m][n % 8] + (float)bias[go];
        }
    }
}

// Same simdgroup-matrix matmul, but writing f16 — for prefill's k/v projections,
// which land directly in the (f16) KV cache.
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
    uint row0 = tgid.y * MM_TM;
    uint out0 = tgid.x * MM_TN;

    threadgroup float Xs[MM_TM][MM_TK];
    threadgroup float Ws[MM_TN][MM_TK];
    threadgroup float Cs[4][MM_TM][8];

    simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.in_dim; k0 += MM_TK) {
        for (uint idx = tid; idx < MM_TM * MM_TK; idx += MM_THREADS) {
            uint lm = idx / MM_TK;
            uint lk = idx % MM_TK;
            uint gr = row0 + lm;
            uint gk = k0 + lk;
            Xs[lm][lk] = (gr < p.n_rows && gk < p.in_dim) ? x[(ulong)gr * p.in_dim + gk] : 0.0f;
        }
        for (uint idx = tid; idx < MM_TN * MM_TK; idx += MM_THREADS) {
            uint ln = idx / MM_TK;
            uint lk = idx % MM_TK;
            uint go = out0 + ln;
            uint gk = k0 + lk;
            Ws[ln][lk] = (go < p.out_dim && gk < p.in_dim) ? (float)w[(ulong)go * p.in_dim + gk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < MM_TK; kk += 8) {
            simdgroup_float8x8 A;
            simdgroup_load(A, &Xs[0][kk], MM_TK);
            simdgroup_float8x8 B;
            simdgroup_load(B, &Ws[sgid * 8][kk], MM_TK, ulong2(0), true);
            simdgroup_multiply_accumulate(C, A, B, C);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(C, &Cs[sgid][0][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < MM_TM * MM_TN; idx += MM_THREADS) {
        uint m = idx / MM_TN;
        uint n = idx % MM_TN;
        uint gr = row0 + m;
        uint go = out0 + n;
        if (gr < p.n_rows && go < p.out_dim) {
            y[(ulong)gr * p.out_dim + go] = (half)(Cs[n / 8][m][n % 8] + (float)bias[go]);
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
    float sum = simd_sum(dot_wx(w + (ulong)row * p.in_dim, x, p.in_dim, lane));
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

// ---------- attention (model.rs::attention) ----------
// One threadgroup per (query head, query row): the same three phases as the CPU code —
// scores → softmax → weighted sum of V — with barriers making each phase's results
// visible before the next. The causal mask falls out of the loop bound: the row at
// position q_pos only iterates over 0..=q_pos.

#define ATTN_TG 256

struct AttnParams {
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint pos0;
    uint max_seq;
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

    // Phase 1: q·k score for every position 0..=q_pos, tracking the max for a stable exp.
    float local_max = -INFINITY;
    for (uint t = tid; t <= q_pos; t += ATTN_TG) {
        device const half *k_t = k_cache + (ulong)t * kvd + kv_off;
        float dot = 0.0f;
        for (uint i = 0; i < hd; i++) {
            dot += q_head[i] * (float)k_t[i];
        }
        dot *= scale;
        sc[t] = dot;
        local_max = max(local_max, dot);
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

    // Phase 2: exponentiate and sum (the softmax denominator).
    float local_sum = 0.0f;
    for (uint t = tid; t <= q_pos; t += ATTN_TG) {
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

    // Phase 3: output = weighted average of v — one thread per output dimension.
    for (uint i = tid; i < hd; i += ATTN_TG) {
        float acc = 0.0f;
        for (uint t = 0; t <= q_pos; t++) {
            acc += sc[t] * (float)v_cache[(ulong)t * kvd + kv_off + i];
        }
        out[(ulong)row * p.n_heads * hd + head * hd + i] = acc / score_sum;
    }
}

// ---------- decode attention: flash-decoding split over positions ----------
// The kernel above dispatches one threadgroup per (head, row). During decode that is
// n_heads threadgroups total (9 for SmolLM2) — most of the GPU idles, and the cost per
// token grows linearly with context. Here the cached positions are split into windows
// of ATTN_SPLIT: one threadgroup computes a *partial* softmax-weighted sum per
// (head, window), all in parallel, and a second tiny kernel merges the windows with
// the online-softmax rule. Scores never touch device memory.
//
// Requires head_dim <= DEC_TG (the Rust side falls back to the kernel above otherwise).

#define ATTN_SPLIT 128 // cached positions per window
#define DEC_TG 128     // threads per threadgroup: one per position in the window

struct AttnDecParams {
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint pos;      // the query's position (context length = pos + 1)
    uint n_splits; // ceil((pos + 1) / ATTN_SPLIT)
};

// Partial per (head, window): [max, sum, acc[head_dim]] — acc is the exp-weighted
// V sum relative to this window's own max; the reduce step rescales.
kernel void attention_decode_partial(
    device const float *q [[buffer(0)]],
    device const half *k_cache [[buffer(1)]],
    device const half *v_cache [[buffer(2)]],
    device float *partials [[buffer(3)]],
    constant AttnDecParams &p [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], // x = head, y = window
    uint2 tpos [[thread_position_in_threadgroup]])
{
    uint tid = tpos.x;
    uint head = tg.x;
    uint hd = p.head_dim;
    uint kvd = p.n_kv_heads * hd;
    uint kv_off = (head / (p.n_heads / p.n_kv_heads)) * hd;
    uint t0 = tg.y * ATTN_SPLIT;
    uint t_end = min(t0 + ATTN_SPLIT, p.pos + 1); // exclusive
    device const float *q_head = q + head * hd;
    float scale = rsqrt((float)hd);

    threadgroup float es[ATTN_SPLIT]; // this window's scores, then exp(score - max)
    threadgroup float red[DEC_TG];

    // One thread = one position: score, then cooperative max / sum reductions.
    uint t = t0 + tid;
    float score = -INFINITY;
    if (t < t_end) {
        // Wide loads: head_dim is a multiple of 4 in every supported model.
        device const half4 *k_t = (device const half4 *)(k_cache + (ulong)t * kvd + kv_off);
        device const float4 *q4 = (device const float4 *)q_head;
        float d = 0.0f;
        for (uint i = 0; i < hd / 4; i++) {
            d += dot(q4[i], float4(k_t[i]));
        }
        score = d * scale;
    }
    // Simdgroup-first max/sum (see rmsnorm) — red's tail doubles as phase-3 scratch.
    float sg_max = simd_max(score);
    if (tid % 32 == 0) {
        red[tid / 32] = sg_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float mm = -INFINITY;
        for (uint j = 0; j < DEC_TG / 32; j++) {
            mm = max(mm, red[j]);
        }
        red[0] = mm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float e = (t < t_end) ? exp(score - m) : 0.0f;
    es[tid] = e;
    float sg_sum = simd_sum(e);
    if (tid % 32 == 0) {
        red[tid / 32] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint j = 0; j < DEC_TG / 32; j++) {
            total += red[j];
        }
        red[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float l = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weighted V sum, threads reshaped as (position lane × output dim):
    // tid = pl * hd + di, so each output dim is covered by P position lanes.
    uint P = DEC_TG / hd;
    uint pl = tid / hd;
    uint di = tid % hd;
    float acc = 0.0f;
    if (pl < P) {
        for (uint tt = t0 + pl; tt < t_end; tt += P) {
            acc += es[tt - t0] * (float)v_cache[(ulong)tt * kvd + kv_off + di];
        }
    }
    red[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out = partials + ((ulong)head * p.n_splits + tg.y) * (hd + 2);
    if (tid == 0) {
        out[0] = m;
        out[1] = l;
    }
    if (tid < hd) {
        float a = 0.0f;
        for (uint j = 0; j < P; j++) {
            a += red[j * hd + tid];
        }
        out[2 + tid] = a;
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
