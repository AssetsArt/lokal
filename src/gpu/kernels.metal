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
// Data-type convention: weights are half (f16) to halve memory traffic;
// activations and the KV cache stay float (f32); accumulation is always float.
//
// Every params struct must match its #[repr(C)] counterpart in gpu/metal.rs exactly.

#include <metal_stdlib>
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
    device const half *w_row = w + (ulong)row * p.in_dim;
    float acc = 0.0f;
    for (uint i = lane; i < p.in_dim; i += 32) {
        acc += (float)w_row[i] * x[i];
    }
    float sum = simd_sum(acc);
    if (lane == 0) {
        y[row] = sum + (float)bias[row];
    }
}

// ---------- matmul: Y = X·Wᵀ + bias — used during prefill (many tokens at once) ----------
// Why batch prefill is fast: matvec re-reads all of W for *every* token, while matmul
// tiles the work and stages tiles in threadgroup memory (on-chip SRAM, far faster than
// device memory) — one tile of W is read from device memory once and reused by every
// token in the tile.
//
//   TM×TK of X (tokens × k-slice)  +  TN×TK of W (outputs × k-slice)  → TM×TN of Y
//   256 threads: each computes one (m,n) cell of the output tile.

#define MM_TM 8   // tokens per tile
#define MM_TN 32  // outputs per tile
#define MM_TK 32  // k-dimension slice staged per iteration

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
    uint2 tpos [[thread_position_in_threadgroup]]) // 256 threads (MSL requires matching vector widths)
{
    uint tid = tpos.x;
    uint m = tid / MM_TN; // token within the tile (0..7)
    uint n = tid % MM_TN; // output within the tile (0..31)
    uint row0 = tgid.y * MM_TM;
    uint out0 = tgid.x * MM_TN;

    threadgroup float Xs[MM_TM][MM_TK];
    threadgroup half Ws[MM_TN][MM_TK];

    float acc = 0.0f;
    for (uint k0 = 0; k0 < p.in_dim; k0 += MM_TK) {
        // Cooperatively stage both tiles into shared memory (out-of-bounds → zero, harmless).
        {
            uint lm = tid / MM_TK; // Xs has 8×32 = 256 cells → one per thread
            uint lk = tid % MM_TK;
            uint gr = row0 + lm;
            uint gk = k0 + lk;
            Xs[lm][lk] = (gr < p.n_rows && gk < p.in_dim) ? x[(ulong)gr * p.in_dim + gk] : 0.0f;
        }
        for (uint i = 0; i < 4; i++) { // Ws has 32×32 = 1024 cells → four per thread
            uint idx = tid + i * 256;
            uint ln = idx / MM_TK;
            uint lk = idx % MM_TK;
            uint go = out0 + ln;
            uint gk = k0 + lk;
            Ws[ln][lk] = (go < p.out_dim && gk < p.in_dim) ? w[(ulong)go * p.in_dim + gk] : (half)0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < MM_TK; k++) {
            acc += Xs[m][k] * (float)Ws[n][k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // everyone done before tiles are overwritten
    }

    uint gr = row0 + m;
    uint go = out0 + n;
    if (gr < p.n_rows && go < p.out_dim) {
        y[(ulong)gr * p.out_dim + go] = acc + (float)bias[go];
    }
}

// ---------- rmsnorm (math.rs::rmsnorm) ----------
// One threadgroup per row (token): cooperative sum of squares via tree reduction.

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

    threadgroup float partial[NORM_TG];

    float acc = 0.0f;
    for (uint i = tid; i < p.dim; i += NORM_TG) {
        acc += xr[i] * xr[i];
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = NORM_TG / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

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
    float c = cos(angle);
    float s = sin(angle);
    device float *head = x + (ulong)row * p.n_heads * p.head_dim + h * p.head_dim;
    float a = head[i];
    float b = head[i + half_dim];
    head[i] = a * c - b * s;
    head[i + half_dim] = a * s + b * c;
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
    device const float *k_cache [[buffer(1)]],
    device const float *v_cache [[buffer(2)]],
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

    threadgroup float red[ATTN_TG];

    // Phase 1: q·k score for every position 0..=q_pos, tracking the max for a stable exp.
    float local_max = -INFINITY;
    for (uint t = tid; t <= q_pos; t += ATTN_TG) {
        device const float *k_t = k_cache + (ulong)t * kvd + kv_off;
        float dot = 0.0f;
        for (uint i = 0; i < hd; i++) {
            dot += q_head[i] * k_t[i];
        }
        dot *= scale;
        sc[t] = dot;
        local_max = max(local_max, dot);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ATTN_TG / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] = max(red[tid], red[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
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
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ATTN_TG / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float score_sum = red[0];

    // Phase 3: output = weighted average of v — one thread per output dimension.
    for (uint i = tid; i < hd; i += ATTN_TG) {
        float acc = 0.0f;
        for (uint t = 0; t <= q_pos; t++) {
            acc += sc[t] * v_cache[(ulong)t * kvd + kv_off + i];
        }
        out[(ulong)row * p.n_heads * hd + head * hd + i] = acc / score_sum;
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
