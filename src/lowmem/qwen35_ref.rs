//! The numerics oracle for qwen35's linear-attention (gated-deltanet) blocks
//! and the full-attention twists — lane C's GPU kernels are tested BIT-FOR-BIT
//! against these functions (dequant_row_ref's role, one subsystem up).
//!
//! Strict IEEE f32 with f64 accumulation exactly where ggml accumulates in
//! ggml_float (l2_norm, rmsnorm): every formula is transcribed from the ggml
//! CPU reference (ggml/src/ggml-cpu/ops.cpp, unary-ops.cpp) and llama.cpp's
//! delta-net decode form (src/models/delta-net-base.cpp,
//! build_delta_net_autoregressive), not from memory. docs/qwen35.md is the map.
//!
//! Shapes follow ggml's state convention: the delta state is [S, S, H] with
//! s[i + j*S + h*S*S] — i the contraction index, j the output index. K-heads
//! broadcast to V-heads (H_v % H_k == 0), matching the graph's repeat.

/// The linear-block geometry. Real 27B values: d_state 128, dt_rank 48,
/// n_group 16, d_conv 4 → d_inner 6144, conv_channels 10240.
#[derive(Clone, Copy)]
pub struct DeltaDims {
    pub d_state: usize,
    /// V heads (llama.cpp's ssm_dt_rank).
    pub n_v_heads: usize,
    /// K heads (llama.cpp's ssm_n_group).
    pub n_k_heads: usize,
    pub d_conv: usize,
}

impl DeltaDims {
    pub fn d_inner(&self) -> usize {
        self.n_v_heads * self.d_state
    }
    /// conv channels C = 2·key_dim + value_dim.
    pub fn conv_channels(&self) -> usize {
        2 * self.n_k_heads * self.d_state + self.d_inner()
    }
    /// Elements of rolling conv state per layer: (d_conv−1)·C.
    pub fn conv_state_elems(&self) -> usize {
        (self.d_conv - 1) * self.conv_channels()
    }
    /// Elements of delta state per layer: S·S·H_v (== d_state·d_inner).
    pub fn delta_state_elems(&self) -> usize {
        self.d_state * self.d_state * self.n_v_heads
    }
}

// ---- exact ggml scalar forms ----

/// unary-ops.cpp op_softplus: x > 20 ? x : ln(1 + eˣ).
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// ops.cpp l2_norm: y = x / max(√(Σx²), eps), sum in f64 (ggml_float).
pub fn l2_norm(x: &mut [f32], eps: f32) {
    let sum: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let scale = 1.0 / (sum.sqrt() as f32).max(eps);
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// RMSNorm with ggml's accumulation: mean of squares in f64, then
/// x · w / √(mean + eps).
pub fn rmsnorm(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mean = (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    for (v, wi) in x.iter_mut().zip(w) {
        *v = *v * scale * wi;
    }
}

/// The per-token gate scalar: g = ssm_a · softplus(alpha + dt_bias), where
/// ssm_a stores −exp(A_log) (LLM_TENSOR_SSM_A_NOSCAN). One value per V head.
pub fn delta_gate(alpha: f32, dt_bias: f32, a: f32) -> f32 {
    a * softplus(alpha + dt_bias)
}

/// One decode step of the depthwise causal conv + SiLU, updating the rolling
/// state in place. `state` holds the last (d_conv−1) input columns per channel
/// (layout [channel][d_conv−1], oldest first); `x` is this token's projected
/// qkv_mixed row [C]; `kernel` is [channel][d_conv]. Returns the activated
/// conv output [C].
pub fn conv_step(dims: &DeltaDims, state: &mut [f32], x: &[f32], kernel: &[f32]) -> Vec<f32> {
    let c_all = dims.conv_channels();
    let k = dims.d_conv;
    assert_eq!(state.len(), c_all * (k - 1));
    assert_eq!(x.len(), c_all);
    assert_eq!(kernel.len(), c_all * k);
    let mut out = vec![0f32; c_all];
    for c in 0..c_all {
        let st = &mut state[c * (k - 1)..(c + 1) * (k - 1)];
        // window = [state.., x[c]] ∘ kernel[c]
        let mut acc = 0f32;
        for (j, &s) in st.iter().enumerate() {
            acc += s * kernel[c * k + j];
        }
        acc += x[c] * kernel[c * k + (k - 1)];
        out[c] = silu(acc);
        // roll: shift left, append the new input
        st.rotate_left(1);
        st[k - 2] = x[c];
    }
    out
}

/// Split the conv output row [C] into per-head q, k, v (graph order:
/// [key_dim | key_dim | value_dim]) and l2-normalize each q/k head.
pub fn split_qkv(dims: &DeltaDims, conv_out: &[f32], eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (s, hk) = (dims.d_state, dims.n_k_heads);
    let key_dim = s * hk;
    let mut q = conv_out[..key_dim].to_vec();
    let mut k = conv_out[key_dim..2 * key_dim].to_vec();
    let v = conv_out[2 * key_dim..2 * key_dim + dims.d_inner()].to_vec();
    for h in 0..hk {
        l2_norm(&mut q[h * s..(h + 1) * s], eps);
        l2_norm(&mut k[h * s..(h + 1) * s], eps);
    }
    (q, k, v)
}

/// One decode step of the gated delta rule
/// (build_delta_net_autoregressive, per V head h with its broadcast K head):
///   q ← q/√S;  s ← s·eᵍ;  sk_j = Σ_i s[i,j]·k_i;
///   d_j = (v_j − sk_j)·β;  s[i,j] += k_i·d_j;  o_j = Σ_i s[i,j]·q_i.
/// `state` is [S·S·H_v]; q/k are per-K-head [S·H_k]; v [S·H_v];
/// g, beta one scalar per V head. Returns o [S·H_v].
pub fn delta_decode_step(
    dims: &DeltaDims,
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    let (s_dim, hv, hk) = (dims.d_state, dims.n_v_heads, dims.n_k_heads);
    assert_eq!(state.len(), dims.delta_state_elems());
    assert_eq!(q.len(), s_dim * hk);
    assert_eq!(k.len(), s_dim * hk);
    assert_eq!(v.len(), s_dim * hv);
    assert_eq!(g.len(), hv);
    assert_eq!(beta.len(), hv);
    let group = hv / hk;
    let scale = 1.0 / (s_dim as f32).sqrt();
    let mut out = vec![0f32; s_dim * hv];
    for h in 0..hv {
        let kh = &k[(h / group) * s_dim..(h / group + 1) * s_dim];
        let qh: Vec<f32> = q[(h / group) * s_dim..(h / group + 1) * s_dim]
            .iter()
            .map(|&x| x * scale)
            .collect();
        let vh = &v[h * s_dim..(h + 1) * s_dim];
        let st = &mut state[h * s_dim * s_dim..(h + 1) * s_dim * s_dim];
        let ge = g[h].exp();
        for x in st.iter_mut() {
            *x *= ge;
        }
        let oh = &mut out[h * s_dim..(h + 1) * s_dim];
        for j in 0..s_dim {
            let col = j * s_dim;
            let mut sk = 0f32;
            for i in 0..s_dim {
                sk += st[col + i] * kh[i];
            }
            let d = (vh[j] - sk) * beta[h];
            let mut o = 0f32;
            for i in 0..s_dim {
                st[col + i] += kh[i] * d;
                o += st[col + i] * qh[i];
            }
            oh[j] = o;
        }
    }
    out
}

/// The linear block's output stage: per-head RMSNorm(o; ssm_norm) · silu(z),
/// in place on `o` (build_norm_gated).
pub fn gated_output_norm(dims: &DeltaDims, o: &mut [f32], w: &[f32], z: &[f32], eps: f32) {
    let s = dims.d_state;
    assert_eq!(w.len(), s);
    assert_eq!(o.len(), z.len());
    for h in 0..dims.n_v_heads {
        let oh = &mut o[h * s..(h + 1) * s];
        rmsnorm(oh, w, eps);
        for (ov, &zv) in oh.iter_mut().zip(&z[h * s..(h + 1) * s]) {
            *ov *= silu(zv);
        }
    }
}

/// The full-attention blocks' output gate: attn · sigmoid(gate), elementwise —
/// the gate rides interleaved in the joint Q projection ([q(hd)|gate(hd)] per
/// head); this applies it after attention, before wo (build_layer_attn).
pub fn attn_out_gate(attn: &mut [f32], gate: &[f32]) {
    for (a, &g) in attn.iter_mut().zip(gate) {
        *a *= sigmoid(g);
    }
}

// MRoPE note, pinned as a fact rather than an implementation: for TEXT input
// every position section carries the same position id, so multi-section rope
// degenerates to plain rope over the first 2·Σsections dims — proven
// source-cited + empirically in the qwen35-kernels lane (Detoro's ruling
// removing the MRoPE kernel); lane C therefore reuses the existing rope, and
// THIS module's oracle for the attention blocks' rope is the existing
// implementation, not a sectioned twin.

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> DeltaDims {
        DeltaDims { d_state: 2, n_v_heads: 2, n_k_heads: 1, d_conv: 2 }
    }

    #[test]
    fn scalar_forms_match_ggml() {
        assert_eq!(softplus(25.0), 25.0); // the >20 shortcut, exact
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        assert!(softplus(-100.0) >= 0.0 && softplus(-100.0) < 1e-6); // no -inf
        let mut v = [3.0f32, 4.0];
        l2_norm(&mut v, 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-7 && (v[1] - 0.8).abs() < 1e-7);
        let mut z = [0.0f32, 0.0];
        l2_norm(&mut z, 1e-6); // zero vector: scale = 1/eps, stays finite zero
        assert_eq!(z, [0.0, 0.0]);
    }

    #[test]
    fn delta_decode_hand_vector() {
        let d = tiny();
        let mut s = vec![0f32; d.delta_state_elems()];
        // step 1: empty state — s becomes β·k⊗v, o = (s q)/√S
        let o = delta_decode_step(&d, &mut s, &[1.0, 0.0], &[1.0, 0.0],
            &[3.0, 4.0, 5.0, 6.0], &[0.0, 0.0], &[1.0, 1.0]);
        let r2 = 2f32.sqrt();
        // head 0: s[i,j] = k_i·v_j → col j: s[0,j]=v_j; o_j = s[0,j]·(1/√2·... q=[1,0]→qs=[1/√2,0]
        assert!((o[0] - 3.0 / r2).abs() < 1e-6);
        assert!((o[1] - 4.0 / r2).abs() < 1e-6);
        assert!((o[2] - 5.0 / r2).abs() < 1e-6);
        assert!((o[3] - 6.0 / r2).abs() < 1e-6);
        // step 2: decay by e^ln(0.5)=0.5, new k orthogonal to old
        let g = 0.5f32.ln();
        let o2 = delta_decode_step(&d, &mut s, &[0.0, 1.0], &[0.0, 1.0],
            &[1.0, 1.0, 1.0, 1.0], &[g, g], &[0.5, 0.5]);
        // head 0 after decay: s[0,0]=1.5, s[0,1]=2.0; k=[0,1] → sk_j = s[1,j] = 0
        // d_j = 0.5; s[1,j] += 0.5; qs=[0,1/√2] → o_j = 0.5/√2
        assert!((o2[0] - 0.5 / r2).abs() < 1e-6);
        assert!((o2[1] - 0.5 / r2).abs() < 1e-6);
        // state kept the decayed first row
        assert!((s[0] - 1.5).abs() < 1e-6); // s[i=0,j=0,h=0]
        assert!((s[2] - 2.0).abs() < 1e-6); // s[i=0,j=1,h=0]
    }

    #[test]
    fn delta_state_layout_is_asymmetric() {
        // Negative control: transposing the state must CHANGE the output —
        // an oracle blind to the [i, j] order would wave through a kernel
        // with swapped indices.
        let d = tiny();
        let mut s1 = vec![0f32; d.delta_state_elems()];
        let _ = delta_decode_step(&d, &mut s1, &[1.0, 0.5], &[1.0, 0.25],
            &[3.0, 4.0, 5.0, 6.0], &[0.0, 0.0], &[1.0, 0.75]);
        let mut s2 = s1.clone();
        for h in 0..d.n_v_heads {
            for i in 0..2 {
                for j in 0..2 {
                    s2[h * 4 + j * 2 + i] = s1[h * 4 + i * 2 + j];
                }
            }
        }
        let q = [0.3, 0.9];
        let o1 = delta_decode_step(&d, &mut s1.clone(), &q, &[0.6, 0.2],
            &[1.0; 4], &[-0.1, -0.2], &[0.5, 0.5]);
        let o2 = delta_decode_step(&d, &mut s2, &q, &[0.6, 0.2],
            &[1.0; 4], &[-0.1, -0.2], &[0.5, 0.5]);
        assert_ne!(o1, o2);
    }

    #[test]
    fn conv_step_hand_vector_and_roll() {
        let d = DeltaDims { d_state: 1, n_v_heads: 1, n_k_heads: 1, d_conv: 2 };
        assert_eq!(d.conv_channels(), 3);
        let mut state = vec![10.0, 20.0, 30.0]; // one past column per channel
        let kernel = vec![1.0, 2.0, 0.5, 0.5, -1.0, 1.0]; // [c][k]
        let out = conv_step(&d, &mut state, &[1.0, 2.0, 3.0], &kernel);
        // c0: 10·1 + 1·2 = 12 → silu(12) ≈ 12·σ(12)
        assert!((out[0] - 12.0 * sigmoid(12.0)).abs() < 1e-4);
        // c1: 20·0.5 + 2·0.5 = 11
        assert!((out[1] - 11.0 * sigmoid(11.0)).abs() < 1e-4);
        // c2: 30·(−1) + 3·1 = −27 → silu(−27) ~ −27·σ(−27) ≈ −5e−11, near zero
        assert!(out[2].abs() < 1e-6);
        assert_eq!(state, vec![1.0, 2.0, 3.0]); // rolled
    }

    #[test]
    fn gated_norm_and_attn_gate() {
        let d = tiny();
        let mut o = vec![1.0f32, 2.0, 3.0, 4.0];
        let z = vec![0.0f32, 1.0, -1.0, 10.0];
        gated_output_norm(&d, &mut o, &[1.0, 1.0], &z, 1e-6);
        // head 0: rms = √((1+4)/2) → normed [1,2]/rms; · silu(z)
        let rms = (2.5f32 + 1e-6).sqrt();
        assert!((o[0] - 1.0 / rms * silu(0.0)).abs() < 1e-6);
        assert!((o[1] - 2.0 / rms * silu(1.0)).abs() < 1e-6);
        let mut a = vec![2.0f32, 2.0];
        attn_out_gate(&mut a, &[0.0, 100.0]);
        assert!((a[0] - 1.0).abs() < 1e-7); // σ(0)=0.5
        assert!((a[1] - 2.0).abs() < 1e-6); // σ(100)=1
    }

    #[test]
    fn adversarial_values_stay_finite() {
        let d = tiny();
        let mut s = vec![1e30f32; d.delta_state_elems()];
        // brutal decay: exp(-1e4) underflows to zero — state must go to β·k⊗v,
        // never NaN (0·inf shapes)
        let o = delta_decode_step(&d, &mut s, &[1.0, 1.0], &[1.0, 1.0],
            &[1.0; 4], &[-1e4, -1e4], &[1.0, 1.0]);
        assert!(o.iter().all(|x| x.is_finite()));
        assert!(s.iter().all(|x| x.is_finite()));
        // β = 0: state unchanged by the write, output = decayed-state read only
        let mut s2 = vec![0.5f32; d.delta_state_elems()];
        let before = s2.clone();
        let _ = delta_decode_step(&d, &mut s2, &[1.0, 0.0], &[1.0, 0.0],
            &[1.0; 4], &[0.0, 0.0], &[0.0, 0.0]);
        assert_eq!(s2, before);
        // denormal inputs stay finite through l2_norm
        let mut tiny_v = [1e-40f32, 1e-41];
        l2_norm(&mut tiny_v, 1e-6);
        assert!(tiny_v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn real_27b_dims_arithmetic() {
        let d = DeltaDims { d_state: 128, n_v_heads: 48, n_k_heads: 16, d_conv: 4 };
        assert_eq!(d.d_inner(), 6144);
        assert_eq!(d.conv_channels(), 10240);
        assert_eq!(d.conv_state_elems(), 30720);
        assert_eq!(d.delta_state_elems(), 786_432);
    }
}
