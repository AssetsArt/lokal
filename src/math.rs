//! All the arithmetic a transformer actually needs, in one small file:
//! matrix-vector multiply, RMSNorm, softmax, and SiLU.

use rayon::prelude::*;

/// y = W·x where W has shape [out_dim, in_dim], stored row-major.
///
/// This is where 90%+ of inference time goes, so the rows are split across
/// cores — each row of W is an independent dot product.
pub fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    assert_eq!(w.len(), out_dim * in_dim);
    assert_eq!(x.len(), in_dim);
    let mut y = vec![0.0; out_dim];
    y.par_iter_mut().enumerate().for_each(|(o, yo)| {
        *yo = dot(&w[o * in_dim..(o + 1) * in_dim], x);
    });
    y
}

/// Dot product of two vectors.
///
/// Accumulates into 8 independent lanes and combines them at the end. Breaking
/// the sequential addition chain lets the compiler emit SIMD instructions
/// (strict left-to-right float addition cannot be vectorized).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        for i in 0..8 {
            acc[i] += a[c * 8 + i] * b[c * 8 + i];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for i in chunks * 8..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// RMSNorm (from "Root Mean Square Layer Normalization").
///
/// Rescales the vector to unit RMS, then multiplies by a learned per-dimension
/// weight. Unlike LayerNorm it skips mean subtraction — cheaper with no quality
/// loss, which is why modern LLMs standardized on it.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = dot(x, x) / x.len() as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt(); // eps guards against division by zero
    x.iter().zip(weight).map(|(xi, wi)| xi * scale * wi).collect()
}

/// In-place softmax: turns raw scores (logits) into probabilities that sum to 1.
///
/// The max is subtracted before exp to prevent overflow — shifting by a constant
/// does not change the result.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// SiLU (a.k.a. swish): x·sigmoid(x) — the activation Llama uses in its MLP.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_matches_hand_computation() {
        // W = [[1,2,3],[4,5,6]] (2 rows, 3 cols), x = [1,1,1] → y = [6, 15]
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = matvec(&w, &[1.0, 1.0, 1.0], 2, 3);
        assert_eq!(y, vec![6.0, 15.0]);
    }

    #[test]
    fn dot_handles_length_not_divisible_by_eight() {
        let a: Vec<f32> = (1..=11).map(|i| i as f32).collect();
        let expected: f32 = a.iter().map(|v| v * v).sum();
        assert_eq!(dot(&a, &a), expected);
    }

    #[test]
    fn softmax_sums_to_one_and_keeps_order() {
        let mut x = [1.0, 3.0, 2.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[1] > x[2] && x[2] > x[0]); // relative order must be preserved
    }

    #[test]
    fn rmsnorm_normalizes_magnitude() {
        // Every element 3 with unit weights → RMS is 3 → every element becomes ~1.
        let x = [3.0; 4];
        let w = [1.0; 4];
        let y = rmsnorm(&x, &w, 1e-6);
        for v in y {
            assert!((v - 1.0).abs() < 1e-3);
        }
    }
}
