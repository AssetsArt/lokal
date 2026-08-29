//! Picks the next token from the logits — the only place in the whole system
//! with any randomness (the model itself is fully deterministic).

use crate::math::softmax;

pub struct Sampler {
    temperature: f32,
    top_p: f32,
    rng_state: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Self {
        Self { temperature, top_p, rng_state: seed.max(1) } // xorshift state must be nonzero
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        // temperature 0 → greedy: always take the top score (bit-for-bit reproducible).
        if self.temperature <= 0.0 {
            return argmax(logits);
        }

        // Higher temperature flattens the distribution → more adventurous choices.
        for v in logits.iter_mut() {
            *v /= self.temperature;
        }
        softmax(logits);

        // Nucleus (top-p) sampling: sort by probability, keep only the head of the
        // distribution whose cumulative mass reaches top_p, then sample from that.
        // This trims the long tail of nonsense tokens.
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        idx.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));

        let mut cum = 0.0;
        let mut cut = idx.len();
        for (i, &id) in idx.iter().enumerate() {
            cum += logits[id as usize];
            if cum >= self.top_p {
                cut = i + 1;
                break;
            }
        }

        // Throw a dart at the cumulative probability bar [0, cum) and see where it lands.
        let r = self.next_f32() * cum;
        let mut acc = 0.0;
        for &id in &idx[..cut] {
            acc += logits[id as usize];
            if r < acc {
                return id;
            }
        }
        idx[cut - 1] // guard against float rounding at the boundary
    }

    /// xorshift64* — a tiny, adequate RNG in four lines; no crate needed.
    fn next_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        let x = self.rng_state.wrapping_mul(0x2545F4914F6CDD1D);
        (x >> 40) as f32 / (1u64 << 24) as f32 // top 24 bits → value in [0,1)
    }
}

fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in x.iter().enumerate() {
        if *v > x[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_the_highest_logit() {
        let mut s = Sampler::new(0.0, 0.9, 42);
        assert_eq!(s.sample(&mut [0.1, 5.0, -2.0, 1.0]), 1);
    }

    #[test]
    fn tiny_top_p_collapses_to_greedy() {
        // A tiny top_p leaves a single token in the nucleus → must match argmax.
        for seed in 1..50 {
            let mut s = Sampler::new(1.0, 0.01, seed);
            assert_eq!(s.sample(&mut [0.0, 0.0, 10.0, 0.0]), 2);
        }
    }

    #[test]
    fn same_seed_gives_same_sequence() {
        let logits = [1.0f32, 0.9, 0.8, 0.7];
        let mut a = Sampler::new(1.0, 0.95, 7);
        let mut b = Sampler::new(1.0, 0.95, 7);
        for _ in 0..20 {
            assert_eq!(a.sample(&mut logits.clone()), b.sample(&mut logits.clone()));
        }
    }
}
