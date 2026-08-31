//! WeightManifest — the mmap layer under -b lowmem.
//!
//! Opens every safetensors shard once, keeps the mmaps plus a name → location
//! table, and hands out bytes on demand. Nothing is read up front: the OS pages
//! data in as it is touched and stays free to drop clean pages under pressure,
//! which is what lets a model larger than RAM open at all.

use memmap2::{Advice, Mmap};
use metal::{Buffer, Device, MTLResourceOptions};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

/// Where one tensor's bytes live: shard index + absolute byte range, plus the
/// dtype and shape needed to interpret them.
pub(crate) struct TensorMeta {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub shard: usize,
    pub offset: usize,
    pub len: usize,
}

pub(crate) struct WeightManifest {
    /// No-copy MTLBuffers wrapping each shard's mmap, so kernels read weight
    /// bytes straight from the page cache — the pattern llama.cpp ships for
    /// larger-than-RAM models on Apple silicon: file-backed pages stay
    /// reclaimable, fault in on first GPU touch, and never exist twice.
    /// (Human-directed exception to the spec's original never-hand-mmap-to-
    /// Metal rule, recorded on the task.) Declared before `shards` so the
    /// buffers drop before the mappings they alias.
    views: Vec<Buffer>,
    shards: Vec<Mmap>,
    tensors: HashMap<String, TensorMeta>,
    /// Total parameter count, summed from the headers — no data was read for it.
    pub n_params: usize,
}

// The views alias plain readonly mmaps; Metal buffer handles are documented
// thread-safe (same justification as the engines).
unsafe impl Send for WeightManifest {}
unsafe impl Sync for WeightManifest {}

impl WeightManifest {
    pub fn open(dir: &Path) -> crate::Result<Self> {
        let mut shards = Vec::new();
        let mut tensors = HashMap::new();
        for (si, name) in crate::weights::shard_files(dir)?.into_iter().enumerate() {
            let file = std::fs::File::open(dir.join(&name))?;
            let mmap = unsafe { Mmap::map(&file)? };
            // Staging sweeps run front to back within a shard; tell the pager so
            // readahead works with us (random faults are what kill disk-backed runs).
            let _ = mmap.advise(Advice::Sequential);
            let st = SafeTensors::deserialize(&mmap)?;
            let base = mmap.as_ptr() as usize;
            for (tname, view) in st.tensors() {
                let data = view.data();
                tensors.insert(
                    tname,
                    TensorMeta {
                        dtype: view.dtype(),
                        shape: view.shape().to_vec(),
                        shard: si,
                        offset: data.as_ptr() as usize - base,
                        len: data.len(),
                    },
                );
            }
            drop(st);
            shards.push(mmap);
        }
        let n_params = tensors.values().map(|m| m.shape.iter().product::<usize>()).sum();
        Ok(Self { views: Vec::new(), shards, tensors, n_params })
    }

    /// Create the GPU views (once, at engine build). mmap bases are page
    /// aligned; the length rounds up to the page the mapping already spans.
    pub fn make_gpu_views(&mut self, device: &Device) {
        const PAGE: usize = 16384;
        self.views = self
            .shards
            .iter()
            .map(|m| {
                device.new_buffer_with_bytes_no_copy(
                    m.as_ptr() as *const _,
                    m.len().next_multiple_of(PAGE) as u64,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
            })
            .collect();
    }

    /// GPU view + absolute byte offset for rows r0..r1 of a 2-D tensor, or
    /// None when views are absent or the span can't be read as ushorts.
    pub fn gpu_span(
        &self,
        name: &str,
        r0: usize,
        r1: usize,
    ) -> crate::Result<Option<(&Buffer, usize)>> {
        let m = self.meta(name)?;
        let n_rows = m.shape.first().copied().unwrap_or(0);
        if m.shape.len() != 2 || r1 > n_rows || r0 >= r1 {
            return Err(format!("gpu_span({name}, {r0}..{r1}): tensor has shape {:?}", m.shape).into());
        }
        let off = m.offset + r0 * (m.len / n_rows);
        Ok(match self.views.get(m.shard) {
            Some(b) if off % 2 == 0 => Some((b, off)),
            _ => None,
        })
    }

    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn meta(&self, name: &str) -> crate::Result<&TensorMeta> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("tensor {name} not found in the weight files").into())
    }

    /// The tensor's raw on-disk bytes.
    fn bytes(&self, m: &TensorMeta) -> &[u8] {
        &self.shards[m.shard][m.offset..m.offset + m.len]
    }

    /// Whole tensor converted to f32 — norms, biases, and the phase-1 eager
    /// path. The paged path must never call this on a large matrix.
    pub fn read_f32(&self, name: &str) -> crate::Result<Vec<f32>> {
        let m = self.meta(name)?;
        crate::weights::to_f32(m.dtype, self.bytes(m))
    }

    /// Raw bytes of rows r0..r1 of a 2-D tensor — one contiguous mmap slice
    /// (rows are contiguous on disk), plus the dtype to interpret it. This is
    /// the read the pager stages pages through; the slice may be unaligned, so
    /// callers convert via byte chunks, never typed pointers.
    pub fn read_rows(&self, name: &str, r0: usize, r1: usize) -> crate::Result<(&[u8], Dtype)> {
        let m = self.meta(name)?;
        let n_rows = m.shape.first().copied().unwrap_or(0);
        if m.shape.len() != 2 || r1 > n_rows || r0 >= r1 {
            return Err(format!(
                "read_rows({name}, {r0}..{r1}): tensor has shape {:?}",
                m.shape
            )
            .into());
        }
        let row_bytes = m.len / n_rows;
        Ok((&self.bytes(m)[r0 * row_bytes..r1 * row_bytes], m.dtype))
    }
}

// ---------- The GGUF seam (frozen: lane gguf-kernels builds against this) ----------
//
// Layouts and semantics below are transcribed from llama.cpp's
// ggml/src/ggml-common.h and ggml-quants.c (checked out read-only at review
// time), NOT from memory. `dequant_row_ref` is the numerics oracle: the GPU
// dequant must match it bit-for-bit (through f16 rounding where the kernel
// dequants to f16).

/// The ggml tensor types lokal runs. Discriminants are the on-disk GGUF ids
/// (enum ggml_type in ggml.h) — `from_gguf` is the only constructor.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)] // names match ggml's exactly — grep-ability beats style
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
}

impl GgmlType {
    /// GGUF tensor-info type id → supported type, or the type's NAME for the
    /// refusal message ("re-download as Q4_K_M/Q8_0").
    pub fn from_gguf(id: u32) -> Result<Self, &'static str> {
        Ok(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            // Names for everything ggml defines, so the error can say what the
            // file actually holds (ggml.h enum ggml_type).
            3 => return Err("Q4_1"),
            6 => return Err("Q5_0"),
            7 => return Err("Q5_1"),
            9 => return Err("Q8_1"),
            10 => return Err("Q2_K"),
            11 => return Err("Q3_K"),
            15 => return Err("Q8_K"),
            16 => return Err("IQ2_XXS"),
            17 => return Err("IQ2_XS"),
            18 => return Err("IQ3_XXS"),
            19 => return Err("IQ1_S"),
            20 => return Err("IQ4_NL"),
            21 => return Err("IQ3_S"),
            22 => return Err("IQ2_S"),
            23 => return Err("IQ4_XS"),
            24..=27 => return Err("integer tensor"),
            28 => return Err("F64"),
            29 => return Err("IQ1_M"),
            30 => return Err("BF16"),
            _ => return Err("unknown ggml type"),
        })
    }

    /// Elements per quantization block (1 for the float types).
    pub fn blk_elems(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q8_0 => 32,          // QK4_0 / QK8_0
            Self::Q4_K | Self::Q5_K | Self::Q6_K => 256, // QK_K
        }
    }

    /// Bytes per block, matching sizeof(block_*) in ggml-common.h exactly.
    pub fn blk_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 2 + 16,            // f16 d + 32 nibbles
            Self::Q8_0 => 2 + 32,            // f16 d + 32 i8
            Self::Q4_K => 2 + 2 + 12 + 128,  // d + dmin + scales[12] + qs[QK_K/2]
            Self::Q5_K => 2 + 2 + 12 + 32 + 128, // … + qh[QK_K/8]
            Self::Q6_K => 128 + 64 + 16 + 2, // ql[QK_K/2] + qh[QK_K/4] + scales[16] + d
        }
    }

    /// Byte size of a row of `n` elements (n must be a whole number of blocks).
    pub fn row_bytes(self, n: usize) -> usize {
        n / self.blk_elems() * self.blk_bytes()
    }
}

/// One tensor as the GGUF file stores it: `data` is the raw (possibly
/// quantized) bytes inside the file's mmap. `dims` is ROW-MAJOR — GGUF's `ne`
/// order (fastest-varying first) is reversed here so `dims == [rows, cols]`
/// for a 2-D weight, matching TensorMeta.shape and every safetensors path in
/// the tree. A row is `dims.last()` contiguous elements, quantized block by
/// block along the row.
pub struct GgufTensor<'a> {
    pub name: String,
    pub dims: Vec<usize>,
    pub ty: GgmlType,
    pub data: &'a [u8],
}

fn f16_bits_at(src: &[u8], off: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([src[off], src[off + 1]])).to_f32()
}

/// The 6-bit scale/min unpack for Q4_K/Q5_K super-blocks —
/// get_scale_min_k4 in ggml-quants.c, verbatim.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// CPU reference dequantization of one row: exact ggml semantics per type
/// (dequantize_row_* in ggml-quants.c). `src` holds `out.len()` elements'
/// worth of blocks; panics on size mismatch — callers size via `row_bytes`.
pub fn dequant_row_ref(ty: GgmlType, src: &[u8], out: &mut [f32]) {
    assert_eq!(src.len(), ty.row_bytes(out.len()), "src/out size mismatch for {ty:?}");
    match ty {
        GgmlType::F32 => {
            for (i, y) in out.iter_mut().enumerate() {
                *y = f32::from_le_bytes(src[4 * i..4 * i + 4].try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (i, y) in out.iter_mut().enumerate() {
                *y = f16_bits_at(src, 2 * i);
            }
        }
        GgmlType::Q4_0 => {
            for (b, y) in out.chunks_exact_mut(32).enumerate() {
                let blk = &src[b * 18..b * 18 + 18];
                let d = f16_bits_at(blk, 0);
                for j in 0..16 {
                    let q = blk[2 + j];
                    y[j] = ((q & 0x0F) as i32 - 8) as f32 * d;
                    y[j + 16] = ((q >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
        GgmlType::Q8_0 => {
            for (b, y) in out.chunks_exact_mut(32).enumerate() {
                let blk = &src[b * 34..b * 34 + 34];
                let d = f16_bits_at(blk, 0);
                for j in 0..32 {
                    y[j] = (blk[2 + j] as i8) as f32 * d;
                }
            }
        }
        GgmlType::Q4_K => {
            for (b, y) in out.chunks_exact_mut(256).enumerate() {
                let blk = &src[b * 144..b * 144 + 144];
                let d = f16_bits_at(blk, 0);
                let dmin = f16_bits_at(blk, 2);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                let mut is = 0;
                for j in (0..256).step_by(64) {
                    let q = &qs[j / 2..j / 2 + 32];
                    let (sc, m) = scale_min_k4(is, scales);
                    let (d1, m1) = (d * sc as f32, dmin * m as f32);
                    let (sc, m) = scale_min_k4(is + 1, scales);
                    let (d2, m2) = (d * sc as f32, dmin * m as f32);
                    for l in 0..32 {
                        y[j + l] = d1 * (q[l] & 0x0F) as f32 - m1;
                        y[j + 32 + l] = d2 * (q[l] >> 4) as f32 - m2;
                    }
                    is += 2;
                }
            }
        }
        GgmlType::Q5_K => {
            for (b, y) in out.chunks_exact_mut(256).enumerate() {
                let blk = &src[b * 176..b * 176 + 176];
                let d = f16_bits_at(blk, 0);
                let dmin = f16_bits_at(blk, 2);
                let scales = &blk[4..16];
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                let mut is = 0;
                let (mut u1, mut u2) = (1u8, 2u8);
                for j in (0..256).step_by(64) {
                    let ql = &qs[j / 2..j / 2 + 32];
                    let (sc, m) = scale_min_k4(is, scales);
                    let (d1, m1) = (d * sc as f32, dmin * m as f32);
                    let (sc, m) = scale_min_k4(is + 1, scales);
                    let (d2, m2) = (d * sc as f32, dmin * m as f32);
                    for l in 0..32 {
                        let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                        let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                        y[j + l] = d1 * ((ql[l] & 0x0F) + hi1) as f32 - m1;
                        y[j + 32 + l] = d2 * ((ql[l] >> 4) + hi2) as f32 - m2;
                    }
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        GgmlType::Q6_K => {
            for (b, y) in out.chunks_exact_mut(256).enumerate() {
                let blk = &src[b * 210..b * 210 + 210];
                let d = f16_bits_at(blk, 208);
                for n in (0..256).step_by(128) {
                    let ql = &blk[n / 2..n / 2 + 64];
                    let qh = &blk[128 + n / 4..128 + n / 4 + 32];
                    let sc = &blk[192 + n / 16..192 + n / 16 + 8];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i8 - 32;
                        let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                        y[n + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        y[n + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        y[n + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        y[n + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod gguf_seam_tests {
    use super::*;

    fn f16b(x: f32) -> [u8; 2] {
        half::f16::from_f32(x).to_bits().to_le_bytes()
    }

    #[test]
    fn blk_bytes_match_ggml_struct_sizes() {
        // The static_asserts in ggml-common.h pin these exact sizes.
        assert_eq!(GgmlType::Q4_0.blk_bytes(), 18);
        assert_eq!(GgmlType::Q8_0.blk_bytes(), 34);
        assert_eq!(GgmlType::Q4_K.blk_bytes(), 144);
        assert_eq!(GgmlType::Q5_K.blk_bytes(), 176);
        assert_eq!(GgmlType::Q6_K.blk_bytes(), 210);
        assert_eq!(GgmlType::Q4_K.row_bytes(512), 288);
    }

    #[test]
    fn q4_0_hand_vector() {
        // d = 2.0; qs[0] = 0xA3: lo nibble 3 -> (3-8)*2 = -10 at y[0],
        // hi nibble 0xA -> (10-8)*2 = 4 at y[16]. All other bytes 0x88 -> 0.
        let mut blk = vec![0u8; 18];
        blk[..2].copy_from_slice(&f16b(2.0));
        blk[2] = 0xA3;
        for b in &mut blk[3..18] {
            *b = 0x88;
        }
        let mut y = [9.9f32; 32];
        dequant_row_ref(GgmlType::Q4_0, &blk, &mut y);
        assert_eq!(y[0], -10.0);
        assert_eq!(y[16], 4.0);
        assert!(y[1..16].iter().chain(&y[17..]).all(|&v| v == 0.0));
    }

    #[test]
    fn q8_0_hand_vector_and_roundtrip_within_one_lsb() {
        let mut blk = vec![0u8; 34];
        blk[..2].copy_from_slice(&f16b(0.5));
        blk[2] = (-7i8) as u8;
        blk[33] = 100;
        let mut y = [0f32; 32];
        dequant_row_ref(GgmlType::Q8_0, &blk, &mut y);
        assert_eq!(y[0], -3.5);
        assert_eq!(y[31], 50.0);

        // Property: any |x| <= 127*d round-trips through Q8_0 within one LSB (= d).
        let d = 0.03125f32; // exactly representable in f16
        let mut xs = [0f32; 32];
        let mut state = 0x243F6A88u32; // deterministic LCG, no rand dep
        for x in xs.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *x = ((state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * 127.0 * d;
        }
        let mut blk = vec![0u8; 34];
        blk[..2].copy_from_slice(&f16b(d));
        for (i, &x) in xs.iter().enumerate() {
            blk[2 + i] = (x / d).round().clamp(-127.0, 127.0) as i8 as u8;
        }
        let mut y = [0f32; 32];
        dequant_row_ref(GgmlType::Q8_0, &blk, &mut y);
        for (x, y) in xs.iter().zip(&y) {
            assert!((x - y).abs() <= d * 0.5 + 1e-7, "x={x} y={y}");
        }
    }

    #[test]
    fn q4_k_hand_vector() {
        // d=1.0, dmin=0.5; scales[0]=3 (group-0 scale), scales[4]=2 (group-0 min);
        // qs[0]=0x51: y[0] = 3*1 - 0.5*2 = 2.0; hi nibble 5 lands in group 1,
        // whose scale/min are 0 -> y[32] = 0.
        let mut blk = vec![0u8; 144];
        blk[..2].copy_from_slice(&f16b(1.0));
        blk[2..4].copy_from_slice(&f16b(0.5));
        blk[4] = 3;
        blk[8] = 2;
        blk[16] = 0x51;
        let mut y = [0f32; 256];
        dequant_row_ref(GgmlType::Q4_K, &blk, &mut y);
        assert_eq!(y[0], 2.0);
        assert_eq!(y[1], -1.0); // 3*0 - 0.5*2
        assert_eq!(y[32], 0.0);
    }

    #[test]
    fn q5_k_hand_vector() {
        // Same scales as the Q4_K vector; ql[0]=1 with qh[0] bit0 set -> value
        // 1+16 = 17: y[0] = 3*17 - 0.5*2 = 50.0.
        let mut blk = vec![0u8; 176];
        blk[..2].copy_from_slice(&f16b(1.0));
        blk[2..4].copy_from_slice(&f16b(0.5));
        blk[4] = 3;
        blk[8] = 2;
        blk[16] = 0x01; // qh[0], bit 0
        blk[48] = 0x01; // ql[0]
        let mut y = [0f32; 256];
        dequant_row_ref(GgmlType::Q5_K, &blk, &mut y);
        assert_eq!(y[0], 50.0);
        assert_eq!(y[1], -1.0);
    }

    #[test]
    fn q6_k_hand_vector() {
        // d=0.25; scales[0]=2, scales[4]=1; ql[0]=0x21, qh[0]=0:
        // q1 = 1-32 = -31  -> y[0]  = 0.25*2*(-31) = -15.5
        // q3 = 2-32 = -30  -> y[64] = 0.25*1*(-30) = -7.5
        let mut blk = vec![0u8; 210];
        blk[0] = 0x21; // ql[0]
        blk[192] = 2; // scales[0]
        blk[196] = 1; // scales[4]
        blk[208..210].copy_from_slice(&f16b(0.25));
        let mut y = [0f32; 256];
        dequant_row_ref(GgmlType::Q6_K, &blk, &mut y);
        assert_eq!(y[0], -15.5);
        assert_eq!(y[64], -7.5);
        // ql[32] = 0, qh[32-region] = 0 -> q2 = -32, scales[2] = 0 -> y[32] = 0
        assert_eq!(y[32], 0.0);
    }

    #[test]
    fn from_gguf_names_the_unsupported_type() {
        assert_eq!(GgmlType::from_gguf(12).unwrap(), GgmlType::Q4_K);
        assert_eq!(GgmlType::from_gguf(10).unwrap_err(), "Q2_K");
        assert_eq!(GgmlType::from_gguf(23).unwrap_err(), "IQ4_XS");
    }
}
