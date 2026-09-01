//! Shared fixtures for the gguf test modules: synthetic GGUF builders and
//! the local real-checkpoint paths the `--ignored` gates read.

#![allow(dead_code)]

    /// A tiny synthetic v3 file: one string KV, one Q8_0 tensor of ne=[32, 2]
    /// at offset 0. Exercises the header walk, alignment, dim reversal, and
    /// bounds checks without touching a real checkpoint.
    pub(crate) fn tiny_gguf(version: u32, align: Option<u64>) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&version.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        let n_kv = 1 + align.is_some() as u64;
        b.extend_from_slice(&n_kv.to_le_bytes());
        let put_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        put_str(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes()); // string
        put_str(&mut b, "llama");
        if let Some(a) = align {
            put_str(&mut b, "general.alignment");
            b.extend_from_slice(&4u32.to_le_bytes()); // u32
            b.extend_from_slice(&(a as u32).to_le_bytes());
        }
        // tensor info: name, n_dims=2, ne=[32,2] (fastest first), type Q8_0, offset 0
        put_str(&mut b, "t");
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&32u64.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        // pad to alignment, then 2 rows x 34 bytes of Q8_0 data
        let align = align.unwrap_or(32) as usize;
        while b.len() % align != 0 {
            b.push(0);
        }
        b.extend_from_slice(&vec![0u8; 68]);
        b
    }

    pub(crate) fn write_tmp(label: &str, bytes: &[u8]) -> std::path::PathBuf {
        // One file per TEST, not per process: the tests run in parallel and a
        // shared name makes one test open another's bytes.
        let p = std::env::temp_dir()
            .join(format!("lokal-gguf-test-{}-{label}.gguf", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    /// A file mixing several types this build cannot run, the shape unsloth's
    /// UD 1-2-bit checkpoints actually have: 3 unsupported types across 4 of 5
    /// tensors, plus one that IS supported.
    pub(crate) fn mixed_lowbit_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&5u64.to_le_bytes()); // n_tensors
        b.extend_from_slice(&1u64.to_le_bytes()); // n_kv
        let put_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        put_str(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes());
        put_str(&mut b, "llama");
        // (name, ggml type id): Q8_1=9, Q5_1=7, Q4_1=3, Q8_0=8. The whole
        // K and i-quant families run now, so the refused set is the legacy
        // non-K types — this list has shrunk every time the lane landed one.
        // Deliberately types this build still refuses — as the lane implements
        // more, this list moves rather than the assertions weakening.
        for (name, ty) in [
            ("token_embd.weight", 9u32),
            ("blk.0.ffn_down.weight", 9),
            ("blk.0.attn_q.weight", 7),
            ("blk.1.attn_q.weight", 3),
            ("output_norm.weight", 8),
        ] {
            put_str(&mut b, name);
            b.extend_from_slice(&2u32.to_le_bytes());
            b.extend_from_slice(&256u64.to_le_bytes());
            b.extend_from_slice(&2u64.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
        }
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend_from_slice(&vec![0u8; 4096]);
        b
    }

    pub(crate) fn qwen_gguf() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join(".cache/gguf/qwen2.5-0.5b-instruct-fp16.gguf")
    }

    pub(crate) fn qwen_hf_tokenizer() -> std::path::PathBuf {
        let snaps = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join(".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots");
        std::fs::read_dir(snaps)
            .unwrap()
            .flatten()
            .map(|e| e.path().join("tokenizer.json"))
            .find(|p| p.is_file())
            .expect("Qwen HF snapshot with tokenizer.json")
    }

    pub(crate) fn qwen35_kv(nextn: u64, with_array: bool) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // one dummy tensor
        let n_kv = 9 + (nextn > 0) as u64 + with_array as u64;
        b.extend_from_slice(&n_kv.to_le_bytes());
        let put_str = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };
        let put_u32 = |b: &mut Vec<u8>, k: &str, v: u32| {
            put_str(b, k);
            b.extend_from_slice(&4u32.to_le_bytes());
            b.extend_from_slice(&v.to_le_bytes());
        };
        put_str(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes());
        put_str(&mut b, "qwen35");
        put_u32(&mut b, "qwen35.block_count", 9);
        if nextn > 0 {
            put_u32(&mut b, "qwen35.nextn_predict_layers", nextn as u32);
        }
        put_u32(&mut b, "qwen35.full_attention_interval", 4);
        put_u32(&mut b, "qwen35.ssm.conv_kernel", 4);
        put_u32(&mut b, "qwen35.ssm.state_size", 128);
        put_u32(&mut b, "qwen35.ssm.group_count", 16);
        put_u32(&mut b, "qwen35.ssm.time_step_rank", 48);
        put_u32(&mut b, "qwen35.ssm.inner_size", 48 * 128);
        // rope sections [11, 11, 10, 0]
        put_str(&mut b, "qwen35.rope.dimension_sections");
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&5u32.to_le_bytes()); // i32 elements
        b.extend_from_slice(&4u64.to_le_bytes());
        for v in [11i32, 11, 10, 0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        if with_array {
            put_str(&mut b, "qwen35.attention.recurrent_layers");
            b.extend_from_slice(&9u32.to_le_bytes());
            b.extend_from_slice(&7u32.to_le_bytes()); // bool elements
            b.extend_from_slice(&8u64.to_le_bytes());
            for v in [1u8, 0, 1, 1, 1, 0, 1, 1] {
                b.push(v);
            }
        }
        // dummy tensor so the parser has a table
        put_str(&mut b, "t");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&32u64.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes()); // Q8_0
        b.extend_from_slice(&0u64.to_le_bytes());
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend_from_slice(&vec![0u8; 34]);
        b
    }

    pub(crate) fn hf_tokenizer_json(repo_dir: &str) -> std::path::PathBuf {
        let snaps = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join(".cache/huggingface/hub")
            .join(repo_dir)
            .join("snapshots");
        std::fs::read_dir(snaps)
            .unwrap()
            .flatten()
            .map(|e| e.path().join("tokenizer.json"))
            .find(|p| p.is_file())
            .expect("snapshot with tokenizer.json")
    }

/// SERIALISES EVERY TEST THAT DRIVES THE METAL DEVICE.
///
/// Diagnosed in lane gpu-test-serialization rather than assumed, because the
/// plan asked which of two causes it was. It is NOT a missing completion wait:
/// every GPU test already commits and waits (17 sites in metal.rs, 1 in
/// pool.rs), and with a status assertion added the command buffer reports
/// `Completed`. It is the device itself under concurrent pressure — with two
/// tests driving it at once, a dispatch reports Completed and the ENTIRE output
/// buffer is still zero, reproduced 3/3 (768 of 768 elements, quant_oracle).
/// A freshly allocated Metal buffer is zero-filled, so an unexecuted dispatch is
/// indistinguishable from a kernel that wrote zeros — which is why this went
/// unnoticed until a test with ~1024 small dispatches joined the suite.
///
/// Hold it around the device work, not around a whole test body: callers that
/// take it inside their GPU helper still let the suite run in parallel
/// everywhere else, which is the point.
///
/// Poisoning is deliberately IGNORED. A panicking GPU test poisons this mutex,
/// and propagating that would turn one real failure into a cascade of unrelated
/// ones — exactly the false-red habit the workspace has a protocol against. The
/// guard protects a device, not data, so there is no invariant a panic can leave
/// broken.
pub(crate) fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
