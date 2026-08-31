//! GGUF: a checkpoint format, not a backend detail (spec Core Principle —
//! docs/gguf-design.md). Every backend reads checkpoints through this module:
//! the container parser, the arch mapping, the embedded tokenizer, and the
//! reference dequant layer the GPU pipelines are verified against.

pub(crate) mod arch;
pub(crate) mod container;
pub(crate) mod dequant;
pub(crate) mod tokenizer;

pub use arch::{
    hf_name, model_config, qwen35_meta, unpermute_llama_qk, GgufArch, Qwen35Meta,
};
pub use container::{expanded_f32_bytes, phys_ram_bytes, quant_bytes, summary, GgufFile};
pub use dequant::{dequant_row_ref, load_f32, GgmlType, GgufTensor};
pub use tokenizer::build_tokenizer;

#[cfg(test)]
pub(crate) mod testutil;
