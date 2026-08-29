//! GPU backends — one file per vendor API.
//!
//! Adding a new backend (e.g. CUDA for NVIDIA, ROCm/Vulkan for AMD):
//!   1. Create src/gpu/<name>.rs and implement the Engine + Session traits (see engine.rs)
//!   2. Add an arm to engine::create()
//!   3. Gate it with #[cfg] for its platform and add a target-specific dependency in Cargo.toml
//!
//! Invariants every backend should keep (lessons baked into metal.rs):
//!   - Decode is memory-bandwidth-bound: upload weights once at engine creation,
//!     keep them resident on the device as f16 — never re-send them per token
//!   - Encode a whole token's forward pass as one submission; syncing per op
//!     (~450 times/token) costs more than the math itself
//!   - Cross the CPU↔device boundary as little as possible: token ids in, logits out

#[cfg(target_os = "macos")]
pub mod metal;
