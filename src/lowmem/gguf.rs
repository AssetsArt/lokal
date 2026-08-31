//! GGUF checkpoint reading (lane gguf-loader). The parser, the ggml type
//! table, and the CPU reference dequant land here; src/lowmem/manifest.rs
//! carries the shared seam types (GgmlType, GgufTensor, dequant_row_ref).
