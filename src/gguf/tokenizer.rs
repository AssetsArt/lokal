//! The tokenizer built from the GGUF's embedded vocab: NFC → Split(per-arch
//! regex) → ByteLevel, BPE from tokenizer.ggml.tokens/merges, control tokens
//! registered as special. Regexes are transcribed from a read-only llama.cpp
//! checkout (src/llama-vocab.cpp), not from memory.

use crate::gguf::container::GgufFile;

// ---------- tokenizer from the embedded vocab (D5) ----------

/// Build a `tokenizers::Tokenizer` equivalent to the model's HF tokenizer.json:
/// NFC → Split(arch regex) → ByteLevel, BPE from tokenizer.ggml.tokens/merges,
/// control tokens (token_type 3) registered as special so they never round-trip
/// through byte decoding (llama3's reserved tokens break naive decoders).
pub fn build_tokenizer(g: &GgufFile) -> crate::Result<tokenizers::Tokenizer> {
    use tokenizers::models::bpe::BPE;
    use tokenizers::normalizers::unicode::NFC;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::pre_tokenizers::sequence::Sequence;
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
    use tokenizers::{AddedToken, SplitDelimiterBehavior};

    let model = g.get_str("tokenizer.ggml.model")?;
    if model != "gpt2" {
        return Err(format!(
            "GGUF tokenizer model \"{model}\" is not supported yet — byte-level BPE (\"gpt2\") only; \
             use the safetensors checkpoint for this model"
        )
        .into());
    }
    let tokens = g.get_arr_str("tokenizer.ggml.tokens")?;
    let merges_raw = g.get_arr_str("tokenizer.ggml.merges")?;
    let token_type = g.get_arr_i64("tokenizer.ggml.token_type").unwrap_or(&[]);

    let vocab: tokenizers::models::bpe::Vocab =
        tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    // Merges are "left right" pairs; byte-level tokens never contain a real
    // space (space is Ġ), so the first space is the separator.
    let merges: Vec<(String, String)> = merges_raw
        .iter()
        .map(|m| {
            m.split_once(' ')
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .ok_or_else(|| format!("malformed merge entry {m:?}"))
        })
        .collect::<Result<_, _>>()?;

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| format!("BPE build: {e}"))?;
    let mut tok = tokenizers::Tokenizer::new(bpe);

    // The split regex is per pre-tokenizer family (tokenizer.ggml.pre), each
    // string lifted from the model's own tokenizer.json via llama-vocab.cpp.
    let pre = g.get_str("tokenizer.ggml.pre").unwrap_or("gpt2");
    let regex = match pre {
        "qwen2" => {
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        }
        // qwen2's split plus \p{M}: combining marks travel with their letters
        // (llama-vocab.cpp PRE_TYPE_QWEN35, lifted from the model's tokenizer.json).
        "qwen35" => {
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        }
        "llama3" | "llama-bpe" => {
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        }
        // GPT-2's original — the ecosystem default for unmarked BPE files.
        _ => r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    };
    let split = Split::new(SplitPattern::Regex(regex.into()), SplitDelimiterBehavior::Isolated, false)
        .map_err(|e| format!("pre-tokenizer regex: {e}"))?;
    let _ = tok.with_normalizer(Some(NFC));
    let _ = tok.with_pre_tokenizer(Some(Sequence::new(vec![
        split.into(),
        ByteLevel::new(false, false, false).into(),
    ])));
    let _ = tok.with_decoder(Some(ByteLevel::new(false, false, false)));

    // token_type 3 = control (llama.cpp's LLAMA_TOKEN_TYPE_CONTROL).
    let specials: Vec<AddedToken> = token_type
        .iter()
        .enumerate()
        .filter(|&(_, &t)| t == 3)
        .filter_map(|(i, _)| tokens.get(i))
        .map(|s| AddedToken::from(s.clone(), true))
        .collect();
    if !specials.is_empty() {
        let _ = tok.add_special_tokens(specials);
    }
    Ok(tok)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "needs the local GGUF + HF checkpoints"]
    fn gguf_tokenizer_matches_hf_on_mixed_corpus() {
        let smol_gguf = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(
            ".cache/huggingface/hub/models--unsloth--SmolLM2-135M-Instruct-GGUF/snapshots/9e6855bc4be717fca1ef21360a1db4b29d5c559a/SmolLM2-135M-Instruct-F16.gguf",
        );
        for (gguf, hf_json) in [
            (crate::gguf::testutil::qwen_gguf(), crate::gguf::testutil::hf_tokenizer_json("models--Qwen--Qwen2.5-0.5B-Instruct")),
            (smol_gguf, crate::gguf::testutil::hf_tokenizer_json("models--HuggingFaceTB--SmolLM2-135M-Instruct")),
        ] {
            check_tokenizer_pair(&gguf, &hf_json);
        }
    }

    fn check_tokenizer_pair(gguf: &std::path::Path, hf_json: &std::path::Path) {
        let g = GgufFile::open(gguf).unwrap();
        let ours = build_tokenizer(&g).unwrap();
        let hf = tokenizers::Tokenizer::from_file(hf_json).unwrap();
        let corpus = [
            "สวัสดีครับ วันนี้อากาศดีมาก ๆ เลยนะครับ",
            "ภาษาไทยไม่มีการเว้นวรรคระหว่างคำ ทำให้ tokenizer ต้องทำงานหนัก",
            "Hello, world! I'll say it again: don't panic.",
            "Mixed ไทย English และ 中文 plus ελληνικά in one line",
            "🎉🚀 emoji test 👨‍👩‍👧‍👦 with a ZWJ family and 🇹🇭 a flag",
            "numbers 1234567890, 3.14159, 1e-9, 0xDEADBEEF",
            "code: fn main() { println!(\"hi\"); }\n\tindented\ttabs  and   runs of spaces",
            "trailing spaces   \nและบรรทัดใหม่\r\nwindows line endings",
            "",
            " leading space",
        ];
        for line in corpus {
            let a = ours.encode(line, false).unwrap();
            let b = hf.encode(line, false).unwrap();
            assert_eq!(a.get_ids(), b.get_ids(), "encode mismatch on {line:?}");
            let da = ours.decode(a.get_ids(), true).unwrap();
            let db = hf.decode(b.get_ids(), true).unwrap();
            assert_eq!(da, db, "decode mismatch on {line:?}");
        }
    }
}
