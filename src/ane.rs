//! The "ane" backend — a hybrid: prefill on the Apple Neural Engine, decode on Metal.
//!
//! The ANE has no direct programming API — the only public road in is Core ML. The
//! prefill graph is exported ahead of time to a .mlmodelc file (see
//! tools/export_prefill.py); this module loads and invokes it through the
//! Objective-C bindings (objc2-core-ml).
//!
//! Division of labor:
//!   prompt ──→ [Core ML → ANE] computes K,V for every position at once (up to 512)
//!          ──→ memcpy K,V into Metal's KV cache (unified memory — nearly free)
//!          ──→ the prompt's final token + anything past 512 + all decoding → Metal
//!
//! Why bother: the ANE draws far less power than the GPU, and moving prefill off the
//! GPU leaves it free to decode for other requests (which matters in serve mode) —
//! not because the ANE is "faster".
//!
//! Caveat to know: Core ML *decides* whether the graph actually lands on the ANE; we
//! only request it (CPUAndNeuralEngine = never the GPU; anything the ANE can't run
//! falls back to the CPU). Verify for real with:
//!   sudo powermetrics --samplers ane_power        (while generating)

use crate::config::ModelConfig;
use crate::engine::{BatchRow, Batcher, Engine, Session};
use crate::gpu::metal::{MetalBatcher, MetalEngine, MetalSession};
use half::f16;
use crate::model::Model;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AnyThread;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};
use std::path::Path;
use std::time::Instant;

/// The compiled prefill graph (.mlmodelc), loaded through Core ML.
struct CoreMlPrefill {
    model: Retained<MLModel>,
    seq: usize, // the graph's fixed length — shorter prompts are zero-padded at the end
}

// Apple documents MLModel prediction as thread-safe.
unsafe impl Send for CoreMlPrefill {}
unsafe impl Sync for CoreMlPrefill {}

impl CoreMlPrefill {
    fn load(path: &Path, seq: usize) -> crate::Result<Self> {
        let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
        let config = unsafe { MLModelConfiguration::new() };
        // Request CPU+ANE only (no GPU) — the GPU stays free for decoding.
        unsafe { config.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine) };
        let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
            .map_err(|e| format!("failed to load {}: {e:?}", path.display()))?;
        Ok(Self { model, seq })
    }

    /// Run the whole prompt → flat (K, V) of shape [layers × seq × kv_dim] as f32.
    fn predict(&self, ids: &[u32]) -> crate::Result<(Vec<f32>, Vec<f32>)> {
        autoreleasepool(|_| {
            // Build the input: an int32 MLMultiArray [1, seq], zero-padded at the tail.
            // (The causal mask inside the graph keeps pad positions from affecting the
            // K,V of the real positions before them.)
            let shape = NSArray::from_retained_slice(&[
                NSNumber::new_usize(1),
                NSNumber::new_usize(self.seq),
            ]);
            let arr = unsafe {
                MLMultiArray::initWithShape_dataType_error(
                    MLMultiArray::alloc(),
                    &shape,
                    MLMultiArrayDataType::Int32,
                )
            }
            .map_err(|e| format!("failed to create MLMultiArray: {e:?}"))?;
            // dataPointer is deprecated (Apple prefers the block-based accessors), but the
            // plain pointer is far more readable here and still fully functional.
            #[allow(deprecated)]
            unsafe {
                let p = arr.dataPointer().as_ptr() as *mut i32;
                for i in 0..self.seq {
                    *p.add(i) = if i < ids.len() { ids[i] as i32 } else { 0 };
                }
            }

            // Wrap it the way Core ML expects: a feature provider for {"ids": multiarray}.
            let value = unsafe { MLFeatureValue::featureValueWithMultiArray(&arr) };
            let value: Retained<AnyObject> = Retained::into_super(Retained::into_super(value));
            let key = NSString::from_str("ids");
            let dict = NSDictionary::from_retained_objects(&[&*key], &[value]);
            let provider = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    &dict,
                )
            }
            .map_err(|e| format!("feature provider: {e:?}"))?;

            // This is the call where the work actually enters the ANE.
            let out = unsafe {
                self.model
                    .predictionFromFeatures_error(ProtocolObject::from_ref(&*provider))
            }
            .map_err(|e| format!("Core ML prediction failed: {e:?}"))?;

            Ok((fetch_f32(&out, "k_cache")?, fetch_f32(&out, "v_cache")?))
        })
    }
}

/// A windowed prefill graph: one chunk of `s` tokens attending to up to `p` past
/// positions fed in as K/V inputs. This is how prompts longer than the plain graphs
/// stay on the ANE — the runtime accumulates each chunk's K/V and feeds it back as
/// the next chunk's past. The fp16 path holds its numeric envelope through
/// p + s = 8,192 (measured against torch f32 on natural text); what limits wider
/// windows is the one-time first-load ANECompilerService cost, which scales
/// 99 s at 6,144 → 250 s at 8,192 → 21+ min at 16,384.
struct CoreMlWindowed {
    model: Retained<MLModel>,
    s: usize,
    p: usize,
}

unsafe impl Send for CoreMlWindowed {}
unsafe impl Sync for CoreMlWindowed {}

fn ml_array(shape: &[usize], dtype: MLMultiArrayDataType) -> crate::Result<Retained<MLMultiArray>> {
    let dims: Vec<_> = shape.iter().map(|&d| NSNumber::new_usize(d)).collect();
    let shape_ns = NSArray::from_retained_slice(&dims);
    unsafe { MLMultiArray::initWithShape_dataType_error(MLMultiArray::alloc(), &shape_ns, dtype) }
        .map_err(|e| format!("failed to create MLMultiArray: {e:?}").into())
}

impl CoreMlWindowed {
    fn load(path: &Path, s: usize, p: usize) -> crate::Result<Self> {
        let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
        let config = unsafe { MLModelConfiguration::new() };
        unsafe { config.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine) };
        let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
            .map_err(|e| format!("failed to load {}: {e:?}", path.display()))?;
        Ok(Self { model, s, p })
    }

    /// One chunk: `ids` (≤ s tokens) at absolute position `pos0`, with `pos0` (≤ p)
    /// valid rows of accumulated K/V ([layers × p × kvd], f16 bits). RoPE cos/sin
    /// are computed here in f32 and fed as inputs — the graph must not derive
    /// positions itself, because the fp16 pipeline cannot represent integers above
    /// 2,048 exactly and RoPE breaks from that position on.
    #[allow(clippy::too_many_arguments)]
    fn predict(
        &self,
        ids: &[u32],
        pos0: usize,
        k_past: &[u16],
        v_past: &[u16],
        n_layers: usize,
        kvd: usize,
        head_dim: usize,
        theta: f32,
    ) -> crate::Result<(Vec<f32>, Vec<f32>)> {
        autoreleasepool(|_| {
            let ids_arr = ml_array(&[1, self.s], MLMultiArrayDataType::Int32)?;
            let cos_arr = ml_array(&[self.s, head_dim], MLMultiArrayDataType::Float16)?;
            let sin_arr = ml_array(&[self.s, head_dim], MLMultiArrayDataType::Float16)?;
            let k_arr = ml_array(&[n_layers, self.p, kvd], MLMultiArrayDataType::Float16)?;
            let v_arr = ml_array(&[n_layers, self.p, kvd], MLMultiArrayDataType::Float16)?;
            let valid_arr = ml_array(&[1, self.p], MLMultiArrayDataType::Float16)?;
            #[allow(deprecated)] // dataPointer: plain pointers, same reasoning as above
            unsafe {
                let idp = ids_arr.dataPointer().as_ptr() as *mut i32;
                for i in 0..self.s {
                    *idp.add(i) = if i < ids.len() { ids[i] as i32 } else { 0 };
                }
                let cp = cos_arr.dataPointer().as_ptr() as *mut u16;
                let sp = sin_arr.dataPointer().as_ptr() as *mut u16;
                let half = head_dim / 2;
                for r in 0..self.s {
                    let pos = (pos0 + r) as f32;
                    for i in 0..half {
                        let freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
                        let (s_v, c_v) = (pos * freq).sin_cos();
                        let (c_b, s_b) = (f16::from_f32(c_v).to_bits(), f16::from_f32(s_v).to_bits());
                        *cp.add(r * head_dim + i) = c_b;
                        *cp.add(r * head_dim + i + half) = c_b; // HF layout: both halves
                        *sp.add(r * head_dim + i) = s_b;
                        *sp.add(r * head_dim + i + half) = s_b;
                    }
                }
                std::ptr::copy_nonoverlapping(
                    k_past.as_ptr(),
                    k_arr.dataPointer().as_ptr() as *mut u16,
                    k_past.len(),
                );
                std::ptr::copy_nonoverlapping(
                    v_past.as_ptr(),
                    v_arr.dataPointer().as_ptr() as *mut u16,
                    v_past.len(),
                );
                let vp = valid_arr.dataPointer().as_ptr() as *mut u16;
                let one = f16::from_f32(1.0).to_bits();
                for i in 0..self.p {
                    *vp.add(i) = if i < pos0 { one } else { 0 };
                }
            }

            let names = ["ids", "cos", "sin", "k_past", "v_past", "past_valid"];
            let arrays = [ids_arr, cos_arr, sin_arr, k_arr, v_arr, valid_arr];
            let keys: Vec<_> = names.iter().map(|n| NSString::from_str(n)).collect();
            let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
            let values: Vec<Retained<AnyObject>> = arrays
                .into_iter()
                .map(|a| {
                    let v = unsafe { MLFeatureValue::featureValueWithMultiArray(&a) };
                    Retained::into_super(Retained::into_super(v))
                })
                .collect();
            let dict = NSDictionary::from_retained_objects(&key_refs, &values);
            let provider = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    &dict,
                )
            }
            .map_err(|e| format!("feature provider: {e:?}"))?;

            let out = unsafe {
                self.model
                    .predictionFromFeatures_error(ProtocolObject::from_ref(&*provider))
            }
            .map_err(|e| format!("Core ML prediction failed: {e:?}"))?;
            Ok((fetch_f32(&out, "k_cache")?, fetch_f32(&out, "v_cache")?))
        })
    }
}

/// Pull an f32 multiarray output into a Vec<f32>.
fn fetch_f32(out: &ProtocolObject<dyn MLFeatureProvider>, name: &str) -> crate::Result<Vec<f32>> {
    let fv = unsafe { out.featureValueForName(&NSString::from_str(name)) }
        .ok_or_else(|| format!("no output named {name}"))?;
    let arr = unsafe { fv.multiArrayValue() }.ok_or_else(|| format!("{name} is not a multiarray"))?;
    let count = unsafe { arr.count() } as usize;
    let mut v = vec![0f32; count];
    #[allow(deprecated)] // dataPointer: same reasoning as on the input side
    unsafe {
        std::ptr::copy_nonoverlapping(arr.dataPointer().as_ptr() as *const f32, v.as_mut_ptr(), count)
    };
    Ok(v)
}

pub struct AneEngine {
    metal: MetalEngine,
    graphs: Vec<CoreMlPrefill>, // sorted by seq, ascending — one fixed shape each
    windowed: Option<CoreMlWindowed>,
}

impl AneEngine {
    pub fn new(model: Model, model_dir: &Path) -> crate::Result<Self> {
        // Collect every prefill graph next to the model: prefill-<seq>.mlmodelc
        // (plain) and prefill-<s>w<p>.mlmodelc (windowed).
        let mut found = Vec::new();
        let mut win = None;
        for entry in std::fs::read_dir(model_dir)? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            let Some(spec) = name.strip_prefix("prefill-").and_then(|s| s.strip_suffix(".mlmodelc"))
            else {
                continue;
            };
            if let Some((s, p)) = spec.split_once('w') {
                if let (Ok(s), Ok(p)) = (s.parse::<usize>(), p.parse::<usize>()) {
                    win = Some((model_dir.join(&name), s, p));
                }
            } else if let Ok(seq) = spec.parse::<usize>() {
                found.push((model_dir.join(&name), seq));
            }
        }
        if found.is_empty() {
            return Err(format!(
                "the ane backend needs prefill-<seq>.mlmodelc graphs in the model directory — build them once with:\n  \
                 uv run --python 3.12 --with torch --with coremltools --with safetensors --with numpy --with tokenizers \\\n    \
                 tools/export_prefill.py {} --shapes 512,2048",
                model_dir.display()
            )
            .into());
        }
        found.sort_by_key(|&(_, seq)| seq);
        // First load on a machine sends each graph through ANECompilerService, which
        // is silent and can take minutes for the windowed graph (~250 s at 8,192);
        // without a notice that reads as a hang. Later loads hit the cache in seconds.
        eprintln!("ANE: loading prefill graphs (first load on this machine compiles them — can take minutes)");
        let graphs = found
            .iter()
            .map(|(path, seq)| CoreMlPrefill::load(path, *seq))
            .collect::<crate::Result<Vec<_>>>()?;
        let windowed = match win {
            Some((path, s, p)) => Some(CoreMlWindowed::load(&path, s, p)?),
            None => None,
        };
        let shapes: Vec<usize> = graphs.iter().map(|g| g.seq).collect();
        match &windowed {
            Some(w) => eprintln!(
                "ANE: loaded prefill graphs {shapes:?} + windowed {}x{} (Core ML, compute = CPU+ANE)",
                w.s, w.p
            ),
            None => eprintln!("ANE: loaded prefill graphs {shapes:?} (Core ML, compute = CPU+ANE)"),
        }
        Ok(Self { metal: MetalEngine::new(model)?, graphs, windowed })
    }
}

impl Engine for AneEngine {
    fn name(&self) -> &'static str {
        "ane+metal"
    }
    fn config(&self) -> &ModelConfig {
        self.metal.config()
    }
    fn session(&self, max_seq: usize) -> crate::Result<Box<dyn Session + '_>> {
        Ok(Box::new(AneSession {
            n_layers: self.config().num_hidden_layers,
            kvd: self.config().kv_dim(),
            metal: self.metal.raw_session(max_seq),
            graphs: &self.graphs,
            windowed: self.windowed.as_ref(),
        }))
    }
    fn batcher(&self, n_slots: usize, max_seq: usize) -> Option<Box<dyn Batcher + '_>> {
        let inner = self.metal.make_batcher(n_slots, max_seq)?;
        Some(Box::new(AneBatcher { engine: self, inner }))
    }
}

/// Continuous batching with the hybrid backend: decode is Metal's batched step;
/// each admitted request's prompt goes through the Core ML graphs straight into
/// its pooled KV slot (the slot-backed session makes write_kv land in the pool).
struct AneBatcher<'a> {
    engine: &'a AneEngine,
    inner: MetalBatcher<'a>,
}

impl Batcher for AneBatcher<'_> {
    fn prefill(&mut self, slot: usize, ids: &[u32]) -> crate::Result<Vec<f32>> {
        let mut s = AneSession {
            n_layers: self.engine.config().num_hidden_layers,
            kvd: self.engine.config().kv_dim(),
            metal: self.inner.slot_session(slot),
            graphs: &self.engine.graphs,
            windowed: self.engine.windowed.as_ref(),
        };
        s.prefill(ids)
    }
    fn decode_step(&mut self, rows: &[BatchRow]) -> crate::Result<Vec<Vec<f32>>> {
        self.inner.decode_step(rows)
    }
}

/// Below this many prompt tokens the GPU prefills faster than even the smallest
/// padded ANE graph (~46 ms for S=512 vs ~1 ms per token on Metal).
const ANE_MIN: usize = 64;

struct AneSession<'a> {
    metal: MetalSession<'a>,
    graphs: &'a [CoreMlPrefill], // sorted by seq, ascending
    windowed: Option<&'a CoreMlWindowed>,
    n_layers: usize,
    kvd: usize,
}

impl AneSession<'_> {
    /// Copy `n` rows of a graph result (per-layer stride `src_stride`) into Metal's
    /// cache — and, when chunking, into the host-side f16 past that feeds the next
    /// windowed chunk (rows at index ≥ p can never be attended to again and are
    /// skipped).
    #[allow(clippy::too_many_arguments)]
    fn absorb(
        &mut self,
        k: &[f32],
        v: &[f32],
        src_stride: usize,
        pos0: usize,
        n: usize,
        past: Option<(&mut [u16], &mut [u16], usize)>,
    ) {
        let kvd = self.kvd;
        for l in 0..self.n_layers {
            let off = l * src_stride * kvd;
            let used = n * kvd;
            self.metal.write_kv(l, pos0, &k[off..off + used], &v[off..off + used]);
        }
        if let Some((past_k, past_v, p)) = past {
            let copy = (n * kvd).min(p.saturating_sub(pos0) * kvd);
            for l in 0..self.n_layers {
                let off = l * src_stride * kvd;
                let dst = l * p * kvd + pos0 * kvd;
                for i in 0..copy {
                    past_k[dst + i] = f16::from_f32(k[off + i]).to_bits();
                    past_v[dst + i] = f16::from_f32(v[off + i]).to_bits();
                }
            }
        }
    }

    /// Prompts longer than the largest plain graph: head chunk through the plain
    /// graph, then windowed chunks that attend to the accumulated past — all on the
    /// ANE up to s+p total positions; Metal takes anything beyond that.
    fn prefill_chunked(&mut self, ids: &[u32], want: usize) -> crate::Result<Vec<f32>> {
        let w = self.windowed.expect("caller checked");
        let ane_total = want.min(w.s + w.p);
        let (n_layers, kvd) = (self.n_layers, self.kvd);
        let mut past_k = vec![0u16; n_layers * w.p * kvd];
        let mut past_v = vec![0u16; n_layers * w.p * kvd];

        let t = Instant::now();
        let head_graph = self.graphs.last().expect("AneEngine::new guarantees one");
        let head = ane_total.min(head_graph.seq);
        let (k, v) = head_graph.predict(&ids[..head])?;
        self.absorb(&k, &v, head_graph.seq, 0, head, Some((&mut past_k, &mut past_v, w.p)));

        let cfg = self.metal.config_ref();
        let (head_dim, theta) = (cfg.head_dim(), cfg.rope_theta);
        let mut pos = head;
        let mut chunks = 1;
        while pos < ane_total {
            let n = (ane_total - pos).min(w.s);
            let (k, v) =
                w.predict(&ids[pos..pos + n], pos, &past_k, &past_v, n_layers, kvd, head_dim, theta)?;
            self.absorb(&k, &v, w.s, pos, n, Some((&mut past_k, &mut past_v, w.p)));
            pos += n;
            chunks += 1;
        }
        eprintln!(
            "  ANE prefill: {ane_total} tokens ({chunks} chunks, windowed S={} P={}) in {:.2}s",
            w.s,
            w.p,
            t.elapsed().as_secs_f64()
        );
        self.metal.prefill_from(&ids[ane_total..], ane_total)
    }
}

/// Pick the graph for a `want`-token prompt. Prefer the smallest graph that fits,
/// but only "upgrade" to a bigger graph when the prompt fills at least half of it —
/// ANE time grows superlinearly with S (attention is S²), so below half-full it is
/// cheaper to fill a smaller graph completely and let Metal take the overflow.
fn pick_graph(graphs: &[CoreMlPrefill], want: usize) -> &CoreMlPrefill {
    let fits = graphs.iter().find(|g| g.seq >= want);
    let under = graphs.iter().rev().find(|g| g.seq <= want);
    match (fits, under) {
        (Some(f), Some(u)) if f.seq != u.seq && want * 2 < f.seq => u,
        (Some(f), _) => f,
        (None, Some(u)) => u, // longer than the largest graph — Metal takes the rest
        (None, None) => unreachable!("AneEngine::new guarantees at least one graph"),
    }
}

impl Session for AneSession<'_> {
    fn forward(&mut self, token: u32, pos: usize) -> crate::Result<Vec<f32>> {
        self.metal.forward(token, pos) // decoding stays entirely on Metal
    }

    fn forward_batch(&mut self, ids: &[u32], pos0: usize) -> crate::Result<Vec<Vec<f32>>> {
        self.metal.forward_batch(ids, pos0) // speculative verification too
    }

    /// Division of labor: the ANE takes the prompt's head (up to the chosen graph's
    /// seq), Metal takes any overflow — and always the final token, because we need
    /// the last position's logits and the ANE graphs deliberately omit lm_head (one
    /// Metal step is cheaper than shipping a vocab-sized matmul through Core ML).
    fn prefill(&mut self, ids: &[u32]) -> crate::Result<Vec<f32>> {
        let want = ids.len().saturating_sub(1);
        if want < ANE_MIN {
            // Say so — the user picked -b ane and would otherwise wonder why the
            // Neural Engine stays idle on a short prompt.
            eprintln!(
                "  ANE skipped: {}-token prompt (< {ANE_MIN}) — the GPU prefills it faster than the smallest padded graph",
                ids.len()
            );
            return self.metal.prefill_from(ids, 0);
        }
        let largest = self.graphs.last().map(|g| g.seq).unwrap_or(0);
        if want > largest && self.windowed.is_some() {
            return self.prefill_chunked(ids, want);
        }
        let g = pick_graph(self.graphs, want);
        let ane_n = want.min(g.seq);
        let t = Instant::now();
        let (k, v) = g.predict(&ids[..ane_n])?;
        // Move K,V into Metal's cache — keep only the ane_n real rows, drop the padding.
        self.absorb(&k, &v, g.seq, 0, ane_n, None);
        eprintln!(
            "  ANE prefill: {ane_n} tokens (S={} graph) in {:.2}s",
            g.seq,
            t.elapsed().as_secs_f64()
        );
        self.metal.prefill_from(&ids[ane_n..], ane_n)
    }
}
