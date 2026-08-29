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
use crate::engine::{Engine, Session};
use crate::gpu::metal::{MetalEngine, MetalSession};
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
}

impl AneEngine {
    pub fn new(model: Model, model_dir: &Path) -> crate::Result<Self> {
        // Collect every prefill-<seq>.mlmodelc next to the model (seq is in the name).
        let mut found = Vec::new();
        for entry in std::fs::read_dir(model_dir)? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            if let Some(seq) = name
                .strip_prefix("prefill-")
                .and_then(|s| s.strip_suffix(".mlmodelc"))
                .and_then(|s| s.parse::<usize>().ok())
            {
                found.push((model_dir.join(&name), seq));
            }
        }
        if found.is_empty() {
            return Err(format!(
                "the ane backend needs prefill-<seq>.mlmodelc graphs in the model directory — build them once with:\n  \
                 uv run --python 3.12 --with torch --with coremltools --with safetensors --with numpy \\\n    \
                 tools/export_prefill.py {} --shapes 512,2048",
                model_dir.display()
            )
            .into());
        }
        found.sort_by_key(|&(_, seq)| seq);
        let graphs = found
            .iter()
            .map(|(path, seq)| CoreMlPrefill::load(path, *seq))
            .collect::<crate::Result<Vec<_>>>()?;
        let shapes: Vec<usize> = graphs.iter().map(|g| g.seq).collect();
        eprintln!("ANE: loaded prefill graphs {shapes:?} (Core ML, compute = CPU+ANE)");
        Ok(Self { metal: MetalEngine::new(model)?, graphs })
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
        }))
    }
}

/// Below this many prompt tokens the GPU prefills faster than even the smallest
/// padded ANE graph (~46 ms for S=512 vs ~1 ms per token on Metal).
const ANE_MIN: usize = 64;

struct AneSession<'a> {
    metal: MetalSession<'a>,
    graphs: &'a [CoreMlPrefill], // sorted by seq, ascending
    n_layers: usize,
    kvd: usize,
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

    /// Division of labor: the ANE takes the prompt's head (up to the chosen graph's
    /// seq), Metal takes any overflow — and always the final token, because we need
    /// the last position's logits and the ANE graphs deliberately omit lm_head (one
    /// Metal step is cheaper than shipping a vocab-sized matmul through Core ML).
    fn prefill(&mut self, ids: &[u32]) -> crate::Result<Vec<f32>> {
        let want = ids.len().saturating_sub(1);
        let mut ane_n = 0;
        if want >= ANE_MIN {
            let g = pick_graph(self.graphs, want);
            ane_n = want.min(g.seq);
            let t = Instant::now();
            let (k, v) = g.predict(&ids[..ane_n])?;
            // Move K,V into Metal's cache — keep only the ane_n real rows, drop the padding.
            let per_layer = g.seq * self.kvd;
            let used = ane_n * self.kvd;
            for layer in 0..self.n_layers {
                let off = layer * per_layer;
                self.metal.write_kv(layer, 0, &k[off..off + used], &v[off..off + used]);
            }
            eprintln!(
                "  ANE prefill: {ane_n} tokens (S={} graph) in {:.2}s",
                g.seq,
                t.elapsed().as_secs_f64()
            );
        }
        self.metal.prefill_from(&ids[ane_n..], ane_n)
    }
}
