//! The "hybrid" backend (`-b hybrid`, formerly `-b ane`): prefill on the Apple
//! Neural Engine — or split across the ANE and the GPU — and decode on Metal.
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

/// Build the feature provider both windowed graphs share: one chunk of `ids` at
/// absolute position `pos0`, the RoPE tables for those positions, and the
/// validity-masked past K/V for `past_layers` layers ([past_layers × p × kvd]
/// f16 bits). RoPE is computed here in f32 rather than in the graph — the fp16
/// pipeline cannot represent integers above 2,048 exactly, which corrupts RoPE
/// from that position on.
#[allow(clippy::too_many_arguments)]
fn windowed_provider(
    s: usize,
    p: usize,
    ids: &[u32],
    pos0: usize,
    k_past: &[u16],
    v_past: &[u16],
    past_layers: usize,
    kvd: usize,
    head_dim: usize,
    theta: f32,
) -> crate::Result<Retained<MLDictionaryFeatureProvider>> {
    let ids_arr = ml_array(&[1, s], MLMultiArrayDataType::Int32)?;
    let cos_arr = ml_array(&[s, head_dim], MLMultiArrayDataType::Float16)?;
    let sin_arr = ml_array(&[s, head_dim], MLMultiArrayDataType::Float16)?;
    // A P=0 graph — the ladder rung for the first chunk, which has no past and
    // pays no past attention — takes only ids/cos/sin.
    let past_arrs = if p > 0 {
        Some((
            ml_array(&[past_layers, p, kvd], MLMultiArrayDataType::Float16)?,
            ml_array(&[past_layers, p, kvd], MLMultiArrayDataType::Float16)?,
            ml_array(&[1, p], MLMultiArrayDataType::Float16)?,
        ))
    } else {
        None
    };
    #[allow(deprecated)] // dataPointer: plain pointers, same reasoning as above
    unsafe {
        let idp = ids_arr.dataPointer().as_ptr() as *mut i32;
        for i in 0..s {
            *idp.add(i) = if i < ids.len() { ids[i] as i32 } else { 0 };
        }
        let cp = cos_arr.dataPointer().as_ptr() as *mut u16;
        let sp = sin_arr.dataPointer().as_ptr() as *mut u16;
        let half = head_dim / 2;
        for r in 0..s {
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
        if let Some((k_arr, v_arr, valid_arr)) = &past_arrs {
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
            for i in 0..p {
                *vp.add(i) = if i < pos0 { one } else { 0 };
            }
        }
    }

    let mut names = vec!["ids", "cos", "sin"];
    let mut arrays = vec![ids_arr, cos_arr, sin_arr];
    if let Some((k_arr, v_arr, valid_arr)) = past_arrs {
        names.extend(["k_past", "v_past", "past_valid"]);
        arrays.extend([k_arr, v_arr, valid_arr]);
    }
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
    Ok(provider)
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
    /// valid rows of accumulated K/V ([layers × p × kvd], f16 bits).
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
            let provider = windowed_provider(
                self.s, self.p, ids, pos0, k_past, v_past, n_layers, kvd, head_dim, theta,
            )?;
            let out = unsafe {
                self.model
                    .predictionFromFeatures_error(ProtocolObject::from_ref(&*provider))
            }
            .map_err(|e| format!("Core ML prediction failed: {e:?}"))?;
            Ok((fetch_f32(&out, "k_cache")?, fetch_f32(&out, "v_cache")?))
        })
    }
}


/// The layer-split FRONT graph: like the windowed graph, but it runs only layers
/// 0..`n_front` and additionally returns the hidden state, so another device can
/// finish the remaining layers for the same chunk. This is what lets the ANE and
/// the GPU work on one prompt at the same time — see `AneSession::prefill_split`.
struct CoreMlFront {
    model: Retained<MLModel>,
    s: usize,
    p: usize,
    n_front: usize,
}

unsafe impl Send for CoreMlFront {}
unsafe impl Sync for CoreMlFront {}

impl CoreMlFront {
    fn load(path: &Path, s: usize, p: usize, n_front: usize) -> crate::Result<Self> {
        let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
        let config = unsafe { MLModelConfiguration::new() };
        unsafe { config.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine) };
        let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
            .map_err(|e| format!("failed to load {}: {e:?}", path.display()))?;
        Ok(Self { model, s, p, n_front })
    }

    /// One chunk through layers 0..n_front → (K, V) for those layers as the
    /// cache's own f16 bits ([n_front × s × kvd]) plus the hidden state
    /// ([s × hidden] f32, the dtype Metal's activations use).
    #[allow(clippy::too_many_arguments)]
    fn predict(
        &self,
        ids: &[u32],
        pos0: usize,
        k_past: &[u16],
        v_past: &[u16],
        kvd: usize,
        head_dim: usize,
        theta: f32,
    ) -> crate::Result<(Vec<u16>, Vec<u16>, Vec<f32>)> {
        autoreleasepool(|_| {
            let provider = windowed_provider(
                self.s, self.p, ids, pos0, k_past, v_past, self.n_front, kvd, head_dim, theta,
            )?;
            let out = unsafe {
                self.model
                    .predictionFromFeatures_error(ProtocolObject::from_ref(&*provider))
            }
            .map_err(|e| format!("Core ML prediction failed: {e:?}"))?;
            Ok((
                fetch_f16_bits(&out, "k_cache")?,
                fetch_f16_bits(&out, "v_cache")?,
                fetch_f32(&out, "x_out")?,
            ))
        })
    }
}

/// Split prefill (see `AneSession::prefill_split`) is what the hybrid backend
/// does whenever the front-graph ladder is exported next to the model.
/// `LOKAL_SPLIT_PREFILL=0` is the kill switch — kept for A/B measurement and
/// debugging, not a mode; any other value, or none, means on.
fn split_enabled() -> bool {
    std::env::var("LOKAL_SPLIT_PREFILL").map_or(true, |v| v != "0")
}

/// Pull an fp16 multiarray output into its raw bits — the KV cache's own dtype,
/// so nothing on the hot path has to convert.
fn fetch_f16_bits(out: &ProtocolObject<dyn MLFeatureProvider>, name: &str) -> crate::Result<Vec<u16>> {
    let fv = unsafe { out.featureValueForName(&NSString::from_str(name)) }
        .ok_or_else(|| format!("no output named {name}"))?;
    let arr = unsafe { fv.multiArrayValue() }.ok_or_else(|| format!("{name} is not a multiarray"))?;
    let count = unsafe { arr.count() } as usize;
    let mut v = vec![0u16; count];
    #[allow(deprecated)] // dataPointer: same reasoning as on the input side
    unsafe {
        std::ptr::copy_nonoverlapping(arr.dataPointer().as_ptr() as *const u16, v.as_mut_ptr(), count)
    };
    Ok(v)
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

/// The prefill-*.mlmodelc entries in a directory, empty if it doesn't exist.
fn graph_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut v = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let n = name.to_string_lossy();
            if n.starts_with("prefill-") && n.ends_with(".mlmodelc") {
                v.push(entry.path());
            }
        }
    }
    v
}

/// Graphs used to be written next to the weights, inside the HF snapshot
/// directory — where an upstream revision bump orphans them and `hf cache
/// delete` removes gigabytes it never created. Move anything found there into
/// the lokal-owned graph dir, once. Move means `rename`, never copy: a copied
/// .mlmodelc loses its ANE compile-cache entry (keyed on the file identity)
/// and silently recompiles for minutes on the next load.
fn migrate_graphs(model_dir: &Path, loc: &crate::hub::GraphLocation) {
    let legacy = graph_files(model_dir);
    if legacy.is_empty() || !graph_files(&loc.dir).is_empty() {
        return;
    }
    if std::fs::create_dir_all(&loc.dir).is_err() {
        return; // unwritable graph dir — the model-dir graphs still load in place
    }
    let mut moved = 0usize;
    for src in &legacy {
        let Some(name) = src.file_name() else { continue };
        let dest = loc.dir.join(name);
        if dest.exists() {
            continue; // never clobber — rename-onto-existing differs across platforms
        }
        if let Err(e) = std::fs::rename(src, &dest) {
            // Cross-device is the expected failure. Leave everything where it
            // is (the model dir keeps working, D5) and say how to move it.
            eprintln!(
                "ANE: could not move graphs out of the model cache ({e}) — using them in place; \
                 relocate manually with:  mv {}/prefill-*.mlmodelc {}/",
                model_dir.display(),
                loc.dir.display()
            );
            break;
        }
        moved += 1;
    }
    if moved > 0 {
        eprintln!(
            "ANE: moved {moved} prefill graph(s) out of the model cache into {}",
            loc.dir.display()
        );
        write_graph_manifest(loc, model_dir);
    }
}

/// graphs.json — which model, snapshot revision, and resolved directory the
/// graphs in this dir were built from, so a later load can refuse graphs whose
/// weights have moved on. Written via temp file + rename in the same dir: an
/// export and a serve run may both write it.
fn write_graph_manifest(loc: &crate::hub::GraphLocation, model_dir: &Path) {
    let manifest = serde_json::json!({
        "model": loc.model,
        "resolved_dir": model_dir
            .canonicalize()
            .unwrap_or_else(|_| model_dir.to_path_buf())
            .display()
            .to_string(),
        "revision": loc.revision,
        "exported_at": iso8601_utc_now(),
    });
    let tmp = loc.dir.join(format!(".graphs.json.tmp{}", std::process::id()));
    let ok = std::fs::write(&tmp, format!("{manifest:#}\n")).is_ok()
        && std::fs::rename(&tmp, loc.dir.join("graphs.json")).is_ok();
    if !ok {
        let _ = std::fs::remove_file(&tmp); // manifest is advisory — never fail a load over it
    }
}

/// The manifest fields load-time enforcement reads; the rest is for humans.
#[derive(serde::Deserialize)]
struct GraphManifest {
    revision: Option<String>,
}

/// Current UTC time as ISO 8601, from std only (civil-from-days per Howard
/// Hinnant's algorithm — no chrono in the tree).
fn iso8601_utc_now() -> String {
    iso8601_utc(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    )
}

fn iso8601_utc(secs: u64) -> String {
    let (days, rem) = (secs / 86400, secs % 86400);
    let (hh, mi, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days as i64 + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mi:02}:{ss:02}Z")
}

/// Collect prefill graphs from one directory into the three families. The
/// `seen` set makes the FIRST directory scanned win per file name (graph dir
/// before model dir — D5).
fn scan_graph_dir(
    dir: &Path,
    seen: &mut std::collections::HashSet<std::ffi::OsString>,
    found: &mut Vec<(std::path::PathBuf, usize)>,
    win: &mut Option<(std::path::PathBuf, usize, usize)>,
    fronts: &mut Vec<(std::path::PathBuf, usize, usize, usize)>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let os_name = entry.file_name();
        let name = os_name.to_string_lossy().into_owned();
        let Some(spec) = name.strip_prefix("prefill-").and_then(|s| s.strip_suffix(".mlmodelc"))
        else {
            continue;
        };
        if !seen.insert(os_name) {
            continue; // an earlier directory already provides this graph
        }
        // prefill-f<layers>-<s>w<p>: the layer-split front half.
        if let Some((layers, rest)) = spec.strip_prefix('f').and_then(|r| r.split_once('-')) {
            if let (Ok(a), Some((s, p))) = (layers.parse::<usize>(), rest.split_once('w')) {
                if let (Ok(s), Ok(p)) = (s.parse::<usize>(), p.parse::<usize>()) {
                    fronts.push((dir.join(&name), s, p, a));
                }
            }
        } else if let Some((s, p)) = spec.split_once('w') {
            if let (Ok(s), Ok(p)) = (s.parse::<usize>(), p.parse::<usize>()) {
                if win.is_none() {
                    *win = Some((dir.join(&name), s, p));
                }
            }
        } else if let Ok(seq) = spec.parse::<usize>() {
            found.push((dir.join(&name), seq));
        }
    }
}

/// No graphs anywhere: say exactly what was searched for which model and how
/// to fix it. If OTHER models have graphs, list them — the classic mistake is
/// exporting for `…-Instruct` and then running the base model.
fn missing_graphs_error(loc: &crate::hub::GraphLocation, model_dir: &Path) -> String {
    let mut msg = format!(
        "the hybrid backend has no prefill graphs for {}\n  searched: {}  (lokal graph cache)\n            {}  (model directory)\n  build them once with:  ./run.sh export-hybrid {}",
        loc.model,
        loc.dir.display(),
        model_dir.display(),
        loc.model
    );
    let own = loc.dir.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    let others: Vec<String> = std::fs::read_dir(crate::hub::graph_cache_base())
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_name() != own && !graph_files(&e.path()).is_empty())
        .take(3)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    if !others.is_empty() {
        msg.push_str(&format!("\n  graphs exist for: {}", others.join(", ")));
    }
    msg
}

pub struct AneEngine {
    metal: MetalEngine,
    graphs: Vec<CoreMlPrefill>, // sorted by seq, ascending — one fixed shape each
    windowed: Option<CoreMlWindowed>,
    // Split prefill, loaded whenever the ladder is exported (LOKAL_SPLIT_PREFILL=0
    // skips it). A ladder like the plain graphs: a windowed graph costs its full
    // P+S attention on every chunk whatever the prompt length, so a short prompt
    // wants a short window.
    front: Vec<CoreMlFront>, // sorted by reach (s + p), ascending
}

impl AneEngine {
    pub fn new(model: Model, model_dir: &Path) -> crate::Result<Self> {
        // Graphs live in the lokal-owned graph directory (hub::graph_location),
        // with the model directory as the legacy fallback — graphs exported
        // straight into a snapshot dir keep working forever, sitting next to
        // the exact weights they were built from.
        let loc = crate::hub::graph_location(model_dir);
        migrate_graphs(model_dir, &loc);

        // graphs.json says which snapshot the graphs were built from. On a
        // mismatch the model moved on — silently running graphs built from
        // different weights is the one failure worse than being slow, so
        // refuse the whole directory and carry on without it. A local-dir
        // model records no revision and is not enforced.
        let mut refused_stale = false;
        let graph_dir_current = match std::fs::read(loc.dir.join("graphs.json")) {
            Err(_) => true, // no manifest — nothing recorded, nothing to enforce
            Ok(bytes) => match serde_json::from_slice::<GraphManifest>(&bytes) {
                Err(e) => {
                    eprintln!(
                        "ANE: {}/graphs.json is unreadable ({e}) — using the graphs anyway",
                        loc.dir.display()
                    );
                    true
                }
                Ok(m) => match (m.revision, &loc.revision) {
                    (Some(built), Some(now)) if built != *now => {
                        eprintln!(
                            "ANE: graphs in {} were built from revision {built}, but {} now \
                             resolves to {now} — refusing them (re-export: ./run.sh export-hybrid {})",
                            loc.dir.display(),
                            loc.model,
                            loc.model
                        );
                        refused_stale = true;
                        false
                    }
                    _ => true,
                },
            },
        };

        // Collect every prefill graph: prefill-<seq>.mlmodelc (plain),
        // prefill-<s>w<p>.mlmodelc (windowed), prefill-f<A>-<s>w<p>.mlmodelc
        // (split front half). Graph dir first, model dir second — the graph
        // dir wins per file name.
        let mut found = Vec::new();
        let mut win = None;
        let mut fronts = Vec::new();
        let mut seen = std::collections::HashSet::new();
        if graph_dir_current {
            scan_graph_dir(&loc.dir, &mut seen, &mut found, &mut win, &mut fronts);
        }
        scan_graph_dir(model_dir, &mut seen, &mut found, &mut win, &mut fronts);
        if found.is_empty() {
            if refused_stale {
                // The stale graphs were refused and nothing else exists: stay
                // up — every prompt runs on Metal until a re-export.
                eprintln!("ANE: no usable prefill graphs — running everything on Metal");
            } else {
                return Err(missing_graphs_error(&loc, model_dir).into());
            }
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
        let win_on_disk = win.is_some();
        // The windowed graph lost its reason to run (measured 2026-08-30, M1 Pro,
        // Qwen2.5-0.5B): its 8,192-token head alone takes 7.94 s where Metal does
        // the same head in ~4.5 s — and past the head both paths run the same
        // Metal tail, so it cannot win at ANY prompt length on this hardware. It
        // was the right choice when Metal prefill ran at 1,461 tok/s; Metal is
        // now ~7x that. The graph also weighs ~0.9 GB in memory, which on a
        // 16 GB machine with other servers resident is the difference between
        // fitting and a SIGKILL. Loading it is opt-in for A/B archaeology:
        // LOKAL_WINDOWED_PREFILL=1.
        let windowed = match win {
            Some((path, s, p))
                if std::env::var("LOKAL_WINDOWED_PREFILL").is_ok_and(|v| v != "0" && !v.is_empty()) =>
            {
                Some(CoreMlWindowed::load(&path, s, p)?)
            }
            _ => None,
        };
        // The ladder loads whenever it exists: split prefill is what this backend
        // does, not a mode. A model with no front graphs (nothing to load) stays
        // silent; LOKAL_SPLIT_PREFILL=0 is the kill switch for A/B runs.
        let mut front = Vec::new();
        if !fronts.is_empty() && !split_enabled() {
            eprintln!("ANE: split prefill disabled (LOKAL_SPLIT_PREFILL=0) — plain ANE prefill");
        } else {
            fronts.sort_by_key(|&(_, s, p, _)| s + p);
            for (path, s, p, a) in fronts {
                // Every new shape is its own ANECompilerService pass on this machine —
                // silent and slow the first time, cached afterwards. Say so when it
                // happens; a cached load takes well under a second and stays quiet.
                let t = Instant::now();
                front.push(CoreMlFront::load(&path, s, p, a)?);
                let dt = t.elapsed().as_secs_f64();
                if dt >= 1.0 {
                    eprintln!("ANE: front graph {s}x{p} layers 0..{a} compiled in {dt:.1}s");
                }
            }
        }
        let shapes: Vec<usize> = graphs.iter().map(|g| g.seq).collect();
        match &windowed {
            Some(w) => eprintln!(
                "ANE: loaded prefill graphs {shapes:?} + windowed {}x{} (Core ML, compute = CPU+ANE)",
                w.s, w.p
            ),
            None => eprintln!("ANE: loaded prefill graphs {shapes:?} (Core ML, compute = CPU+ANE)"),
        }
        if windowed.is_none() && win_on_disk {
            eprintln!(
                "ANE: windowed graph present but idle — Metal outruns it at every length on this \
                 hardware (LOKAL_WINDOWED_PREFILL=1 loads it anyway)"
            );
        }
        if !front.is_empty() {
            // Announce the routing once, here — prefill_split itself reports per
            // prompt, and nobody should have to read code to learn which path ran.
            let rungs: Vec<String> =
                front.iter().map(|g| format!("{}x{}@{}", g.s, g.p, g.n_front)).collect();
            eprintln!(
                "ANE: split prefill on — ladder [{}]; prompts pipeline ANE+GPU (LOKAL_SPLIT_PREFILL=0 disables)",
                rungs.join(", ")
            );
        }
        Ok(Self { metal: MetalEngine::new(model)?, graphs, windowed, front })
    }
}

impl Engine for AneEngine {
    fn name(&self) -> &'static str {
        "hybrid (ANE + Metal)"
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
            front: &self.front,
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
            front: &self.engine.front,
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
    front: &'a [CoreMlFront], // sorted by reach (s + p), ascending; empty = split off
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


    /// Split prefill — the two-device pipeline, the default whenever a front-graph
    /// ladder is exported. Returns `Ok(None)` when the ladder cannot serve this
    /// session at all (nothing has been touched — the caller falls back to the
    /// plain path); a mid-run `Err` may leave partial KV rows behind, which is
    /// safe because the plain path rewrites every row from position 0.
    ///
    /// The prompt is cut into chunks of the front graph's S. The ANE runs chunk c
    /// through layers 0..A while Metal runs chunk c-1 through layers A..L, so in
    /// steady state both engines are working on the same prompt instead of one
    /// waiting for the other. Causality holds because each half only ever needs
    /// its OWN layers' past K/V — which that device already produced for every
    /// earlier chunk — plus this chunk's hidden state from the stage before it.
    ///
    /// The ANE's front-layer K/V still land in Metal's cache: decoding runs every
    /// layer on the GPU and needs all of them. Conversion to the cache's f16
    /// happens on the ANE thread, which needs those same rows for the next
    /// chunk's past anyway, so the thread driving the GPU only memcpys.
    fn prefill_split(&mut self, ids: &[u32]) -> crate::Result<Option<Vec<f32>>> {
        let kvd = self.kvd;
        let cfg = self.metal.config_ref();
        let (head_dim, theta, hidden) = (cfg.head_dim(), cfg.rope_theta, cfg.hidden_size);
        // The hidden state is written straight into Metal's per-chunk scratch, so a
        // graph wider than that scratch could never be used.
        let room = self.metal.max_chunk_rows();
        let front = self.front;

        // Pick the chunk stride first. Two chunks is all fill and drain — the
        // pipeline only starts paying for itself once there are stages in the
        // middle where both engines are busy — so take the widest exported stride
        // that still cuts this prompt into at least MIN_CHUNKS pieces. The count
        // that matters is the REAL chunk count, div_ceil — demanding len >= 3*s
        // would reject a 764-token prompt at stride 256 even though it cuts into
        // three chunks, and push the whole 513..767 band onto a narrower stride
        // whose ladder reaches shorter.
        const MIN_CHUNKS: usize = 3;
        let mut strides: Vec<usize> =
            front.iter().map(|g| g.s).filter(|&s| s <= room).collect();
        strides.sort_unstable();
        strides.dedup();
        if strides.is_empty() {
            // Every exported rung is wider than Metal's per-chunk scratch — the
            // ladder cannot feed this session. Not an error on a default path:
            // the caller degrades to the plain ANE prefill.
            return Ok(None);
        }
        // Per stride: the ladder as the ANE thread will see it (one front-layer
        // count — a leftover graph exported at a different split would make the
        // chunks disagree about where layer A is), and the span the ANE takes.
        // The span is NOT trimmed: handing a runt last chunk to Metal was
        // measured slower every time (582 tokens: 256+256+70 does 10.2k tok/s,
        // 2x256 + a 70-token Metal tail 9.1k, 4x128 + tail 9.4k) — the pipeline
        // hides a runt's rung cost better than a serial layer-0 tail repays it.
        let plan_for = |s: usize| {
            let mut rungs: Vec<&CoreMlFront> = front.iter().filter(|g| g.s == s).collect();
            rungs.sort_by_key(|g| g.p);
            let a = rungs[0].n_front;
            rungs.retain(|g| g.n_front == a);
            let p_max = rungs.last().expect("at least one rung").p;
            let span = ids.len().min(s + p_max);
            (rungs, a, span)
        };
        // Widest stride whose ANE span still cuts into >= MIN_CHUNKS chunks —
        // two chunks is all fill and drain, the pipeline only pays for itself
        // with stages in the middle where both engines are busy. The count is
        // the REAL chunk count, div_ceil: a len >= 3*s guard would reject a
        // 764-token prompt at stride 256 despite its three real chunks, and
        // push the whole 513..767 band onto a narrower stride whose ladder
        // reaches shorter.
        let s = strides
            .iter()
            .rev()
            .copied()
            .find(|&s| plan_for(s).2.div_ceil(s) >= MIN_CHUNKS)
            .unwrap_or(strides[0]);
        let (rungs, a, ane_total) = plan_for(s);
        let p_max = rungs.last().expect("at least one rung").p;
        let ladder: &[&CoreMlFront] = &rungs; // shared with the Core ML thread

        struct Chunk {
            pos: usize,
            n: usize,
            k: Vec<u16>, // [A × n × kvd] f16 bits, ready for the cache
            v: Vec<u16>,
            x: Vec<f32>, // [n × hidden] — the layer-A input for the other device
            ane_s: f64,  // how long this chunk's front half took, for the balance report
        }

        let t = Instant::now();
        let (tx, rx) = std::sync::mpsc::channel::<crate::Result<Chunk>>();
        let (logits, chunks, ane_s, metal_s) =
            std::thread::scope(|scope| -> crate::Result<(Vec<f32>, usize, f64, f64)> {
            scope.spawn(move || {
                // One master past at the widest rung's stride; a narrower rung gets
                // the first p rows of each layer staged into its own layout.
                let mut past_k = vec![0u16; a * p_max * kvd];
                let mut past_v = vec![0u16; a * p_max * kvd];
                let (mut stage_k, mut stage_v) = (Vec::new(), Vec::new());
                let mut pos = 0;
                while pos < ane_total {
                    let n = (ane_total - pos).min(s);
                    let t_chunk = Instant::now();
                    let g = ladder.iter().find(|g| g.p >= pos).unwrap_or(&ladder[ladder.len() - 1]);
                    let narrow = g.p != p_max;
                    if narrow {
                        let span = g.p * kvd;
                        stage_k.resize(a * span, 0);
                        stage_v.resize(a * span, 0);
                        for l in 0..a {
                            let src = l * p_max * kvd;
                            let dst = l * span;
                            stage_k[dst..dst + span].copy_from_slice(&past_k[src..src + span]);
                            stage_v[dst..dst + span].copy_from_slice(&past_v[src..src + span]);
                        }
                    }
                    let (kin, vin) = if narrow {
                        (&stage_k[..], &stage_v[..])
                    } else {
                        (&past_k[..], &past_v[..])
                    };
                    let (k, v, x) = match g.predict(
                        &ids[pos..pos + n], pos, kin, vin, kvd, head_dim, theta,
                    ) {
                        Ok(out) => out,
                        Err(e) => {
                            let _ = tx.send(Err(e));
                            return;
                        }
                    };
                    // K/V arrive as the cache's own f16 bits — take the real rows out
                    // of each layer's s-row slab, no conversion anywhere.
                    let rows = n * kvd;
                    let (mut kb, mut vb) = (vec![0u16; a * rows], vec![0u16; a * rows]);
                    for l in 0..a {
                        let src = l * g.s * kvd;
                        kb[l * rows..(l + 1) * rows].copy_from_slice(&k[src..src + rows]);
                        vb[l * rows..(l + 1) * rows].copy_from_slice(&v[src..src + rows]);
                    }
                    // Rows at index ≥ p_max can never be attended to again — skip them.
                    let keep = rows.min(p_max.saturating_sub(pos) * kvd);
                    for l in 0..a {
                        let dst = l * p_max * kvd + pos * kvd;
                        past_k[dst..dst + keep].copy_from_slice(&kb[l * rows..l * rows + keep]);
                        past_v[dst..dst + keep].copy_from_slice(&vb[l * rows..l * rows + keep]);
                    }
                    let chunk = Chunk {
                        pos,
                        n,
                        k: kb,
                        v: vb,
                        x: x[..n * hidden].to_vec(),
                        ane_s: t_chunk.elapsed().as_secs_f64(),
                    };
                    if tx.send(Ok(chunk)).is_err() {
                        return; // the consumer gave up (an error downstream)
                    }
                    pos += n;
                }
            });

            let (mut logits, mut chunks) = (Vec::new(), 0usize);
            let (mut ane_s, mut metal_s) = (0.0, 0.0);
            for msg in rx {
                let c = msg?;
                let t_stage = Instant::now();
                ane_s += c.ane_s;
                let rows = c.n * kvd;
                for l in 0..a {
                    let (lo, hi) = (l * rows, (l + 1) * rows);
                    self.metal.write_kv_bits(l, c.pos, &c.k[lo..hi], &c.v[lo..hi]);
                }
                let last = c.pos + c.n == ids.len();
                logits = self.metal.prefill_tail_layers(&c.x, c.pos, c.n, a, last)?;
                metal_s += t_stage.elapsed().as_secs_f64();
                chunks += 1;
            }
            Ok((logits, chunks, ane_s, metal_s))
        })?;

        // The balance of the two sides is the whole story: a pipeline is only worth
        // its complexity while neither engine is waiting much on the other.
        eprintln!(
            "  ANE split prefill: {ane_total} tokens ({chunks} chunks of {s}, layers 0..{a} on ANE ‖ {a}.. on Metal, windows {:?}) \
             in {:.3}s [ANE {:.3}s, Metal {:.3}s, {:.0}% overlapped]",
            rungs.iter().map(|g| g.p).collect::<Vec<_>>(),
            t.elapsed().as_secs_f64(),
            ane_s,
            metal_s,
            100.0 * (1.0 - t.elapsed().as_secs_f64() / (ane_s + metal_s)).max(0.0)
        );
        if ane_total < ids.len() {
            return self.metal.prefill_from(&ids[ane_total..], ane_total).map(Some);
        }
        Ok(Some(logits))
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
            // Say so — the user picked -b hybrid and would otherwise wonder why the
            // Neural Engine stays idle on a short prompt.
            eprintln!(
                "  ANE skipped: {}-token prompt (< {ANE_MIN}) — the GPU prefills it faster than the smallest padded graph",
                ids.len()
            );
            return self.metal.prefill_from(ids, 0);
        }
        // Split prefill is the default whenever the ladder is loaded. A ladder
        // that cannot serve this prompt (Ok(None)) falls through silently; a
        // mid-run failure redoes the whole prompt on the plain path below, which
        // is safe because that path rewrites every KV row from position 0.
        if !self.front.is_empty() {
            match self.prefill_split(ids) {
                Ok(Some(logits)) => return Ok(logits),
                Ok(None) => {}
                Err(e) => {
                    eprintln!("  ANE split prefill failed ({e}) — redoing the prompt on the plain path")
                }
            }
        }
        let largest = self.graphs.last().map(|g| g.seq).unwrap_or(0);
        if want > largest && self.windowed.is_some() {
            return self.prefill_chunked(ids, want);
        }
        // Plain-head guard: a padded head graph only pays while it beats Metal on
        // the same rows, and that inverted for the big shapes when Metal got its
        // 7x (measured 2026-08-30, Qwen: the S=2048 graph takes 0.86 s where
        // Metal does those rows in ~0.66 s — so every no-ladder prompt that
        // reached for it lost to `-b metal` outright, while the S=512 head still
        // wins on both models). Past twice the smallest graph the head costs
        // more than it saves: hand the whole prompt to Metal instead. A model
        // with a split ladder never reaches this point for such lengths.
        let smallest = self.graphs.first().map(|g| g.seq).unwrap_or(0);
        if want > 2 * smallest {
            eprintln!(
                "  ANE skipped: {}-token prompt — Metal outruns the padded {largest}-token graph here \
                 (export a split ladder to put the ANE back to work)",
                ids.len()
            );
            return self.metal.prefill_from(ids, 0);
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

#[cfg(test)]
mod tests {
    #[test]
    fn iso8601_matches_known_epochs() {
        assert_eq!(super::iso8601_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(super::iso8601_utc(1_000_000_000), "2001-09-09T01:46:40Z");
        // A leap-year date past February.
        assert_eq!(super::iso8601_utc(1_709_251_200), "2024-03-01T00:00:00Z");
    }
}
