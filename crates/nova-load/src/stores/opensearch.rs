//! OpenSearch load backend (feature `opensearch`).
//!
//! Maps each [`Point`] to an OpenSearch document: dense vectors become
//! `knn_vector` fields, the payload lands as ordinary (dynamically-mapped)
//! fields, and the point id becomes the document `_id`. Upserts go through the
//! bulk API.
//!
//! Scope for now: **dense vectors + payload**. Sparse and multivector values
//! error out — OpenSearch models those differently (neural sparse /
//! `rank_features`, and no real multivector), and this backend exists to get
//! dense corpora into an OpenSearch cluster.
//!
//! Modelled on the qdrant backend rather than the elastic one, because
//! OpenSearch's indexing lifecycle is genuinely qdrant-shaped:
//! `index.knn.advanced.approximate_threshold: -1` suppresses ANN structure
//! building during bulk load exactly like qdrant's `indexing_threshold: 0`, so
//! the graph build really is a distinct, timeable post-upload phase. (Elastic
//! has no equivalent — it builds inline during ingest.) Concretely that means:
//! typed config structs with a nested `params:` block, request bodies assembled
//! by pure functions ([`build_create_index`] / [`build_update_settings`]) that
//! are unit-tested without a server, a typed [`OpenSearchConfigError`], and a
//! post-settle [`OpenSearchStore::verify_params`] sanity check.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use opensearch::auth::Credentials;
use opensearch::cert::CertificateValidation;
use opensearch::http::Method;
use opensearch::http::headers::HeaderMap;
use opensearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
use opensearch::indices::{
    IndicesCreateParts, IndicesDeleteParts, IndicesExistsParts, IndicesGetMappingParts,
    IndicesGetSettingsParts, IndicesPutSettingsParts, IndicesRefreshParts, IndicesStatsParts,
};
use opensearch::{BulkOperation, BulkParts, ExistsParts, OpenSearch};
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Value, json};
use url::Url;

use crate::config::{VectorKind, VectorSpec};
use crate::stores::{CollectionSchema, Point, PointId, StoreError, VectorStore, VectorValue};

/// Connection + store settings for an OpenSearch backend, as written under
/// `vectorstore:` in the YAML.
///
/// Note what is *not* here: the per-vector schema (distance, size, datatype,
/// on_disk) lives in the top-level `vectors:` section, because the vector name
/// is the key shared between extraction (which parquet column) and storage (the
/// field mapping). The create-index request is assembled in
/// [`build_create_index`] from those vector specs plus the index-wide
/// [`OpenSearchParams`] here — the same split the qdrant backend uses.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenSearchConfig {
    /// Node URL, e.g. `https://localhost:9200`.
    pub url: String,
    /// Basic-auth username (paired with `password`). An out-of-the-box
    /// OpenSearch node uses `admin` plus the password from
    /// `OPENSEARCH_INITIAL_ADMIN_PASSWORD`.
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub password: Option<String>,
    /// Bearer token (JWT), an alternative to username/password. OpenSearch has
    /// no Elasticsearch-style encoded API key, so there is no `api_key` here.
    #[serde(default)]
    pub bearer_token: Option<String>,
    /// Target index. Defaults to `default`.
    #[serde(default = "default_index")]
    pub index_name: String,
    /// Skip TLS certificate validation. Needed for a default OpenSearch node,
    /// which serves HTTPS with a self-signed cert. DEV ONLY.
    #[serde(default)]
    pub tls_insecure: bool,
    /// Per-request timeout in seconds. Bulk upserts into a cluster under load
    /// routinely outlive a short client timeout; when that fires, the server
    /// most likely still applies the write and the retry re-sends the same
    /// documents (duplicate work for an already-slow cluster), so a tight
    /// timeout *amplifies* backpressure instead of surviving it. Default 120,
    /// matching the qdrant backend.
    #[serde(default = "default_timeout_s")]
    pub timeout_s: u64,
    /// Index-wide creation params. All optional; OpenSearch defaults apply,
    /// except for `space_type`, which this backend always writes explicitly —
    /// see [`parse_space_type`].
    #[serde(default)]
    pub params: Option<OpenSearchParams>,
}

/// Index-wide knobs — attributes of the whole index, not of any single vector.
/// (Per-vector knobs belong on [`VectorSpec`] in the `vectors:` section.)
///
/// The `method` is index-wide here even though OpenSearch accepts it per field,
/// mirroring how the qdrant backend treats HNSW/quantization as collection-wide:
/// a load writes one corpus with one index configuration, and per-field method
/// tuning has no caller.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenSearchParams {
    /// `index.number_of_shards`. Static — settable only at creation.
    #[serde(default)]
    pub shards: Option<u32>,
    /// `index.number_of_replicas`. Dynamic — [`reindex`](VectorStore::reindex)
    /// can patch it.
    #[serde(default)]
    pub replicas: Option<u32>,
    /// `index.refresh_interval` in steady state (e.g. `30s`, or `-1` to leave
    /// periodic refresh off). Unset = OpenSearch's 1s default. Bulk loading
    /// always forces `-1` for the duration regardless — see
    /// [`defer_indexing`](VectorStore::defer_indexing).
    #[serde(default)]
    pub refresh_interval: Option<String>,
    /// `index.knn.algo_param.ef_search`, the index-level default search breadth.
    /// Dynamic. A storm's `search_params.ef_search` overrides it per query on
    /// the faiss and nmslib engines; the lucene engine ignores both and uses
    /// `k`.
    #[serde(default)]
    pub ef_search: Option<u32>,
    /// ANN method + engine (`hnsw`/`faiss` unless set). IMMUTABLE after index
    /// creation — every field of it is, per OpenSearch's own mapping tables —
    /// so `reindex` rejects a change rather than silently no-op'ing.
    #[serde(default)]
    pub method: Option<MethodConfig>,
    /// Vector workload mode: `in_memory` (lowest latency) or `on_disk` (lowest
    /// cost). Per-vector `on_disk: true` sets this too; an explicit `mode` here
    /// wins. Immutable after creation.
    #[serde(default)]
    pub mode: Option<String>,
    /// Quantization encoder shorthand that pairs with `mode`, one of `1x`, `2x`,
    /// `4x`, `8x`, `16x`, `32x`. Immutable after creation.
    #[serde(default)]
    pub compression_level: Option<String>,
    /// If the index already exists, drop + recreate it instead of reusing it.
    /// Consumed by this backend, not part of the create request itself.
    #[serde(default)]
    pub recreate: bool,
}

/// The ANN method definition (`params.method`), flattened one level relative to
/// OpenSearch's own JSON: the engine-specific knobs sit directly on this struct
/// rather than under a nested `parameters` object, so `deny_unknown_fields`
/// catches a typo'd knob at config-parse time instead of letting OpenSearch
/// silently ignore it. [`method_json`] re-nests them on the way out.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodConfig {
    /// `hnsw` (default), `ivf`, `flat`, or `disk_ann`.
    #[serde(default)]
    pub name: Option<String>,
    /// `faiss` (OpenSearch's own default), `lucene`, `nmslib` (deprecated), or
    /// `jvector` (needs the `opensearch-jvector` plugin).
    #[serde(default)]
    pub engine: Option<String>,
    /// HNSW: bidirectional links per node. Keep between 2 and 100.
    #[serde(default)]
    pub m: Option<u32>,
    /// HNSW: build-time candidate list size.
    #[serde(default)]
    pub ef_construction: Option<u32>,
    /// faiss HNSW: the engine-level default search breadth, baked into the
    /// mapping. Distinct from `params.ef_search`, which is the (dynamic) index
    /// setting; a query's own `method_parameters.ef_search` overrides both.
    #[serde(default)]
    pub ef_search: Option<u32>,
    /// IVF: number of buckets to partition vectors into. Note IVF requires a
    /// trained model, which this backend does not create.
    #[serde(default)]
    pub nlist: Option<u32>,
    /// IVF: buckets examined per query.
    #[serde(default)]
    pub nprobes: Option<u32>,
    /// Vector encoder — `flat` (default, no compression), `sq` (scalar, fp16),
    /// or `pq` (product; requires a trained model).
    #[serde(default)]
    pub encoder: Option<EncoderConfig>,
}

/// A faiss encoder definition (`params.method.encoder`). Same flattening as
/// [`MethodConfig`]: `pq`'s and `sq`'s knobs sit side by side here and are
/// re-nested under `parameters` by [`encoder_json`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderConfig {
    /// `flat`, `sq`, or `pq`.
    pub name: String,
    /// `pq`: subvectors to split each vector into (the dimension must divide by
    /// it). Also the `sq`/`pq` shared spelling, hence the shared field.
    #[serde(default)]
    pub m: Option<u32>,
    /// `pq`: bits per subvector code. Must be 8 for the `hnsw` method.
    #[serde(default)]
    pub code_size: Option<u32>,
    /// `sq`: quantization type. Only `fp16` exists today.
    #[serde(default, rename = "type")]
    pub sq_type: Option<String>,
    /// `sq`: round out-of-range values into the representable range instead of
    /// rejecting the document.
    #[serde(default)]
    pub clip: Option<bool>,
    /// `sq`: bits per dimension (1 or 16). Required from OpenSearch 3.6 on.
    #[serde(default)]
    pub bits: Option<u32>,
}

fn default_index() -> String {
    "default".to_string()
}

fn default_timeout_s() -> u64 {
    120
}

/// Manual `Debug` so secrets never land in logs, errors, or `--dry-run` output
/// — only whether one is set.
impl fmt::Debug for OpenSearchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenSearchConfig")
            .field("url", &self.url)
            .field("username", &self.username)
            .field("password", &self.password.as_ref().map(|_| "<redacted>"))
            .field(
                "bearer_token",
                &self.bearer_token.as_ref().map(|_| "<redacted>"),
            )
            .field("index_name", &self.index_name)
            .field("tls_insecure", &self.tls_insecure)
            .field("timeout_s", &self.timeout_s)
            .field("params", &self.params)
            .finish()
    }
}

/// Errors that surface only when translating the backend-agnostic config into a
/// concrete OpenSearch request body.
#[derive(Debug, thiserror::Error)]
pub enum OpenSearchConfigError {
    #[error(
        "vector `{0}` is dense but has no size: set `size:` in the vectors config or ensure the loader can infer it from the parquet schema"
    )]
    MissingSize(String),
    #[error(
        "vector `{name}`: unknown distance `{value}` (expected one of: cosine, dot, euclid, manhattan)"
    )]
    UnknownDistance { name: String, value: String },
    #[error(
        "vector `{name}`: unknown datatype `{value}` (expected one of: float32, byte, binary)"
    )]
    UnknownDatatype { name: String, value: String },
    #[error(
        "vector `{name}`: datatype `float16` has no OpenSearch `data_type` — 16-bit storage is a faiss encoder, not an element type. Set `params.method.encoder: {{name: sq, type: fp16, bits: 16}}` and leave the vector's `datatype` at float32."
    )]
    Float16Datatype { name: String },
    #[error(
        "vector `{name}`: datatype `uint8` does not map onto OpenSearch's `byte` data_type — `byte` is SIGNED (-128..127) while `uint8` is 0..255, so half the range would be rejected or wrap. Use `datatype: byte` if the corpus really is signed 8-bit, or keep float32 and quantize with `params.method.encoder`."
    )]
    Uint8Datatype { name: String },
    #[error(
        "vector `{name}` is {kind:?}: the opensearch backend supports dense vectors only for now"
    )]
    UnsupportedVectorKind { name: String, kind: VectorKind },
    #[error("unknown method name `{0}` (expected one of: hnsw, ivf, flat, disk_ann)")]
    UnknownMethodName(String),
    #[error("unknown engine `{0}` (expected one of: faiss, lucene, nmslib, jvector)")]
    UnknownEngine(String),
    #[error("unknown vector mode `{0}` (expected one of: in_memory, on_disk)")]
    UnknownMode(String),
    #[error("unknown compression_level `{0}` (expected one of: 1x, 2x, 4x, 8x, 16x, 32x)")]
    UnknownCompressionLevel(String),
    #[error("unknown encoder name `{0}` (expected one of: flat, sq, pq)")]
    UnknownEncoder(String),
}

/// A nova distance name → an OpenSearch `space_type`.
///
/// Unset means cosine, NOT OpenSearch's own `l2` default. The whole point of
/// the shared `vectors:` block is that one config means the same thing on every
/// backend, and the qdrant backend reads an unset distance as cosine; inheriting
/// `l2` here would make two configs that look identical silently measure
/// different metrics. So this backend always writes `space_type` explicitly.
fn parse_space_type(name: &str, value: Option<&str>) -> Result<&'static str, OpenSearchConfigError> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        None | Some("cosine" | "cosinesimil") => Ok("cosinesimil"),
        Some("dot" | "innerproduct" | "ip") => Ok("innerproduct"),
        Some("euclid" | "euclidean" | "l2") => Ok("l2"),
        Some("manhattan" | "l1") => Ok("l1"),
        Some(other) => Err(OpenSearchConfigError::UnknownDistance {
            name: name.to_string(),
            value: other.to_string(),
        }),
    }
}

/// A nova datatype name → an OpenSearch `data_type`. `None` leaves the field
/// unset (OpenSearch defaults to `float`).
///
/// `float16` and `uint8` are rejected with pointed errors rather than mapped
/// onto a near-miss: OpenSearch has no 16-bit element type (that is the faiss
/// `sq` encoder), and its `byte` is signed, so silently accepting `uint8` would
/// corrupt the top half of the range.
fn parse_data_type(
    name: &str,
    value: Option<&str>,
) -> Result<Option<&'static str>, OpenSearchConfigError> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        None => Ok(None),
        Some("float32" | "f32" | "float") => Ok(Some("float")),
        Some("byte" | "int8" | "i8") => Ok(Some("byte")),
        Some("binary") => Ok(Some("binary")),
        Some("float16" | "f16") => Err(OpenSearchConfigError::Float16Datatype {
            name: name.to_string(),
        }),
        Some("uint8" | "u8") => Err(OpenSearchConfigError::Uint8Datatype {
            name: name.to_string(),
        }),
        Some(other) => Err(OpenSearchConfigError::UnknownDatatype {
            name: name.to_string(),
            value: other.to_string(),
        }),
    }
}

fn parse_method_name(value: Option<&str>) -> Result<&'static str, OpenSearchConfigError> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        None | Some("hnsw") => Ok("hnsw"),
        Some("ivf") => Ok("ivf"),
        Some("flat") => Ok("flat"),
        Some("disk_ann" | "diskann") => Ok("disk_ann"),
        Some(other) => Err(OpenSearchConfigError::UnknownMethodName(other.to_string())),
    }
}

/// The engine, defaulting to `faiss`. That is already OpenSearch's own default,
/// but it is pinned explicitly so a config keeps meaning the same thing across
/// server versions (the default was `nmslib` before 2.19) and so the coverage
/// test can assert it.
fn parse_engine(value: Option<&str>) -> Result<&'static str, OpenSearchConfigError> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        None | Some("faiss") => Ok("faiss"),
        Some("lucene") => Ok("lucene"),
        Some("nmslib") => Ok("nmslib"),
        Some("jvector") => Ok("jvector"),
        Some(other) => Err(OpenSearchConfigError::UnknownEngine(other.to_string())),
    }
}

fn parse_mode(value: Option<&str>) -> Result<Option<&'static str>, OpenSearchConfigError> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        None => Ok(None),
        Some("in_memory") => Ok(Some("in_memory")),
        Some("on_disk") => Ok(Some("on_disk")),
        Some(other) => Err(OpenSearchConfigError::UnknownMode(other.to_string())),
    }
}

fn parse_compression_level(
    value: Option<&str>,
) -> Result<Option<&'static str>, OpenSearchConfigError> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        None => Ok(None),
        Some("1x") => Ok(Some("1x")),
        Some("2x") => Ok(Some("2x")),
        Some("4x") => Ok(Some("4x")),
        Some("8x") => Ok(Some("8x")),
        Some("16x") => Ok(Some("16x")),
        Some("32x") => Ok(Some("32x")),
        Some(other) => Err(OpenSearchConfigError::UnknownCompressionLevel(
            other.to_string(),
        )),
    }
}

/// An encoder config → its OpenSearch JSON (`{name, parameters: {...}}`).
/// Unset knobs are omitted so the engine keeps its own defaults.
fn encoder_json(enc: &EncoderConfig) -> Result<Value, OpenSearchConfigError> {
    let name = match enc.name.to_ascii_lowercase().as_str() {
        "flat" => "flat",
        "sq" => "sq",
        "pq" => "pq",
        other => return Err(OpenSearchConfigError::UnknownEncoder(other.to_string())),
    };
    let mut params = serde_json::Map::new();
    if let Some(m) = enc.m {
        params.insert("m".into(), json!(m));
    }
    if let Some(cs) = enc.code_size {
        params.insert("code_size".into(), json!(cs));
    }
    if let Some(t) = &enc.sq_type {
        params.insert("type".into(), json!(t));
    }
    if let Some(c) = enc.clip {
        params.insert("clip".into(), json!(c));
    }
    if let Some(b) = enc.bits {
        params.insert("bits".into(), json!(b));
    }
    let mut out = serde_json::Map::from_iter([("name".to_string(), json!(name))]);
    if !params.is_empty() {
        out.insert("parameters".to_string(), Value::Object(params));
    }
    Ok(Value::Object(out))
}

/// A [`MethodConfig`] → the OpenSearch `method` object, re-nesting the flat
/// engine knobs under `parameters`. `space_type` is NOT set here — it is written
/// at the top level of the field mapping, which is the form that also works when
/// no `method` is configured at all.
fn method_json(method: &MethodConfig) -> Result<Value, OpenSearchConfigError> {
    let mut params = serde_json::Map::new();
    if let Some(m) = method.m {
        params.insert("m".into(), json!(m));
    }
    if let Some(ef) = method.ef_construction {
        params.insert("ef_construction".into(), json!(ef));
    }
    if let Some(ef) = method.ef_search {
        params.insert("ef_search".into(), json!(ef));
    }
    if let Some(nlist) = method.nlist {
        params.insert("nlist".into(), json!(nlist));
    }
    if let Some(nprobes) = method.nprobes {
        params.insert("nprobes".into(), json!(nprobes));
    }
    if let Some(enc) = &method.encoder {
        params.insert("encoder".into(), encoder_json(enc)?);
    }

    let mut out = serde_json::Map::from_iter([
        (
            "name".to_string(),
            json!(parse_method_name(method.name.as_deref())?),
        ),
        (
            "engine".to_string(),
            json!(parse_engine(method.engine.as_deref())?),
        ),
    ]);
    if !params.is_empty() {
        out.insert("parameters".to_string(), Value::Object(params));
    }
    Ok(Value::Object(out))
}

/// One dense vector spec → its `knn_vector` field mapping.
fn knn_vector_field(
    name: &str,
    spec: &VectorSpec,
    params: &OpenSearchParams,
    dims: &HashMap<String, u64>,
) -> Result<Value, OpenSearchConfigError> {
    if spec.kind != VectorKind::Dense {
        return Err(OpenSearchConfigError::UnsupportedVectorKind {
            name: name.to_string(),
            kind: spec.kind,
        });
    }
    let dimension = spec
        .size
        .or_else(|| dims.get(name).copied())
        .ok_or_else(|| OpenSearchConfigError::MissingSize(name.to_string()))?;

    let mut field = serde_json::Map::from_iter([
        ("type".to_string(), json!("knn_vector")),
        ("dimension".to_string(), json!(dimension)),
        (
            "space_type".to_string(),
            json!(parse_space_type(name, spec.distance.as_deref())?),
        ),
    ]);
    if let Some(dt) = parse_data_type(name, spec.datatype.as_deref())? {
        field.insert("data_type".to_string(), json!(dt));
    }
    // An explicit index-wide `mode` wins; otherwise the per-vector `on_disk`
    // flag selects it, which is how the same `vectors:` block expresses "keep
    // this off the heap" on qdrant.
    let mode = match parse_mode(params.mode.as_deref())? {
        Some(m) => Some(m),
        None => spec.on_disk.map(|d| if d { "on_disk" } else { "in_memory" }),
    };
    if let Some(m) = mode {
        field.insert("mode".to_string(), json!(m));
    }
    if let Some(c) = parse_compression_level(params.compression_level.as_deref())? {
        field.insert("compression_level".to_string(), json!(c));
    }
    if let Some(method) = &params.method {
        field.insert("method".to_string(), method_json(method)?);
    }
    Ok(Value::Object(field))
}

/// Build the create-index body by gathering fields from two sources:
/// - per-vector schema from `vectors` (the top-level specs) + `dims`, one
///   `knn_vector` field each
/// - index-wide knobs from `params`
///
/// `dims` supplies resolved dimensionality by name; the loader fills it from the
/// parquet schema, and an explicit `size:` on a [`VectorSpec`] takes precedence.
///
/// `index.knn: true` is always set — it is a STATIC setting, so an index created
/// without it can never do approximate search, and there is no recovery short of
/// recreating it.
pub fn build_create_index(
    vectors: &HashMap<String, VectorSpec>,
    params: &OpenSearchParams,
    dims: &HashMap<String, u64>,
) -> Result<Value, OpenSearchConfigError> {
    let mut properties = serde_json::Map::new();
    for (name, spec) in vectors {
        properties.insert(name.clone(), knn_vector_field(name, spec, params, dims)?);
    }

    let mut index = serde_json::Map::from_iter([("knn".to_string(), json!(true))]);
    if let Some(v) = params.shards {
        index.insert("number_of_shards".to_string(), json!(v));
    }
    if let Some(v) = params.replicas {
        index.insert("number_of_replicas".to_string(), json!(v));
    }
    if let Some(v) = &params.refresh_interval {
        index.insert("refresh_interval".to_string(), json!(v));
    }
    if let Some(v) = params.ef_search {
        index.insert("knn.algo_param.ef_search".to_string(), json!(v));
    }

    Ok(json!({
        "settings": { "index": Value::Object(index) },
        "mappings": { "properties": Value::Object(properties) },
    }))
}

/// Build the settings body that patches index knobs on an *already-existing*
/// index, from the same [`OpenSearchParams`] shape [`build_create_index`] reads
/// at creation time.
///
/// Only the DYNAMIC subset appears here. Everything else OpenSearch documents as
/// "Updatable after index creation: No" — the whole `method` object, `mode`,
/// `compression_level`, `number_of_shards`, and every field's `dimension` /
/// `space_type` / `data_type`. Those are deliberately not read here, the same
/// way qdrant's `build_update_collection` deliberately ignores structural
/// params; [`OpenSearchStore::reindex`] refuses a run that asks to change one
/// rather than silently no-op'ing.
///
/// Returns `None` when nothing dynamic is configured, so the caller can skip the
/// request entirely instead of PUTting an empty body.
pub fn build_update_settings(params: &OpenSearchParams) -> Option<Value> {
    let mut index = serde_json::Map::new();
    if let Some(v) = params.replicas {
        index.insert("number_of_replicas".to_string(), json!(v));
    }
    if let Some(v) = &params.refresh_interval {
        index.insert("refresh_interval".to_string(), json!(v));
    }
    if let Some(v) = params.ef_search {
        index.insert("knn.algo_param.ef_search".to_string(), json!(v));
    }
    (!index.is_empty()).then(|| json!({ "index": Value::Object(index) }))
}

impl OpenSearchConfig {
    /// Build the client once. Consumes the config to avoid cloning its fields.
    pub async fn connect(self) -> Result<OpenSearchStore, StoreError> {
        let creds = if let Some(token) = self.bearer_token {
            Some(Credentials::Bearer(token))
        } else if let (Some(u), Some(p)) = (self.username, self.password) {
            Some(Credentials::Basic(u, p))
        } else {
            None
        };

        // Always go through TransportBuilder so auth, timeout and TLS
        // validation are configurable (Transport::single_node can do none).
        let pool = SingleNodeConnectionPool::new(Url::parse(&self.url).map_err(to_store_err)?);
        let mut builder =
            TransportBuilder::new(pool).timeout(Duration::from_secs(self.timeout_s));
        if let Some(creds) = creds {
            builder = builder.auth(creds);
        }
        if self.tls_insecure {
            builder = builder.cert_validation(CertificateValidation::None);
        }
        let transport = builder.build().map_err(to_store_err)?;

        Ok(OpenSearchStore {
            client: OpenSearch::new(transport),
            index_name: self.index_name,
            params: self.params.unwrap_or_default(),
            expected_vectors: Mutex::new(HashMap::new()),
        })
    }
}

/// Flatten an error and its `source()` chain into one message — the top-level
/// opensearch/reqwest message is often just "error sending request for url
/// (...)", with the real cause (connection refused, invalid certificate, TLS
/// handshake) one or two links down.
fn flatten<E: std::error::Error>(e: &E) -> String {
    let mut msg = e.to_string();
    let mut src = e.source();
    while let Some(s) = src {
        msg.push_str(&format!(": {s}"));
        src = s.source();
    }
    msg
}

/// A transport-level client error → a [`StoreError`], preserving the HTTP status
/// when there is one so [`StoreError::is_retryable`] can classify it.
fn to_store_err<E: std::error::Error>(e: E) -> StoreError {
    StoreError::OpenSearch {
        status: None,
        message: flatten(&e),
    }
}

/// An OpenSearch client error, keeping its status code for retry classification.
fn client_err(e: opensearch::Error) -> StoreError {
    StoreError::OpenSearch {
        status: e.status_code().map(|s| s.as_u16()),
        message: flatten(&e),
    }
}

/// A non-2xx response → a [`StoreError`] carrying the status, so a 400 (bad
/// mapping, malformed document) aborts immediately instead of burning the whole
/// upsert retry budget re-sending what the server already rejected, while a 429
/// or 503 still retries.
async fn status_err(context: &str, resp: opensearch::http::response::Response) -> StoreError {
    let status = resp.status_code().as_u16();
    let detail = resp.text().await.unwrap_or_default();
    StoreError::OpenSearch {
        status: Some(status),
        message: format!("{context} failed (HTTP {status}): {detail}"),
    }
}

/// A connected OpenSearch backend. Like qdrant (and unlike the data sources,
/// where the config itself implements the trait), the store holds an initialized
/// client reused across every `upsert_batch`, so it is a distinct runtime object
/// built once.
pub struct OpenSearchStore {
    client: OpenSearch,
    index_name: String,
    params: OpenSearchParams,
    /// The per-vector schema the last `enable_indexing`/`reindex` asked for,
    /// checked against the live mapping by [`OpenSearchStore::verify_params`]
    /// once indexing settles.
    ///
    /// It is held here rather than threaded through because
    /// [`wait_for_indexing`](VectorStore::wait_for_indexing) takes no schema —
    /// the same reason the elastic backend keeps an `expected` list. Empty
    /// (e.g. a distributed `finalize` that never ran `enable_indexing` on this
    /// process) means the mapping half of the check is skipped; the settings
    /// half still runs, since those come from `params`.
    expected_vectors: Mutex<HashMap<String, VectorSpec>>,
}

impl fmt::Display for OpenSearchStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "opensearch({})", self.index_name)
    }
}

impl OpenSearchStore {
    fn index(&self) -> &str {
        self.index_name.as_str()
    }

    /// PUT an index settings body.
    async fn put_settings(&self, body: Value) -> Result<(), StoreError> {
        let resp = self
            .client
            .indices()
            .put_settings(IndicesPutSettingsParts::Index(&[self.index()]))
            .body(body)
            .send()
            .await
            .map_err(client_err)?;
        if !resp.status_code().is_success() {
            return Err(status_err("put settings", resp).await);
        }
        Ok(())
    }

    /// Force a refresh so newly-indexed documents become searchable now.
    async fn refresh(&self) -> Result<(), StoreError> {
        let resp = self
            .client
            .indices()
            .refresh(IndicesRefreshParts::Index(&[self.index()]))
            .send()
            .await
            .map_err(client_err)?;
        if !resp.status_code().is_success() {
            return Err(status_err("refresh", resp).await);
        }
        Ok(())
    }

    /// Trigger an ASYNCHRONOUS force-merge to a single segment.
    ///
    /// This is what actually builds the ANN structures for everything loaded
    /// while `approximate_threshold` was `-1`: that setting is consulted when a
    /// segment is written, so flipping it back only affects *future* segments —
    /// the already-written ones have to be rewritten. Rewriting them also
    /// consolidates the index into one segment, which is the state a benchmark
    /// wants to query.
    ///
    /// `wait_for_completion=false` is passed through
    /// [`OpenSearch::send`](opensearch::OpenSearch::send) because the client's
    /// typed `forcemerge` builder does not expose that parameter, and a blocking
    /// merge would put the entire graph build inside `enable_indexing` — where
    /// the trait's timing contract cannot see it. `wait_for_indexing` is what
    /// waits for it to settle.
    async fn force_merge(&self) -> Result<(), StoreError> {
        let path = format!("/{}/_forcemerge", self.index());
        let resp = self
            .client
            .send(
                Method::Post,
                &path,
                HeaderMap::new(),
                Some(&[
                    ("max_num_segments", "1"),
                    ("wait_for_completion", "false"),
                ]),
                None::<()>,
                None,
            )
            .await
            .map_err(client_err)?;
        if !resp.status_code().is_success() {
            return Err(status_err("force merge", resp).await);
        }
        Ok(())
    }

    /// Number of merges currently running on the index (`_stats/merge` →
    /// `indices.<name>.total.merges.current`). The ANN structures are written as
    /// segments are written and rewritten as segments merge, so a settled merge
    /// count is the "no more index building happening" signal — the OpenSearch
    /// analog of qdrant's green status and milvus's `pending_index_rows`.
    async fn merges_current(&self) -> Result<i64, StoreError> {
        let index = self.index();
        let resp = self
            .client
            .indices()
            .stats(IndicesStatsParts::IndexMetric(&[index], &["merge"]))
            .send()
            .await
            .map_err(client_err)?;
        if !resp.status_code().is_success() {
            return Err(status_err("merge stats", resp).await);
        }
        let body: Value = resp.json().await.map_err(client_err)?;
        // Prefer the per-index total; fall back to the cluster-wide `_all`.
        Ok(body["indices"][index]["total"]["merges"]["current"]
            .as_i64()
            .or_else(|| body["_all"]["total"]["merges"]["current"].as_i64())
            .unwrap_or(0))
    }

    /// The live `mappings.properties` object for this index.
    async fn live_properties(&self) -> Result<Value, StoreError> {
        let index = self.index();
        let resp = self
            .client
            .indices()
            .get_mapping(IndicesGetMappingParts::Index(&[index]))
            .send()
            .await
            .map_err(client_err)?;
        if !resp.status_code().is_success() {
            return Err(status_err("get mapping", resp).await);
        }
        let body: Value = resp.json().await.map_err(client_err)?;
        Ok(body[index]["mappings"]["properties"].clone())
    }

    /// The live `settings.index` object for this index.
    async fn live_settings(&self) -> Result<Value, StoreError> {
        let index = self.index();
        let resp = self
            .client
            .indices()
            .get_settings(IndicesGetSettingsParts::Index(&[index]))
            .send()
            .await
            .map_err(client_err)?;
        if !resp.status_code().is_success() {
            return Err(status_err("get settings", resp).await);
        }
        let body: Value = resp.json().await.map_err(client_err)?;
        Ok(body[index]["settings"]["index"].clone())
    }

    /// Sanity check: confirm the live mapping and settings reflect what we asked
    /// for. Every field set in config must match what the server reports; unset
    /// fields are skipped (OpenSearch keeps its own defaults). Runs after
    /// indexing settles in `wait_for_indexing`, mirroring qdrant's
    /// `verify_params`.
    ///
    /// The mapping side is a SUBSET check on `method.parameters`: OpenSearch
    /// fills in defaults for anything we left unset, so only the knobs the
    /// config named are compared.
    async fn verify_params(&self) -> Result<(), StoreError> {
        let expected = self.expected_vectors.lock().expect("mapping lock").clone();
        let props = if expected.is_empty() {
            Value::Null
        } else {
            self.live_properties().await?
        };

        for (name, spec) in &expected {
            let got = &props[name];
            if got.is_null() {
                return Err(StoreError::Other(format!(
                    "sanity check FAILED on {self}: field `{name}` missing from the live mapping"
                )));
            }
            let want_space = parse_space_type(name, spec.distance.as_deref())
                .map_err(|e| StoreError::Other(e.to_string()))?;
            // `space_type` may be reported at the field's top level or inside
            // `method`, depending on where it was written and the server
            // version — accept either, they mean the same thing.
            let got_space = got["space_type"]
                .as_str()
                .or_else(|| got["method"]["space_type"].as_str())
                .unwrap_or_default();
            if got_space != want_space {
                return Err(StoreError::Other(format!(
                    "sanity check FAILED on {self} field `{name}`: requested \
                     space_type={want_space} but the live mapping has {got_space}"
                )));
            }

            if let Some(method) = &self.params.method {
                let want_engine = parse_engine(method.engine.as_deref())
                    .map_err(|e| StoreError::Other(e.to_string()))?;
                let got_engine = got["method"]["engine"].as_str().unwrap_or_default();
                if got_engine != want_engine {
                    return Err(StoreError::Other(format!(
                        "sanity check FAILED on {self} field `{name}`: requested \
                         engine={want_engine} but the live mapping has {got_engine}"
                    )));
                }
                let live_params = &got["method"]["parameters"];
                check_u64_json(
                    self,
                    name,
                    "method.parameters.m",
                    method.m,
                    &live_params["m"],
                )?;
                check_u64_json(
                    self,
                    name,
                    "method.parameters.ef_construction",
                    method.ef_construction,
                    &live_params["ef_construction"],
                )?;
            }
            tracing::info!(
                "{self} field `{name}` verified: space_type={got_space} method={}",
                got["method"]
            );
        }

        // Settings side: the dynamic knobs we set. OpenSearch reports every
        // index setting as a STRING, hence the parse rather than `as_u64`.
        let settings = self.live_settings().await?;
        check_u64_setting(
            self,
            "number_of_replicas",
            self.params.replicas,
            &lookup_setting(&settings, "number_of_replicas"),
        )?;
        check_u64_setting(
            self,
            "knn.algo_param.ef_search",
            self.params.ef_search,
            &lookup_setting(&settings, "knn.algo_param.ef_search"),
        )?;

        tracing::info!("{self} index params verified against the live index config");
        Ok(())
    }
}

/// Read one dotted index setting out of a `GET _settings` response.
///
/// OpenSearch does NOT return settings in a single consistent shape: it echoes
/// each key grouped the way it happens to store it, so a real 3.8 response
/// contains `"knn": "true"` (that is `index.knn`) and `"knn.algo_param": {
/// "ef_search": "200" }` SIDE BY SIDE in the same object. A plain nested walk
/// for `knn` → `algo_param` → `ef_search` therefore lands on the string `"true"`
/// and reports the setting as missing.
///
/// So flatten every nested object into dotted paths first and do one exact
/// lookup, which is correct whichever grouping the server picked. Values come
/// back as strings (`"200"`, not `200`) — see [`check_u64_setting`].
fn lookup_setting(settings: &Value, dotted: &str) -> Value {
    fn flatten(prefix: &str, value: &Value, out: &mut HashMap<String, Value>) {
        match value {
            Value::Object(map) => {
                for (k, v) in map {
                    let key = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{prefix}.{k}")
                    };
                    flatten(&key, v, out);
                }
            }
            leaf => {
                out.insert(prefix.to_string(), leaf.clone());
            }
        }
    }
    let mut flat = HashMap::new();
    flatten("", settings, &mut flat);
    flat.get(dotted).cloned().unwrap_or(Value::Null)
}

/// Compare a requested `u64` knob against a live mapping value (a real JSON
/// number). Unset requests are skipped — the server keeps its default.
fn check_u64_json(
    store: &OpenSearchStore,
    field: &str,
    knob: &str,
    want: Option<u32>,
    got: &Value,
) -> Result<(), StoreError> {
    if let Some(w) = want
        && got.as_u64() != Some(w as u64)
    {
        return Err(StoreError::Other(format!(
            "sanity check FAILED on {store} field `{field}` {knob}: requested {w} but the \
             live mapping has {got}"
        )));
    }
    Ok(())
}

/// A live index setting as a number. OpenSearch returns settings as STRINGS
/// (`"1"`, not `1`), so a numeric value is accepted too rather than assuming one
/// representation.
fn setting_as_u64(got: &Value) -> Option<u64> {
    got.as_u64()
        .or_else(|| got.as_str().and_then(|s| s.parse::<u64>().ok()))
}

/// Compare a requested `u64` knob against a live index SETTING.
fn check_u64_setting(
    store: &OpenSearchStore,
    knob: &str,
    want: Option<u32>,
    got: &Value,
) -> Result<(), StoreError> {
    if let Some(w) = want {
        let live = setting_as_u64(got);
        if live != Some(w as u64) {
            return Err(StoreError::Other(format!(
                "sanity check FAILED on {store} {knob}: requested {w} but the live index \
                 config has {got}"
            )));
        }
    }
    Ok(())
}

/// One bulk document: the point's payload fields, plus its dense vectors.
///
/// The vectors are held as `Vec<f32>` and serialized straight into the request
/// instead of being folded into the payload [`Value`] first, and that is the
/// whole reason this type exists rather than a plain `Value::Object`.
/// `serde_json::Number` is f64-backed, so an `f32` stored in a `Value` is
/// re-emitted as the shortest form of the *promoted f64*: `0.1f32` goes out as
/// `0.10000000149011612` where `0.1` round-trips to the identical f32. That is
/// roughly twice the bytes for every component of every vector on every bulk
/// request, plus the matching parse cost on the server — a self-inflicted
/// handicap in a benchmarking tool, and a misleading one, since it makes the
/// backend's wire cost look worse than JSON actually requires.
///
/// Serializing the slice directly reaches serde's `serialize_f32`, which keeps
/// the f32 shortest form. Purely a size/cost fix: both spellings parse back to
/// the same f32, so indexed vectors are unchanged.
struct BulkDoc {
    payload: serde_json::Map<String, Value>,
    /// `(vector name, values)`, in `HashMap` iteration order — field order in
    /// a JSON object is not significant to OpenSearch.
    vectors: Vec<(String, Vec<f32>)>,
}

impl Serialize for BulkDoc {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.payload.len() + self.vectors.len()))?;
        for (key, value) in &self.payload {
            map.serialize_entry(key, value)?;
        }
        for (name, values) in &self.vectors {
            // `values` is `&Vec<f32>`, so this reaches `serialize_f32` per
            // component — see the type docs.
            map.serialize_entry(name, values)?;
        }
        map.end()
    }
}

#[async_trait]
impl VectorStore for OpenSearchStore {
    async fn ensure_collection(&self, schema: &CollectionSchema) -> Result<(), StoreError> {
        let index = self.index();
        let exists = self
            .client
            .indices()
            .exists(IndicesExistsParts::Index(&[index]))
            .send()
            .await
            .map_err(client_err)?
            .status_code()
            .is_success();

        if exists {
            if !self.params.recreate {
                // Same posture as the qdrant backend: an existing index is
                // taken at face value rather than diffed against the config.
                return Ok(());
            }
            let resp = self
                .client
                .indices()
                .delete(IndicesDeleteParts::Index(&[index]))
                .send()
                .await
                .map_err(client_err)?;
            if !resp.status_code().is_success() {
                return Err(status_err("delete index", resp).await);
            }
        }

        let body = build_create_index(&schema.vectors, &self.params, &schema.dims)
            .map_err(|e| StoreError::Other(e.to_string()))?;
        let resp = self
            .client
            .indices()
            .create(IndicesCreateParts::Index(index))
            .body(body)
            .send()
            .await
            .map_err(client_err)?;
        if !resp.status_code().is_success() {
            return Err(status_err(&format!("create index `{index}`"), resp).await);
        }
        Ok(())
    }

    async fn upsert_batch(&self, points: Vec<Point>) -> Result<(), StoreError> {
        if points.is_empty() {
            return Ok(());
        }
        let mut ops: Vec<BulkOperation<BulkDoc>> = Vec::with_capacity(points.len());
        for point in points {
            let id = match &point.id {
                PointId::Integer(n) => n.to_string(),
                PointId::String(s) => s.clone(),
            };
            let mut vectors = Vec::with_capacity(point.vectors.len());
            for (name, value) in point.vectors {
                match value {
                    VectorValue::Dense(d) => vectors.push((name, d)),
                    // Guarded at the point of use, so no separate check can
                    // drift out of sync with what the mapping builder accepts.
                    VectorValue::Sparse { .. } | VectorValue::Multi(_) => {
                        return Err(StoreError::Other(format!(
                            "the opensearch backend supports dense vectors only; \
                             vector `{name}` is not dense"
                        )));
                    }
                }
            }
            let doc = BulkDoc {
                payload: point.payload,
                vectors,
            };
            ops.push(BulkOperation::index(doc).id(id).into());
        }

        let resp = self
            .client
            .bulk(BulkParts::Index(self.index()))
            .body(ops)
            .send()
            .await
            .map_err(client_err)?;
        if !resp.status_code().is_success() {
            return Err(status_err("bulk upsert", resp).await);
        }
        // A 200 can still carry per-item failures; surface the first one, with
        // its own status so retryability is judged on the item, not the
        // envelope.
        let body: Value = resp.json().await.map_err(client_err)?;
        if body["errors"].as_bool() == Some(true) {
            let first = body["items"]
                .as_array()
                .and_then(|items| items.iter().find(|it| !it["index"]["error"].is_null()));
            let status = first.and_then(|it| it["index"]["status"].as_u64()).map(|s| s as u16);
            let detail = first
                .map(|it| it["index"]["error"].to_string())
                .unwrap_or_else(|| "unknown bulk item error".to_string());
            return Err(StoreError::OpenSearch {
                status,
                message: format!("bulk upsert had item errors: {detail}"),
            });
        }
        Ok(())
    }

    async fn point_exists(&self, id: &PointId) -> Result<bool, StoreError> {
        // `HEAD /<index>/_doc/<id>` — no payload, no vector, no body at all:
        // pure existence, which is what the `--continue` resume probe needs
        // (the same reasoning as the qdrant backend's payload-free
        // `get_points`).
        let id = match id {
            PointId::Integer(n) => n.to_string(),
            PointId::String(s) => s.clone(),
        };
        let resp = self
            .client
            .exists(ExistsParts::IndexId(self.index(), &id))
            .send()
            .await
            .map_err(client_err)?;
        let status = resp.status_code();
        if status.as_u16() == 404 {
            return Ok(false);
        }
        if !status.is_success() {
            return Err(status_err("document exists probe", resp).await);
        }
        Ok(true)
    }

    async fn close(&self) -> Result<(), StoreError> {
        // The HTTP client cleans up on drop; nothing to do.
        Ok(())
    }

    async fn defer_indexing(&self) -> Result<(), StoreError> {
        // Two levers, both dynamic:
        //
        // `knn.advanced.approximate_threshold: -1` stops OpenSearch building
        // ANN data structures for the segments written during the load. This is
        // the direct analog of qdrant's `indexing_threshold: 0` — it is what
        // makes the graph build a distinct, timeable phase afterwards instead of
        // a cost smeared through ingestion.
        //
        // `refresh_interval: -1` stops paying to publish a new searchable
        // segment on every interval while bulk indexing.
        //
        // The index is created by `ensure_collection` before this runs, so the
        // PUT targets an existing index. Both are restored in `enable_indexing`.
        self.put_settings(json!({
            "index": {
                "knn.advanced.approximate_threshold": -1,
                "refresh_interval": "-1",
            }
        }))
        .await
    }

    async fn enable_indexing(&self, schema: &CollectionSchema) -> Result<(), StoreError> {
        // Record what the mapping should be so `wait_for_indexing` can verify
        // it once the build settles.
        *self.expected_vectors.lock().expect("mapping lock") = schema.vectors.clone();
        // Restore both deferral levers: `0` = always build ANN structures (the
        // OpenSearch default), and the configured refresh interval or `null` to
        // fall back to the 1s default.
        let refresh = match &self.params.refresh_interval {
            Some(v) => json!(v),
            None => Value::Null,
        };
        self.put_settings(json!({
            "index": {
                "knn.advanced.approximate_threshold": 0,
                "refresh_interval": refresh,
            }
        }))
        .await?;
        // Publish everything bulk-loaded, then rewrite the segments so the ANN
        // structures skipped during the load actually get built. Async — the
        // build is what `wait_for_indexing` times.
        self.refresh().await?;
        self.force_merge().await
    }

    async fn wait_for_indexing(&self) -> Result<Instant, StoreError> {
        // Reach a genuine steady state, like the qdrant and milvus backends —
        // not just "docs are searchable". OpenSearch writes each segment's ANN
        // structures as the segment is written and rebuilds them as segments
        // merge, so "indexing is really done" means no merges are running —
        // HELD for a window, so a brief lull between merge waves doesn't read as
        // completion. Return the instant the merges first settled (the start of
        // the hold), so the caller measures real build time excluding the hold.
        const POLL_INTERVAL: Duration = Duration::from_secs(1);
        const SETTLED_HOLD: Duration = Duration::from_secs(5);
        const TIMEOUT: Duration = Duration::from_secs(1800);

        let start = Instant::now();
        let mut settled_since: Option<Instant> = None;
        loop {
            if start.elapsed() >= TIMEOUT {
                return Err(StoreError::Other(format!(
                    "timed out after {}s waiting for merges to settle on {self}",
                    TIMEOUT.as_secs()
                )));
            }

            let current = self.merges_current().await?;
            tracing::info!("{self} indexing: merges_current={current}");

            if current == 0 {
                let since = *settled_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= SETTLED_HOLD {
                    tracing::info!(
                        "{self} indexing settled (no merges running for {}s)",
                        SETTLED_HOLD.as_secs()
                    );
                    // Sanity-check the live mapping + settings against what was
                    // requested. Runs after `since` (the converged instant we
                    // return), so it is not counted in the caller's build
                    // timing — same placement as the qdrant backend's.
                    self.verify_params().await?;
                    return Ok(since);
                }
            } else if settled_since.take().is_some() {
                tracing::warn!(
                    "{self} merges resumed (merges_current={current}); restarting the {}s window",
                    SETTLED_HOLD.as_secs()
                );
            }

            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    async fn reindex(&self, schema: &CollectionSchema) -> Result<(), StoreError> {
        // Patch what OpenSearch documents as dynamic; refuse what it documents
        // as immutable rather than silently no-op'ing.
        //
        // Every `method` field (name, engine, space_type, m, ef_construction,
        // …), plus `mode`, `compression_level` and the per-field `dimension` /
        // `data_type`, are all "Updatable after index creation: No". Unlike
        // Elasticsearch — where an Update Mapping + force-merge can widen
        // `index_options` in place — OpenSearch has no in-place path for them at
        // all; changing one needs a new index and a reload. So a config that
        // asks for a different method here is a mistake worth stopping on: the
        // alternative is a `reindex` that reports success while changing
        // nothing, and a sweep that attributes identical numbers to different
        // settings.
        let live = self.live_properties().await?;
        for (name, spec) in &schema.vectors {
            let got = &live[name];
            if got.is_null() {
                return Err(StoreError::Other(format!(
                    "reindex: field `{name}` not found in the mapping on {self}"
                )));
            }
            let want_space = parse_space_type(name, spec.distance.as_deref())
                .map_err(|e| StoreError::Other(e.to_string()))?;
            let got_space = got["space_type"]
                .as_str()
                .or_else(|| got["method"]["space_type"].as_str())
                .unwrap_or_default();
            if got_space != want_space {
                return Err(StoreError::Other(format!(
                    "reindex: opensearch cannot change `space_type` in place on `{name}` \
                     ({got_space} → {want_space}); that needs a new index + a reload"
                )));
            }
            if let Some(method) = &self.params.method {
                let want = method_json(method).map_err(|e| StoreError::Other(e.to_string()))?;
                let got_method = &got["method"];
                if !method_matches(&want, got_method) {
                    return Err(StoreError::Other(format!(
                        "reindex: opensearch cannot change the ANN `method` in place on \
                         `{name}` (live {got_method}, requested {want}); every method field \
                         is immutable after index creation, so this needs a new index + a \
                         reload. Sweep it on a fresh-load axis, not an in-place one."
                    )));
                }
            }
        }

        *self.expected_vectors.lock().expect("mapping lock") = schema.vectors.clone();
        match build_update_settings(&self.params) {
            Some(body) => {
                tracing::info!("{self}: reindex patching dynamic settings → {body}");
                self.put_settings(body).await
            }
            None => {
                tracing::warn!(
                    "{self}: no dynamic settings configured (replicas / refresh_interval / \
                     ef_search); reindex has nothing to patch"
                );
                Ok(())
            }
        }
    }

    async fn delete_collection(&self) -> Result<(), StoreError> {
        // Best-effort: a 404 (index absent) is fine.
        let resp = self
            .client
            .indices()
            .delete(IndicesDeleteParts::Index(&[self.index()]))
            .send()
            .await
            .map_err(client_err)?;
        let status = resp.status_code();
        if !status.is_success() && status.as_u16() != 404 {
            return Err(status_err("delete index", resp).await);
        }
        Ok(())
    }
}

/// Whether a live `method` object satisfies the requested one: same `name` and
/// `engine`, and every requested `parameters` knob present with that value.
/// A SUBSET check on the parameters, because OpenSearch materializes its own
/// defaults for anything the config left unset.
fn method_matches(want: &Value, got: &Value) -> bool {
    if got.is_null() {
        return false;
    }
    if want["name"] != got["name"] || want["engine"] != got["engine"] {
        return false;
    }
    let Some(want_params) = want.get("parameters").and_then(Value::as_object) else {
        return true;
    };
    let got_params = &got["parameters"];
    want_params
        .iter()
        .all(|(k, v)| got_params.get(k) == Some(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::LoadConfig;
    use crate::stores::VectorStoreConfig;

    /// Deserialize the "every param" fixture at the repo root into a
    /// `LoadConfig`.
    fn load_fixture() -> LoadConfig {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/configs/opensearch_all_params.yaml"
        );
        let yaml =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"));
        serde_yaml::from_str(&yaml).expect("fixture should deserialize into LoadConfig")
    }

    fn opensearch_store(cfg: &LoadConfig) -> &OpenSearchConfig {
        match &cfg.vectorstore {
            VectorStoreConfig::OpenSearch(store) => store.as_ref(),
            _ => panic!("test fixture must be an opensearch vectorstore"),
        }
    }

    fn dense_spec(distance: Option<&str>, datatype: Option<&str>) -> VectorSpec {
        VectorSpec {
            kind: VectorKind::Dense,
            column: "c".into(),
            size: Some(8),
            distance: distance.map(str::to_string),
            comparator: None,
            datatype: datatype.map(str::to_string),
            on_disk: None,
            modifier: None,
        }
    }

    fn one_vector(spec: VectorSpec) -> HashMap<String, VectorSpec> {
        HashMap::from([("dense".to_string(), spec)])
    }

    /// Every index param we support must round-trip from YAML into the
    /// create-index request. When you add a knob to `VectorSpec` or
    /// `OpenSearchParams`, add it to `tests/configs/opensearch_all_params.yaml`
    /// and assert it here — that pairing is what keeps "all params available"
    /// honest. (Mirrors the qdrant backend's own coverage test.)
    #[test]
    fn all_index_params_flow_through() {
        let cfg = load_fixture();
        let store = opensearch_store(&cfg);

        // Store-level settings (not part of the create-index request).
        assert_eq!(store.url, "https://localhost:9200");
        assert_eq!(store.username.as_deref(), Some("admin"));
        assert_eq!(store.password.as_deref(), Some("secret"));
        assert_eq!(store.index_name, "everything");
        assert!(store.tls_insecure);
        assert_eq!(store.timeout_s, 240);

        let params = store.params.as_ref().expect("params present");
        assert!(params.recreate);

        // Sizes are explicit in the fixture, so no inference is needed.
        let body = build_create_index(&cfg.vectors, params, &HashMap::new())
            .expect("build should succeed");

        // Index-wide settings.
        let index = &body["settings"]["index"];
        assert_eq!(index["knn"], json!(true));
        assert_eq!(index["number_of_shards"], json!(3));
        assert_eq!(index["number_of_replicas"], json!(1));
        assert_eq!(index["refresh_interval"], json!("30s"));
        assert_eq!(index["knn.algo_param.ef_search"], json!(200));

        // The dense field mapping.
        let field = &body["mappings"]["properties"]["dense"];
        assert_eq!(field["type"], json!("knn_vector"));
        assert_eq!(field["dimension"], json!(384));
        assert_eq!(field["space_type"], json!("cosinesimil"));
        assert_eq!(field["data_type"], json!("float"));
        assert_eq!(field["mode"], json!("on_disk"));
        assert_eq!(field["compression_level"], json!("16x"));

        // The method, with its flat knobs re-nested under `parameters`.
        let method = &field["method"];
        assert_eq!(method["name"], json!("hnsw"));
        assert_eq!(method["engine"], json!("faiss"));
        assert_eq!(method["parameters"]["m"], json!(16));
        assert_eq!(method["parameters"]["ef_construction"], json!(256));
        assert_eq!(method["parameters"]["ef_search"], json!(128));
        assert_eq!(method["parameters"]["encoder"]["name"], json!("sq"));
        assert_eq!(
            method["parameters"]["encoder"]["parameters"]["type"],
            json!("fp16")
        );
        assert_eq!(
            method["parameters"]["encoder"]["parameters"]["bits"],
            json!(16)
        );
        assert_eq!(
            method["parameters"]["encoder"]["parameters"]["clip"],
            json!(true)
        );
    }

    /// `index.knn` is a STATIC setting — an index created without it can never
    /// do approximate search — so it must be present even for a bare config.
    #[test]
    fn knn_is_always_enabled_and_space_type_always_written() {
        let body = build_create_index(
            &one_vector(dense_spec(None, None)),
            &OpenSearchParams::default(),
            &HashMap::new(),
        )
        .expect("build should succeed");

        assert_eq!(body["settings"]["index"]["knn"], json!(true));
        // An unset distance means cosine, NOT OpenSearch's own `l2` default.
        assert_eq!(
            body["mappings"]["properties"]["dense"]["space_type"],
            json!("cosinesimil")
        );
        // Nothing else is invented: no method, no mode, no data_type.
        assert!(body["mappings"]["properties"]["dense"]["method"].is_null());
        assert!(body["mappings"]["properties"]["dense"]["mode"].is_null());
        assert!(body["mappings"]["properties"]["dense"]["data_type"].is_null());
    }

    /// The four nova distances all have an OpenSearch space, including
    /// `manhattan` → `l1`, and the aliases the qdrant backend accepts work here
    /// too.
    #[test]
    fn distances_map_onto_space_types() {
        for (distance, space) in [
            (None, "cosinesimil"),
            (Some("cosine"), "cosinesimil"),
            (Some("dot"), "innerproduct"),
            (Some("euclid"), "l2"),
            (Some("euclidean"), "l2"),
            (Some("l2"), "l2"),
            (Some("manhattan"), "l1"),
            (Some("l1"), "l1"),
            (Some("COSINE"), "cosinesimil"),
        ] {
            assert_eq!(
                parse_space_type("dense", distance).expect("known distance"),
                space,
                "{distance:?}"
            );
        }
        assert!(matches!(
            parse_space_type("dense", Some("hamming-ish")),
            Err(OpenSearchConfigError::UnknownDistance { .. })
        ));
    }

    /// float16 and uint8 are rejected with pointed errors rather than mapped
    /// onto a near-miss — OpenSearch has no 16-bit element type, and its `byte`
    /// is signed.
    #[test]
    fn lossy_datatypes_are_rejected_with_guidance() {
        let err = parse_data_type("dense", Some("float16")).unwrap_err();
        assert!(matches!(err, OpenSearchConfigError::Float16Datatype { .. }));
        assert!(err.to_string().contains("sq"), "{err}");

        let err = parse_data_type("dense", Some("uint8")).unwrap_err();
        assert!(matches!(err, OpenSearchConfigError::Uint8Datatype { .. }));
        assert!(err.to_string().contains("SIGNED"), "{err}");

        assert_eq!(parse_data_type("dense", Some("float32")).unwrap(), Some("float"));
        assert_eq!(parse_data_type("dense", Some("byte")).unwrap(), Some("byte"));
        assert_eq!(parse_data_type("dense", None).unwrap(), None);
    }

    /// A per-vector `on_disk` selects the workload mode, and an explicit
    /// index-wide `mode` overrides it.
    #[test]
    fn on_disk_selects_mode_and_explicit_mode_wins() {
        let mut spec = dense_spec(None, None);
        spec.on_disk = Some(true);
        let body = build_create_index(
            &one_vector(spec),
            &OpenSearchParams::default(),
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(body["mappings"]["properties"]["dense"]["mode"], json!("on_disk"));

        let mut spec = dense_spec(None, None);
        spec.on_disk = Some(true);
        let params = OpenSearchParams {
            mode: Some("in_memory".into()),
            ..Default::default()
        };
        let body = build_create_index(&one_vector(spec), &params, &HashMap::new()).unwrap();
        assert_eq!(
            body["mappings"]["properties"]["dense"]["mode"],
            json!("in_memory")
        );
    }

    /// Sparse and multivector are refused at mapping-build time with a clear
    /// message, not silently dropped.
    #[test]
    fn non_dense_vectors_are_rejected() {
        for kind in [VectorKind::Sparse, VectorKind::Multivector] {
            let mut spec = dense_spec(None, None);
            spec.kind = kind;
            let err = build_create_index(
                &one_vector(spec),
                &OpenSearchParams::default(),
                &HashMap::new(),
            )
            .expect_err("non-dense must be rejected");
            assert!(matches!(
                err,
                OpenSearchConfigError::UnsupportedVectorKind { .. }
            ));
        }
    }

    /// A dense vector with no `size:` and nothing inferable from the parquet
    /// schema is an error, not a zero-dimension field.
    #[test]
    fn missing_dimension_is_an_error() {
        let mut spec = dense_spec(None, None);
        spec.size = None;
        let err = build_create_index(
            &one_vector(spec),
            &OpenSearchParams::default(),
            &HashMap::new(),
        )
        .expect_err("missing size must be rejected");
        assert!(matches!(err, OpenSearchConfigError::MissingSize(_)));
    }

    /// `build_update_settings` carries only the DYNAMIC knobs — never the
    /// method, mode, compression_level or shard count, which OpenSearch cannot
    /// change in place.
    #[test]
    fn update_settings_carries_only_dynamic_knobs() {
        let params = OpenSearchParams {
            shards: Some(9),
            replicas: Some(2),
            refresh_interval: Some("5s".into()),
            ef_search: Some(512),
            method: Some(MethodConfig {
                m: Some(64),
                ..Default::default()
            }),
            mode: Some("on_disk".into()),
            compression_level: Some("32x".into()),
            recreate: true,
        };
        let body = build_update_settings(&params).expect("something dynamic is set");
        let index = &body["index"];
        assert_eq!(index["number_of_replicas"], json!(2));
        assert_eq!(index["refresh_interval"], json!("5s"));
        assert_eq!(index["knn.algo_param.ef_search"], json!(512));
        assert!(index["number_of_shards"].is_null());
        assert!(index["method"].is_null());
        assert!(index["mode"].is_null());
        assert!(index["compression_level"].is_null());

        // Nothing dynamic configured → no request at all, rather than an empty
        // settings PUT.
        assert!(build_update_settings(&OpenSearchParams::default()).is_none());
    }

    /// OpenSearch returns index settings with inconsistent grouping — a real
    /// 3.8 response has `"knn": "true"` and `"knn.algo_param": {"ef_search":
    /// "200"}` side by side — so the lookup flattens to dotted paths rather than
    /// walking nested keys. (A nested walk read `knn` as the string `"true"` and
    /// reported ef_search as missing; caught by the post-load sanity check on
    /// the first live run against 3.8.)
    #[test]
    fn settings_lookup_handles_opensearchs_mixed_key_grouping() {
        // Verbatim from `GET /nova_smoke/_settings` on OpenSearch 3.8.0.
        let live = json!({
            "replication": { "type": "DOCUMENT" },
            "knn.advanced": { "approximate_threshold": "0" },
            "number_of_shards": "1",
            "knn.algo_param": { "ef_search": "200" },
            "provided_name": "nova_smoke",
            "knn": "true",
            "number_of_replicas": "0",
        });
        assert_eq!(
            lookup_setting(&live, "knn.algo_param.ef_search"),
            json!("200")
        );
        assert_eq!(
            lookup_setting(&live, "knn.advanced.approximate_threshold"),
            json!("0")
        );
        assert_eq!(lookup_setting(&live, "number_of_replicas"), json!("0"));
        assert_eq!(lookup_setting(&live, "knn"), json!("true"));
        assert_eq!(lookup_setting(&live, "nope.not.here"), Value::Null);

        // The fully-nested grouping must read identically, so this does not
        // depend on which shape a given server version happens to emit.
        let nested = json!({ "knn": { "algo_param": { "ef_search": "200" } } });
        assert_eq!(
            lookup_setting(&nested, "knn.algo_param.ef_search"),
            json!("200")
        );
    }

    /// Settings come back as STRINGS, so the coercion must accept both — and
    /// must not read a missing or non-numeric setting as a value.
    #[test]
    fn setting_comparison_accepts_string_or_number() {
        assert_eq!(setting_as_u64(&json!("200")), Some(200));
        assert_eq!(setting_as_u64(&json!(200)), Some(200));
        assert_eq!(setting_as_u64(&Value::Null), None);
        assert_eq!(setting_as_u64(&json!("30s")), None);
        assert_eq!(setting_as_u64(&json!("true")), None);
    }

    /// The reindex immutability guard is a SUBSET check on `parameters`: the
    /// server materializes defaults for knobs the config never named, and those
    /// must not read as a mismatch.
    #[test]
    fn method_matches_ignores_server_filled_defaults() {
        let want = method_json(&MethodConfig {
            m: Some(16),
            ..Default::default()
        })
        .unwrap();
        let live = json!({
            "name": "hnsw",
            "engine": "faiss",
            "space_type": "cosinesimil",
            "parameters": { "m": 16, "ef_construction": 100, "encoder": { "name": "flat" } },
        });
        assert!(method_matches(&want, &live));

        // A genuine difference is caught.
        let live_m32 = json!({
            "name": "hnsw",
            "engine": "faiss",
            "parameters": { "m": 32 },
        });
        assert!(!method_matches(&want, &live_m32));

        // So is a different engine, and a missing mapping.
        let live_lucene = json!({
            "name": "hnsw",
            "engine": "lucene",
            "parameters": { "m": 16 },
        });
        assert!(!method_matches(&want, &live_lucene));
        assert!(!method_matches(&want, &Value::Null));
    }

    /// Dense vectors must reach the bulk body as f32, not as promoted f64s.
    ///
    /// Folding them into the payload `Value` (what `json!` did) re-emits every
    /// component as the shortest form of `f as f64` — ~2x the bytes per bulk
    /// request and a longer parse server-side, for an identical indexed vector.
    /// See `BulkDoc`.
    #[test]
    fn bulk_documents_serialize_vectors_as_f32_not_promoted_f64() {
        let mut payload = serde_json::Map::new();
        payload.insert("label".to_string(), json!("a"));
        let doc = BulkDoc {
            payload,
            vectors: vec![("dense".to_string(), vec![0.1f32, 0.2, -0.3])],
        };

        let body = serde_json::to_string(&doc).expect("bulk doc serializes");
        assert_eq!(body, r#"{"label":"a","dense":[0.1,0.2,-0.3]}"#);

        // The premise, pinned: this is what the `json!`/`Value` route produced,
        // and what a regression here would silently go back to.
        assert_eq!(
            json!(vec![0.1f32, 0.2, -0.3]).to_string(),
            "[0.10000000149011612,0.20000000298023224,-0.30000001192092896]"
        );
    }

}
