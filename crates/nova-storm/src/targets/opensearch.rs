//! OpenSearch implementation of [`QueryTarget`] (feature `opensearch`).
//!
//! Approximate kNN via the `knn` query clause; a batch of N queries is one
//! `_msearch` round-trip (the client has no batched-kNN endpoint). The metric is
//! fixed at index time by the field's `space_type`, so nothing metric-related is
//! set per query — only the search-breadth knobs, which live under
//! `method_parameters` and `rescore`. Filters are not supported yet (rejected at
//! construction). Recall uses the returned `_id`s.
//!
//! Modelled on the qdrant target: search params are parsed into a typed struct
//! and baked into the target once at construction, scores are parsed alongside
//! ids so tie-aware recall works, and client deadlines are classified apart from
//! real errors.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use opensearch::auth::Credentials;
use opensearch::cert::CertificateValidation;
use opensearch::http::request::JsonBody;
use opensearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
use opensearch::indices::IndicesGetMappingParts;
use opensearch::{MsearchParts, OpenSearch};
use serde::Deserialize;
use serde_json::{Value, json};
use url::Url;

use super::{BatchOutcome, QueryTarget, ScoringProfile};
use crate::config::{QueryConfig, WithPayload};
use crate::errors::TargetError;
use crate::queries::QueryVector;

/// Connection + target settings for an OpenSearch backend (`type: opensearch`).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenSearchConfig {
    /// Node URL, e.g. `https://localhost:9200`.
    pub url: String,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub password: Option<String>,
    /// Bearer token (JWT), an alternative to username/password. OpenSearch has
    /// no Elasticsearch-style encoded API key, so there is no `api_key` here.
    #[serde(default)]
    pub bearer_token: Option<String>,
    #[serde(default = "default_index")]
    pub index_name: String,
    /// Skip TLS cert validation (a default OpenSearch node serves a self-signed
    /// cert). DEV ONLY.
    #[serde(default)]
    pub tls_insecure: bool,
    /// Per-request timeout in seconds. Same generous-default reasoning as the
    /// qdrant target's `timeout_s`: a load test's whole job is to push a cluster
    /// into multi-second latencies, and a tight timeout turns honest slow
    /// dispatches into errors, corrupting the error counts. Unset = 300s.
    #[serde(default = "default_timeout_s")]
    pub timeout_s: u64,
}

fn default_timeout_s() -> u64 {
    300
}

fn default_index() -> String {
    "default".to_string()
}

/// Manual `Debug` so secrets never reach logs.
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
            .finish()
    }
}

/// OpenSearch search-time tuning (`query.search_params` for an `opensearch`
/// target). All fields optional; the server applies its own defaults for
/// anything left unset. `deny_unknown_fields` rejects a key that isn't an
/// OpenSearch search param — so a qdrant/milvus/elastic knob under an
/// `opensearch` target is caught at startup.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenSearchSearchParams {
    /// HNSW candidate-list size at query time — OpenSearch's analog of qdrant's
    /// `hnsw_ef`. Overrides the index's own `knn.algo_param.ef_search` on the
    /// faiss and nmslib engines. NOTE: the lucene engine ignores it and uses the
    /// larger of `k` and `ef_search`.
    #[serde(default)]
    pub ef_search: Option<u64>,
    /// IVF buckets examined per query. Only meaningful on an `ivf` index.
    #[serde(default)]
    pub nprobes: Option<u64>,
    /// Rescoring of quantized candidates against full-precision vectors — the
    /// direct analog of qdrant's `quantization: {rescore, oversampling}`.
    #[serde(default)]
    pub rescore: Option<RescoreConfig>,
}

/// Rescoring behavior at query time (`search_params.rescore`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RescoreConfig {
    /// Turn rescoring on/off outright. Defaults off for `in_memory` fields and
    /// on (with a compression-derived factor) for `on_disk` ones.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Candidates retrieved before ranking, as a multiple of `k`. Valid range
    /// `[1.0, 100.0]`. Qdrant spells this `oversampling`.
    #[serde(default)]
    pub oversample_factor: Option<f64>,
}

/// The `rescore` value to put in a query body, or `None` to omit the key
/// entirely and let the field's mode decide.
///
/// OpenSearch accepts either a bare boolean or an object, so `enabled: false`
/// has to serialize as `false` rather than `{"enabled": false}` — which the
/// server would not understand.
fn rescore_json(rescore: &RescoreConfig) -> Option<Value> {
    match (rescore.enabled, rescore.oversample_factor) {
        (Some(false), _) => Some(json!(false)),
        (_, Some(factor)) => Some(json!({ "oversample_factor": factor })),
        (Some(true), None) => Some(json!(true)),
        (None, None) => None,
    }
}

pub struct OpenSearchTarget {
    client: OpenSearch,
    index_name: String,
    vector_field: String,
    top_k: u64,
    /// `method_parameters` for every query, built once at construction — the
    /// same "bake the knobs in once" pattern as `top_k`/`vector_field`. `None`
    /// when nothing is configured, so the key is omitted and the index's own
    /// `knn.algo_param.ef_search` applies.
    method_parameters: Option<Value>,
    /// The `rescore` clause, built once. `None` omits the key.
    rescore: Option<Value>,
    with_payload: WithPayload,
    collect_ids: bool,
    /// Whether to materialize per-point scores. Atomic because the runner can
    /// turn it off after `scoring_profile()` if the run turns out not to compare
    /// scores — score collection happens inside the measured latency window.
    collect_scores: AtomicBool,
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

fn to_other<E: std::error::Error>(e: E) -> TargetError {
    TargetError::Other(flatten(&e))
}

impl OpenSearchConfig {
    pub async fn into_target(self, query: &QueryConfig) -> Result<OpenSearchTarget, TargetError> {
        if query.filter.is_some() {
            // OpenSearch DOES support a `filter` inside the knn clause on the
            // faiss and lucene engines; translating nova's backend-agnostic
            // `Filter` into one is a follow-up, not something to half-do here.
            return Err(TargetError::Other(
                "filters are not yet supported for the opensearch target".to_string(),
            ));
        }
        // OpenSearch always searches a named `knn_vector` field; there is no
        // unnamed default like qdrant, so `vector_name` (the field) is required.
        let vector_field = query.vector_name.clone().ok_or_else(|| {
            TargetError::Other(
                "opensearch target requires `query.vector_name` (the knn_vector field to search)"
                    .to_string(),
            )
        })?;

        let sp: OpenSearchSearchParams = query
            .search_params
            .as_ref()
            .map(|v| serde_yaml::from_value(v.clone()))
            .transpose()
            .map_err(|e| TargetError::Other(format!("opensearch search_params: {e}")))?
            .unwrap_or_default();

        // OpenSearch requires ef_search >= k on the faiss/nmslib engines —
        // validate at startup for a clear error rather than failing every
        // dispatch for the run's whole duration.
        if let Some(ef) = sp.ef_search
            && ef < query.top_k
        {
            return Err(TargetError::Other(format!(
                "opensearch ef_search ({ef}) must be >= top_k ({})",
                query.top_k
            )));
        }
        if let Some(factor) = sp.rescore.as_ref().and_then(|r| r.oversample_factor)
            && !(1.0..=100.0).contains(&factor)
        {
            return Err(TargetError::Other(format!(
                "opensearch rescore.oversample_factor ({factor}) must be in [1.0, 100.0]"
            )));
        }

        let mut method_parameters = serde_json::Map::new();
        if let Some(ef) = sp.ef_search {
            method_parameters.insert("ef_search".into(), json!(ef));
        }
        if let Some(n) = sp.nprobes {
            method_parameters.insert("nprobes".into(), json!(n));
        }

        let creds = if let Some(token) = self.bearer_token {
            Some(Credentials::Bearer(token))
        } else if let (Some(u), Some(p)) = (self.username, self.password) {
            Some(Credentials::Basic(u, p))
        } else {
            None
        };
        let pool = SingleNodeConnectionPool::new(Url::parse(&self.url).map_err(to_other)?);
        let mut builder =
            TransportBuilder::new(pool).timeout(Duration::from_secs(self.timeout_s));
        if let Some(creds) = creds {
            builder = builder.auth(creds);
        }
        if self.tls_insecure {
            builder = builder.cert_validation(CertificateValidation::None);
        }
        let transport = builder.build().map_err(to_other)?;

        Ok(OpenSearchTarget {
            client: OpenSearch::new(transport),
            index_name: self.index_name,
            vector_field,
            top_k: query.top_k,
            method_parameters: (!method_parameters.is_empty())
                .then(|| Value::Object(method_parameters)),
            rescore: sp.rescore.as_ref().and_then(rescore_json),
            with_payload: query.with_payload.clone(),
            collect_ids: query.source.ground_truth_column.is_some(),
            collect_scores: AtomicBool::new(query.source.ground_truth_score_column.is_some()),
        })
    }
}

impl fmt::Display for OpenSearchTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "opensearch({})", self.index_name)
    }
}

/// An OpenSearch `space_type` → the distance vocabulary [`ScoringProfile`] uses.
/// `None` for anything this build does not recognize: an unknown space must NOT
/// read as "larger is better", since a wrong orientation turns every honest miss
/// into a `missing_from_gt` alarm.
///
/// Every mapped space scores higher-is-better on OpenSearch (they are all
/// monotone transforms of distance — `1/(1+d)` and friends), which is the
/// orientation the recall code assumes for these names.
fn distance_for_space(space: &str) -> Option<&'static str> {
    match space {
        "cosinesimil" => Some("cosine"),
        "innerproduct" => Some("dot"),
        "l2" => Some("euclid"),
        "l1" => Some("manhattan"),
        _ => None,
    }
}

/// An OpenSearch `data_type` → the datatype vocabulary [`ScoringProfile`] uses,
/// which drives the default score tie tolerance.
fn datatype_for_data_type(data_type: Option<&str>) -> &'static str {
    match data_type {
        // Absent is the common case and means 32-bit floats, OpenSearch's own
        // default element type.
        None | Some("float") => "float32",
        Some("byte") => "uint8",
        // A value this build cannot parse takes the conservative tolerance
        // rather than the tight float32 one — see the qdrant target.
        Some(_) => "unknown",
    }
}

/// A whole-batch failure outcome (no per-query ids).
fn fail(started: Instant, n: usize, error: String, timed_out: bool) -> BatchOutcome {
    BatchOutcome {
        latency: started.elapsed(),
        ok: false,
        ids: vec![None; n],
        scores: vec![None; n],
        error: Some(error),
        timed_out,
    }
}

#[async_trait]
impl QueryTarget for OpenSearchTarget {
    fn disable_score_collection(&self) {
        self.collect_scores.store(false, Ordering::Relaxed);
    }

    async fn scoring_profile(&self) -> ScoringProfile {
        // One request for all three answers, so a single failure cannot leave
        // them inconsistent. Best-effort throughout: anything undeterminable
        // stays `None`/`false`, and the caller treats an unknown distance as
        // "don't compare scores" rather than picking a side.
        let mut out = ScoringProfile::default();
        let index = self.index_name.as_str();
        let Ok(resp) = self
            .client
            .indices()
            .get_mapping(IndicesGetMappingParts::Index(&[index]))
            .send()
            .await
        else {
            return out;
        };
        if !resp.status_code().is_success() {
            return out;
        }
        let Ok(body) = resp.json::<Value>().await else {
            return out;
        };
        let field = &body[index]["mappings"]["properties"][&self.vector_field];
        if field.is_null() {
            return out;
        }

        // `space_type` may sit at the field's top level or inside `method`,
        // depending on how the index was created — accept either.
        out.distance = field["space_type"]
            .as_str()
            .or_else(|| field["method"]["space_type"].as_str())
            .and_then(distance_for_space)
            .map(str::to_string);
        out.datatype = Some(datatype_for_data_type(field["data_type"].as_str()).to_string());
        // Quantization here means a non-`flat` faiss encoder (`sq`/`pq`), or
        // the `mode`/`compression_level` shorthand that selects one.
        let encoder = field["method"]["parameters"]["encoder"]["name"].as_str();
        out.quantized = matches!(encoder, Some("sq") | Some("pq"))
            || field["compression_level"]
                .as_str()
                .is_some_and(|c| c != "1x");
        out
    }

    async fn query_batch(&self, queries: &[&QueryVector]) -> BatchOutcome {
        let started = Instant::now();
        if queries.is_empty() {
            return BatchOutcome {
                latency: started.elapsed(),
                ok: true,
                ids: Vec::new(),
                scores: Vec::new(),
                error: None,
                timed_out: false,
            };
        }

        // What to return in `_source`. OpenSearch 2.17+ keeps vectors out of
        // `_source` by default (`index.knn.derived_source.enabled`), but an
        // index created without that — or an older cluster — would refetch and
        // re-serialize the vector on every hit and inflate the measured cost,
        // so the vector field is excluded explicitly. That also matches qdrant's
        // `with_payload`, which returns payload and never the vector.
        let source = match &self.with_payload {
            WithPayload::Enable(true) => json!({ "excludes": [self.vector_field.as_str()] }),
            WithPayload::Enable(false) => json!(false),
            // Field include-list (the RAG shape): return exactly these fields;
            // the vector is excluded implicitly by not being listed.
            WithPayload::Fields(fields) => json!({ "includes": fields }),
        };

        // `_msearch` body: one header line + one knn search-body line per query.
        let mut body: Vec<JsonBody<Value>> = Vec::with_capacity(queries.len() * 2);
        for q in queries {
            // Dense-only target: guarded at the point of use, so no separate
            // check can drift out of sync — a sparse query is a per-dispatch
            // data error, never a panic.
            let Some(dense) = q.vector.as_dense() else {
                return fail(
                    started,
                    queries.len(),
                    "the opensearch target does not support sparse queries".to_string(),
                    false,
                );
            };
            body.push(json!({ "index": self.index_name }).into());
            let mut knn = json!({
                "vector": dense,
                "k": self.top_k,
            });
            if let Some(params) = &self.method_parameters {
                knn["method_parameters"] = params.clone();
            }
            if let Some(rescore) = &self.rescore {
                knn["rescore"] = rescore.clone();
            }
            body.push(
                json!({
                    "query": { "knn": { self.vector_field.as_str(): knn } },
                    "_source": source,
                    "size": self.top_k,
                })
                .into(),
            );
        }

        let resp = match self
            .client
            .msearch(MsearchParts::None)
            .body(body)
            .send()
            .await
        {
            Ok(r) => r,
            // A CLIENT-side deadline ("the query was too slow for `timeout_s`")
            // is a saturation finding, not a broken target — kept apart from
            // other errors all the way into the summary, same as the qdrant
            // target does with gRPC CANCELLED/DEADLINE_EXCEEDED.
            Err(e) => {
                let timed_out = e.is_timeout();
                return fail(started, queries.len(), flatten(&e), timed_out);
            }
        };
        // Check the HTTP status before parsing JSON — a proxy 502 / auth 401 may
        // return non-JSON (HTML) that would otherwise surface as an opaque parse
        // error instead of the real status.
        let status = resp.status_code();
        if !status.is_success() {
            let detail = resp.text().await.unwrap_or_default();
            return fail(
                started,
                queries.len(),
                format!("msearch HTTP {status}: {detail}"),
                false,
            );
        }
        let val: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return fail(started, queries.len(), flatten(&e), false),
        };

        let Some(responses) = val["responses"].as_array() else {
            return fail(
                started,
                queries.len(),
                format!("msearch: no `responses` array: {val}"),
                false,
            );
        };
        // A count mismatch means responses can't be zipped positionally against
        // the submitted queries without risking one query's recall being scored
        // against another's results — treat as failure, as the qdrant target
        // does.
        if responses.len() != queries.len() {
            return fail(
                started,
                queries.len(),
                format!(
                    "msearch returned {} responses for {} queries",
                    responses.len(),
                    queries.len()
                ),
                false,
            );
        }

        let collect_scores = self.collect_scores.load(Ordering::Relaxed);
        let mut ids = Vec::with_capacity(queries.len());
        let mut scores = Vec::with_capacity(queries.len());
        for (i, r) in responses.iter().enumerate() {
            // Any of these means this query's hits are incomplete or wrong —
            // fail the whole batch rather than score recall against a partial or
            // malformed result (a silently understated recall would look like an
            // engine problem, not the infra failure it is).
            if let Some(err) = r.get("error").filter(|e| !e.is_null()) {
                return fail(
                    started,
                    queries.len(),
                    format!("msearch item {i} error: {err}"),
                    false,
                );
            }
            // A SERVER-side timeout, distinct from the client deadline above but
            // the same kind of finding.
            if r["timed_out"].as_bool().unwrap_or(false) {
                return fail(
                    started,
                    queries.len(),
                    format!("msearch item {i} timed out: {r}"),
                    true,
                );
            }
            let failed_shards = r["_shards"]["failed"].as_u64().unwrap_or(0);
            if failed_shards > 0 {
                return fail(
                    started,
                    queries.len(),
                    format!(
                        "msearch item {i}: {failed_shards} failed shard(s): {}",
                        r["_shards"]
                    ),
                    false,
                );
            }
            let Some(hits) = r["hits"]["hits"].as_array() else {
                return fail(
                    started,
                    queries.len(),
                    format!("msearch item {i}: no `hits.hits` array: {r}"),
                    false,
                );
            };

            if !self.collect_ids {
                ids.push(None);
                scores.push(None);
                continue;
            }
            let mut query_ids = Vec::with_capacity(hits.len());
            // Collected alongside the ids, from the SAME hit, so the two stay
            // positionally aligned by construction.
            let mut query_scores = if collect_scores {
                Vec::with_capacity(hits.len())
            } else {
                Vec::new()
            };
            for h in hits {
                let Some(id) = h["_id"].as_str() else {
                    return fail(
                        started,
                        queries.len(),
                        format!("msearch item {i}: a hit has no string `_id`: {h}"),
                        false,
                    );
                };
                query_ids.push(id.to_string());
                if collect_scores {
                    let Some(score) = h["_score"].as_f64() else {
                        return fail(
                            started,
                            queries.len(),
                            format!("msearch item {i}: a hit has no numeric `_score`: {h}"),
                            false,
                        );
                    };
                    query_scores.push(score as f32);
                }
            }
            ids.push(Some(query_ids));
            scores.push(collect_scores.then_some(query_scores));
        }

        BatchOutcome {
            latency: started.elapsed(),
            ok: true,
            ids,
            scores,
            error: None,
            timed_out: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::StormConfig;
    use crate::targets::TargetConfig;

    /// A storm config with an `opensearch` target, plus whatever extra `query:`
    /// keys the test needs.
    fn cfg(extra_query: &str) -> StormConfig {
        let yaml = format!(
            "
target:
  type: opensearch
  url: http://localhost:9200
  index_name: fiqa
query:
  vector_name: dense
  top_k: 10
{extra_query}
  source:
    uri: /tmp/q.parquet
    column: dense_embedding
load:
  concurrency: 1
  duration_s: 1
"
        );
        StormConfig::from_yaml(&yaml).expect("config should parse")
    }

    /// `Result::expect_err` needs `Debug` on the Ok type; `OpenSearchTarget`
    /// deliberately has none (it holds a live client), so unwrap the error side
    /// by hand.
    fn err(result: Result<OpenSearchTarget, TargetError>) -> TargetError {
        match result {
            Ok(_) => panic!("expected the target build to fail"),
            Err(e) => e,
        }
    }

    fn opensearch_config(target: TargetConfig) -> OpenSearchConfig {
        match target {
            TargetConfig::OpenSearch(c) => *c,
            _ => panic!("expected an opensearch target"),
        }
    }

    /// The `type: opensearch` tag dispatches to this backend, and the target
    /// builds from a minimal config.
    #[tokio::test]
    async fn builds_target() {
        let cfg = cfg("");
        let target = opensearch_config(cfg.target)
            .into_target(&cfg.query)
            .await
            .expect("target should build");
        assert_eq!(target.index_name, "fiqa");
        assert_eq!(target.vector_field, "dense");
        assert_eq!(target.top_k, 10);
        // Nothing configured → no `method_parameters`, no `rescore`, so the
        // index's own defaults apply rather than anything invented here.
        assert!(target.method_parameters.is_none());
        assert!(target.rescore.is_none());
        assert_eq!(target.to_string(), "opensearch(fiqa)");
    }

    /// Search params parse into `method_parameters` / `rescore` and are baked in
    /// once at construction.
    #[tokio::test]
    async fn search_params_parse_and_convert() {
        let cfg = cfg("  search_params:\n    ef_search: 128\n    nprobes: 32\n    rescore:\n      oversample_factor: 2.0");
        let target = opensearch_config(cfg.target)
            .into_target(&cfg.query)
            .await
            .expect("target should build");
        let params = target.method_parameters.expect("method_parameters set");
        assert_eq!(params["ef_search"], json!(128));
        assert_eq!(params["nprobes"], json!(32));
        assert_eq!(
            target.rescore.expect("rescore set")["oversample_factor"],
            json!(2.0)
        );
    }

    /// A foreign backend's search param is rejected at startup, not silently
    /// ignored — `hnsw_ef` is qdrant's spelling, `num_candidates` is elastic's.
    #[tokio::test]
    async fn foreign_search_params_are_rejected() {
        for foreign in ["  search_params:\n    hnsw_ef: 128", "  search_params:\n    num_candidates: 128"] {
            let cfg = cfg(foreign);
            let err = err(opensearch_config(cfg.target).into_target(&cfg.query).await);
            assert!(err.to_string().contains("search_params"), "{err}");
        }
    }

    /// `ef_search` below `top_k` is rejected at construction rather than
    /// failing every dispatch for the run's whole duration.
    #[tokio::test]
    async fn ef_search_below_top_k_is_rejected() {
        let cfg = cfg("  search_params:\n    ef_search: 5");
        let err = err(opensearch_config(cfg.target).into_target(&cfg.query).await);
        assert!(err.to_string().contains("must be >= top_k"), "{err}");
    }

    /// An out-of-range oversample factor is caught at startup too — OpenSearch
    /// accepts only [1.0, 100.0].
    #[tokio::test]
    async fn oversample_factor_range_is_validated() {
        let cfg = cfg("  search_params:\n    rescore:\n      oversample_factor: 0.5");
        let err = err(opensearch_config(cfg.target).into_target(&cfg.query).await);
        assert!(err.to_string().contains("[1.0, 100.0]"), "{err}");
    }

    /// The target searches a named `knn_vector` field, so `vector_name` is
    /// required — there is no unnamed default like qdrant's.
    #[tokio::test]
    async fn vector_name_is_required() {
        let yaml = "
target:
  type: opensearch
  url: http://localhost:9200
  index_name: fiqa
query:
  top_k: 10
  source:
    uri: /tmp/q.parquet
    column: dense_embedding
";
        let cfg = StormConfig::from_yaml(yaml).expect("config should parse");
        let err = err(opensearch_config(cfg.target).into_target(&cfg.query).await);
        assert!(err.to_string().contains("vector_name"), "{err}");
    }

    /// Filters are rejected at construction, not per dispatch — the per-dispatch
    /// guard would fail in microseconds and spin every worker flat out for the
    /// whole run while still exiting 0.
    #[tokio::test]
    async fn filters_are_rejected() {
        let cfg = cfg("  filter:\n    must:\n      - field: lang\n        match: en");
        let err = err(opensearch_config(cfg.target).into_target(&cfg.query).await);
        assert!(err.to_string().contains("filters are not yet supported"), "{err}");
    }

    /// `rescore` serializes to the shape OpenSearch actually accepts: a bare
    /// boolean when switched off, an object when a factor is given.
    #[test]
    fn rescore_serializes_as_bool_or_object() {
        assert_eq!(
            rescore_json(&RescoreConfig {
                enabled: Some(false),
                oversample_factor: None
            }),
            Some(json!(false))
        );
        // An explicit `false` wins over a factor — turning rescoring off and
        // also asking to oversample is contradictory, and `{enabled: false}` is
        // not a body OpenSearch understands.
        assert_eq!(
            rescore_json(&RescoreConfig {
                enabled: Some(false),
                oversample_factor: Some(2.0)
            }),
            Some(json!(false))
        );
        assert_eq!(
            rescore_json(&RescoreConfig {
                enabled: Some(true),
                oversample_factor: None
            }),
            Some(json!(true))
        );
        assert_eq!(
            rescore_json(&RescoreConfig {
                enabled: None,
                oversample_factor: Some(3.5)
            }),
            Some(json!({ "oversample_factor": 3.5 }))
        );
        // Nothing set → omit the key entirely so the field's mode decides.
        assert_eq!(
            rescore_json(&RescoreConfig {
                enabled: None,
                oversample_factor: None
            }),
            None
        );
    }

    /// Every space type this backend writes maps back into the distance
    /// vocabulary recall uses; an unrecognized one stays `None` rather than
    /// defaulting to "larger is better".
    #[test]
    fn space_types_map_back_to_distances() {
        assert_eq!(distance_for_space("cosinesimil"), Some("cosine"));
        assert_eq!(distance_for_space("innerproduct"), Some("dot"));
        assert_eq!(distance_for_space("l2"), Some("euclid"));
        assert_eq!(distance_for_space("l1"), Some("manhattan"));
        assert_eq!(distance_for_space("hamming"), None);
        assert_eq!(distance_for_space(""), None);
    }

    /// An absent `data_type` means float32 (OpenSearch's default element type),
    /// not "unknown" — the two take different tie tolerances.
    #[test]
    fn data_types_map_back_to_datatypes() {
        assert_eq!(datatype_for_data_type(None), "float32");
        assert_eq!(datatype_for_data_type(Some("float")), "float32");
        assert_eq!(datatype_for_data_type(Some("byte")), "uint8");
        assert_eq!(datatype_for_data_type(Some("binary")), "unknown");
    }

    /// Score collection follows the run's ground-truth-score config, and the
    /// runner can turn it off afterwards.
    #[tokio::test]
    async fn score_collection_follows_config_and_can_be_disabled() {
        let plain = cfg("");
        let target = opensearch_config(plain.target)
            .into_target(&plain.query)
            .await
            .expect("target should build");
        // No ground_truth_column in the fixture → neither ids nor scores.
        assert!(!target.collect_ids);
        assert!(!target.collect_scores.load(Ordering::Relaxed));

        let with_gt = cfg("");
        let mut query = with_gt.query;
        query.source.ground_truth_column = Some("hit_ids".into());
        query.source.ground_truth_score_column = Some("hit_scores".into());
        let target = opensearch_config(with_gt.target)
            .into_target(&query)
            .await
            .expect("target should build");
        assert!(target.collect_ids);
        assert!(target.collect_scores.load(Ordering::Relaxed));
        target.disable_score_collection();
        assert!(!target.collect_scores.load(Ordering::Relaxed));
        // Disabling scores must not disable id collection — recall still needs
        // the ids.
        assert!(target.collect_ids);
    }
}
