// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.

//! Capture RPC to the RL-owned CPU service. No model rendering or TQ dependency.

use std::time::{Duration, Instant};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, value::RawValue};
use std::collections::BTreeMap;

const MAX_BODY_BYTES: usize = 512 * 1024 * 1024;

/// Internal extension set only after a successful authenticated /prepare RPC.
#[derive(Clone)]
pub(crate) struct ManagedCapture;

/// Opaque prefill routes carried alongside the unchanged decode response body.
#[derive(Clone)]
pub(crate) struct PrefillCapture(pub std::sync::Arc<RawValue>);

impl AsRef<[u8]> for PrefillCapture {
    fn as_ref(&self) -> &[u8] {
        self.0.get().as_bytes()
    }
}

pub struct CaptureClient {
    url: String,
    token: String,
    client: reqwest::Client,
}

impl CaptureClient {
    pub fn new(url: String, token: String, timeout: Duration) -> Result<Self, String> {
        let parsed = reqwest::Url::parse(&url).map_err(|e| e.to_string())?;
        if parsed.scheme() != "http"
            || parsed.host_str() != Some("127.0.0.1")
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.query().is_some()
            || parsed.fragment().is_some()
            || parsed.path() != "/"
            || timeout.is_zero()
        {
            return Err("PD capture requires a loopback HTTP URL and a positive timeout".into());
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .timeout(timeout)
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self {
            url: url.trim_end_matches('/').into(),
            token,
            client,
        })
    }

    async fn rpc(
        &self,
        operation: &'static str,
        body: impl Into<reqwest::Body>,
    ) -> Result<Bytes, Response> {
        // Include body transfer, connection waits, service time, and response
        // receipt/validation, not just time to HTTP headers.
        let started = Instant::now();
        let result = self.rpc_inner(operation, body, started).await;
        metrics::histogram!(
            "vllm_router_pd_capture_rpc_duration_seconds",
            "operation" => operation,
            "outcome" => if result.is_ok() { "success" } else { "error" },
        )
        .record(started.elapsed().as_secs_f64());
        result
    }

    async fn rpc_inner(
        &self,
        operation: &str,
        body: impl Into<reqwest::Body>,
        started: Instant,
    ) -> Result<Bytes, Response> {
        let result = self
            .client
            .post(format!("{}/{operation}", self.url))
            .bearer_auth(&self.token)
            .header(header::CONTENT_TYPE, "application/json")
            .body(body)
            .send()
            .await;
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                log_socket_exhaustion(operation, "send", &error);
                tracing::error!(
                    %operation,
                    phase = "send",
                    elapsed_seconds = started.elapsed().as_secs_f64(),
                    error = ?error,
                    is_timeout = error.is_timeout(),
                    is_connect = error.is_connect(),
                    is_body = error.is_body(),
                    is_decode = error.is_decode(),
                    "PD capture RPC failed; will not retry"
                );
                return Err(failure(StatusCode::BAD_GATEWAY, "PD capture RPC failed"));
            }
        };
        if !response.status().is_success() {
            let status = response.status();
            tracing::error!(%operation, %status, "PD capture RPC rejected the call");
            // Validation before dispatch is a request error. A failure after
            // inference/staging is ambiguous and must close managed admission.
            return Err(failure(
                if operation == "prepare" && status == StatusCode::BAD_REQUEST {
                    StatusCode::BAD_REQUEST
                } else {
                    StatusCode::BAD_GATEWAY
                },
                "PD capture RPC rejected the call; see RL service logs",
            ));
        }
        let bytes = response.bytes().await.map_err(|error| {
            log_socket_exhaustion(operation, "response_body", &error);
            // Debug includes reqwest's nested transport source; Display hides it.
            // Do not log headers or the capture payload (tokens and credentials).
            tracing::error!(
                %operation,
                phase = "response_body",
                elapsed_seconds = started.elapsed().as_secs_f64(),
                error = ?error,
                is_timeout = error.is_timeout(),
                is_connect = error.is_connect(),
                is_body = error.is_body(),
                is_decode = error.is_decode(),
                "Invalid PD capture RPC response; will not retry"
            );
            failure(StatusCode::BAD_GATEWAY, "Invalid PD capture RPC response")
        })?;
        // Inspect the envelope, never materialize the token/logprob fields.
        let fields: BTreeMap<String, &RawValue> =
            serde_json::from_slice(&bytes).map_err(|error| {
                tracing::error!(%operation, phase = "response_body", error = ?error,
                "Invalid PD capture RPC response; will not retry");
                failure(StatusCode::BAD_GATEWAY, "Invalid PD capture RPC response")
            })?;
        if operation == "prepare" && fields.contains_key("ng_capture") {
            return Err(failure(
                StatusCode::BAD_GATEWAY,
                "Invalid prepared capture request",
            ));
        }
        if operation == "complete"
            && !fields
                .get("ng_commit_coords")
                .is_some_and(|v| v.get().starts_with('{'))
        {
            return Err(failure(
                StatusCode::BAD_GATEWAY,
                "Capture response lacks commit coordinates",
            ));
        }
        Ok(bytes)
    }

    pub async fn run(&self, request: Request, next: Next, weight_version: u64) -> Response {
        if request.uri().path() != "/v1/chat/completions" {
            return next.run(request).await;
        }
        let (mut parts, body) = request.into_parts();
        let bytes = match to_bytes(body, MAX_BODY_BYTES).await {
            Ok(bytes) => bytes,
            Err(_) => return failure(StatusCode::PAYLOAD_TOO_LARGE, "Cannot read PD request body"),
        };
        let original: BTreeMap<String, &RawValue> = match serde_json::from_slice(&bytes) {
            Ok(body) => body,
            Err(_) => return failure(StatusCode::BAD_REQUEST, "Invalid chat JSON"),
        };
        let Some(admission) = original
            .get("ng_capture")
            .filter(|value| value.get() != "null")
        else {
            if original.contains_key("nemo_rl_capture") {
                return failure(
                    StatusCode::BAD_REQUEST,
                    "nemo_rl_capture is owned by managed capture admission",
                );
            }
            return next
                .run(Request::from_parts(parts, Body::from(bytes)))
                .await;
        };
        if original
            .get("stream")
            .is_some_and(|value| value.get() == "true")
        {
            return failure(
                StatusCode::BAD_REQUEST,
                "Managed PD capture requires stream=false",
            );
        }
        let admission = (*admission).to_owned();
        drop(original);
        let prepared = match self.rpc("prepare", bytes).await {
            Ok(body) => body,
            Err(error) => return error,
        };
        let prepared = match attach_engine_context(&prepared, &admission, weight_version) {
            Ok(body) => body,
            Err(_) => return failure(StatusCode::BAD_GATEWAY, "Invalid prepared capture context"),
        };
        parts.headers.remove(header::CONTENT_LENGTH);
        parts.extensions.insert(ManagedCapture);
        let response = next
            .run(Request::from_parts(parts, Body::from(prepared)))
            .await;
        if !response.status().is_success() {
            // Preserve context-length and other engine errors verbatim. Nothing
            // was staged during prepare; there is no request registry to abort.
            return response;
        }
        let (mut parts, body) = response.into_parts();
        // Stream the engine body into the envelope. Backpressure from the
        // capture worker bounds buffering instead of retaining a full response
        // for each of the potentially thousands of active model calls.
        // Emit authoritative fields last so an invalid engine JSON fragment
        // cannot override the router's admission or policy epoch in the envelope.
        let mut suffix = vec![Ok::<_, std::io::Error>(Bytes::from(format!(
            ",\"admission\":{},\"weight_version\":{}",
            admission.get(),
            weight_version,
        )))];
        if let Some(prefill) = parts.extensions.remove::<PrefillCapture>() {
            suffix.push(Ok(Bytes::from_static(b",\"prefill_capture\":")));
            // Keep the original allocation alive through the upload; R3 routes
            // can be large, so avoid copying them into an envelope string.
            suffix.push(Ok(Bytes::from_owner(prefill)));
        }
        suffix.push(Ok(Bytes::from_static(b"}")));
        let mut received = 0usize;
        let response_stream = body.into_data_stream().map(move |chunk| {
            let chunk = chunk.map_err(std::io::Error::other)?;
            received = received.saturating_add(chunk.len());
            if received > MAX_BODY_BYTES {
                return Err(std::io::Error::other(
                    "PD capture response exceeds body limit",
                ));
            }
            Ok(chunk)
        });
        let envelope = reqwest::Body::wrap_stream(
            futures_util::stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(
                b"{\"response\":",
            ))])
            .chain(response_stream)
            .chain(futures_util::stream::iter(suffix)),
        );
        let completed = match self.rpc("complete", envelope).await {
            Ok(body) => body,
            Err(error) => return error,
        };
        parts.headers.remove(header::CONTENT_LENGTH);
        parts.headers.remove(header::CONTENT_ENCODING);
        parts.headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        Response::from_parts(parts, Body::from(completed))
    }
}

/// Add router-owned context while preserving large prepared fields as raw JSON.
fn attach_engine_context(
    prepared: &[u8],
    admission: &RawValue,
    weight_version: u64,
) -> Result<Vec<u8>, serde_json::Error> {
    let mut fields: BTreeMap<String, &RawValue> = serde_json::from_slice(prepared)?;
    let top_logprobs = fields.get("top_logprobs").map_or("0", |value| value.get());
    let routed_experts = fields
        .get("nemo_rl_pd_capture")
        .map_or("false", |value| value.get());
    let context = RawValue::from_string(format!(
        "{{\"version\":1,\"admission\":{},\"weight_version\":{},\"phase\":\"decode\",\"top_logprobs\":{},\"routed_experts\":{}}}",
        admission.get(), weight_version, top_logprobs, routed_experts,
    ))?;
    fields.insert("nemo_rl_capture".into(), &context);
    serde_json::to_vec(&fields)
}

/// Inspect nested transport errors: reqwest's top-level kind can just be Request.
fn log_socket_exhaustion(operation: &str, phase: &str, error: &(dyn std::error::Error + 'static)) {
    let mut cause = Some(error);
    while let Some(error) = cause {
        if let Some(io_error) = error.downcast_ref::<std::io::Error>() {
            #[cfg(unix)]
            let limit = match io_error.raw_os_error() {
                Some(libc::ENOBUFS) => Some("socket_buffers"),
                Some(libc::EMFILE) => Some("process_file_descriptors"),
                Some(libc::ENFILE) => Some("system_file_descriptors"),
                _ => None,
            };
            #[cfg(not(unix))]
            let limit: Option<&str> = None;
            if let Some(limit) = limit {
                tracing::error!(
                    %operation, %phase, %limit,
                    os_error = io_error.raw_os_error(),
                    detail = %io_error,
                    "PD capture hit a local socket resource limit; check OS socket buffers/file-descriptor limits and capture-worker backlog. Connection pooling remains enabled; uploads are not concurrency-capped"
                );
                return;
            }
        }
        cause = error.source();
    }
}

fn failure(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({"error": {"message": message}}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::post, Router};
    use metrics_exporter_prometheus::PrometheusBuilder;
    use std::{
        convert::Infallible,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };
    use tokio::{net::TcpListener, task::JoinHandle};

    async fn mock_capture(
        status: StatusCode,
        payload: &'static str,
        delay: Duration,
        timeout: Duration,
    ) -> (CaptureClient, Arc<AtomicUsize>, JoinHandle<()>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let handler_calls = calls.clone();
        let app = Router::new().route(
            "/{operation}",
            post(move || {
                handler_calls.fetch_add(1, Ordering::SeqCst);
                async move {
                    // Send headers first, then delay the body: timing only
                    // send().await would miss this part of the round trip.
                    let body = Body::from_stream(futures_util::stream::once(async move {
                        tokio::time::sleep(delay).await;
                        Ok::<_, Infallible>(payload)
                    }));
                    Response::builder().status(status).body(body).unwrap()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let client = CaptureClient::new(format!("http://{addr}"), "test".into(), timeout).unwrap();
        (client, calls, server)
    }

    fn sample(text: &str, suffix: &str, operation: &str, outcome: &str) -> f64 {
        let name = format!("vllm_router_pd_capture_rpc_duration_seconds_{suffix}{{");
        let lines: Vec<_> = text
            .lines()
            .filter(|line| {
                line.starts_with(&name)
                    && line.contains(&format!("operation=\"{operation}\""))
                    && line.contains(&format!("outcome=\"{outcome}\""))
            })
            .collect();
        assert_eq!(lines.len(), 1, "missing or duplicate sample: {text}");
        lines[0].split_whitespace().last().unwrap().parse().unwrap()
    }

    #[tokio::test]
    async fn rpc_timing_includes_response_body_for_both_operations() {
        let recorder = PrometheusBuilder::new()
            .set_buckets(&[0.01, 0.1, 1.0])
            .unwrap()
            .build_recorder();
        // The default test runtime is current-thread, keeping the recorder
        // scoped to this test even while awaiting HTTP responses.
        let _guard = metrics::set_default_local_recorder(&recorder);
        let delay = Duration::from_millis(30);
        let (client, calls, server) = mock_capture(
            StatusCode::OK,
            r#"{"ng_commit_coords":{}}"#,
            delay,
            Duration::from_secs(5),
        )
        .await;
        for operation in ["prepare", "complete"] {
            assert_eq!(
                client.rpc(operation, "{}").await.unwrap(),
                r#"{"ng_commit_coords":{}}"#
            );
        }
        server.abort();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let text = recorder.handle().render();
        for operation in ["prepare", "complete"] {
            assert_eq!(sample(&text, "count", operation, "success"), 1.0);
            assert!(sample(&text, "sum", operation, "success") >= delay.as_secs_f64());
        }
        assert!(!text.contains("outcome=\"error\""));
    }

    #[tokio::test]
    async fn failed_rpc_is_timed_once_without_retrying() {
        for (status, payload, delay, timeout, operation, expected_status) in [
            (
                StatusCode::BAD_REQUEST,
                "{}",
                0,
                5000,
                "prepare",
                StatusCode::BAD_REQUEST,
            ),
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "{}",
                0,
                5000,
                "complete",
                StatusCode::BAD_GATEWAY,
            ),
            (
                StatusCode::OK,
                "not json",
                30,
                5000,
                "complete",
                StatusCode::BAD_GATEWAY,
            ),
            (
                StatusCode::OK,
                "{}",
                1000,
                30,
                "prepare",
                StatusCode::BAD_GATEWAY,
            ),
        ] {
            let recorder = PrometheusBuilder::new()
                .set_buckets(&[0.01, 0.1, 1.0])
                .unwrap()
                .build_recorder();
            let _guard = metrics::set_default_local_recorder(&recorder);
            let (client, calls, server) = mock_capture(
                status,
                payload,
                Duration::from_millis(delay),
                Duration::from_millis(timeout),
            )
            .await;
            assert_eq!(
                client.rpc(operation, "{}").await.unwrap_err().status(),
                expected_status
            );
            server.abort();
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            let text = recorder.handle().render();
            assert_eq!(sample(&text, "count", operation, "error"), 1.0);
            if delay > 0 {
                assert!(sample(&text, "sum", operation, "error") >= 0.03);
            }
            assert!(!text.contains("outcome=\"success\""));
        }
    }

    #[derive(Clone, Default)]
    struct LogWriter(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for LogWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn transport_error_logs_include_sources_and_phase_without_payloads() {
        let logs = LogWriter::default();
        let writer = logs.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_max_level(tracing::Level::ERROR)
            .with_writer(move || writer.clone())
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);

        // Accept the connection, read the HTTP request, then close it without
        // responding: reqwest's Display hides the underlying hyper error.
        use tokio::io::AsyncReadExt;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = [0; 4096];
            assert!(stream.read(&mut bytes).await.unwrap() > 0);
        });
        let client = CaptureClient::new(
            format!("http://{addr}"),
            "private-token-marker".into(),
            Duration::from_secs(2),
        )
        .unwrap();
        let failure = client
            .rpc("complete", r#"{"secret":"private-body-marker"}"#)
            .await
            .unwrap_err();
        assert_eq!(failure.status(), StatusCode::BAD_GATEWAY);
        server.await.unwrap();
        let text = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
        assert!(text.contains("operation=complete"), "{text}");
        assert!(text.contains("phase=\"send\""), "{text}");
        assert!(
            text.contains("source:") && text.contains("hyper::Error"),
            "{text}"
        );
        assert!(text.contains("elapsed_seconds="), "{text}");
        assert!(text.contains("is_timeout=false"), "{text}");
        assert!(!text.contains("private-token-marker") && !text.contains("private-body-marker"));

        logs.0.lock().unwrap().clear();
        let (client, calls, server) = mock_capture(
            StatusCode::OK,
            "{}",
            Duration::from_secs(1),
            Duration::from_millis(30),
        )
        .await;
        assert_eq!(
            client.rpc("prepare", "{}").await.unwrap_err().status(),
            StatusCode::BAD_GATEWAY
        );
        server.abort();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let text = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
        assert!(text.contains("phase=\"response_body\""), "{text}");
        assert!(
            text.contains("is_timeout=true") && text.contains("source:"),
            "{text}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn socket_limit_diagnostic_identifies_nested_os_errors() {
        let logs = LogWriter::default();
        let writer = logs.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_max_level(tracing::Level::ERROR)
            .with_writer(move || writer.clone())
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);
        for (errno, limit) in [
            (libc::ENOBUFS, Some("socket_buffers")),
            (libc::EMFILE, Some("process_file_descriptors")),
            (libc::ENFILE, Some("system_file_descriptors")),
            (libc::ECONNRESET, None),
        ] {
            logs.0.lock().unwrap().clear();
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let _ = tokio::io::copy(&mut socket, &mut tokio::io::sink()).await;
            });
            let client =
                CaptureClient::new(url, "private-token-marker".into(), Duration::from_secs(2))
                    .unwrap();
            // Inject a body-stream OS error to exercise the actual reqwest /
            // hyper source chain without exhausting this machine's sockets.
            let body = reqwest::Body::wrap_stream(futures_util::stream::iter([Err::<Bytes, _>(
                std::io::Error::from_raw_os_error(errno),
            )]));
            assert_eq!(
                client.rpc("complete", body).await.unwrap_err().status(),
                StatusCode::BAD_GATEWAY
            );
            server.abort();
            let text = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
            if let Some(limit) = limit {
                assert!(
                    text.contains("PD capture hit a local socket resource limit"),
                    "{text}"
                );
                assert!(text.contains(&format!("limit={limit}")), "{text}");
                assert!(text.contains(&format!("os_error={errno}")), "{text}");
                assert!(
                    text.contains("operation=complete") && text.contains("phase=send"),
                    "{text}"
                );
            } else {
                assert!(!text.contains("local socket resource limit"), "{text}");
            }
            assert!(!text.contains("private-token-marker"));
        }
    }

    #[tokio::test]
    async fn raw_capture_preserves_prepared_and_engine_bytes_and_waits_for_ack() {
        use tower::ServiceExt;
        let prefix = format!("[{}]", vec!["12345"; 100_000].join(", "));
        let prepared = format!(r#"{{"messages":[],"required_prefix_token_ids":{prefix}}}"#);
        let engine = r#"{"prompt_token_ids":[1, 2, 3],"choices":[{"index":0,"token_ids":[4, 5],"logprobs":{"content":[{"logprob":-0.123456789012345678901,"token":"token_id:\u0034"}]}}]}"#;
        let prefill =
            r#"{"prefill":{"prompt_token_ids":[1, 2, 3],"routed_experts":"opaque\u0034"}}"#;
        let incoming = r#"{ "messages": [], "ng_capture": {"model_call_id":"raw"} }"#;
        let completing = Arc::new(tokio::sync::Semaphore::new(0));
        let ack = Arc::new(tokio::sync::Semaphore::new(0));
        let completing_handler = completing.clone();
        let ack_handler = ack.clone();
        let prepared_handler = prepared.clone();
        let service = Router::new()
            .route(
                "/prepare",
                post(move |body: Bytes| {
                    let prepared = prepared_handler.clone();
                    async move {
                        assert_eq!(body, incoming);
                        prepared
                    }
                }),
            )
            .route(
                "/complete",
                post(move |body: Bytes| {
                    let completing = completing_handler.clone();
                    let ack = ack_handler.clone();
                    async move {
                        let fields: BTreeMap<String, &RawValue> =
                            serde_json::from_slice(&body).unwrap();
                        assert_eq!(fields["response"].get(), engine);
                        assert_eq!(fields["prefill_capture"].get(), prefill);
                        assert_eq!(fields["weight_version"].get(), "7");
                        assert_eq!(fields["admission"].get(), r#"{"model_call_id":"raw"}"#);
                        completing.add_permits(1);
                        ack.acquire().await.unwrap().forget();
                        r#"{"choices":[],"ng_commit_coords":{"disposition":"staged"}}"#
                    }
                }),
            );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, service).await.unwrap() });
        let capture =
            Arc::new(CaptureClient::new(url, "test".into(), Duration::from_secs(5)).unwrap());
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(move |request: Request| {
                    let prepared = prepared.clone();
                    async move {
                        assert!(request.extensions().get::<ManagedCapture>().is_some());
                        let body = to_bytes(request.into_body(), MAX_BODY_BYTES).await.unwrap();
                        let mut fields: BTreeMap<String, &RawValue> =
                            serde_json::from_slice(&body).unwrap();
                        let context: serde_json::Value =
                            serde_json::from_str(fields.remove("nemo_rl_capture").unwrap().get())
                                .unwrap();
                        assert_eq!(context["weight_version"], 7);
                        assert_eq!(context["admission"]["model_call_id"], "raw");
                        let original: BTreeMap<String, &RawValue> =
                            serde_json::from_str(&prepared).unwrap();
                        assert_eq!(fields.len(), original.len());
                        for (key, value) in original {
                            assert_eq!(fields[&key].get(), value.get());
                        }
                        // Exercise chunked forwarding, including a split inside a JSON string.
                        let mut response =
                            Response::new(Body::from_stream(futures_util::stream::iter([
                                Ok::<_, Infallible>(&engine[..100]),
                                Ok(&engine[100..]),
                            ])));
                        response.extensions_mut().insert(PrefillCapture(Arc::from(
                            RawValue::from_string(prefill.into()).unwrap(),
                        )));
                        response
                    }
                }),
            )
            .layer(axum::middleware::from_fn(
                move |request: Request, next: Next| {
                    let capture = capture.clone();
                    async move { capture.run(request, next, 7).await }
                },
            ));
        let task = tokio::spawn(
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(incoming))
                    .unwrap(),
            ),
        );
        tokio::time::timeout(Duration::from_secs(5), completing.acquire())
            .await
            .unwrap()
            .unwrap()
            .forget();
        assert!(
            !task.is_finished(),
            "response must wait for staging acknowledgement"
        );
        ack.add_permits(1);
        let response = task.await.unwrap().unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            to_bytes(response.into_body(), MAX_BODY_BYTES)
                .await
                .unwrap(),
            r#"{"choices":[],"ng_commit_coords":{"disposition":"staged"}}"#
        );
        server.abort();
    }
    #[tokio::test]
    async fn client_context_cannot_bypass_managed_admission() {
        use tower::ServiceExt;
        let client = Arc::new(
            CaptureClient::new(
                "http://127.0.0.1:1".into(),
                "test".into(),
                Duration::from_secs(1),
            )
            .unwrap(),
        );
        let app = Router::new()
            .route("/v1/chat/completions", post(|| async { StatusCode::OK }))
            .layer(axum::middleware::from_fn(
                move |request: Request, next: Next| {
                    let client = client.clone();
                    async move { client.run(request, next, 7).await }
                },
            ));
        for (body, status) in [
            (r#"{"messages":[]}"#, StatusCode::OK),
            (
                r#"{"messages":[],"nemo_rl_capture":{"weight_version":999}}"#,
                StatusCode::BAD_REQUEST,
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/chat/completions")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), status);
        }
    }

    #[tokio::test]
    async fn completion_uploads_have_no_worker_count_concurrency_cap() {
        const CALLS: u32 = 64;
        let arrived = Arc::new(tokio::sync::Semaphore::new(0));
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let seen = arrived.clone();
        let gate = release.clone();
        let service = Router::new().route(
            "/complete",
            post(move || {
                let seen = seen.clone();
                let gate = gate.clone();
                async move {
                    seen.add_permits(1);
                    gate.acquire().await.unwrap().forget();
                    r#"{"ng_commit_coords":{"disposition":"staged"}}"#
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, service).await.unwrap() });
        let capture =
            Arc::new(CaptureClient::new(url, "test".into(), Duration::from_secs(10)).unwrap());
        let mut calls = tokio::task::JoinSet::new();
        for _ in 0..CALLS {
            let capture = capture.clone();
            calls.spawn(async move { capture.rpc("complete", "{}").await });
        }
        tokio::time::timeout(Duration::from_secs(5), arrived.acquire_many(CALLS))
            .await
            .expect("all uploads must reach the service before any acknowledgement")
            .unwrap()
            .forget();
        release.add_permits(CALLS as usize);
        while let Some(result) = calls.join_next().await {
            assert!(result.unwrap().is_ok());
        }
        server.abort();
    }
}
