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
use serde_json::{json, Value};

const MAX_BODY_BYTES: usize = 512 * 1024 * 1024;

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

    async fn rpc(&self, operation: &'static str, body: &Value) -> Result<Value, Response> {
        // Include request serialization, connection waits, service time, and
        // response-body receipt/JSON decoding, not just time to HTTP headers.
        let started = Instant::now();
        let result = self.rpc_inner(operation, body).await;
        metrics::histogram!(
            "vllm_router_pd_capture_rpc_duration_seconds",
            "operation" => operation,
            "outcome" => if result.is_ok() { "success" } else { "error" },
        )
        .record(started.elapsed().as_secs_f64());
        result
    }

    async fn rpc_inner(&self, operation: &str, body: &Value) -> Result<Value, Response> {
        let result = self
            .client
            .post(format!("{}/{operation}", self.url))
            .bearer_auth(&self.token)
            .json(body)
            .send()
            .await;
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                tracing::error!(%operation, %error, "PD capture RPC failed; will not retry");
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
        response.json::<Value>().await.map_err(|error| {
            tracing::error!(%operation, %error, "Invalid PD capture RPC response");
            failure(StatusCode::BAD_GATEWAY, "Invalid PD capture RPC response")
        })
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
        let original: Value = match serde_json::from_slice(&bytes) {
            Ok(body) => body,
            Err(_) => return failure(StatusCode::BAD_REQUEST, "Invalid chat JSON"),
        };
        let Some(admission) = original.get("ng_capture").filter(|value| !value.is_null()) else {
            return next
                .run(Request::from_parts(parts, Body::from(bytes)))
                .await;
        };
        if original.get("stream").and_then(Value::as_bool) == Some(true) {
            return failure(
                StatusCode::BAD_REQUEST,
                "Managed PD capture requires stream=false",
            );
        }
        let prepared = match self.rpc("prepare", &original).await {
            Ok(body) if body.is_object() && body.get("ng_capture").is_none() => body,
            Ok(_) => return failure(StatusCode::BAD_GATEWAY, "Invalid prepared capture request"),
            Err(error) => return error,
        };
        parts.headers.remove(header::CONTENT_LENGTH);
        let response = next
            .run(Request::from_parts(parts, Body::from(prepared.to_string())))
            .await;
        if !response.status().is_success() {
            // Preserve context-length and other engine errors verbatim. Nothing
            // was staged during prepare; there is no request registry to abort.
            return response;
        }
        let (mut parts, body) = response.into_parts();
        let bytes = match to_bytes(body, MAX_BODY_BYTES).await {
            Ok(bytes) => bytes,
            Err(_) => return failure(StatusCode::BAD_GATEWAY, "Cannot read PD capture response"),
        };
        let payload: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(_) => return failure(StatusCode::BAD_GATEWAY, "Invalid PD capture response JSON"),
        };
        let completed = match self
            .rpc(
                "complete",
                &json!({
                    "admission": admission, "response": payload, "weight_version": weight_version,
                }),
            )
            .await
        {
            Ok(value) if value.get("ng_commit_coords").is_some_and(Value::is_object) => value,
            Ok(_) => {
                return failure(
                    StatusCode::BAD_GATEWAY,
                    "Capture response lacks commit coordinates",
                )
            }
            Err(error) => return error,
        };
        parts.headers.remove(header::CONTENT_LENGTH);
        parts.headers.remove(header::CONTENT_ENCODING);
        parts.headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        Response::from_parts(parts, Body::from(completed.to_string()))
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
        let (client, calls, server) =
            mock_capture(StatusCode::OK, "{}", delay, Duration::from_secs(5)).await;
        for operation in ["prepare", "complete"] {
            assert_eq!(client.rpc(operation, &json!({})).await.unwrap(), json!({}));
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
                client
                    .rpc(operation, &json!({}))
                    .await
                    .unwrap_err()
                    .status(),
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
}
