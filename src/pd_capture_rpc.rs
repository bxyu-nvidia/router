// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.

//! Capture RPC to the RL-owned CPU service. No model rendering or TQ dependency.

use std::time::Duration;

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

    async fn rpc(&self, operation: &str, body: &Value) -> Result<Value, Response> {
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
