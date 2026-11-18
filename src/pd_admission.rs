// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.

//! Managed RL admission spans both PD phases and the entire response stream.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use futures_util::StreamExt;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{mpsc, oneshot, Notify};
use tokio_stream::wrappers::ReceiverStream;

use axum::{routing::post, Router};

#[derive(Clone)]
pub(crate) struct ManagedCapture;

pub const WEIGHT_VERSION_HEADER: &str = "x-nemo-rl-weight-version";

#[derive(Clone, Serialize)]
struct AdmissionState {
    paused: bool,
    active: usize,
    weight_version: u64,
    failure: Option<String>,
}

pub struct PDAdmission {
    token: String,
    engine_capture: Mutex<Option<(u32, bool)>>,
    state: Mutex<AdmissionState>,
    changed: Notify,
}

impl PDAdmission {
    pub fn new(token: String) -> Result<Arc<Self>, String> {
        if token.trim().is_empty() {
            return Err("NEMO_RL_PD_CONTROL_TOKEN must not be empty".into());
        }
        Ok(Arc::new(Self {
            token,
            engine_capture: Mutex::new(None),
            state: Mutex::new(AdmissionState {
                paused: false,
                active: 0,
                weight_version: 0,
                failure: None,
            }),
            changed: Notify::new(),
        }))
    }

    async fn admit(self: &Arc<Self>) -> Result<Admission, String> {
        loop {
            let wake = self.changed.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            {
                let mut state = self.state.lock();
                if let Some(failure) = &state.failure {
                    return Err(failure.clone());
                }
                if !state.paused {
                    state.active += 1;
                    return Ok(Admission {
                        gate: self.clone(),
                        weight_version: state.weight_version,
                        complete: false,
                    });
                }
            }
            wake.await;
        }
    }

    async fn pause(&self) -> Result<(), String> {
        self.state.lock().paused = true;
        self.changed.notify_waiters();
        loop {
            let wake = self.changed.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            {
                let state = self.state.lock();
                if let Some(failure) = &state.failure {
                    return Err(failure.clone());
                }
                if state.active == 0 {
                    return Ok(());
                }
            }
            wake.await;
        }
    }

    fn set_version(&self, version: u64) -> Result<(), String> {
        let mut state = self.state.lock();
        if state.failure.is_some() || !state.paused || state.active != 0 {
            return Err("Pause and drain the router before publishing a policy version".into());
        }
        if version < state.weight_version {
            return Err("Policy weight version must not decrease".into());
        }
        state.weight_version = version;
        Ok(())
    }

    fn resume(&self) -> Result<(), String> {
        let mut state = self.state.lock();
        if state.failure.is_some() || state.active != 0 {
            return Err("Cannot resume admission without a successful drain".into());
        }
        state.paused = false;
        self.changed.notify_waiters();
        Ok(())
    }

    fn fail(&self, message: &str) {
        let mut state = self.state.lock();
        state.paused = true;
        state.failure.get_or_insert_with(|| message.into());
        self.changed.notify_waiters();
    }
}

struct Admission {
    gate: Arc<PDAdmission>,
    weight_version: u64,
    complete: bool,
}

impl Drop for Admission {
    fn drop(&mut self) {
        if !self.complete {
            self.gate
                .fail("Router request ended without proving both PD phases drained");
        }
        self.gate.state.lock().active -= 1;
        self.gate.changed.notify_waiters();
    }
}

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum ControlRequest {
    Status,
    ConfigureCapture {
        top_logprobs: u32,
        routed_experts: bool,
    },
    Pause,
    SetVersion {
        weight_version: u64,
    },
    Resume,
}

pub fn routes<S: Clone + Send + Sync + 'static>(gate: Option<Arc<PDAdmission>>) -> Router<S> {
    Router::new()
        .route("/nemo_rl/admission", post(control))
        .with_state(gate)
}

async fn control(
    State(gate): State<Option<Arc<PDAdmission>>>,
    headers: HeaderMap,
    Json(request): Json<ControlRequest>,
) -> Response {
    let Some(gate) = gate else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let token = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "));
    if token != Some(gate.token.as_str()) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let result = match request {
        ControlRequest::Status => Ok(()),
        ControlRequest::ConfigureCapture {
            top_logprobs,
            routed_experts,
        } => {
            let state = gate.state.lock();
            if !state.paused || state.active != 0 || state.failure.is_some() {
                Err("Pause and drain before configuring engine capture".into())
            } else {
                *gate.engine_capture.lock() = Some((top_logprobs, routed_experts));
                Ok(())
            }
        }
        ControlRequest::Pause => gate.pause().await,
        ControlRequest::SetVersion { weight_version } => gate.set_version(weight_version),
        ControlRequest::Resume => gate.resume(),
    };
    match result {
        Ok(()) => Json(json!({"ok": true, "version": 1, "capture_protocol": gate.engine_capture.lock().as_ref().map(|_| 2), "state": gate.state.lock().clone()}))
            .into_response(),
        Err(error) => (StatusCode::CONFLICT, Json(json!({"error": error}))).into_response(),
    }
}

pub async fn middleware(
    State(gate): State<Option<Arc<PDAdmission>>>,
    request: Request,
    next: Next,
) -> Response {
    let Some(gate) = gate else {
        return next.run(request).await;
    };
    let path = request.uri().path();
    if !matches!(
        path,
        "/generate"
            | "/inference/v1/generate"
            | "/v1/chat/completions"
            | "/v1/completions"
            | "/v1/responses"
            | "/v1/embeddings"
            | "/rerank"
            | "/v1/rerank"
            | "/health_generate"
    ) {
        return next.run(request).await;
    }
    let mut admission = match gate.admit().await {
        Ok(admission) => admission,
        Err(error) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": error})),
            )
                .into_response()
        }
    };
    // A disconnected client must not cancel work between prefill and decode.
    // This task owns the admission until the upstream body reaches EOF, even
    // when the response or its stream receiver has already been dropped.
    let (headers_tx, headers_rx) = oneshot::channel();
    tokio::spawn(async move {
        let engine_capture = *admission.gate.engine_capture.lock();
        let response = if let Some((top_logprobs, routed_experts)) = engine_capture {
            run_engine_capture(
                request,
                next,
                admission.weight_version,
                top_logprobs,
                routed_experts,
            )
            .await
        } else {
            next.run(request).await
        };
        if response.status().is_server_error() {
            admission
                .gate
                .fail("PD router returned a server error; drain is uncertain");
        }
        let (mut parts, body) = response.into_parts();
        parts.headers.insert(
            WEIGHT_VERSION_HEADER,
            HeaderValue::from(admission.weight_version),
        );
        let (tx, rx) = mpsc::channel(8);
        let response = Response::from_parts(parts, Body::from_stream(ReceiverStream::new(rx)));
        let _ = headers_tx.send(response);
        let mut stream = body.into_data_stream();
        while let Some(chunk) = stream.next().await {
            if chunk.is_err() {
                admission
                    .gate
                    .fail("PD response stream failed before completion");
            }
            // send fails immediately after downstream disconnect; keep draining.
            let _ = tx.send(chunk).await;
        }
        admission.complete = true;
    });
    headers_rx
        .await
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// Stamp only small coordination fields. Token arrays never enter this service.
async fn run_engine_capture(
    request: Request,
    next: Next,
    version: u64,
    top_logprobs: u32,
    routed_experts: bool,
) -> Response {
    use serde_json::{value::RawValue, Value};
    use std::collections::BTreeMap;
    if request.uri().path() != "/v1/chat/completions" {
        return next.run(request).await;
    }
    let (mut parts, body) = request.into_parts();
    let bytes = match axum::body::to_bytes(body, 512 * 1024 * 1024).await {
        Ok(bytes) => bytes,
        Err(_) => return StatusCode::PAYLOAD_TOO_LARGE.into_response(),
    };
    let mut fields: BTreeMap<String, Box<RawValue>> = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let Some(admission) = fields
        .remove("ng_capture")
        .filter(|value| value.get() != "null")
    else {
        if fields.contains_key("nemo_rl_capture") {
            return (
                StatusCode::BAD_REQUEST,
                "nemo_rl_capture is owned by managed admission",
            )
                .into_response();
        }
        return next
            .run(Request::from_parts(parts, Body::from(bytes)))
            .await;
    };
    // These fields are owned by admission, not by a caller or earlier retry.
    for key in [
        "nemo_rl_capture",
        "nemo_rl_pd_capture",
        "required_prefix_token_ids",
        "routed_experts_prompt_start",
    ] {
        fields.remove(key);
    }
    for (key, expected) in [
        ("stream", json!(false)),
        ("n", json!(1)),
        ("best_of", json!(1)),
    ] {
        if let Some(value) = fields.get(key) {
            if serde_json::from_str::<Value>(value.get())
                .ok()
                .is_some_and(|value| !value.is_null() && value != expected)
            {
                return (
                    StatusCode::BAD_REQUEST,
                    "Engine capture requires stream=false, n=1, best_of=1",
                )
                    .into_response();
            }
        }
    }
    let mut admission: Value = match serde_json::from_str(admission.get()) {
        Ok(Value::Object(value)) => Value::Object(value),
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };
    if admission.get("prev_len").is_none() {
        admission["prev_len"] = json!(0);
    }
    let context = json!({"version": 1, "admission": admission,
        "weight_version": version, "phase": "decode", "top_logprobs": top_logprobs, "routed_experts": routed_experts});
    fields.insert(
        "nemo_rl_capture".into(),
        serde_json::value::to_raw_value(&context).unwrap(),
    );
    fields.remove("top_logprobs");
    for key in ["logprobs", "return_token_ids"] {
        fields.insert(key.into(), serde_json::value::to_raw_value(&false).unwrap());
    }
    parts.headers.remove(axum::http::header::CONTENT_LENGTH);
    parts.extensions.insert(crate::pd_admission::ManagedCapture);
    next.run(Request::from_parts(
        parts,
        Body::from(serde_json::to_vec(&fields).unwrap()),
    ))
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        http::Request as HttpRequest, middleware::from_fn_with_state, routing::post, Router,
    };
    use futures_util::future;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Semaphore;
    use tower::ServiceExt;

    fn gate() -> Arc<PDAdmission> {
        PDAdmission::new("control-secret".into()).unwrap()
    }

    fn app(gate: Option<Arc<PDAdmission>>, downstream: Router) -> Router {
        downstream
            .merge(routes(gate.clone()))
            .layer(from_fn_with_state(gate, middleware))
    }

    async fn command(app: &Router, value: serde_json::Value) -> Response {
        app.clone()
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/nemo_rl/admission")
                    .header("authorization", "Bearer control-secret")
                    .header("content-type", "application/json")
                    .body(Body::from(value.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    fn request() -> HttpRequest<Body> {
        HttpRequest::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .body(Body::empty())
            .unwrap()
    }

    async fn consume(response: Response) {
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn engine_capture_preserves_requests_without_capture_admission() {
        let gate = gate();
        *gate.engine_capture.lock() = Some((8, true));
        let downstream = Router::new().route(
            "/v1/chat/completions",
            post(|request: Request| async move {
                assert!(request
                    .extensions()
                    .get::<crate::pd_admission::ManagedCapture>()
                    .is_none());
                let bytes = axum::body::to_bytes(request.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(body["required_prefix_token_ids"], json!([1, 2]));
                assert_eq!(body["logprobs"], true);
                assert!(body.get("nemo_rl_capture").is_none());
                StatusCode::OK
            }),
        );
        let app = app(Some(gate), downstream);
        let request = HttpRequest::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .body(Body::from(
                r#"{"messages":[],"required_prefix_token_ids":[1,2],"logprobs":true}"#,
            ))
            .unwrap();
        assert_eq!(app.oneshot(request).await.unwrap().status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn engine_capture_stamps_epoch_without_http_capture_service() {
        let gate = gate();
        let downstream = Router::new().route("/v1/chat/completions", post(|request: Request| async move {
            assert!(request.extensions().get::<crate::pd_admission::ManagedCapture>().is_some());
            let bytes = axum::body::to_bytes(request.into_body(), usize::MAX).await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert!(body.get("ng_capture").is_none());
            assert!(body.get("required_prefix_token_ids").is_none());
            assert_eq!(body["messages"][0]["content"], "hello");
            assert_eq!(body["nemo_rl_capture"], json!({"version":1,
                "admission":{"model_call_id":"c1","prev_len":0}, "weight_version":7,
                "phase":"decode", "top_logprobs":8, "routed_experts":true}));
            Json(json!({"choices":[{"message":{"content":"ok"}}],"ng_commit_coords":{"disposition":"staged"}}))
        }));
        let app = app(Some(gate), downstream);
        assert_eq!(
            command(
                &app,
                json!({"operation":"configure_capture", "top_logprobs":8,"routed_experts":true})
            )
            .await
            .status(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            command(&app, json!({"operation":"pause"})).await.status(),
            StatusCode::OK
        );
        assert_eq!(
            command(
                &app,
                json!({"operation":"configure_capture", "top_logprobs":8,"routed_experts":true})
            )
            .await
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            command(&app, json!({"operation":"set_version", "weight_version":7}))
                .await
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            command(&app, json!({"operation":"resume"})).await.status(),
            StatusCode::OK
        );
        let response = app
            .oneshot(captured_request(
                json!({"nemo_rl_capture":{"weight_version":999},"required_prefix_token_ids":[99]}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["ng_commit_coords"]["disposition"], "staged");
    }

    #[tokio::test]
    async fn policy_version_changes_only_after_both_phases_drain() {
        let gate = gate();
        let entered = Arc::new(Semaphore::new(0));
        let prefill_done = Arc::new(Semaphore::new(0));
        let decode_done = Arc::new(Semaphore::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        let route = {
            let (entered, prefill_done, decode_done, calls) = (
                entered.clone(),
                prefill_done.clone(),
                decode_done.clone(),
                calls.clone(),
            );
            post(move || {
                let (entered, prefill_done, decode_done, calls) = (
                    entered.clone(),
                    prefill_done.clone(),
                    decode_done.clone(),
                    calls.clone(),
                );
                async move {
                    if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                        entered.add_permits(1);
                        prefill_done.acquire().await.unwrap().forget();
                        entered.add_permits(1);
                        decode_done.acquire().await.unwrap().forget();
                    }
                    "complete"
                }
            })
        };
        let app = app(
            Some(gate.clone()),
            Router::new().route("/v1/chat/completions", route),
        );
        let running = tokio::spawn(app.clone().oneshot(request()));
        entered.acquire().await.unwrap().forget();
        let pause = command(&app, json!({"operation": "pause"}));
        tokio::pin!(pause);
        assert!(futures::poll!(&mut pause).is_pending());
        assert!(gate.state.lock().paused);
        assert_eq!(
            command(
                &app,
                json!({"operation": "set_version", "weight_version": 7})
            )
            .await
            .status(),
            StatusCode::CONFLICT
        );
        let waiting = app.clone().oneshot(request());
        tokio::pin!(waiting);
        assert!(futures::poll!(&mut waiting).is_pending());
        prefill_done.add_permits(1);
        entered.acquire().await.unwrap().forget();
        assert!(futures::poll!(&mut pause).is_pending());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        decode_done.add_permits(1);
        let response = running.await.unwrap().unwrap();
        assert_eq!(response.headers()[WEIGHT_VERSION_HEADER], "0");
        consume(response).await;
        assert_eq!(pause.await.status(), StatusCode::OK);
        assert_eq!(
            command(
                &app,
                json!({"operation": "set_version", "weight_version": 7})
            )
            .await
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            command(&app, json!({"operation": "resume"})).await.status(),
            StatusCode::OK
        );
        let response = waiting.await.unwrap();
        assert_eq!(response.headers()[WEIGHT_VERSION_HEADER], "7");
        consume(response).await;
    }

    #[tokio::test]
    async fn disconnect_before_headers_does_not_release_admission() {
        let gate = gate();
        let entered = Arc::new(Semaphore::new(0));
        let finish = Arc::new(Semaphore::new(0));
        let route = {
            let (entered, finish) = (entered.clone(), finish.clone());
            post(move || {
                let (entered, finish) = (entered.clone(), finish.clone());
                async move {
                    entered.add_permits(1);
                    finish.acquire().await.unwrap().forget();
                    "complete"
                }
            })
        };
        let app = app(
            Some(gate.clone()),
            Router::new().route("/v1/chat/completions", route),
        );
        let running = tokio::spawn(app.clone().oneshot(request()));
        entered.acquire().await.unwrap().forget();
        running.abort();
        assert!(running.await.unwrap_err().is_cancelled());
        assert_eq!(gate.state.lock().active, 1);
        let pause = command(&app, json!({"operation": "pause"}));
        tokio::pin!(pause);
        assert!(futures::poll!(&mut pause).is_pending());
        finish.add_permits(1);
        assert_eq!(pause.await.status(), StatusCode::OK);
        assert_eq!(gate.state.lock().active, 0);
    }

    #[tokio::test]
    async fn disconnected_response_stream_is_drained_to_eof() {
        let gate = gate();
        let (tx, rx) = mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(1);
        let source = Arc::new(Mutex::new(Some(rx)));
        let app = app(
            Some(gate.clone()),
            Router::new().route(
                "/v1/chat/completions",
                post(move || {
                    let body =
                        Body::from_stream(ReceiverStream::new(source.lock().take().unwrap()));
                    future::ready(Response::new(body))
                }),
            ),
        );
        let response = app.clone().oneshot(request()).await.unwrap();
        drop(response);
        let pause = command(&app, json!({"operation": "pause"}));
        tokio::pin!(pause);
        assert!(futures::poll!(&mut pause).is_pending());
        tx.send(Ok(bytes::Bytes::from_static(b"tail")))
            .await
            .unwrap();
        drop(tx);
        assert_eq!(pause.await.status(), StatusCode::OK);
        assert!(gate.state.lock().failure.is_none());
    }

    #[tokio::test]
    async fn cancelled_pause_stays_closed_and_versions_cannot_regress() {
        let gate = gate();
        let mut admitted = gate.admit().await.unwrap();
        {
            let pause = gate.pause();
            tokio::pin!(pause);
            assert!(futures::poll!(&mut pause).is_pending());
        }
        assert!(gate.state.lock().paused);
        assert!(gate.resume().is_err());
        admitted.complete = true;
        drop(admitted);
        gate.pause().await.unwrap();
        gate.set_version(8).unwrap();
        assert!(gate.set_version(7).is_err());
        gate.resume().unwrap();
        assert!(gate.set_version(9).is_err());
    }

    #[tokio::test]
    async fn upstream_failure_prevents_refit_and_reopening() {
        let gate = gate();
        let app = app(
            Some(gate.clone()),
            Router::new().route(
                "/v1/chat/completions",
                post(|| async { StatusCode::BAD_GATEWAY }),
            ),
        );
        consume(app.clone().oneshot(request()).await.unwrap()).await;
        assert!(gate.pause().await.is_err());
        assert!(gate.resume().is_err());
        assert!(gate.set_version(9).is_err());
        assert_eq!(
            app.clone().oneshot(request()).await.unwrap().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            command(&app, json!({"operation": "status"})).await.status(),
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn control_requires_authentication_and_is_opt_in() {
        let gate = gate();
        let app = app(Some(gate.clone()), Router::new());
        let response = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/nemo_rl/admission")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"operation":"pause"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert!(!gate.state.lock().paused);
        let disabled = super::routes::<()>(None);
        assert_eq!(
            command(&disabled, json!({"operation": "pause"}))
                .await
                .status(),
            StatusCode::NOT_FOUND
        );
    }

    fn captured_request(extra: serde_json::Value) -> Request {
        let mut body = json!({"messages": [{"role":"user", "content":"hello"}], "tools": [{"type":"function"}], "ng_capture": {"model_call_id":"c1"}});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("x-session-id", "session")
            .header("content-length", body.to_string().len())
            .body(Body::from(body.to_string()))
            .unwrap()
    }
}
