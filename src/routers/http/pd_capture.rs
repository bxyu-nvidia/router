// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact-token R3 metadata for NeMo RL's completion and chat transports.
use std::io;

use bytes::{Bytes, BytesMut};
use futures_util::{Stream, StreamExt};
use serde_json::{json, Value};

pub fn enabled(request: &Value) -> bool {
    request.get("nemo_rl_pd_capture").and_then(Value::as_bool) == Some(true)
}

fn token_ids(value: Option<&Value>) -> Result<&Vec<Value>, String> {
    value
        .and_then(Value::as_array)
        .filter(|ids| {
            !ids.is_empty()
                && ids
                    .iter()
                    .all(|t| t.as_u64().is_some_and(|t| t <= u32::MAX as u64))
        })
        .ok_or_else(|| "PD capture requires nonempty integer prompt token IDs".into())
}

fn choice(response: &Value) -> Result<&Value, String> {
    response
        .get("choices")
        .and_then(Value::as_array)
        .filter(|choices| choices.len() == 1)
        .and_then(|choices| choices.first())
        .filter(|choice| choice.get("index").and_then(Value::as_u64) == Some(0))
        .ok_or_else(|| "PD capture requires one response choice with index 0".into())
}

fn prompt_ids(response: &Value) -> Result<&Vec<Value>, String> {
    token_ids(
        response
            .get("prompt_token_ids")
            .filter(|v| !v.is_null())
            .or_else(|| response.pointer("/choices/0/prompt_token_ids")),
    )
}

/// Reject unsupported capture requests before submitting any GPU work.
pub fn validate(request: &Value, path: &str, sequential_nixl: bool) -> Result<(), String> {
    if !enabled(request) {
        return Ok(());
    }
    if !sequential_nixl {
        return Err("PD capture requires sequential NIXL dispatch".into());
    }
    if request
        .get("n")
        .filter(|v| !v.is_null())
        .and_then(Value::as_u64)
        .unwrap_or(1)
        != 1
        || request
            .get("best_of")
            .filter(|v| !v.is_null())
            .and_then(Value::as_u64)
            .unwrap_or(1)
            != 1
    {
        return Err("PD capture requires n=1 and best_of=1".into());
    }
    match path {
        "/v1/chat/completions" => {
            if request
                .get("messages")
                .and_then(Value::as_array)
                .is_none_or(|m| m.is_empty())
            {
                return Err("PD chat capture requires messages".into());
            }
            if request.get("include_reasoning").and_then(Value::as_bool) == Some(false) {
                return Err(
                    "PD capture requires include_reasoning=true for complete token metadata".into(),
                );
            }
        }
        "/v1/completions" => {
            token_ids(request.get("prompt"))?;
            if request.get("stream").and_then(Value::as_bool) == Some(true) {
                return Err(
                    "PD completion capture requires stream=false; use chat for streaming capture"
                        .into(),
                );
            }
            if request.get("echo").and_then(Value::as_bool) == Some(true) {
                return Err("PD completion capture requires echo=false".into());
            }
        }
        _ => return Err("PD capture supports chat/completions endpoints only".into()),
    }
    Ok(())
}

pub fn prepare_prefill(request: &mut Value) {
    if enabled(request) {
        request["return_token_ids"] = json!(true);
    }
    if let Some(object) = request.as_object_mut() {
        object.remove("nemo_rl_pd_capture");
    }
}

pub fn prepare_decode(request: &mut Value, prefill: &Value) -> Result<(), String> {
    if enabled(request) {
        let start = prefill
            .pointer("/nemo_rl_pd_capture/decode_route_start")
            .ok_or("PD capture requires sequential NIXL prefill metadata")?;
        request["routed_experts_prompt_start"] = start.clone();
        request["return_token_ids"] = json!(true);
    }
    if let Some(object) = request.as_object_mut() {
        object.remove("nemo_rl_pd_capture");
    }
    Ok(())
}

pub fn attach_prefill(request: &Value, response: &mut Value) -> Result<(), String> {
    if !enabled(request) {
        return Ok(());
    }
    let prompt = prompt_ids(response)?;
    let prompt_len = if request.get("messages").is_some() {
        // Chat rendering and prefix alignment belong to vLLM, never the router.
        if let Some(prefix) = request
            .get("required_prefix_token_ids")
            .and_then(Value::as_array)
        {
            if !prompt.starts_with(prefix) {
                return Err("PD prefill did not preserve required_prefix_token_ids".into());
            }
        }
        prompt.len()
    } else {
        let submitted = token_ids(request.get("prompt"))?;
        if !submitted.starts_with(prompt) {
            return Err("PD prefill changed the submitted prompt token IDs".into());
        }
        submitted.len()
    };
    let end = response
        .pointer("/kv_transfer_params/remote_num_tokens")
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .filter(|end| *end <= prompt.len())
        .ok_or("PD capture requires valid NIXL remote_num_tokens")?;
    let start = (prompt_len - 1).min(end);
    let prompt_start = request
        .get("routed_experts_prompt_start")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if prompt_start >= prompt_len as u64 {
        return Err("PD routed_experts_prompt_start exceeds the last prompt token".into());
    }
    // KV transfer can back off before the committed parent (e.g. Mamba block
    // alignment). Decode recomputes those tokens, but their routes are already
    // staged. Return only the suffix needed by this call, including its boundary.
    let start = start.max(prompt_start as usize);
    let routes = choice(response)?
        .get("routed_experts")
        .filter(|v| v.is_string())
        .ok_or("PD capture prefill response lacks routed experts")?;
    response["nemo_rl_pd_capture"] = json!({
        "version": 1, "decode_route_start": start, "prefill_route_end": end,
        "prefill": {"prompt_token_ids": prompt, "routed_experts": routes}
    });
    Ok(())
}

pub fn metadata(prefill: &Value) -> Option<&Value> {
    prefill.get("nemo_rl_pd_capture")
}

fn validate_decode_prompt(metadata: &Value, response: &Value) -> Result<(), String> {
    let prefill_ids = token_ids(metadata.pointer("/prefill/prompt_token_ids"))?;
    let decode_ids = prompt_ids(response)?;
    let start = metadata["decode_route_start"]
        .as_u64()
        .ok_or("Missing PD route boundary")?;
    if !decode_ids.starts_with(prefill_ids) || start >= decode_ids.len() as u64 {
        return Err("PD prefill/decode prompt token IDs or boundaries differ".into());
    }
    Ok(())
}

/// Preserve decode's sample, logprobs, top-k and tool calls verbatim.
pub fn merge(prefill: &Value, decode: &mut Value) -> Result<(), String> {
    if let Some(metadata) = metadata(prefill) {
        // Backend errors remain errors, never successful captures.
        if decode.get("error").is_some() {
            return Ok(());
        }
        validate_decode_prompt(metadata, decode)?;
        if !choice(decode)?
            .get("routed_experts")
            .is_some_and(Value::is_string)
        {
            return Err("PD capture decode response lacks routed experts".into());
        }
        decode["nemo_rl_pd_capture"] = metadata.clone();
    }
    Ok(())
}

struct CaptureStream<S> {
    upstream: std::pin::Pin<Box<S>>,
    buffer: BytesMut,
    scanned: usize,
    metadata: Value,
    prompt_seen: bool,
    final_seen: bool,
    failed: bool,
    done: bool,
}

impl<S> CaptureStream<S> {
    fn frame(&mut self, bytes: Bytes) -> Result<Bytes, io::Error> {
        let text = std::str::from_utf8(&bytes).map_err(io::Error::other)?;
        let data = text
            .lines()
            .filter_map(|l| l.strip_prefix("data:").map(|s| s.trim_start()))
            .collect::<Vec<_>>()
            .join("\n");
        if data.is_empty() {
            return Ok(bytes);
        } // SSE comments/keepalives.
        if data == "[DONE]" {
            if !self.failed && (!self.prompt_seen || !self.final_seen) {
                return Err(io::Error::other(
                    "PD capture stream ended without prompt IDs or expert routes",
                ));
            }
            self.done = true;
            return Ok(bytes);
        }
        let mut chunk: Value = serde_json::from_str(&data).map_err(io::Error::other)?;
        if chunk.get("error").is_some() {
            self.failed = true;
            return Ok(bytes);
        }
        if chunk.get("prompt_token_ids").is_some_and(|v| !v.is_null()) {
            validate_decode_prompt(&self.metadata, &chunk).map_err(io::Error::other)?;
            self.prompt_seen = true;
        }
        if chunk
            .pointer("/choices/0/finish_reason")
            .is_some_and(|v| !v.is_null())
        {
            if !self.prompt_seen
                || !choice(&chunk)
                    .map_err(io::Error::other)?
                    .get("routed_experts")
                    .is_some_and(Value::is_string)
            {
                return Err(io::Error::other(
                    "PD capture terminal chunk lacks prompt IDs or expert routes",
                ));
            }
            chunk["nemo_rl_pd_capture"] = self.metadata.clone();
            self.final_seen = true;
            return Ok(Bytes::from(format!("data: {}\n\n", chunk)));
        }
        Ok(bytes)
    }
}

/// Attach prefill capture to the terminal chat chunk, even across HTTP frame splits.
pub fn stream<S, E>(upstream: S, metadata: Value) -> impl Stream<Item = Result<Bytes, io::Error>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    let state = CaptureStream {
        upstream: Box::pin(upstream),
        buffer: BytesMut::new(),
        scanned: 0,
        metadata,
        prompt_seen: false,
        final_seen: false,
        failed: false,
        done: false,
    };
    futures_util::stream::try_unfold(state, |mut state| async move {
        if state.done {
            return Ok(None);
        }
        loop {
            let suffix = &state.buffer[state.scanned..];
            let boundary = suffix
                .windows(2)
                .position(|w| w == b"\n\n")
                .map(|p| p + 2)
                .into_iter()
                .chain(
                    suffix
                        .windows(4)
                        .position(|w| w == b"\r\n\r\n")
                        .map(|p| p + 4),
                )
                .min()
                .map(|p| p + state.scanned);
            if let Some(end) = boundary {
                let frame = state.buffer.split_to(end).freeze();
                state.scanned = 0;
                let output = state.frame(frame)?;
                return Ok(Some((output, state)));
            }
            state.scanned = state.buffer.len().saturating_sub(3);
            match state.upstream.next().await {
                Some(Ok(bytes)) => state.buffer.extend_from_slice(&bytes),
                Some(Err(error)) => return Err(io::Error::other(error)),
                None => return Err(io::Error::other("PD capture stream closed before [DONE]")),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::TryStreamExt;

    fn chat_request() -> Value {
        json!({"messages": [{"role": "user", "content": "hi"}],
            "required_prefix_token_ids": [11, 12], "nemo_rl_pd_capture": true})
    }

    fn prefill() -> Value {
        json!({"prompt_token_ids": [11, 12, 13],
            "kv_transfer_params": {"remote_num_tokens": 2},
            "choices": [{"index": 0, "token_ids": [999], "routed_experts": "prefill-routes"}]})
    }

    fn decode() -> Value {
        json!({"prompt_token_ids": [11, 12, 13], "choices": [{"index": 0,
            "token_ids": [14, 15], "routed_experts": "decode-routes", "finish_reason": "tool_calls",
            "message": {"tool_calls": [{"id": "call-1", "function": {"name": "tool", "arguments": "{}"}}]},
            "logprobs": {"content": [{"token": "token_id:14", "logprob": -0.2,
                "top_logprobs": [{"token": "token_id:15", "logprob": -0.3}]}]}}]})
    }

    #[test]
    fn chat_capture_preserves_exact_prefix_and_decode_metadata_without_prefill_sample() {
        let request = chat_request();
        validate(&request, "/v1/chat/completions", true).unwrap();
        let mut response = prefill();
        attach_prefill(&request, &mut response).unwrap();
        let mut pre = request.clone();
        prepare_prefill(&mut pre);
        let mut dec = request.clone();
        prepare_decode(&mut dec, &response).unwrap();
        for stage in [&pre, &dec] {
            assert_eq!(stage["required_prefix_token_ids"], json!([11, 12]));
            assert_eq!(stage["messages"], request["messages"]);
            assert_eq!(stage["return_token_ids"], true);
            assert!(stage.get("nemo_rl_pd_capture").is_none());
        }
        assert_eq!(dec["routed_experts_prompt_start"], 2);
        let mut final_response = decode();
        let original = final_response.clone();
        merge(&response, &mut final_response).unwrap();
        assert_eq!(final_response["choices"], original["choices"]);
        assert_eq!(
            final_response["prompt_token_ids"],
            original["prompt_token_ids"]
        );
        assert_eq!(final_response["nemo_rl_pd_capture"]["version"], 1);
        assert_eq!(final_response["nemo_rl_pd_capture"]["prefill_route_end"], 2);
        assert!(final_response["nemo_rl_pd_capture"]["prefill"]
            .get("token_ids")
            .is_none());
    }

    #[test]
    fn completion_capture_retains_existing_boundary_for_truncated_prefill() {
        let request = json!({"prompt": [11, 12, 13], "nemo_rl_pd_capture": true});
        let mut response = json!({"kv_transfer_params": {"remote_num_tokens": 2},
            "choices": [{"index": 0, "prompt_token_ids": [11, 12], "routed_experts": "pre"}]});
        validate(&request, "/v1/completions", true).unwrap();
        attach_prefill(&request, &mut response).unwrap();
        let mut final_response = json!({"choices": [{"index": 0, "prompt_token_ids": [11, 12, 13],
            "token_ids": [14], "routed_experts": "dec"}]});
        merge(&response, &mut final_response).unwrap();
        assert_eq!(
            final_response["nemo_rl_pd_capture"]["decode_route_start"],
            2
        );
    }

    #[test]
    fn kv_backoff_before_parent_captures_only_the_needed_decode_suffix() {
        for transfer_end in [0, 1, 2, 3] {
            let mut request = chat_request();
            request["routed_experts_prompt_start"] = json!(1);
            let mut response = prefill();
            response["kv_transfer_params"]["remote_num_tokens"] = json!(transfer_end);
            attach_prefill(&request, &mut response).unwrap();
            let start = transfer_end.clamp(1, 2);
            assert_eq!(response["nemo_rl_pd_capture"]["decode_route_start"], start);
            assert_eq!(
                response["nemo_rl_pd_capture"]["prefill_route_end"],
                transfer_end
            );
            prepare_decode(&mut request, &response).unwrap();
            assert_eq!(request["routed_experts_prompt_start"], start);
            let mut final_response = decode();
            merge(&response, &mut final_response).unwrap();
        }
        let mut request = chat_request();
        request["routed_experts_prompt_start"] = json!(3);
        assert!(attach_prefill(&request, &mut prefill()).is_err());
    }

    #[test]
    fn invalid_capture_requests_and_mismatched_engine_metadata_fail() {
        for (key, value) in [("n", json!(2)), ("include_reasoning", json!(false))] {
            let mut request = chat_request();
            request[key] = value;
            assert!(validate(&request, "/v1/chat/completions", true).is_err());
        }
        assert!(validate(&chat_request(), "/v1/chat/completions", false).is_err());
        let mut bad = prefill();
        bad["kv_transfer_params"]["remote_num_tokens"] = json!(4);
        assert!(attach_prefill(&chat_request(), &mut bad).is_err());
        let mut bad = prefill();
        bad["prompt_token_ids"] = json!([11, 99, 13]);
        assert!(attach_prefill(&chat_request(), &mut bad).is_err());
        let mut response = prefill();
        attach_prefill(&chat_request(), &mut response).unwrap();
        let mut bad = decode();
        bad["prompt_token_ids"] = json!([11, 12, 99]);
        assert!(merge(&response, &mut bad).is_err());
        let mut bad = decode();
        bad["choices"][0]["routed_experts"] = Value::Null;
        assert!(merge(&response, &mut bad).is_err());
        let mut error = json!({"error": {"message": "backend failed"}});
        let original = error.clone();
        merge(&response, &mut error).unwrap();
        assert_eq!(error, original);
        let mut ordinary = json!({"messages": [], "stream": true});
        let original = ordinary.clone();
        prepare_prefill(&mut ordinary);
        prepare_decode(&mut ordinary, &Value::Null).unwrap();
        assert_eq!(ordinary, original);
    }

    #[tokio::test]
    async fn streamed_capture_handles_split_frames_and_preserves_ids_logprobs_and_usage() {
        let mut prefill = prefill();
        attach_prefill(&chat_request(), &mut prefill).unwrap();
        let terminal = json!({"id": "chat-1", "object": "chat.completion.chunk", "choices": [{"index": 0,
            "delta": {}, "finish_reason": "stop", "routed_experts": "decode-routes"}]});
        let first = "data: {\"prompt_token_ids\":[11,12,13],\"choices\":[]}\r\n\r\n";
        let sample = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hé\"},\"token_ids\":[14],\"logprobs\":{\"content\":[{\"token\":\"token_id:14\",\"logprob\":-0.2}]}}]}\n\n";
        let usage = "data: {\"choices\":[],\"usage\":{\"completion_tokens\":1}}\n\n";
        let raw =
            format!(": keepalive\n\n{first}{sample}data: {terminal}\n\n{usage}data: [DONE]\n\n");
        for size in [1, 17, raw.len()] {
            let chunks: Vec<_> = raw
                .as_bytes()
                .chunks(size)
                .map(|b| Ok::<_, io::Error>(Bytes::copy_from_slice(b)))
                .collect();
            let output = stream(
                futures_util::stream::iter(chunks),
                metadata(&prefill).unwrap().clone(),
            )
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .concat();
            let output = String::from_utf8(output).unwrap();
            assert!(output.contains(first));
            assert!(output.contains(sample));
            assert!(output.contains(usage));
            assert!(output.ends_with("data: [DONE]\n\n"));
            let captured: Value = output
                .split("\n\n")
                .filter_map(|f| f.strip_prefix("data: "))
                .filter_map(|f| serde_json::from_str::<Value>(f).ok())
                .find(|f| f.get("nemo_rl_pd_capture").is_some())
                .unwrap();
            assert_eq!(captured["choices"], terminal["choices"]);
            assert_eq!(
                captured["nemo_rl_pd_capture"],
                prefill["nemo_rl_pd_capture"]
            );
        }
    }

    #[tokio::test]
    async fn incomplete_or_mismatched_capture_stream_is_not_successful() {
        let mut prefill = prefill();
        attach_prefill(&chat_request(), &mut prefill).unwrap();
        for text in [
            "data: [DONE]\n\n",
            "data: {\"prompt_token_ids\":[11,99,13],\"choices\":[]}\n\n",
            "data: {\"prompt_token_ids\":[11,12,13],\"choices\":[]}\n\n",
        ] {
            let upstream = futures_util::stream::iter([Ok::<_, io::Error>(Bytes::from_static(
                text.as_bytes(),
            ))]);
            assert!(stream(upstream, metadata(&prefill).unwrap().clone())
                .try_collect::<Vec<_>>()
                .await
                .is_err());
        }
    }
}
