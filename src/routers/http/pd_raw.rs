// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Preserve chat payloads while forwarding small engine capture metadata.

use std::{
    collections::BTreeMap,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use axum::{body::Body, response::Response};
use serde::{ser::SerializeMap, Serialize, Serializer};
use serde_json::{json, value::RawValue, Value};

type RawObject<'a> = BTreeMap<String, &'a RawValue>;

/// Only coordination fields are materialized. Clones share opaque request data.
#[derive(Clone, Debug)]
pub(crate) struct PdRequest {
    controls: Value,
    opaque: BTreeMap<String, Arc<RawValue>>,
    pub managed: bool,
}

impl From<Value> for PdRequest {
    fn from(controls: Value) -> Self {
        Self {
            controls,
            opaque: BTreeMap::new(),
            managed: false,
        }
    }
}

impl Deref for PdRequest {
    type Target = Value;
    fn deref(&self) -> &Value {
        &self.controls
    }
}

impl DerefMut for PdRequest {
    fn deref_mut(&mut self) -> &mut Value {
        &mut self.controls
    }
}

impl Serialize for PdRequest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if !self.managed {
            return self.controls.serialize(serializer);
        }
        // Retain sorted field order used by the cache-aware routing policy.
        let controls = self
            .controls
            .as_object()
            .expect("managed request is an object");
        let mut keys: Vec<&String> = controls.keys().chain(self.opaque.keys()).collect();
        keys.sort_unstable();
        let mut map = serializer.serialize_map(Some(keys.len()))?;
        for key in keys {
            if let Some(value) = controls.get(key) {
                map.serialize_entry(key, value)?;
            } else {
                map.serialize_entry(key, self.opaque[key].as_ref())?;
            }
        }
        map.end()
    }
}

impl PdRequest {
    pub fn managed(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        let fields: RawObject<'_> = serde_json::from_slice(bytes)?;
        let mut request = Self {
            controls: json!({}),
            opaque: BTreeMap::new(),
            managed: true,
        };
        for (key, value) in fields {
            match key.as_str() {
                "stream"
                | "stream_options"
                | "max_tokens"
                | "max_completion_tokens"
                | "min_tokens"
                | "n"
                | "best_of"
                | "include_reasoning"
                | "logprobs"
                | "echo"
                | "nemo_rl_capture"
                | "routed_experts_prompt_start"
                | "return_token_ids"
                | "kv_transfer_params" => {
                    request.controls[key] = serde_json::from_str(value.get())?;
                }
                _ => {
                    request.opaque.insert(key, Arc::from(value.to_owned()));
                }
            }
        }
        Ok(request)
    }

    pub fn validate_capture(&self, path: &str, sequential_nixl: bool) -> Result<(), String> {
        if !self.managed {
            return Ok(());
        }
        if path != "/v1/chat/completions"
            || self.get("stream").and_then(Value::as_bool) == Some(true)
        {
            return Err("Managed capture requires non-streaming chat completions".into());
        }
        if !sequential_nixl {
            return Err("Engine capture requires sequential NIXL dispatch".into());
        }
        Ok(())
    }

    /// Managed prefill returns only KV coordination and an acknowledged TQ key.
    pub fn prefill(&self, bytes: &[u8]) -> Result<Value, String> {
        if !self.managed {
            return serde_json::from_slice(bytes).map_err(|e| format!("Invalid prefill JSON: {e}"));
        }
        let fields: RawObject<'_> =
            serde_json::from_slice(bytes).map_err(|e| format!("Invalid prefill JSON: {e}"))?;
        let mut controls = json!({});
        if let Some(params) = fields.get("kv_transfer_params") {
            controls["kv_transfer_params"] =
                serde_json::from_str(params.get()).map_err(|e| e.to_string())?;
        }
        Ok(controls)
    }
}

/// Forward an engine response, including errors and acknowledged commit receipts.
pub(crate) fn forward_response(response: reqwest::Response) -> Result<Response, String> {
    let mut builder = Response::builder().status(response.status());
    for (name, value) in response.headers() {
        if name != "content-length" && name != "transfer-encoding" && name != "content-encoding" {
            builder = builder.header(name, value);
        }
    }
    builder
        .body(Body::from_stream(response.bytes_stream()))
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_capture_passes_only_prefill_key_to_decode() {
        let request = PdRequest::managed(br#"{"messages":[],"nemo_rl_capture":{}}"#).unwrap();
        assert!(request
            .validate_capture("/v1/chat/completions", false)
            .is_err());
        let prefill = request.prefill(br#"{"choices":[{"message":{"content":"ignored"}}],"kv_transfer_params":{"remote_num_tokens":4,"nemo_rl_prefill_key":"opaque"}}"#).unwrap();
        let mut decode = request;
        super::super::pd_capture::prepare_decode(&mut decode, &prefill).unwrap();
        assert_eq!(decode["nemo_rl_capture"]["prefill_key"], "opaque");
        assert_eq!(decode["nemo_rl_capture"]["phase"], "decode");
        assert!(decode.get("nemo_rl_pd_capture").is_none());
        assert!(super::super::pd_capture::prepare_decode(&mut decode, &json!({})).is_err());
    }

    #[test]
    fn managed_request_shares_opaque_arrays_and_preserves_wire_values() {
        let tokens = format!("[{}]", vec!["12345"; 100_000].join(", "));
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":"x"}}],"required_prefix_token_ids":{tokens},"stream":false,"max_tokens":1000,"min_tokens":4,"unknown_extension":{{"number":1.00000000000000000001}}}}"#
        );
        let mut request = PdRequest::managed(body.as_bytes()).unwrap();
        assert!(request.controls.get("required_prefix_token_ids").is_none());
        let copy = request.clone();
        assert!(Arc::ptr_eq(
            &request.opaque["required_prefix_token_ids"],
            &copy.opaque["required_prefix_token_ids"]
        ));
        request["max_tokens"] = json!(1);
        let encoded = serde_json::to_string(&request).unwrap();
        assert!(encoded.contains(&tokens));
        assert!(encoded.contains("1.00000000000000000001"));
        let decoded: Value = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded["max_tokens"], 1);
        assert_eq!(decoded["messages"][0]["content"], "x");
        assert_eq!(
            decoded["required_prefix_token_ids"]
                .as_array()
                .unwrap()
                .len(),
            100_000
        );
    }

    #[test]
    fn capture_context_survives_both_legs_without_changing_kv_or_prefix() {
        let body = br#"{"required_prefix_token_ids":[1,2,3],"vllm_xargs":{"extension":42},"kv_transfer_params":{"remote_num_tokens":2},"nemo_rl_capture":{"version":1,"admission":{"model_call_id":"call","prev_len":3,"staging_chain":["parent"]},"weight_version":7,"phase":"decode","top_logprobs":8,"routed_experts":true}}"#;
        let original = PdRequest::managed(body).unwrap();
        for phase in ["prefill", "decode"] {
            let mut request = original.clone();
            if phase == "prefill" {
                super::super::pd_capture::prepare_prefill(&mut request);
            } else {
                super::super::pd_capture::prepare_decode(
                    &mut request,
                    &json!({"kv_transfer_params":{"nemo_rl_prefill_key":"key"}}),
                )
                .unwrap();
            }
            let encoded = serde_json::to_value(request).unwrap();
            assert_eq!(encoded["nemo_rl_capture"]["phase"], phase);
            assert_eq!(encoded["nemo_rl_capture"]["weight_version"], 7);
            assert_eq!(
                encoded["nemo_rl_capture"]["admission"]["staging_chain"],
                json!(["parent"])
            );
            assert_eq!(encoded["required_prefix_token_ids"], json!([1, 2, 3]));
            assert_eq!(encoded["vllm_xargs"]["extension"], 42);
            assert_eq!(encoded["kv_transfer_params"]["remote_num_tokens"], 2);
        }
        assert_eq!(original["nemo_rl_capture"]["phase"], "decode");
    }

    #[test]
    fn managed_capture_rejects_streaming_and_unsupported_connector() {
        let request = PdRequest::managed(br#"{"stream":true}"#).unwrap();
        assert!(request
            .validate_capture("/v1/chat/completions", true)
            .is_err());
        let request = PdRequest::managed(br#"{"nemo_rl_capture":{}}"#).unwrap();
        assert!(request
            .validate_capture("/v1/chat/completions", false)
            .is_err());
    }
}
