// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Managed capture keeps training arrays opaque until they reach the RL worker.

use std::{
    collections::BTreeMap,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use axum::{body::Body, response::Response};
use serde::{
    de::{IgnoredAny, SeqAccess, Visitor},
    ser::SerializeMap,
    Serialize, Serializer,
};
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
                | "nemo_rl_pd_capture"
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
            return super::pd_capture::validate(&self.controls, path, sequential_nixl);
        }
        if path != "/v1/chat/completions"
            || self.get("stream").and_then(Value::as_bool) == Some(true)
        {
            return Err("Managed capture requires non-streaming chat completions".into());
        }
        if self.get("nemo_rl_pd_capture").and_then(Value::as_bool) == Some(true) && !sequential_nixl
        {
            return Err("PD capture requires sequential NIXL dispatch".into());
        }
        // /prepare validates messages, n, best_of and the ledger prefix. The
        // engine renders the chat; /complete validates exact token alignment.
        Ok(())
    }

    /// Return small coordination JSON plus opaque R3 metadata for /complete.
    pub fn prefill(&self, bytes: &[u8]) -> Result<(Value, Option<Box<RawValue>>), String> {
        if !self.managed {
            let mut response =
                serde_json::from_slice(bytes).map_err(|e| format!("Invalid prefill JSON: {e}"))?;
            super::pd_capture::attach_prefill(&self.controls, &mut response)?;
            return Ok((response, None));
        }
        let fields: RawObject<'_> =
            serde_json::from_slice(bytes).map_err(|e| format!("Invalid prefill JSON: {e}"))?;
        let mut controls = json!({});
        if let Some(params) = fields.get("kv_transfer_params") {
            controls["kv_transfer_params"] =
                serde_json::from_str(params.get()).map_err(|e| e.to_string())?;
        }
        if self.get("nemo_rl_pd_capture").and_then(Value::as_bool) != Some(true) {
            return Ok((controls, None));
        }
        let choices: Vec<RawObject<'_>> = serde_json::from_str(
            fields
                .get("choices")
                .ok_or("Missing prefill choices")?
                .get(),
        )
        .map_err(|e| e.to_string())?;
        if choices.len() != 1 || choices[0].get("index").map(|v| v.get()) != Some("0") {
            return Err("PD capture requires one prefill choice with index 0".into());
        }
        let prompt = fields
            .get("prompt_token_ids")
            .filter(|v| v.get() != "null")
            .or_else(|| choices[0].get("prompt_token_ids"))
            .ok_or("Missing prefill prompt IDs")?;
        let prompt_len = array_len(prompt)?;
        let end = controls
            .pointer("/kv_transfer_params/remote_num_tokens")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .filter(|end| *end <= prompt_len)
            .ok_or("PD capture requires valid NIXL remote_num_tokens")?;
        let prompt_start = self
            .get("routed_experts_prompt_start")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if prompt_len == 0 || prompt_start >= prompt_len as u64 {
            return Err("PD routed_experts_prompt_start exceeds the last prompt token".into());
        }
        let start = (prompt_len - 1).min(end).max(prompt_start as usize);
        let routes = choices[0]
            .get("routed_experts")
            .ok_or("Missing prefill routed experts")?;
        #[derive(Serialize)]
        struct Prefill<'a> {
            prompt_token_ids: &'a RawValue,
            routed_experts: &'a RawValue,
        }
        #[derive(Serialize)]
        struct Metadata<'a> {
            version: u8,
            decode_route_start: usize,
            prefill_route_end: usize,
            prefill: Prefill<'a>,
        }
        let metadata = serde_json::value::to_raw_value(&Metadata {
            version: 1,
            decode_route_start: start,
            prefill_route_end: end,
            prefill: Prefill {
                prompt_token_ids: prompt,
                routed_experts: routes,
            },
        })
        .map_err(|e| e.to_string())?;
        controls["nemo_rl_pd_capture"] = json!({"decode_route_start": start});
        Ok((controls, Some(metadata)))
    }
}

// Count positions without allocating or converting individual token IDs. Their
// types, values and prefix consistency are checked by the capture worker.
fn array_len(raw: &RawValue) -> Result<usize, String> {
    struct Count;
    impl<'de> Visitor<'de> for Count {
        type Value = usize;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a token array")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<usize, A::Error> {
            let mut count = 0;
            while seq.next_element::<IgnoredAny>()?.is_some() {
                count += 1;
            }
            Ok(count)
        }
    }
    let mut deserializer = serde_json::Deserializer::from_str(raw.get());
    serde::Deserializer::deserialize_seq(&mut deserializer, Count).map_err(|e| e.to_string())
}

/// Stream decode training data unchanged; the capture worker joins R3 metadata.
pub(crate) fn decode_response(
    response: reqwest::Response,
    metadata: Option<Box<RawValue>>,
) -> Result<Response, String> {
    let status = response.status();
    let mut builder = Response::builder().status(status);
    for (name, value) in response.headers() {
        if name != "content-length" && name != "transfer-encoding" && name != "content-encoding" {
            builder = builder.header(name, value);
        }
    }
    if let Some(metadata) = metadata.filter(|_| status.is_success()) {
        builder = builder.extension(crate::pd_capture_rpc::PrefillCapture(Arc::from(metadata)));
    }
    builder
        .body(Body::from_stream(response.bytes_stream()))
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

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
                super::super::pd_capture::prepare_decode(&mut request, &Value::Null).unwrap();
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
    fn prefill_routes_keep_raw_training_data_and_only_compute_boundaries() {
        let request =
            PdRequest::managed(br#"{"nemo_rl_pd_capture":true,"routed_experts_prompt_start":2}"#)
                .unwrap();
        let prefill = br#"{"prompt_token_ids":[11, 12, 13, 14],"kv_transfer_params":{"remote_num_tokens":1},"choices":[{"index":0,"token_ids":[999],"routed_experts":"abc\u0064"}]}"#;
        let (controls, metadata) = request.prefill(prefill).unwrap();
        assert_eq!(controls["nemo_rl_pd_capture"]["decode_route_start"], 2);
        let raw = metadata.unwrap();
        assert!(raw.get().contains("[11, 12, 13, 14]"));
        assert!(raw.get().contains(r#""abc\u0064""#));
        assert!(!raw.get().contains("999"));
        assert!(!controls.to_string().contains("prompt_token_ids"));
        assert!(request.prefill(br#"{"prompt_token_ids":[],"kv_transfer_params":{"remote_num_tokens":0},"choices":[{"index":0}]}"#).is_err());
        assert!(request.prefill(br#"{"prompt_token_ids":[1,2,3],"kv_transfer_params":{"remote_num_tokens":4},"choices":[{"index":0}]}"#).is_err());
    }

    #[test]
    fn managed_capture_rejects_streaming_and_unsupported_r3_connector() {
        let request = PdRequest::managed(br#"{"stream":true}"#).unwrap();
        assert!(request
            .validate_capture("/v1/chat/completions", true)
            .is_err());
        let request = PdRequest::managed(br#"{"nemo_rl_pd_capture":true}"#).unwrap();
        assert!(request
            .validate_capture("/v1/chat/completions", false)
            .is_err());
    }
}
