// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact-token R3 metadata for NeMo RL's non-streaming completion transport.
use serde_json::{json, Value};

pub fn enabled(request: &Value) -> bool {
    request.get("nemo_rl_pd_capture").and_then(Value::as_bool) == Some(true)
}

fn split(request: &Value) -> Result<usize, String> {
    let prompt = request
        .get("prompt")
        .and_then(Value::as_array)
        .filter(|p| !p.is_empty() && p.iter().all(|t| t.as_u64().is_some()))
        .ok_or("PD capture requires a nonempty token-ID prompt")?;
    if request.get("stream").and_then(Value::as_bool) == Some(true)
        || request.get("n").and_then(Value::as_u64).unwrap_or(1) != 1
    {
        return Err("PD capture requires non-streaming n=1".into());
    }
    Ok(prompt.len() - 1)
}

pub fn prepare_decode(request: &mut Value, prefill: &Value) -> Result<(), String> {
    if enabled(request) {
        split(request)?;
        let start = prefill
            .pointer("/nemo_rl_pd_capture/decode_route_start")
            .ok_or("PD capture requires sequential NIXL prefill metadata")?;
        request["routed_experts_prompt_start"] = start.clone();
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
    let last_prompt = split(request)?;
    let end = response
        .pointer("/kv_transfer_params/remote_num_tokens")
        .and_then(Value::as_u64)
        .ok_or("PD capture requires NIXL remote_num_tokens")? as usize;
    let start = last_prompt.min(end);
    let choice = response
        .get("choices")
        .and_then(Value::as_array)
        .filter(|choices| choices.len() == 1)
        .and_then(|choices| choices.first())
        .ok_or("PD capture prefill response must contain one choice")?;
    let prompt = choice
        .get("prompt_token_ids")
        .filter(|v| v.is_array())
        .ok_or("PD capture prefill response lacks prompt IDs")?;
    let routes = choice
        .get("routed_experts")
        .filter(|v| v.is_string())
        .ok_or("PD capture prefill response lacks routed experts")?;
    response["nemo_rl_pd_capture"] = json!({
        "version": 1, "decode_route_start": start, "prefill_route_end": end,
        "prefill": {"prompt_token_ids": prompt, "routed_experts": routes}
    });
    Ok(())
}

pub fn merge(prefill: &Value, decode: &mut Value) {
    if let Some(metadata) = prefill.get("nemo_rl_pd_capture") {
        // Decode errors remain errors, never successful captures.
        if decode.get("choices").is_some() {
            decode["nemo_rl_pd_capture"] = metadata.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_prefill_routes_and_decode_boundary_without_prefill_sample() {
        let request = json!({"prompt": [11, 12, 13], "nemo_rl_pd_capture": true});
        let mut decode_request = request.clone();
        let mut prefill = json!({"kv_transfer_params": {"remote_num_tokens": 2}, "choices": [{"prompt_token_ids": [11, 12],
            "token_ids": [999], "routed_experts": "prefill-routes"}]});
        attach_prefill(&request, &mut prefill).unwrap();
        prepare_decode(&mut decode_request, &prefill).unwrap();
        assert_eq!(decode_request["routed_experts_prompt_start"], 2);
        assert!(decode_request.get("nemo_rl_pd_capture").is_none());
        let mut decode =
            json!({"choices": [{"token_ids": [14, 15], "routed_experts": "decode-routes"}]});
        merge(&prefill, &mut decode);
        assert_eq!(decode["choices"][0]["token_ids"], json!([14, 15]));
        assert_eq!(
            decode["nemo_rl_pd_capture"]["prefill"]["prompt_token_ids"],
            json!([11, 12])
        );
        assert!(decode["nemo_rl_pd_capture"]["prefill"]
            .get("token_ids")
            .is_none());
    }

    #[test]
    fn ordinary_requests_are_unchanged_and_invalid_capture_fails() {
        let mut request = json!({"prompt": "ordinary"});
        let original = request.clone();
        prepare_decode(&mut request, &Value::Null).unwrap();
        assert_eq!(request, original);
        for invalid in [
            json!({"prompt": "text", "nemo_rl_pd_capture": true}),
            json!({"prompt": [1], "stream": true, "nemo_rl_pd_capture": true}),
        ] {
            assert!(prepare_decode(&mut invalid.clone(), &Value::Null).is_err());
        }
        assert!(attach_prefill(
            &json!({"prompt": [1], "nemo_rl_pd_capture": true}),
            &mut json!({"choices": [{}]})
        )
        .is_err());
    }
}
