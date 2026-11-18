// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small coordination fields for engine-side NeMo RL capture.

use serde_json::{json, Value};

pub fn prepare_prefill(request: &mut Value) {
    if let Some(context) = request
        .get_mut("nemo_rl_capture")
        .and_then(Value::as_object_mut)
    {
        context.insert("phase".into(), json!("prefill"));
    }
}

pub fn prepare_decode(request: &mut Value, prefill: &Value) -> Result<(), String> {
    if let Some(context) = request
        .get_mut("nemo_rl_capture")
        .and_then(Value::as_object_mut)
    {
        let key = prefill
            .pointer("/kv_transfer_params/nemo_rl_prefill_key")
            .and_then(Value::as_str)
            .filter(|key| !key.is_empty())
            .ok_or("Engine capture requires an acknowledged prefill key")?;
        context.insert("phase".into(), json!("decode"));
        context.insert("prefill_key".into(), json!(key));
    }
    Ok(())
}
