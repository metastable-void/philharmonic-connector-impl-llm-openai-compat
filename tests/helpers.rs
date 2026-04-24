use philharmonic_connector_common::{UnixMillis, Uuid};
use philharmonic_connector_impl_api::ConnectorCallContext;
use serde_json::{Value as JsonValue, json};

pub fn call_context() -> ConnectorCallContext {
    ConnectorCallContext {
        tenant_id: Uuid::nil(),
        instance_id: Uuid::nil(),
        step_seq: 1,
        config_uuid: Uuid::nil(),
        issued_at: UnixMillis(0),
        expires_at: UnixMillis(10_000),
    }
}

pub fn config(base_url: &str, api_key: &str, dialect: &str, timeout_ms: u64) -> JsonValue {
    json!({
        "base_url": base_url,
        "api_key": api_key,
        "dialect": dialect,
        "timeout_ms": timeout_ms
    })
}

pub fn request(
    model: &str,
    messages: JsonValue,
    output_schema: JsonValue,
    max_output_tokens: Option<u32>,
) -> JsonValue {
    let mut value = json!({
        "model": model,
        "messages": messages,
        "output_schema": output_schema
    });

    if let Some(tokens) = max_output_tokens {
        value["max_output_tokens"] = json!(tokens);
    }

    value
}
