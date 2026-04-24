use philharmonic_connector_common::{UnixMillis, Uuid};
use philharmonic_connector_impl_api::{ConnectorCallContext, Implementation};
use philharmonic_connector_impl_llm_openai_compat::LlmOpenaiCompat;
use serde_json::{Value as JsonValue, json};

fn ctx() -> ConnectorCallContext {
    ConnectorCallContext {
        tenant_id: Uuid::nil(),
        instance_id: Uuid::nil(),
        step_seq: 1,
        config_uuid: Uuid::nil(),
        issued_at: UnixMillis(0),
        expires_at: UnixMillis(10_000),
    }
}

#[tokio::test]
#[ignore]
async fn vllm_native_smoke() {
    if std::env::var("VLLM_SMOKE_ENABLED").ok().as_deref() != Some("1") {
        return;
    }

    let base_url = std::env::var("VLLM_BASE_URL")
        .expect("VLLM_BASE_URL is required when VLLM_SMOKE_ENABLED=1");
    let api_key = std::env::var("VLLM_API_KEY").unwrap_or_else(|_| "test-key".to_owned());

    let fixture_request: JsonValue = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/vllm/structured_outputs_json_chat_request.json"
    )))
    .unwrap();
    let schema_text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/vllm/sample_json_schema.json"
    ));
    let schema_value: JsonValue = serde_json::from_str(schema_text).unwrap();

    let mut messages = fixture_request["messages"].clone();
    let template = messages[1]["content"].as_str().unwrap();
    messages[1]["content"] = json!(template.replace("<sample_json_schema inline>", schema_text));

    let request = json!({
        "model": fixture_request["model"],
        "messages": messages,
        "output_schema": schema_value,
        "max_output_tokens": fixture_request["max_completion_tokens"]
    });

    let impl_ = LlmOpenaiCompat::new().unwrap();
    let out = impl_
        .execute(
            &json!({
                "base_url": base_url,
                "api_key": api_key,
                "dialect": "vllm_native",
                "timeout_ms": 30_000
            }),
            &request,
            &ctx(),
        )
        .await
        .unwrap();

    assert!(out["output"].is_object());
}
