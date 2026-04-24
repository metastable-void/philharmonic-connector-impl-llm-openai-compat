use philharmonic_connector_common::{UnixMillis, Uuid};
use philharmonic_connector_impl_api::{ConnectorCallContext, Implementation};
use philharmonic_connector_impl_llm_openai_compat::LlmOpenaiCompat;
use serde_json::json;

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

fn enabled() -> bool {
    std::env::var("OPENAI_SMOKE_ENABLED").ok().as_deref() == Some("1")
}

#[tokio::test]
#[ignore]
async fn openai_native_smoke() {
    if !enabled() {
        return;
    }

    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY is required when OPENAI_SMOKE_ENABLED=1");
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_owned());

    let schema = json!({
        "type": "object",
        "properties": {
            "message": {"type": "string"}
        },
        "required": ["message"],
        "additionalProperties": false
    });

    let request = json!({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Return an object with key message."}
        ],
        "output_schema": schema,
        "max_output_tokens": 128
    });

    let impl_ = LlmOpenaiCompat::new().unwrap();
    let out = impl_
        .execute(
            &json!({
                "base_url": base_url,
                "api_key": api_key,
                "dialect": "openai_native",
                "timeout_ms": 30_000
            }),
            &request,
            &ctx(),
        )
        .await
        .unwrap();

    assert!(out["output"].is_object());
}

#[tokio::test]
#[ignore]
async fn tool_call_fallback_smoke() {
    if !enabled() {
        return;
    }

    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY is required when OPENAI_SMOKE_ENABLED=1");
    let base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_owned());

    let schema = json!({
        "type": "object",
        "properties": {
            "message": {"type": "string"}
        },
        "required": ["message"],
        "additionalProperties": false
    });

    let request = json!({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Return an object with key message."}
        ],
        "output_schema": schema,
        "max_output_tokens": 128
    });

    let impl_ = LlmOpenaiCompat::new().unwrap();
    let out = impl_
        .execute(
            &json!({
                "base_url": base_url,
                "api_key": api_key,
                "dialect": "tool_call_fallback",
                "timeout_ms": 30_000
            }),
            &request,
            &ctx(),
        )
        .await
        .unwrap();

    assert!(out["output"].is_object());
}
