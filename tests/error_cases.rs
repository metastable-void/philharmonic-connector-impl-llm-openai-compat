mod helpers;

use helpers::{call_context, config, request};
use philharmonic_connector_impl_api::{Implementation, ImplementationError};
use philharmonic_connector_impl_llm_openai_compat::LlmOpenaiCompat;
use serde_json::{Value as JsonValue, json};
use std::{net::TcpListener, time::Duration};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn openai_request_parts() -> (String, JsonValue, JsonValue) {
    let request_fixture: JsonValue = serde_json::from_str(include_str!(
        "fixtures/openai-chat/openai_native/request.json"
    ))
    .unwrap();
    let schema: JsonValue =
        serde_json::from_str(include_str!("fixtures/openai-chat/sample_json_schema.json")).unwrap();

    (
        request_fixture["model"].as_str().unwrap().to_owned(),
        request_fixture["messages"].clone(),
        schema,
    )
}

#[tokio::test]
async fn invalid_config_when_required_field_missing() {
    let impl_ = LlmOpenaiCompat::new().unwrap();
    let (_model, messages, schema) = openai_request_parts();

    let cfg = json!({
        "base_url": "https://api.openai.com/v1",
        "dialect": "openai_native",
        "timeout_ms": 100
    });

    let err = impl_
        .execute(
            &cfg,
            &request("gpt-4o-mini", messages, schema, None),
            &call_context(),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, ImplementationError::InvalidConfig { .. }));
}

#[tokio::test]
async fn invalid_request_when_output_schema_fails_compile() {
    let server = MockServer::start().await;

    let (model, messages, _schema) = openai_request_parts();
    let impl_ = LlmOpenaiCompat::new().unwrap();

    let err = impl_
        .execute(
            &config(&server.uri(), "test-key", "openai_native", 100),
            &request(&model, messages, json!({"type": 17}), None),
            &call_context(),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, ImplementationError::InvalidRequest { .. }));
}

#[tokio::test]
async fn upstream_error_401_surfaces_status_and_body() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({"error": "bad key"})))
        .mount(&server)
        .await;

    let (model, messages, schema) = openai_request_parts();
    let impl_ = LlmOpenaiCompat::new().unwrap();

    let err = impl_
        .execute(
            &config(&server.uri(), "test-key", "openai_native", 100),
            &request(&model, messages, schema, None),
            &call_context(),
        )
        .await
        .unwrap_err();

    let ImplementationError::UpstreamError { status, body } = err else {
        panic!("expected UpstreamError");
    };
    assert_eq!(status, 401);
    assert!(body.contains("bad key"));
}

#[tokio::test]
async fn upstream_error_500_surfaces_status_and_body_after_retries() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({"error": "boom"})))
        .mount(&server)
        .await;

    let (model, messages, schema) = openai_request_parts();
    let impl_ = LlmOpenaiCompat::new().unwrap();

    let err = impl_
        .execute(
            &config(&server.uri(), "test-key", "openai_native", 100),
            &request(&model, messages, schema, None),
            &call_context(),
        )
        .await
        .unwrap_err();

    let ImplementationError::UpstreamError { status, body } = err else {
        panic!("expected UpstreamError");
    };
    assert_eq!(status, 500);
    assert!(body.contains("boom"));

    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 3);
}

#[tokio::test]
async fn upstream_unreachable_when_connection_refused() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let (model, messages, schema) = openai_request_parts();
    let impl_ = LlmOpenaiCompat::new().unwrap();

    let err = impl_
        .execute(
            &config(&format!("http://{addr}"), "test-key", "openai_native", 100),
            &request(&model, messages, schema, None),
            &call_context(),
        )
        .await
        .unwrap_err();

    assert!(matches!(
        err,
        ImplementationError::UpstreamUnreachable { .. }
    ));
}

#[tokio::test]
async fn upstream_timeout_when_attempt_times_out() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(80)))
        .mount(&server)
        .await;

    let (model, messages, schema) = openai_request_parts();
    let impl_ = LlmOpenaiCompat::new().unwrap();

    let err = impl_
        .execute(
            &config(&server.uri(), "test-key", "openai_native", 10),
            &request(&model, messages, schema, None),
            &call_context(),
        )
        .await
        .unwrap_err();

    assert_eq!(err, ImplementationError::UpstreamTimeout);
}

#[tokio::test]
async fn schema_validation_failed_when_output_is_off_schema() {
    let server = MockServer::start().await;

    let bad_response = json!({
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "{\"age\":\"not-an-integer\"}"
            }
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 5,
            "total_tokens": 10
        }
    });

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(bad_response))
        .mount(&server)
        .await;

    let strict_schema = json!({
        "type": "object",
        "properties": {
            "age": {"type": "integer"}
        },
        "required": ["age"],
        "additionalProperties": false
    });

    let (model, messages, _schema) = openai_request_parts();
    let impl_ = LlmOpenaiCompat::new().unwrap();

    let err = impl_
        .execute(
            &config(&server.uri(), "test-key", "openai_native", 100),
            &request(&model, messages, strict_schema, None),
            &call_context(),
        )
        .await
        .unwrap_err();

    let ImplementationError::SchemaValidationFailed { detail } = err else {
        panic!("expected SchemaValidationFailed");
    };
    assert!(detail.contains("age"));
}

#[tokio::test]
async fn internal_when_provider_payload_is_malformed() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"no_choices": true})))
        .mount(&server)
        .await;

    let (model, messages, schema) = openai_request_parts();
    let impl_ = LlmOpenaiCompat::new().unwrap();

    let err = impl_
        .execute(
            &config(&server.uri(), "test-key", "openai_native", 100),
            &request(&model, messages, schema, None),
            &call_context(),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, ImplementationError::Internal { .. }));
}
