mod helpers;

use helpers::{call_context, config, request};
use philharmonic_connector_impl_api::{Implementation, ImplementationError};
use philharmonic_connector_impl_llm_openai_compat::LlmOpenaiCompat;
use serde_json::{Value as JsonValue, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[tokio::test]
async fn off_schema_provider_output_returns_schema_validation_failed_with_path_detail() {
    let server = MockServer::start().await;

    let response = json!({
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "{\"profile\":{\"age\":\"thirty\"}}"
            }
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12
        }
    });

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;

    let request_fixture: JsonValue = serde_json::from_str(include_str!(
        "fixtures/openai-chat/openai_native/request.json"
    ))
    .unwrap();

    let schema = json!({
        "type": "object",
        "properties": {
            "profile": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer"}
                },
                "required": ["age"],
                "additionalProperties": false
            }
        },
        "required": ["profile"],
        "additionalProperties": false
    });

    let impl_ = LlmOpenaiCompat::new().unwrap();
    let err = impl_
        .execute(
            &config(&server.uri(), "test-key", "openai_native", 100),
            &request(
                request_fixture["model"].as_str().unwrap(),
                request_fixture["messages"].clone(),
                schema,
                None,
            ),
            &call_context(),
        )
        .await
        .unwrap_err();

    let ImplementationError::SchemaValidationFailed { detail } = err else {
        panic!("expected SchemaValidationFailed");
    };

    assert!(detail.contains("age") || detail.contains("/profile/age"));
}
