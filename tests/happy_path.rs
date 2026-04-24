mod helpers;

use helpers::{call_context, config, request};
use philharmonic_connector_impl_api::Implementation;
use philharmonic_connector_impl_llm_openai_compat::LlmOpenaiCompat;
use serde_json::{Value as JsonValue, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[tokio::test]
async fn openai_native_happy_path() {
    let server = MockServer::start().await;
    let response_fixture = include_str!("fixtures/openai-chat/openai_native/response.json");

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_raw(response_fixture, "application/json"),
        )
        .mount(&server)
        .await;

    let openai_request: JsonValue = serde_json::from_str(include_str!(
        "fixtures/openai-chat/openai_native/request.json"
    ))
    .unwrap();
    let schema: JsonValue =
        serde_json::from_str(include_str!("fixtures/openai-chat/sample_json_schema.json")).unwrap();

    let impl_ = LlmOpenaiCompat::new().unwrap();
    let out = impl_
        .execute(
            &config(&server.uri(), "test-key", "openai_native", 500),
            &request(
                openai_request["model"].as_str().unwrap(),
                openai_request["messages"].clone(),
                schema,
                None,
            ),
            &call_context(),
        )
        .await
        .unwrap();

    assert_eq!(out["stop_reason"], json!("end_turn"));
    assert_eq!(out["usage"]["input_tokens"], json!(132));
    assert_eq!(out["usage"]["output_tokens"], json!(154));
    assert!(out["output"].is_object());
}

#[tokio::test]
async fn vllm_native_happy_path() {
    let server = MockServer::start().await;
    let response_fixture = include_str!("fixtures/vllm_native_response.json");

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_raw(response_fixture, "application/json"),
        )
        .mount(&server)
        .await;

    let vllm_request: JsonValue = serde_json::from_str(include_str!(
        "fixtures/vllm/structured_outputs_json_chat_request.json"
    ))
    .unwrap();
    let schema: JsonValue =
        serde_json::from_str(include_str!("fixtures/vllm/sample_json_schema.json")).unwrap();

    let impl_ = LlmOpenaiCompat::new().unwrap();
    let out = impl_
        .execute(
            &config(&server.uri(), "test-key", "vllm_native", 500),
            &request(
                vllm_request["model"].as_str().unwrap(),
                vllm_request["messages"].clone(),
                schema,
                Some(1000),
            ),
            &call_context(),
        )
        .await
        .unwrap();

    assert_eq!(out["stop_reason"], json!("end_turn"));
    assert_eq!(out["usage"]["input_tokens"], json!(120));
    assert_eq!(out["usage"]["output_tokens"], json!(80));
    assert!(out["output"].is_object());
}

#[tokio::test]
async fn tool_call_fallback_happy_path() {
    let server = MockServer::start().await;
    let response_fixture = include_str!("fixtures/openai-chat/tool_call_fallback/response.json");

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_raw(response_fixture, "application/json"),
        )
        .mount(&server)
        .await;

    let openai_request: JsonValue = serde_json::from_str(include_str!(
        "fixtures/openai-chat/tool_call_fallback/request.json"
    ))
    .unwrap();
    let schema: JsonValue =
        serde_json::from_str(include_str!("fixtures/openai-chat/sample_json_schema.json")).unwrap();

    let impl_ = LlmOpenaiCompat::new().unwrap();
    let out = impl_
        .execute(
            &config(&server.uri(), "test-key", "tool_call_fallback", 500),
            &request(
                openai_request["model"].as_str().unwrap(),
                openai_request["messages"].clone(),
                schema,
                None,
            ),
            &call_context(),
        )
        .await
        .unwrap();

    assert_eq!(out["stop_reason"], json!("end_turn"));
    assert_eq!(out["usage"]["input_tokens"], json!(111));
    assert_eq!(out["usage"]["output_tokens"], json!(71));
    assert!(out["output"].is_object());
}
