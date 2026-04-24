mod helpers;

use helpers::{call_context, config, request};
use philharmonic_connector_impl_api::Implementation;
use philharmonic_connector_impl_llm_openai_compat::LlmOpenaiCompat;
use serde_json::{Value as JsonValue, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, method, path},
};

fn reconstructed_vllm_messages(request_fixture: &JsonValue, schema_text: &str) -> JsonValue {
    let mut messages = request_fixture["messages"].clone();
    let user_template = messages[1]["content"].as_str().unwrap();
    let rebuilt = user_template.replace("<sample_json_schema inline>", schema_text);
    messages[1]["content"] = json!(rebuilt);
    messages
}

#[tokio::test]
async fn outbound_body_matches_vllm_fixture_with_documented_carve_outs() {
    let server = MockServer::start().await;

    let fixture_request: JsonValue = serde_json::from_str(include_str!(
        "fixtures/vllm/structured_outputs_json_chat_request.json"
    ))
    .unwrap();
    let schema_text = include_str!("fixtures/vllm/sample_json_schema.json");
    let schema_value: JsonValue = serde_json::from_str(schema_text).unwrap();
    let rebuilt_messages = reconstructed_vllm_messages(&fixture_request, schema_text);

    let mut expected_request = fixture_request.clone();
    expected_request["messages"] = rebuilt_messages.clone();

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_json(expected_request.clone()))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_raw(
                    include_str!("fixtures/vllm_native_response.json"),
                    "application/json",
                ),
        )
        .mount(&server)
        .await;

    let impl_ = LlmOpenaiCompat::new().unwrap();
    impl_
        .execute(
            &config(&server.uri(), "test-key", "vllm_native", 800),
            &request(
                fixture_request["model"].as_str().unwrap(),
                rebuilt_messages,
                schema_value,
                Some(
                    fixture_request["max_completion_tokens"]
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())
                        .unwrap(),
                ),
            ),
            &call_context(),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn structured_outputs_json_subobject_matches_schema_fixture() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_raw(
                    include_str!("fixtures/vllm_native_response.json"),
                    "application/json",
                ),
        )
        .mount(&server)
        .await;

    let fixture_request: JsonValue = serde_json::from_str(include_str!(
        "fixtures/vllm/structured_outputs_json_chat_request.json"
    ))
    .unwrap();
    let schema_text = include_str!("fixtures/vllm/sample_json_schema.json");
    let schema_value: JsonValue = serde_json::from_str(schema_text).unwrap();
    let rebuilt_messages = reconstructed_vllm_messages(&fixture_request, schema_text);

    let impl_ = LlmOpenaiCompat::new().unwrap();
    impl_
        .execute(
            &config(&server.uri(), "test-key", "vllm_native", 800),
            &request(
                fixture_request["model"].as_str().unwrap(),
                rebuilt_messages,
                schema_value.clone(),
                fixture_request["max_completion_tokens"]
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok()),
            ),
            &call_context(),
        )
        .await
        .unwrap();

    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    let body: JsonValue = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["structured_outputs"]["json"], schema_value);
}
