mod helpers;

use helpers::{call_context, config, request};
use philharmonic_connector_impl_api::Implementation;
use philharmonic_connector_impl_llm_openai_compat::LlmOpenaiCompat;
use serde_json::Value as JsonValue;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, method, path},
};

#[tokio::test]
async fn outbound_body_matches_tool_call_fixture() {
    let server = MockServer::start().await;

    let expected_request: JsonValue = serde_json::from_str(include_str!(
        "fixtures/openai-chat/tool_call_fallback/request.json"
    ))
    .unwrap();
    let response_fixture = include_str!("fixtures/openai-chat/tool_call_fallback/response.json");

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_json(expected_request.clone()))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_raw(response_fixture, "application/json"),
        )
        .mount(&server)
        .await;

    let schema: JsonValue =
        serde_json::from_str(include_str!("fixtures/openai-chat/sample_json_schema.json")).unwrap();

    let impl_ = LlmOpenaiCompat::new().unwrap();
    impl_
        .execute(
            &config(&server.uri(), "test-key", "tool_call_fallback", 800),
            &request(
                expected_request["model"].as_str().unwrap(),
                expected_request["messages"].clone(),
                schema,
                None,
            ),
            &call_context(),
        )
        .await
        .unwrap();
}
