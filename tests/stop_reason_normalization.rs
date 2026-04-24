mod helpers;

use helpers::{call_context, config, request};
use philharmonic_connector_impl_api::Implementation;
use philharmonic_connector_impl_llm_openai_compat::LlmOpenaiCompat;
use serde_json::{Value as JsonValue, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn expected_stop_reason(dialect: &str, finish_reason: &str) -> &'static str {
    match (dialect, finish_reason) {
        (_, "stop") => "end_turn",
        (_, "length") => "max_tokens",
        (_, "content_filter") => "content_filter",
        ("tool_call_fallback", "tool_calls") => "end_turn",
        (_, "tool_calls") => "error",
        _ => "error",
    }
}

fn response_for(dialect: &str, finish_reason: &str) -> JsonValue {
    match dialect {
        "tool_call_fallback" => json!({
            "choices": [{
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "type": "function",
                        "id": "call_1",
                        "function": {
                            "name": "emit_output",
                            "arguments": "{\"ok\":true}"
                        }
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 13,
                "total_tokens": 24
            }
        }),
        _ => json!({
            "choices": [{
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": "{\"ok\":true}"
                }
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 13,
                "total_tokens": 24
            }
        }),
    }
}

#[tokio::test]
async fn finish_reason_variants_normalize_per_spec_table() {
    let request_fixture: JsonValue = serde_json::from_str(include_str!(
        "fixtures/openai-chat/openai_native/request.json"
    ))
    .unwrap();

    let schema = json!({
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"}
        },
        "required": ["ok"],
        "additionalProperties": false
    });

    let dialects = ["openai_native", "vllm_native", "tool_call_fallback"];
    let finish_reasons = ["stop", "length", "content_filter", "tool_calls", "other"];

    for dialect in dialects {
        for finish_reason in finish_reasons {
            let server = MockServer::start().await;

            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(
                    ResponseTemplate::new(200).set_body_json(response_for(dialect, finish_reason)),
                )
                .mount(&server)
                .await;

            let impl_ = LlmOpenaiCompat::new().unwrap();
            let out = impl_
                .execute(
                    &config(&server.uri(), "test-key", dialect, 100),
                    &request(
                        request_fixture["model"].as_str().unwrap(),
                        request_fixture["messages"].clone(),
                        schema.clone(),
                        None,
                    ),
                    &call_context(),
                )
                .await
                .unwrap();

            assert_eq!(
                out["stop_reason"],
                json!(expected_stop_reason(dialect, finish_reason)),
                "dialect={dialect} finish_reason={finish_reason}"
            );
        }
    }
}
