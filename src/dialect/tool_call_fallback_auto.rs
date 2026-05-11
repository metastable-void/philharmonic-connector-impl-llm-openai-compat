//! `tool_call_fallback_auto` dialect translation.

use crate::{
    error::Result, request::LlmGenerateRequest, response::LlmGenerateResponse,
    types::ProviderChatResponse,
};
use serde_json::{Value as JsonValue, json};

pub(crate) fn translate_request(request: &LlmGenerateRequest) -> JsonValue {
    super::tool_call_fallback::translate_request_with_tool_choice(request, json!("auto"))
}

/// Extract via the forced-tool fallback path; only request `tool_choice` differs.
pub(crate) fn extract_response(provider: &ProviderChatResponse) -> Result<LlmGenerateResponse> {
    super::tool_call_fallback::extract_response(provider)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        request::{Message, Role},
        response::StopReason,
        types::{
            ProviderChoice, ProviderFunction, ProviderMessage, ProviderToolCall, ProviderUsage,
        },
    };

    fn base_request() -> LlmGenerateRequest {
        LlmGenerateRequest {
            model: "gpt-4o-mini".to_owned(),
            messages: vec![Message {
                role: Role::User,
                content: "hello".to_owned(),
            }],
            output_schema: json!({"type": "object"}),
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
        }
    }

    #[test]
    fn translates_basic_request_to_expected_body() {
        let request = base_request();

        let body = translate_request(&request);
        assert_eq!(body["model"], json!("gpt-4o-mini"));
        assert_eq!(body["tools"][0]["type"], json!("function"));
        assert_eq!(body["tools"][0]["function"]["name"], json!("emit_output"));
        assert_eq!(body["tools"][0]["function"]["strict"], json!(true));
        assert_eq!(
            body["tools"][0]["function"]["parameters"],
            json!({"type": "object"})
        );
        assert_eq!(body["tool_choice"], json!("auto"));
    }

    #[test]
    fn translates_optional_fields_to_expected_body() {
        let request = LlmGenerateRequest {
            max_output_tokens: Some(16),
            temperature: Some(0.3),
            top_p: Some(0.7),
            stop: Some(vec!["END".to_owned()]),
            ..base_request()
        };

        let body = translate_request(&request);
        assert_eq!(body["tool_choice"], json!("auto"));
        assert_eq!(body["max_completion_tokens"], json!(16));
        assert!((body["temperature"].as_f64().unwrap() - 0.3).abs() < 1e-6);
        assert!((body["top_p"].as_f64().unwrap() - 0.7).abs() < 1e-6);
        assert_eq!(body["stop"], json!(["END"]));
    }

    #[test]
    fn extracts_tool_call_arguments_json_via_fallback_delegate() {
        let provider = ProviderChatResponse {
            choices: vec![ProviderChoice {
                finish_reason: Some("tool_calls".to_owned()),
                message: ProviderMessage {
                    content: None,
                    tool_calls: vec![ProviderToolCall {
                        function: ProviderFunction {
                            arguments: "{\"x\":3}".to_owned(),
                        },
                    }],
                },
            }],
            usage: Some(ProviderUsage {
                prompt_tokens: Some(13),
                completion_tokens: Some(17),
            }),
        };

        let out = extract_response(&provider).unwrap();
        assert_eq!(out.output, json!({"x": 3}));
        assert_eq!(out.stop_reason, StopReason::EndTurn);
        assert_eq!(out.usage.input_tokens, 13);
        assert_eq!(out.usage.output_tokens, 17);
    }

    #[test]
    fn finish_reason_mapping_delegates_to_fallback() {
        let provider = ProviderChatResponse {
            choices: vec![ProviderChoice {
                finish_reason: Some("length".to_owned()),
                message: ProviderMessage {
                    content: None,
                    tool_calls: vec![ProviderToolCall {
                        function: ProviderFunction {
                            arguments: "{}".to_owned(),
                        },
                    }],
                },
            }],
            usage: None,
        };

        let out = extract_response(&provider).unwrap();
        assert_eq!(out.output, json!({}));
        assert_eq!(out.stop_reason, StopReason::MaxTokens);
        assert_eq!(out.usage.input_tokens, 0);
        assert_eq!(out.usage.output_tokens, 0);
    }
}
