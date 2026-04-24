//! `tool_call_fallback` dialect translation.

use crate::{
    dialect::{first_choice, normalized_usage, parse_output_json},
    error::{Error, Result},
    request::LlmGenerateRequest,
    response::{LlmGenerateResponse, StopReason},
    types::ProviderChatResponse,
};
use serde_json::{Map, Value as JsonValue, json};

const TOOL_NAME: &str = "emit_output";

pub(crate) fn translate_request(request: &LlmGenerateRequest) -> JsonValue {
    let mut body = Map::new();
    body.insert("model".to_owned(), json!(request.model));
    body.insert("messages".to_owned(), json!(request.messages));
    body.insert(
        "tools".to_owned(),
        json!([
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": "Produce the structured output.",
                    "strict": true,
                    "parameters": request.output_schema,
                }
            }
        ]),
    );
    body.insert(
        "tool_choice".to_owned(),
        json!({
            "type": "function",
            "function": {
                "name": TOOL_NAME,
            }
        }),
    );

    if let Some(value) = request.max_output_tokens {
        body.insert("max_completion_tokens".to_owned(), json!(value));
    }
    if let Some(value) = request.temperature {
        body.insert("temperature".to_owned(), json!(value));
    }
    if let Some(value) = request.top_p {
        body.insert("top_p".to_owned(), json!(value));
    }
    if let Some(value) = &request.stop {
        body.insert("stop".to_owned(), json!(value));
    }

    JsonValue::Object(body)
}

pub(crate) fn extract_response(provider: &ProviderChatResponse) -> Result<LlmGenerateResponse> {
    let choice = first_choice(provider)?;
    let finish_reason = choice.finish_reason.as_deref().ok_or_else(|| {
        Error::MalformedProviderPayload("choices[0].finish_reason missing".to_owned())
    })?;

    let tool_call = choice.message.tool_calls.first().ok_or_else(|| {
        Error::MalformedProviderPayload("choices[0].message.tool_calls[0] missing".to_owned())
    })?;

    let output = parse_output_json(
        "choices[0].message.tool_calls[0].function.arguments",
        &tool_call.function.arguments,
    )?;

    let usage = normalized_usage(provider)?;

    Ok(LlmGenerateResponse {
        output,
        stop_reason: map_finish_reason(finish_reason),
        usage,
    })
}

pub(crate) fn map_finish_reason(finish_reason: &str) -> StopReason {
    match finish_reason {
        "stop" => StopReason::EndTurn,
        "length" => StopReason::MaxTokens,
        "content_filter" => StopReason::ContentFilter,
        "tool_calls" => StopReason::EndTurn,
        "stop_sequence" => StopReason::StopSequence,
        _ => StopReason::Error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        request::{Message, Role},
        types::{
            ProviderChoice, ProviderFunction, ProviderMessage, ProviderToolCall, ProviderUsage,
        },
    };

    #[test]
    fn translates_basic_request_to_expected_body() {
        let request = LlmGenerateRequest {
            model: "gpt-4o-mini".to_owned(),
            messages: vec![Message {
                role: Role::User,
                content: "hello".to_owned(),
            }],
            output_schema: json!({"type": "object"}),
            max_output_tokens: Some(16),
            temperature: Some(0.3),
            top_p: Some(0.7),
            stop: Some(vec!["END".to_owned()]),
        };

        let body = translate_request(&request);
        assert_eq!(body["model"], json!("gpt-4o-mini"));
        assert_eq!(body["tools"][0]["type"], json!("function"));
        assert_eq!(body["tools"][0]["function"]["name"], json!("emit_output"));
        assert_eq!(body["tools"][0]["function"]["strict"], json!(true));
        assert_eq!(
            body["tools"][0]["function"]["parameters"],
            json!({"type": "object"})
        );
        assert_eq!(
            body["tool_choice"]["function"]["name"],
            json!("emit_output")
        );
        assert_eq!(body["max_completion_tokens"], json!(16));
        assert!((body["temperature"].as_f64().unwrap() - 0.3).abs() < 1e-6);
        assert!((body["top_p"].as_f64().unwrap() - 0.7).abs() < 1e-6);
        assert_eq!(body["stop"], json!(["END"]));
    }

    #[test]
    fn finish_reason_maps_per_table() {
        assert_eq!(map_finish_reason("stop"), StopReason::EndTurn);
        assert_eq!(map_finish_reason("length"), StopReason::MaxTokens);
        assert_eq!(
            map_finish_reason("content_filter"),
            StopReason::ContentFilter
        );
        assert_eq!(map_finish_reason("tool_calls"), StopReason::EndTurn);
        assert_eq!(map_finish_reason("other"), StopReason::Error);
    }

    #[test]
    fn extracts_tool_call_arguments_json() {
        let provider = ProviderChatResponse {
            choices: vec![ProviderChoice {
                finish_reason: Some("stop".to_owned()),
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
}
