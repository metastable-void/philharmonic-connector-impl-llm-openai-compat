//! `vllm_native` dialect translation.

use crate::{
    dialect::{first_choice, normalized_usage, parse_output_json},
    error::{Error, Result},
    request::LlmGenerateRequest,
    response::{LlmGenerateResponse, StopReason},
    types::ProviderChatResponse,
};
use serde_json::{Map, Value as JsonValue, json};

pub(crate) fn translate_request(request: &LlmGenerateRequest) -> JsonValue {
    let mut body = Map::new();
    body.insert("model".to_owned(), json!(request.model));
    body.insert("messages".to_owned(), json!(request.messages));
    body.insert(
        "structured_outputs".to_owned(),
        json!({
            "json": request.output_schema,
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

    let content = choice.message.content.as_deref().ok_or_else(|| {
        Error::MalformedProviderPayload("choices[0].message.content missing".to_owned())
    })?;

    let output = parse_output_json("choices[0].message.content", content)?;
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
        "stop_sequence" => StopReason::StopSequence,
        "tool_calls" => StopReason::Error,
        _ => StopReason::Error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        request::{Message, Role},
        types::{ProviderChoice, ProviderMessage, ProviderUsage},
    };

    #[test]
    fn translates_basic_request_to_expected_body() {
        let request = LlmGenerateRequest {
            model: "HuggingFaceH4/zephyr-7b-beta".to_owned(),
            messages: vec![Message {
                role: Role::User,
                content: "hello".to_owned(),
            }],
            output_schema: json!({"type": "object"}),
            max_output_tokens: Some(64),
            temperature: Some(0.1),
            top_p: Some(0.9),
            stop: Some(vec!["END".to_owned()]),
        };

        let body = translate_request(&request);
        assert_eq!(body["model"], json!("HuggingFaceH4/zephyr-7b-beta"));
        assert_eq!(
            body["structured_outputs"]["json"],
            json!({"type": "object"})
        );
        assert_eq!(body["max_completion_tokens"], json!(64));
        assert!((body["temperature"].as_f64().unwrap() - 0.1).abs() < 1e-6);
        assert!((body["top_p"].as_f64().unwrap() - 0.9).abs() < 1e-6);
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
        assert_eq!(map_finish_reason("tool_calls"), StopReason::Error);
        assert_eq!(map_finish_reason("other"), StopReason::Error);
    }

    #[test]
    fn extracts_content_json() {
        let provider = ProviderChatResponse {
            choices: vec![ProviderChoice {
                finish_reason: Some("stop".to_owned()),
                message: ProviderMessage {
                    content: Some("{\"x\":2}".to_owned()),
                    tool_calls: Vec::new(),
                },
            }],
            usage: Some(ProviderUsage {
                prompt_tokens: Some(7),
                completion_tokens: Some(11),
            }),
        };

        let out = extract_response(&provider).unwrap();
        assert_eq!(out.output, json!({"x": 2}));
        assert_eq!(out.stop_reason, StopReason::EndTurn);
        assert_eq!(out.usage.input_tokens, 7);
        assert_eq!(out.usage.output_tokens, 11);
    }
}
