//! Dialect translation/extraction dispatch.

pub(crate) mod openai_native;
pub(crate) mod tool_call_fallback;
pub(crate) mod vllm_native;

use crate::{
    config::Dialect,
    error::{Error, Result},
    request::LlmGenerateRequest,
    response::{LlmGenerateResponse, Usage},
    types::{ProviderChatResponse, ProviderChoice},
};
use serde_json::Value as JsonValue;

pub(crate) fn build_request_body(dialect: Dialect, request: &LlmGenerateRequest) -> JsonValue {
    match dialect {
        Dialect::OpenaiNative => openai_native::translate_request(request),
        Dialect::VllmNative => vllm_native::translate_request(request),
        Dialect::ToolCallFallback => tool_call_fallback::translate_request(request),
    }
}

pub(crate) fn extract_response(
    dialect: Dialect,
    provider_body: &JsonValue,
) -> Result<LlmGenerateResponse> {
    let provider = parse_provider_response(provider_body)?;
    match dialect {
        Dialect::OpenaiNative => openai_native::extract_response(&provider),
        Dialect::VllmNative => vllm_native::extract_response(&provider),
        Dialect::ToolCallFallback => tool_call_fallback::extract_response(&provider),
    }
}

fn parse_provider_response(provider_body: &JsonValue) -> Result<ProviderChatResponse> {
    serde_json::from_value(provider_body.clone()).map_err(|e| {
        Error::MalformedProviderPayload(format!("provider response envelope is malformed: {e}"))
    })
}

pub(crate) fn first_choice(provider: &ProviderChatResponse) -> Result<&ProviderChoice> {
    provider.choices.first().ok_or_else(|| {
        Error::MalformedProviderPayload("provider response has no choices[0]".to_owned())
    })
}

pub(crate) fn parse_output_json(source: &str, payload: &str) -> Result<JsonValue> {
    serde_json::from_str(payload).map_err(|e| {
        Error::MalformedProviderPayload(format!("failed to parse {source} as JSON: {e}"))
    })
}

pub(crate) fn normalized_usage(provider: &ProviderChatResponse) -> Result<Usage> {
    let input_tokens = convert_usage(provider.usage.as_ref().and_then(|u| u.prompt_tokens))?;
    let output_tokens = convert_usage(provider.usage.as_ref().and_then(|u| u.completion_tokens))?;

    Ok(Usage {
        input_tokens,
        output_tokens,
    })
}

fn convert_usage(value: Option<u64>) -> Result<u32> {
    let value = value.unwrap_or(0);
    u32::try_from(value).map_err(|_| {
        Error::MalformedProviderPayload(format!("usage token count {value} exceeds u32 range"))
    })
}
