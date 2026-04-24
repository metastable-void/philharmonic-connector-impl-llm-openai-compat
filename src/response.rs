//! Normalized `llm_generate` response model for `llm_openai_compat`.
//!
//! Dialect parsers produce this provider-agnostic output shape after
//! extracting and validating provider `/chat/completions` responses.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// One normalized generation response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmGenerateResponse {
    /// Parsed JSON output object/array/value from the provider.
    pub output: JsonValue,
    /// Normalized stop reason.
    pub stop_reason: StopReason,
    /// Normalized usage counters.
    pub usage: Usage,
}

/// Normalized stop-reason set for connector consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Model completed normally.
    EndTurn,
    /// Generation stopped due to token limit.
    MaxTokens,
    /// Generation stopped due to stop-sequence match.
    StopSequence,
    /// Provider content filter halted output.
    ContentFilter,
    /// Provider returned an unknown/unsupported stop condition.
    Error,
}

/// Normalized token usage counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt/input tokens.
    pub input_tokens: u32,
    /// Completion/output tokens.
    pub output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_reason_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&StopReason::EndTurn).unwrap(),
            "\"end_turn\""
        );
        assert_eq!(
            serde_json::to_string(&StopReason::MaxTokens).unwrap(),
            "\"max_tokens\""
        );
        assert_eq!(
            serde_json::to_string(&StopReason::StopSequence).unwrap(),
            "\"stop_sequence\""
        );
        assert_eq!(
            serde_json::to_string(&StopReason::ContentFilter).unwrap(),
            "\"content_filter\""
        );
        assert_eq!(
            serde_json::to_string(&StopReason::Error).unwrap(),
            "\"error\""
        );
    }
}
