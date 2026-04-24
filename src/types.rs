//! Internal provider wire-shape types.
//!
//! These structs model the subset of OpenAI-compatible
//! `/chat/completions` response envelopes this crate needs to parse.

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct ProviderChatResponse {
    #[serde(default)]
    pub choices: Vec<ProviderChoice>,
    pub usage: Option<ProviderUsage>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct ProviderChoice {
    pub finish_reason: Option<String>,
    pub message: ProviderMessage,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct ProviderMessage {
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ProviderToolCall>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct ProviderToolCall {
    pub function: ProviderFunction,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct ProviderFunction {
    pub arguments: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct ProviderUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
}
