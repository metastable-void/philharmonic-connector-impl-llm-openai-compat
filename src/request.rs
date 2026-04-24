//! Normalized `llm_generate` request model for `llm_openai_compat`.
//!
//! This is the script-facing request shape from the connector wire
//! protocol. Dialect modules translate this into provider-native
//! `/chat/completions` request bodies.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// One normalized `llm_generate` request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmGenerateRequest {
    /// Provider model identifier.
    pub model: String,
    /// Conversation messages in chronological order.
    pub messages: Vec<Message>,
    /// Required output JSON Schema.
    pub output_schema: JsonValue,
    /// Optional maximum generation token count.
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    /// Optional sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Optional nucleus-sampling parameter.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Optional stop sequences.
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

/// One normalized chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Message role.
    pub role: Role,
    /// UTF-8 message content.
    pub content: String,
}

/// Supported role values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// System instruction message.
    System,
    /// End-user message.
    User,
    /// Assistant message.
    Assistant,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserialize_rejects_unknown_generation_knobs() {
        let value = json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "output_schema": {"type": "object"},
            "temperature": 0.2,
            "unknown_knob": true
        });

        let err = serde_json::from_value::<LlmGenerateRequest>(value).unwrap_err();
        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn role_enum_roundtrips_system_user_assistant() {
        let cases = [Role::System, Role::User, Role::Assistant];
        for role in cases {
            let encoded = serde_json::to_value(role).unwrap();
            let decoded: Role = serde_json::from_value(encoded).unwrap();
            assert_eq!(decoded, role);
        }
    }
}
