//! Configuration model for `llm_openai_compat`.
//!
//! The config carries provider location/auth and the dialect selector
//! that controls request/response translation behavior.

/// Top-level config payload for the `llm_openai_compat` implementation.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LlmOpenaiCompatConfig {
    /// Base URL of the provider endpoint, e.g. `https://api.openai.com/v1`.
    pub base_url: String,
    /// API key sent in `Authorization: Bearer <api_key>`.
    pub api_key: String,
    /// Request/response translation mode.
    pub dialect: Dialect,
    /// Per-attempt timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

const fn default_timeout_ms() -> u64 {
    60_000
}

/// OpenAI-compatible dialect selection.
#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Dialect {
    /// Native OpenAI `response_format: {type: "json_schema"}` mode.
    OpenaiNative,
    /// Native vLLM `structured_outputs: {json: ...}` mode.
    VllmNative,
    /// OpenAI tool-call fallback that carries output via function args.
    ToolCallFallback,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value as JsonValue, json};

    #[test]
    fn deserialize_rejects_unknown_fields() {
        let value = json!({
            "base_url": "https://example.test/v1",
            "api_key": "secret",
            "dialect": "openai_native",
            "timeout_ms": 1000,
            "extra": true
        });

        let err = serde_json::from_value::<LlmOpenaiCompatConfig>(value).unwrap_err();
        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn dialect_enum_roundtrips_all_three() {
        let cases = [
            Dialect::OpenaiNative,
            Dialect::VllmNative,
            Dialect::ToolCallFallback,
        ];

        for dialect in cases {
            let encoded = serde_json::to_value(dialect).unwrap();
            let decoded: Dialect = serde_json::from_value(encoded.clone()).unwrap();
            assert_eq!(decoded, dialect);
        }
    }

    #[test]
    fn default_timeout_ms_is_60000() {
        let value: JsonValue = json!({
            "base_url": "https://example.test/v1",
            "api_key": "secret",
            "dialect": "openai_native"
        });

        let config: LlmOpenaiCompatConfig = serde_json::from_value(value).unwrap();
        assert_eq!(config.timeout_ms, 60_000);
    }
}
