//! Configuration model for `llm_openai_compat`.
//!
//! The config carries provider location/auth and the dialect selector
//! that controls request/response translation behavior.

use crate::error::CustomHeaderError;
use std::collections::BTreeMap;

/// Top-level config payload for the `llm_openai_compat` implementation.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(try_from = "LlmOpenaiCompatConfigRaw")]
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
    /// Caller-supplied HTTP headers to attach to every upstream
    /// request. Useful for per-provider knobs (e.g. Hugging
    /// Face Inference's `X-HF-Bill-To` for org billing,
    /// OpenAI's `OpenAI-Organization` / `OpenAI-Project`,
    /// OpenRouter's `HTTP-Referer` / `X-Title`). Reserved
    /// headers and malformed values are rejected at deserialize
    /// time; see `validate_custom_headers`.
    #[serde(default)]
    pub custom_headers: BTreeMap<String, String>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LlmOpenaiCompatConfigRaw {
    base_url: String,
    api_key: String,
    dialect: Dialect,
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,
    #[serde(default)]
    custom_headers: BTreeMap<String, String>,
}

impl TryFrom<LlmOpenaiCompatConfigRaw> for LlmOpenaiCompatConfig {
    type Error = CustomHeaderError;

    fn try_from(raw: LlmOpenaiCompatConfigRaw) -> Result<Self, Self::Error> {
        validate_custom_headers(&raw.custom_headers)?;
        Ok(Self {
            base_url: raw.base_url,
            api_key: raw.api_key,
            dialect: raw.dialect,
            timeout_ms: raw.timeout_ms,
            custom_headers: raw.custom_headers,
        })
    }
}

const fn default_timeout_ms() -> u64 {
    60_000
}

pub(crate) fn validate_custom_headers(
    custom_headers: &BTreeMap<String, String>,
) -> Result<(), CustomHeaderError> {
    for (name, value) in custom_headers {
        validate_custom_header_name(name)?;
        validate_custom_header_value(name, value)?;
    }

    Ok(())
}

fn validate_custom_header_name(name: &str) -> Result<(), CustomHeaderError> {
    if name.is_empty() {
        return Err(custom_header_error(name, "empty name"));
    }

    if is_reserved_header_name(name) {
        return Err(custom_header_error(name, "reserved name"));
    }

    if !name.bytes().all(is_http_token_char) {
        return Err(custom_header_error(
            name,
            "name has invalid token character",
        ));
    }

    Ok(())
}

fn validate_custom_header_value(name: &str, value: &str) -> Result<(), CustomHeaderError> {
    if value.bytes().any(|byte| matches!(byte, b'\r' | b'\n')) {
        return Err(custom_header_error(name, "contains CR/LF"));
    }

    if value.bytes().any(is_disallowed_header_value_control) {
        return Err(custom_header_error(
            name,
            "value has invalid control character",
        ));
    }

    Ok(())
}

fn custom_header_error(name: &str, reason: &str) -> CustomHeaderError {
    CustomHeaderError {
        name: name.to_owned(),
        reason: reason.to_owned(),
    }
}

fn is_reserved_header_name(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "authorization"
            | "content-type"
            | "content-length"
            | "host"
            | "transfer-encoding"
            | "connection"
    )
}

fn is_http_token_char(byte: u8) -> bool {
    matches!(
        byte,
        b'!' | b'#'
            | b'$'
            | b'%'
            | b'&'
            | b'\''
            | b'*'
            | b'+'
            | b'-'
            | b'.'
            | b'^'
            | b'_'
            | b'`'
            | b'|'
            | b'~'
            | b'0'..=b'9'
            | b'a'..=b'z'
            | b'A'..=b'Z'
    )
}

fn is_disallowed_header_value_control(byte: u8) -> bool {
    matches!(byte, 0x00..=0x08 | 0x0b..=0x0c | 0x0e..=0x1f | 0x7f)
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
    /// OpenAI tool-call fallback using provider-selected tool choice.
    ToolCallFallbackAuto,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value as JsonValue, json};

    fn base_config(custom_headers: JsonValue) -> JsonValue {
        json!({
            "base_url": "https://example.test/v1",
            "api_key": "secret",
            "dialect": "openai_native",
            "timeout_ms": 1000,
            "custom_headers": custom_headers
        })
    }

    fn parse_custom_headers(custom_headers: JsonValue) -> LlmOpenaiCompatConfig {
        serde_json::from_value(base_config(custom_headers)).unwrap()
    }

    fn assert_invalid_custom_header(
        custom_headers: JsonValue,
        expected_name: &str,
        expected_reason: &str,
    ) {
        let err = serde_json::from_value::<LlmOpenaiCompatConfig>(base_config(custom_headers))
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains(&format!("custom header {expected_name:?} is invalid")));
        assert!(
            message.contains(expected_reason),
            "error message {message:?} did not contain {expected_reason:?}"
        );
    }

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
    fn dialect_enum_roundtrips_all_four() {
        let cases = [
            Dialect::OpenaiNative,
            Dialect::VllmNative,
            Dialect::ToolCallFallback,
            Dialect::ToolCallFallbackAuto,
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

    #[test]
    fn default_custom_headers_is_empty_when_omitted() {
        let value: JsonValue = json!({
            "base_url": "https://example.test/v1",
            "api_key": "secret",
            "dialect": "openai_native"
        });

        let config: LlmOpenaiCompatConfig = serde_json::from_value(value).unwrap();
        assert!(config.custom_headers.is_empty());
    }

    #[test]
    fn accepts_well_formed_custom_headers() {
        let config = parse_custom_headers(json!({
            "X-HF-Bill-To": "org_abc"
        }));

        assert_eq!(
            config.custom_headers.get("X-HF-Bill-To"),
            Some(&"org_abc".to_owned())
        );
    }

    #[test]
    fn accepts_multiple_custom_headers() {
        let config = parse_custom_headers(json!({
            "HTTP-Referer": "https://example.test",
            "OpenAI-Organization": "org_abc",
            "X-HF-Bill-To": "billing_org"
        }));

        assert_eq!(config.custom_headers.len(), 3);
    }

    #[test]
    fn rejects_reserved_authorization_case_insensitive() {
        for name in ["Authorization", "authorization", "AUTHORIZATION"] {
            assert_invalid_custom_header(json!({ name: "secret" }), name, "reserved name");
        }
    }

    #[test]
    fn rejects_reserved_content_type() {
        assert_invalid_custom_header(
            json!({ "Content-Type": "application/json" }),
            "Content-Type",
            "reserved name",
        );
    }

    #[test]
    fn rejects_reserved_content_length() {
        assert_invalid_custom_header(
            json!({ "Content-Length": "10" }),
            "Content-Length",
            "reserved name",
        );
    }

    #[test]
    fn rejects_reserved_host() {
        assert_invalid_custom_header(json!({ "Host": "example.test" }), "Host", "reserved name");
    }

    #[test]
    fn rejects_reserved_transfer_encoding() {
        assert_invalid_custom_header(
            json!({ "Transfer-Encoding": "chunked" }),
            "Transfer-Encoding",
            "reserved name",
        );
    }

    #[test]
    fn rejects_reserved_connection() {
        assert_invalid_custom_header(
            json!({ "Connection": "close" }),
            "Connection",
            "reserved name",
        );
    }

    #[test]
    fn rejects_crlf_in_value() {
        assert_invalid_custom_header(
            json!({ "X-Foo": "bad\r\nInjected: yes" }),
            "X-Foo",
            "contains CR/LF",
        );
    }

    #[test]
    fn rejects_lf_only_in_value() {
        assert_invalid_custom_header(json!({ "X-Foo": "bad\nthing" }), "X-Foo", "contains CR/LF");
    }

    #[test]
    fn rejects_cr_only_in_value() {
        assert_invalid_custom_header(json!({ "X-Foo": "bad\rthing" }), "X-Foo", "contains CR/LF");
    }

    #[test]
    fn rejects_control_char_in_value() {
        assert_invalid_custom_header(
            json!({ "X-Foo": "bad\u{0001}" }),
            "X-Foo",
            "value has invalid control character",
        );
    }

    #[test]
    fn rejects_empty_name() {
        assert_invalid_custom_header(json!({ "": "value" }), "", "empty name");
    }

    #[test]
    fn rejects_invalid_token_char_in_name() {
        for name in ["X Foo", "X:Foo", "日本語"] {
            assert_invalid_custom_header(
                json!({ name: "value" }),
                name,
                "name has invalid token character",
            );
        }
    }

    #[test]
    fn accepts_empty_value() {
        let config = parse_custom_headers(json!({
            "X-Empty": ""
        }));

        assert_eq!(config.custom_headers.get("X-Empty"), Some(&String::new()));
    }

    #[test]
    fn accepts_horizontal_tab_in_value() {
        let config = parse_custom_headers(json!({
            "X-Tab": "before\tafter"
        }));

        assert_eq!(
            config.custom_headers.get("X-Tab"),
            Some(&"before\tafter".to_owned())
        );
    }
}
