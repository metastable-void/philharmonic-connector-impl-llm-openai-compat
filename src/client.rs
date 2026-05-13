//! HTTP client plumbing for `llm_openai_compat`.
//!
//! Owns `mechanics_http_client::Client` construction and one-attempt
//! POST execution against `/chat/completions`.

use crate::{
    config::LlmOpenaiCompatConfig,
    error::{Error, Result},
};
use mechanics_http_client::{Client, Error as HttpError, HeaderMap, HeaderName, HeaderValue};
use serde_json::Value as JsonValue;
use std::time::Duration;

pub(crate) fn build_client() -> Result<Client> {
    Client::new().map_err(|e| Error::Internal(format!("failed to build HTTP client: {e}")))
}

pub(crate) async fn execute_one_attempt(
    client: &Client,
    config: &LlmOpenaiCompatConfig,
    body: &JsonValue,
) -> Result<JsonValue> {
    let url = chat_completions_url(&config.base_url);

    let mut request = client
        .post(url)
        .timeout(Duration::from_millis(config.timeout_ms))
        .bearer_auth(&config.api_key)
        .header("content-type", "application/json");

    for (name, value) in &config.custom_headers {
        let header_name = HeaderName::try_from(name.as_str()).map_err(|e| {
            Error::Internal(format!("invalid custom_header could not be applied: {e}"))
        })?;
        let header_value = HeaderValue::try_from(value).map_err(|e| {
            Error::Internal(format!("invalid custom_header could not be applied: {e}"))
        })?;
        request = request.header(header_name, header_value);
    }

    let response = request.json(body).send().await.map_err(map_http_error)?;

    let status = response.status().as_u16();
    let headers = response.headers().clone();
    let bytes = response.bytes().await.map_err(map_http_error)?;

    if !(200..=299).contains(&status) {
        let body = String::from_utf8_lossy(&bytes).into_owned();
        return Err(Error::UpstreamNonSuccess {
            status,
            body,
            retry_after: retry_after_header(&headers),
        });
    }

    serde_json::from_slice::<JsonValue>(&bytes).map_err(|e| {
        Error::MalformedProviderPayload(format!("provider response was not valid JSON: {e}"))
    })
}

fn map_http_error(error: HttpError) -> Error {
    if error.is_timeout() {
        Error::UpstreamTimeout
    } else {
        Error::UpstreamUnreachable(error.to_string())
    }
}

fn chat_completions_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    format!("{trimmed}/chat/completions")
}

fn retry_after_header(headers: &HeaderMap) -> Option<String> {
    headers.get("retry-after").map(|value| {
        value
            .to_str()
            .map(str::to_owned)
            .unwrap_or_else(|_| String::from_utf8_lossy(value.as_bytes()).into_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Dialect;
    use std::collections::BTreeMap;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path},
    };

    #[tokio::test]
    async fn execute_one_attempt_applies_custom_headers() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("X-HF-Bill-To", "org_abc"))
            .and(header("OpenAI-Organization", "org_openai"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(serde_json::json!({"choices": []})),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = build_client().unwrap();
        let config = LlmOpenaiCompatConfig {
            base_url: server.uri(),
            api_key: "test-key".to_owned(),
            dialect: Dialect::OpenaiNative,
            timeout_ms: 500,
            custom_headers: BTreeMap::from([
                ("OpenAI-Organization".to_owned(), "org_openai".to_owned()),
                ("X-HF-Bill-To".to_owned(), "org_abc".to_owned()),
            ]),
        };

        execute_one_attempt(&client, &config, &serde_json::json!({"messages": []}))
            .await
            .unwrap();
    }
}
