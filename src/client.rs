//! HTTP client plumbing for `llm_openai_compat`.
//!
//! This module owns reqwest client construction and one-attempt POST
//! execution against `/chat/completions`.

use crate::{
    config::LlmOpenaiCompatConfig,
    error::{Error, Result},
};
use serde_json::Value as JsonValue;
use std::time::Duration;

pub(crate) fn build_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .build()
        .map_err(|e| Error::Internal(format!("failed to build reqwest client: {e}")))
}

pub(crate) async fn execute_one_attempt(
    client: &reqwest::Client,
    config: &LlmOpenaiCompatConfig,
    body: &JsonValue,
) -> Result<JsonValue> {
    let url = chat_completions_url(&config.base_url);

    let response = client
        .post(url)
        .timeout(Duration::from_millis(config.timeout_ms))
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", config.api_key),
        )
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .json(body)
        .send()
        .await
        .map_err(map_reqwest_error)?;

    let status = response.status().as_u16();
    let headers = response.headers().clone();
    let bytes = response.bytes().await.map_err(map_reqwest_error)?;

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

fn map_reqwest_error(error: reqwest::Error) -> Error {
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

fn retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers.get(reqwest::header::RETRY_AFTER).map(|value| {
        value
            .to_str()
            .map(str::to_owned)
            .unwrap_or_else(|_| String::from_utf8_lossy(value.as_bytes()).into_owned())
    })
}
