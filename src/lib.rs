//! OpenAI-compatible LLM connector implementation for Philharmonic.
//!
//! `llm_openai_compat` implements the shared
//! [`philharmonic_connector_impl_api::Implementation`] trait for the
//! normalized `llm_generate` wire protocol described in the workspace
//! connector architecture docs. It translates normalized requests into
//! OpenAI-compatible `/chat/completions` payloads and normalizes
//! provider responses back into `{output, stop_reason, usage}`.
//!
//! ## Dialects
//!
//! - `openai_native`: uses `response_format: {type: "json_schema"}`.
//! - `vllm_native`: uses top-level `structured_outputs: {json: ...}`.
//! - `tool_call_fallback`: uses a forced synthetic function call and
//!   parses output from `tool_calls[0].function.arguments`.
//!
//! ## Validation and retries
//!
//! The implementation compiles `output_schema` with Draft 2020-12 via
//! `jsonschema`, validates provider output before returning, and applies
//! a minimal hardcoded retry loop for retryable upstream failures
//! (429/5xx/network/timeout).

mod client;
mod config;
mod dialect;
mod error;
mod request;
mod response;
mod retry;
mod schema;
mod types;

pub use crate::config::{Dialect, LlmOpenaiCompatConfig};
pub use crate::request::{LlmGenerateRequest, Message, Role};
pub use crate::response::{LlmGenerateResponse, StopReason, Usage};
pub use philharmonic_connector_impl_api::{
    ConnectorCallContext, Implementation, ImplementationError, JsonValue, async_trait,
};

const NAME: &str = "llm_openai_compat";

/// `llm_openai_compat` connector implementation.
#[derive(Clone, Debug)]
pub struct LlmOpenaiCompat {
    client: reqwest::Client,
}

impl LlmOpenaiCompat {
    /// Builds an instance with a workspace-standard reqwest client.
    pub fn new() -> Result<Self, ImplementationError> {
        let client = client::build_client().map_err(ImplementationError::from)?;
        Ok(Self { client })
    }

    /// Builds an instance with an externally-configured reqwest client.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl Implementation for LlmOpenaiCompat {
    fn name(&self) -> &str {
        NAME
    }

    async fn execute(
        &self,
        config: &JsonValue,
        request: &JsonValue,
        _ctx: &ConnectorCallContext,
    ) -> Result<JsonValue, ImplementationError> {
        let config: LlmOpenaiCompatConfig = serde_json::from_value(config.clone())
            .map_err(|e| error::Error::InvalidConfig(e.to_string()))
            .map_err(ImplementationError::from)?;

        let request: LlmGenerateRequest = serde_json::from_value(request.clone())
            .map_err(|e| error::Error::InvalidRequest(e.to_string()))
            .map_err(ImplementationError::from)?;

        let compiled_schema =
            schema::compile(&request.output_schema).map_err(ImplementationError::from)?;

        let provider_request = dialect::build_request_body(config.dialect, &request);

        let provider_response = retry::execute_with_retry(|| {
            client::execute_one_attempt(&self.client, &config, &provider_request)
        })
        .await
        .map_err(ImplementationError::from)?;

        let response = dialect::extract_response(config.dialect, &provider_response)
            .map_err(ImplementationError::from)?;

        schema::validate(&compiled_schema, &response.output).map_err(ImplementationError::from)?;

        serde_json::to_value(response)
            .map_err(|e| error::Error::Internal(e.to_string()))
            .map_err(ImplementationError::from)
    }
}
