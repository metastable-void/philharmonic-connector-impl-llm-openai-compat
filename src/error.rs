//! Internal error model for `llm_openai_compat`.
//!
//! This module keeps implementation-local failure states typed and
//! explicit, then maps them to the connector-wide `ImplementationError`
//! boundary used by the framework.

use philharmonic_connector_impl_api::ImplementationError;

pub(crate) type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub(crate) enum Error {
    #[error("{0}")]
    InvalidConfig(String),

    #[error("{0}")]
    InvalidRequest(String),

    #[error("upstream returned non-success status {status}")]
    UpstreamNonSuccess {
        status: u16,
        body: String,
        retry_after: Option<String>,
    },

    #[error("{0}")]
    UpstreamUnreachable(String),

    #[error("upstream timeout")]
    UpstreamTimeout,

    #[error("{0}")]
    SchemaValidationFailed(String),

    #[error("{0}")]
    MalformedProviderPayload(String),

    #[error("{0}")]
    Internal(String),
}

impl From<Error> for ImplementationError {
    fn from(value: Error) -> Self {
        match value {
            Error::InvalidConfig(detail) => ImplementationError::InvalidConfig { detail },
            Error::InvalidRequest(detail) => ImplementationError::InvalidRequest { detail },
            Error::UpstreamNonSuccess { status, body, .. } => {
                ImplementationError::UpstreamError { status, body }
            }
            Error::UpstreamUnreachable(detail) => {
                ImplementationError::UpstreamUnreachable { detail }
            }
            Error::UpstreamTimeout => ImplementationError::UpstreamTimeout,
            Error::SchemaValidationFailed(detail) => {
                ImplementationError::SchemaValidationFailed { detail }
            }
            Error::MalformedProviderPayload(detail) | Error::Internal(detail) => {
                ImplementationError::Internal { detail }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_internal_variant_maps_to_wire() {
        let invalid_config = ImplementationError::from(Error::InvalidConfig("cfg".to_owned()));
        assert_eq!(
            invalid_config,
            ImplementationError::InvalidConfig {
                detail: "cfg".to_owned(),
            }
        );

        let invalid_request = ImplementationError::from(Error::InvalidRequest("req".to_owned()));
        assert_eq!(
            invalid_request,
            ImplementationError::InvalidRequest {
                detail: "req".to_owned(),
            }
        );

        let upstream = ImplementationError::from(Error::UpstreamNonSuccess {
            status: 429,
            body: "too many".to_owned(),
            retry_after: Some("1".to_owned()),
        });
        assert_eq!(
            upstream,
            ImplementationError::UpstreamError {
                status: 429,
                body: "too many".to_owned(),
            }
        );

        let unreachable = ImplementationError::from(Error::UpstreamUnreachable("dns".to_owned()));
        assert_eq!(
            unreachable,
            ImplementationError::UpstreamUnreachable {
                detail: "dns".to_owned(),
            }
        );

        let timeout = ImplementationError::from(Error::UpstreamTimeout);
        assert_eq!(timeout, ImplementationError::UpstreamTimeout);

        let schema = ImplementationError::from(Error::SchemaValidationFailed("/x".to_owned()));
        assert_eq!(
            schema,
            ImplementationError::SchemaValidationFailed {
                detail: "/x".to_owned(),
            }
        );

        let malformed =
            ImplementationError::from(Error::MalformedProviderPayload("bad payload".to_owned()));
        assert_eq!(
            malformed,
            ImplementationError::Internal {
                detail: "bad payload".to_owned(),
            }
        );

        let internal = ImplementationError::from(Error::Internal("boom".to_owned()));
        assert_eq!(
            internal,
            ImplementationError::Internal {
                detail: "boom".to_owned(),
            }
        );
    }
}
