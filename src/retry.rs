//! Retry execution for `llm_openai_compat`.
//!
//! v1 uses a hardcoded minimal policy: 3 attempts total, retrying only
//! on 429/5xx, network errors, and per-attempt timeouts.

use crate::error::{Error, Result};
use std::{
    future::Future,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const MAX_ATTEMPTS: usize = 3;
const BASE_BACKOFF_MS: u64 = 1_000;
const CAP_BACKOFF_MS: u64 = 8_000;

pub(crate) async fn execute_with_retry<T, Op, Fut>(mut operation: Op) -> Result<T>
where
    Op: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    for attempt in 0..MAX_ATTEMPTS {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                if !should_retry(&err) || attempt + 1 >= MAX_ATTEMPTS {
                    return Err(err);
                }

                let max_delay_ms = exponential_backoff_ms(attempt);
                let jitter_ms = draw_jitter_ms(max_delay_ms);
                let delay = compute_retry_delay(&err, jitter_ms);

                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    Err(Error::UpstreamTimeout)
}

pub(crate) fn should_retry(err: &Error) -> bool {
    match err {
        Error::UpstreamTimeout | Error::UpstreamUnreachable(_) => true,
        Error::UpstreamNonSuccess { status, .. } => *status == 429 || (500..=599).contains(status),
        Error::InvalidConfig(_)
        | Error::InvalidRequest(_)
        | Error::SchemaValidationFailed(_)
        | Error::MalformedProviderPayload(_)
        | Error::Internal(_) => false,
    }
}

fn exponential_backoff_ms(attempt: usize) -> u64 {
    let exponent = u32::try_from(attempt).unwrap_or(u32::MAX);
    let factor = 2_u64.saturating_pow(exponent);
    BASE_BACKOFF_MS.saturating_mul(factor).min(CAP_BACKOFF_MS)
}

fn compute_retry_delay(err: &Error, jitter_ms: u64) -> Duration {
    if let Error::UpstreamNonSuccess {
        status: 429,
        retry_after,
        ..
    } = err
        && let Some(value) = retry_after
        && let Some(seconds) = parse_retry_after_seconds(value)
    {
        return Duration::from_secs(seconds);
    }

    Duration::from_millis(jitter_ms)
}

fn parse_retry_after_seconds(value: &str) -> Option<u64> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<u64>().ok()
}

fn draw_jitter_ms(max_ms: u64) -> u64 {
    if max_ms == 0 {
        return 0;
    }

    let modulus = match max_ms.checked_add(1) {
        Some(value) => value,
        None => return max_ms,
    };

    next_pseudo_random_u64() % modulus
}

fn next_pseudo_random_u64() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1);

    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_nanos(),
        Err(_) => 0,
    };

    let nanos_bytes = nanos.to_le_bytes();
    let mut low = [0_u8; 8];
    low.copy_from_slice(&nanos_bytes[..8]);
    let mut state = u64::from_le_bytes(low) ^ counter ^ 0x9E37_79B9_7F4A_7C15;

    // SplitMix64 for a fast, dependency-free pseudo-random sample.
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_backoff_respects_cap() {
        assert_eq!(exponential_backoff_ms(0), 1_000);
        assert_eq!(exponential_backoff_ms(1), 2_000);
        assert_eq!(exponential_backoff_ms(2), 4_000);
        assert_eq!(exponential_backoff_ms(3), 8_000);
        assert_eq!(exponential_backoff_ms(10), 8_000);
    }

    #[test]
    fn retry_after_seconds_parsed() {
        assert_eq!(parse_retry_after_seconds("2"), Some(2));
        assert_eq!(parse_retry_after_seconds(" 5 "), Some(5));
        assert_eq!(parse_retry_after_seconds(""), None);
    }

    #[test]
    fn retry_after_http_date_is_ignored() {
        assert_eq!(
            parse_retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT"),
            None
        );
    }

    #[test]
    fn full_jitter_is_bounded() {
        for _ in 0..100 {
            let jitter = draw_jitter_ms(250);
            assert!(jitter <= 250);
        }
    }

    #[test]
    fn retryability_matches_policy() {
        assert!(should_retry(&Error::UpstreamTimeout));
        assert!(should_retry(&Error::UpstreamUnreachable("io".to_owned())));
        assert!(should_retry(&Error::UpstreamNonSuccess {
            status: 429,
            body: "{}".to_owned(),
            retry_after: None,
        }));
        assert!(should_retry(&Error::UpstreamNonSuccess {
            status: 500,
            body: "{}".to_owned(),
            retry_after: None,
        }));
        assert!(!should_retry(&Error::UpstreamNonSuccess {
            status: 401,
            body: "{}".to_owned(),
            retry_after: None,
        }));
        assert!(!should_retry(&Error::InvalidRequest("bad".to_owned())));
    }

    #[test]
    fn retry_after_overrides_jitter_for_429() {
        let err = Error::UpstreamNonSuccess {
            status: 429,
            body: "{}".to_owned(),
            retry_after: Some("3".to_owned()),
        };
        let delay = compute_retry_delay(&err, 17);
        assert_eq!(delay, Duration::from_secs(3));
    }
}
