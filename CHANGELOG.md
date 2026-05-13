# Changelog

All notable changes to this crate are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this crate adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-13

Changed (breaking): outbound HTTP is now driven by `mechanics-http-client`
(hyper-rustls + webpki-roots + aws-lc-rs) instead of `reqwest`.

- `LlmOpenaiCompat::with_client(reqwest::Client)` →
  `LlmOpenaiCompat::with_client(mechanics_http_client::Client)`.
- Removed: the `reqwest` dependency.
- Disabled `jsonschema`'s default features (we don't use its
  `resolve-http` path, which pulled `reqwest` transitively).
- Trust posture: TLS root store is the bundled Mozilla CA bundle
  (`webpki-roots`) only — no OS-native trust, no
  `rustls-platform-verifier`. Crypto provider is `aws-lc-rs`;
  `ring` is no longer in the dep graph.

## [0.1.2] - 2026-05-11

Added a `tool_call_fallback_auto` dialect for providers that reject
forced tool selection but accept `tool_choice: "auto"` with the existing
single `emit_output` tool-call fallback shape.

## [0.1.1] - 2026-05-10

Added a `custom_headers` runtime endpoint config field for
provider-specific upstream HTTP headers. Custom headers now validate at
deserialize time, including reserved-header rejection and CR/LF/control
character guards for header values.

## [0.1.0] - 2026-04-24

Initial release of `llm_openai_compat` with OpenAI-compatible
`/chat/completions` support across `openai_native`, `vllm_native`, and
`tool_call_fallback` dialects, including schema validation,
stop-reason/usage normalization, and wiremock-backed request-vector and
error-path tests.
