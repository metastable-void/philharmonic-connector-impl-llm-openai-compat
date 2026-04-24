# Changelog

All notable changes to this crate are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this crate adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-24

Initial release of `llm_openai_compat` with OpenAI-compatible
`/chat/completions` support across `openai_native`, `vllm_native`, and
`tool_call_fallback` dialects, including schema validation,
stop-reason/usage normalization, and wiremock-backed request-vector and
error-path tests.
