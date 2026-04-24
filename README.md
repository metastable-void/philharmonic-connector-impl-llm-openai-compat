# philharmonic-connector-impl-llm-openai-compat

Part of the Philharmonic workspace: https://github.com/metastable-void/philharmonic-workspace

`philharmonic-connector-impl-llm-openai-compat` implements the
`llm_generate` connector capability for OpenAI-compatible
`/chat/completions` APIs. It supports three translation dialects —
`openai_native`, `vllm_native`, and `tool_call_fallback` — while
normalizing responses into `{output, stop_reason, usage}` and enforcing
`output_schema` validation before returning results.

## Contributing

This crate is developed as a submodule of the Philharmonic
workspace. Workspace-wide development conventions — git workflow,
script wrappers, Rust code rules, versioning, terminology — live
in the workspace meta-repo at
[metastable-void/philharmonic-workspace](https://github.com/metastable-void/philharmonic-workspace),
authoritatively in its
[`CONTRIBUTING.md`](https://github.com/metastable-void/philharmonic-workspace/blob/main/CONTRIBUTING.md).

SPDX-License-Identifier: Apache-2.0 OR MPL-2.0
