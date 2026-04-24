# Test Fixture Provenance

This directory is a synchronized fixture copy of
`<workspace-root>/docs/upstream-fixtures/`, with one synthesized file for
vLLM response coverage.

The copied files are committed in this crate so standalone impl-repo CI and
`cargo package` / publish flows remain self-contained inside the package
boundary.

## Provenance by subtree

- `vllm/`
  - Byte-exact copies from workspace provenance fixtures.
  - Upstream source commit: vLLM
    `cf8a613a87264183058801309868722f9013e101`.
  - See `<workspace-root>/docs/upstream-fixtures/vllm/README.md` for pinned
    blob SHAs and recapture context.

- `openai-chat/`
  - Byte-exact copies from workspace provenance fixtures.
  - Captured on 2026-04-24 against real OpenAI API model
    `gpt-4o-mini-2024-07-18`.
  - See `<workspace-root>/docs/upstream-fixtures/openai-chat/README.md` for
    capture commands and recapture steps.

- `vllm_native_response.json`
  - Synthesized locally to match the OpenAI `/chat/completions` response
    envelope because the pinned upstream vLLM fixture set publishes request
    vectors but not response bodies.

## Sync discipline

When provenance fixtures under `<workspace-root>/docs/upstream-fixtures/`
change, update the copied files here as byte-exact duplicates so request-vector
and smoke tests keep validating against the same authoritative source material.
