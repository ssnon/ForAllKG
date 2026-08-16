# alpha4c.5h.0a — Blind Split SHA Domain Fix

The first alpha4c.5h freeze attempt failed before writing any freeze artifact.

Observed:

- raw `blind_split.json` file SHA256:
  `16c8fe725468a57a7703e13e843c1486176f51a264bb6b82f593cb7abbd956c5`
- historical alpha4c.5f.2 `split_sha256`:
  `4b73127ceb27ff0ec7afeb5362485eecc15fa95fd808377331a57f2b6f497d16`

These values are expected to differ.

## Root cause

Historical alpha4c.5f.2 defines `split_sha256` as the SHA256 of canonical JSON
after removing the `split_sha256` field itself.

The initial alpha4c.5h implementation incorrectly compared that semantic SHA
against `sha256_file(blind_split.json)`, which hashes pretty-printed bytes.

This was an orchestration bug. It does not imply split drift.

## Corrected verification

alpha4c.5h.0a now delegates split identity to the frozen historical
alpha4c.5f.2 validators:

- `validate_pool_manifest(..., verify_source_manifest=False)`
- `validate_blind_split(pool=..., split=...)`

Those validators verify:

- pool semantic integrity;
- blind-split semantic SHA;
- exact deterministic ID-only recomputation;
- partition coverage/disjointness/counts;
- scientific-fields-used = false;
- Reserve B sealed;
- zero split-time LLM calls.

After semantic validation succeeds, alpha4c.5h records the *current raw file
SHA* separately as a byte-lock for post-freeze verification.

Thus the freeze binds both:

1. historical semantic split identity;
2. exact current file bytes.

The two hashes are never conflated again.
