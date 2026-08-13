# Run / Attempt provenance v1

This change keeps the existing deterministic extraction identity intact while
making each physical extraction execution independently addressable.

## Identity contract

- `run_fingerprint`: unchanged deterministic fingerprint of the extraction
  configuration and source state.
- `run_id`: unchanged short form of `run_fingerprint`.
- `attempt_id`: concrete execution identity. Re-running the same `run_id`
  creates a new attempt directory instead of overwriting the previous one.

New extraction layout:

```text
<root>/extracted/<paper>/runs/<run_id>/
  run.json
  latest_attempt.json
  attempts/<attempt_id>/
    run.json
    chunks/
    source_chunks/
    debug/
    documents/
    active_chunks.json
    extraction_quality.json
    lineage.json
    summary.json
    events.jsonl
    manifest.jsonl
```

`latest_run.json` continues to expose `run_directory` as the deterministic run
family directory. New pointers additionally expose `attempt_id` and
`attempt_directory`. Readers first use the concrete attempt when present and
fall back to the legacy flat run directory when no attempt metadata exists.

## Cache compatibility

A new attempt never writes into a prior attempt. When extraction is invoked
without `--force`, validated chunk cache files and non-forced vision cache files
are copied from the previous concrete attempt into the new attempt before the
normal validation/cache checks run. This preserves the old no-force behavior
without sacrificing attempt isolation.

## Downstream provenance

Paper GraphML records `source_extraction_attempt_id`. Broad projection summaries
propagate it as `source_extraction_attempt_id`. Broad freshness checks require an
exact attempt match when the extraction has an attempt identity; legacy
extractions without an attempt identity continue to use the existing
`run_id`/`run_fingerprint` comparison.

`build_paper_graph` and `extract_bridge_graph` accept `--attempt-id` for explicit
historical-attempt selection. The Bridge extraction fingerprint intentionally
continues to depend on deterministic strict-run identity and frozen content
hashes, not `attempt_id`.

## Legacy compatibility

No migration of existing data is required. A legacy layout such as
`runs/<run_id>/active_chunks.json` remains readable. New runs use attempt-scoped
artifacts, while stable paper-level GraphML/projection aliases remain mutable
"latest" views bound to the concrete extraction attempt through provenance.
