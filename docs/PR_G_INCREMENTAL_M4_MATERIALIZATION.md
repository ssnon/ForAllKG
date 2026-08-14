# PR G — Incremental M4 materialization

This PR adds an opt-in M4 entrypoint:

```bash
python -m scripts.materialize_corpus_documents_incremental ...
```

It intentionally does not replace the existing production M4 runner yet. This
keeps the patch additive and lets a real corpus smoke-test the cache semantics
before wiring the new entrypoint into the knowledge-aware backfill coordinator.

## Goals

- Reuse materialized documents independently instead of invalidating a whole
  paper when its SI artifact set changes.
- Preserve existing SI document IDs when new SI artifacts are discovered.
- Verify cached Markdown SHA and package metadata before reuse.
- Store a materialization-context fingerprint covering the policy file and the
  materializer implementations.
- Migrate legacy M4 state when the prior materialization report has the same
  materialization ID and policy ID and the cached outputs verify.
- Retry only failed documents with `--retry-failed`.
- Allow bounded paper-level parallelism with `--workers`; default remains 1
  because Marker can be GPU/RAM intensive.
- Keep the existing M4 report/config/extraction-plan contracts intact.

## Safety

The incremental runner never infers scientific results and never promotes
retrieval metadata to evidence. Cache reuse is content/provenance reuse only.

`--force` deliberately disables cache reuse. Without `--force`, a materialized
record is reused only when source identity/SHA, materialization context,
Markdown SHA, and package metadata all validate.

## Expected use

For an already-materialized corpus, run once with `--workers 1`; most documents
should report `cache_reused` and the migration should be fast. For a future
batch containing new PDFs, start with `--workers 2` only if system memory/VRAM
allows multiple Marker processes safely.
