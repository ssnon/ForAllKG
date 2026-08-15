# alpha4c.4d.2.1 — Derived Locator Provenance Fix

## Observed failure

The v2 input-preparation run successfully re-materialized all eight canonical
graphs and stopped after SERS_11 with:

```text
Alpha4c4d2Error: Kiwook_SERS_11: locator_index appeared.
```

No canonical input lock, projection, corpus, or Trend output had been created.

## Root cause

The alpha4c.4d.2 support code incorrectly classified the run-level
`locator_index.json` as an immutable Strict extraction input.

Repository behavior shows the opposite:

1. provenance backfill refreshes document packages/assets;
2. it builds locator records;
3. it writes run-level `locator_index.json` and `.csv`;
4. `build_paper_graph` then loads that run-level locator index and uses it to
   backfill edge asset provenance.

Therefore `locator_index.json` is a deterministic build-time derived artifact.
It may legitimately be absent before canonical re-materialization and appear
during it.

The true immutable Strict inputs remain:

```text
latest_run.json
run.json
active_chunks.json
active strict-valid chunk JSONs
```

The downstream scientific input is frozen after preparation by SHA-locking the
final canonical GraphML and resolution decisions.

## Repair

`verify_strict_source_unchanged()` no longer compares locator-index
presence/hash before and after canonical construction.

It still verifies all actual frozen Strict inputs and extraction-quality
classification.

The preparation snapshot copier is also made resume-safe: if a pre/post audit
snapshot already exists from the failed preparation, a rerun will not
overwrite it. This preserves the original pre-refreeze evidence.

Before installing, the patch verifies that no canonical input lock, holdout
manifest, or holdout report exists. It also compares any existing
pre-refreeze resolution snapshots against current resolution decisions at the
manual-decision level.

## Holdout status

This is an operational input-preparation correction only. The v2 scientific
holdout has not started and the eight papers remain blind with respect to
Trend outputs.

After installation:

```bash
python -m scripts.prepare_sers_alpha4c4d2_holdout_inputs
```

should normally see the already-refrozen canonical graphs as ready and create
the persistent canonical input lock without re-running the canonical build.

Only after that succeeds:

```bash
python -m scripts.run_sers_alpha4c4d2_trend_holdout
```
