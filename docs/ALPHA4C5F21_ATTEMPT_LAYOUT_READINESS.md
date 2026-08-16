# alpha4c.5f.2.1 — Attempt-aware Strict Source Readiness

## Failure observed before Reserve-A consumption

The alpha4c.5f.2 Reserve-A readiness preflight failed while looking for:

`<run-family>/active_chunks.json`

The new 103-paper Strict extraction uses the current
`run-attempt-provenance-v1` layout:

```text
paper/latest_run.json
       |
       +-- run_directory --------> runs/<run_id>/              (family)
       |
       +-- attempt_directory ----> runs/<run_id>/attempts/<id>/
                                      run.json
                                      active_chunks.json
```

or equivalently:

```text
runs/<run_id>/latest_attempt.json
       -> attempt_directory
```

The historical alpha4c.5f.1 resolver predates this layout and assumes
`run_directory` itself contains `run.json` and `active_chunks.json`.

## Scope of the fix

This patch does **not** modify historical alpha4c.5f.1 or alpha4c.4d2 code.

It adds an alpha4c.5f.2-specific strict-source resolver which:

1. accepts a concrete `attempt_directory` from `latest_run.json`;
2. otherwise follows `latest_attempt.json`;
3. otherwise supports the historical flat layout;
4. snapshots both family/attempt provenance pointers;
5. verifies pointer/run/attempt identity consistency;
6. preserves the same Strict quality and positive-evidence gates;
7. preserves immutable SHA checks for run metadata and active chunk inputs.

The canonical readiness semantics remain
`canonical_readiness_gate_v1_alpha4c5f1`.
Only source-layout resolution changes.

No extraction, Explorer, Maker, Trend, Comparison, or scientific acceptance
semantic is changed.

Reserve A remains unconsumed.
