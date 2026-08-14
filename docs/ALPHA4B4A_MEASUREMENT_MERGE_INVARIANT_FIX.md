# alpha4b.4.1 — Measurement Merge Invariant Fix

## Why this patch exists

The first frozen SERS holdout reached ComparisonContext construction and
failed on `Kiwook_SERS_10 / meas_shell_thickness_range` because the canonical
Measurement had both `value_numeric` and `value_text`.

This is a generic implementation bug, not a paper-specific exception.

Each strict chunk is validated by `MeasurementNode` and therefore contains
exactly one result representation. The paper graph merger previously reused
same-type local IDs across chunks and filled blank attributes independently.
That allowed this invalid transformation:

```
chunk A: value_numeric = 8.4, value_text = ""
chunk B: value_numeric = "",  value_text = "3.6–10.0 nm"

old merge:
value_numeric = 8.4
value_text    = "3.6–10.0 nm"   # invalid
```

## Fix

Same-ID Measurement mentions are now kept as separate chunk-scoped mentions
when their scientific payloads conflict:

- numeric vs textual representation
- different numeric values
- different textual results
- conflicting metric identity
- conflicting subject identity
- conflicting explicit numeric units

Compatible duplicate mentions may still merge.

Incoming edges and embedded references are remapped to the preserved mention,
so provenance is not discarded.

## Hard invariant

`assert_measurement_value_xor` is run:

1. after all strict chunk graphs are merged;
2. after paper-level resolution/domain canonicalization.

Invalid canonical graphs can therefore no longer be silently materialized.

The semantic audit also reports an XOR violation as a blocking metric-semantic
issue, so legacy invalid graphs do not pass the core semantic gate unnoticed.

Invariant ID:

`measurement_payload_isolation_v1_alpha4b4a`

## Holdout discipline

This bug was discovered after holdout execution started. Under the frozen
evaluation protocol, this is a permitted domain-independent invariant fix, but
the previous calibration/holdout epoch is invalidated.

The installer therefore pauses `run_sers_alpha4b4_holdout`.

Run the calibration replay first:

```bash
python -m scripts.replay_sers_alpha4b4_calibration_after_xor_fix \
  --replay-id sers_alpha4b4a_xorfix_calibration_replay_v1
```

The replay:

- rebuilds SERS 1/5/8 canonical graphs from the same frozen strict runs;
- makes no LLM calls;
- reuses existing frozen Bridge materializations;
- rebuilds all three projections;
- creates a new corpus and quality/comparison sidecars under new IDs;
- never overwrites the old calibration corpus/sidecars;
- writes `calibration_replay_report.json`.

Do **not** resume SERS 2/6/10 holdout until that report has been reviewed and a
new calibration freeze protocol has been issued.
