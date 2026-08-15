# alpha4c.4d.2 — Frozen v2 Input Preparation + Blind Trend Runner

Frozen v2 papers:

```text
Kiwook_SERS_21
Kiwook_SERS_38
Kiwook_SERS_12
Kiwook_SERS_28
Kiwook_SERS_17
Kiwook_SERS_22
Kiwook_SERS_23
Kiwook_SERS_11
```

Split SHA256:

```text
6eebae74732070408e920154ba898841c879a8e97cb866a8971cb9feef526966
```

MetricDefinition semantics:

```text
sers_au_ag_metric_definition_v3_alpha4c4c1
```

## Phase A — input preparation

Run first:

```bash
python -m scripts.prepare_sers_alpha4c4d2_holdout_inputs --preflight-only
```

This only reports:

- frozen Strict extraction quality;
- whether `active_chunks.complete` is true/false (diagnostic only);
- current canonical Measurement merge epoch;
- whether canonical refreeze is required.

No files are modified.

Then:

```bash
python -m scripts.prepare_sers_alpha4c4d2_holdout_inputs
```

For every paper:

- `complete` and `partial_acceptable` are allowed;
- `partial_critical` is allowed with `--allow-incomplete`;
- `rejected` fails closed;
- no LLM extraction is run;
- canonical refreeze happens only when the canonical graph is absent, has the
  wrong Measurement merge invariant, or violates Measurement numeric/text XOR;
- existing manual resolution decisions must survive exactly;
- canonical GraphML and `resolution/decisions.jsonl` are SHA-locked.

Successful preparation creates:

```text
evaluation/sers_alpha4c4d2/holdout_v2/canonical_input_lock.json
```

## Phase B — blind v2 holdout

Only after the input lock passes:

```bash
python -m scripts.run_sers_alpha4c4d2_trend_holdout
```

Pipeline:

```text
8 locked canonical graphs
  -> evidence projections (no Bridge)
  -> campaign corpus
  -> MeasurementResultIdentity
  -> MetricDefinition v3
  -> Comparison
  -> TrendEvidence
  -> TrendPrecision
```

If local TrendResults are nonzero:

```text
-> CrossContext
-> CrossContextAssessment
```

If local TrendResults are zero:

```text
CrossContext = not_applicable_zero_local_results
Assessment   = not_applicable_zero_local_results
```

No empty CrossContext source is fabricated and no generic contract is relaxed.

## Acceptance

Pass/fail is invariant-only.

These are valid outcomes:

```text
TrendEvidence = 0
local TrendResults = 0
cross-paper pairs = 0
repeated = 0
reversed = 0
all assessments = insufficient
```

A scientific-contract failure after the real runner starts consumes all eight
v2 papers. If code must change after inspection, v2 is retired and the next
blind epoch may use only the 14-paper v3 reserve.
