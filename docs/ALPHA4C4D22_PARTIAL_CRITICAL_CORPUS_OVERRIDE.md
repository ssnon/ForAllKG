# alpha4c.4d.2.2 — PARTIAL_CRITICAL Corpus Override Propagation

## Observed failure

The v2 input preparation succeeded and created the persistent canonical input
lock. SERS_11 was explicitly classified and locked as:

```text
partial_critical
```

The real holdout runner then built all eight evidence projections successfully
and stopped before corpus construction because `scripts.build_corpus_graph`
uses its default extraction-quality policy unless passed:

```text
--allow-critical-partial
```

No corpus, MeasurementResultIdentity, MetricDefinition, Comparison, Trend,
Precision, CrossContext, or Assessment result was produced.

## Root cause

alpha4c.4d.2 already froze this policy:

```text
partial_critical_allowed_with_allow_incomplete = true
```

and the input preparation accepted SERS_11 with explicit incomplete handling.

The runner failed to propagate the corresponding corpus-level explicit
override. This is an orchestration mismatch, not a change in scientific
eligibility semantics.

## Repair

The runner now reads the already-frozen canonical input lock.

If at least one locked paper has:

```text
graph_materialization_status == "partial_critical"
```

the corpus command adds:

```text
--allow-critical-partial
```

No override is added when all inputs are complete/partial_acceptable.

The runner also explicitly rejects any locked status outside:

```text
complete
partial_acceptable
partial_critical
```

so `rejected` and unknown quality remain fail-closed.

## Holdout status

The same v2 campaign is resumed.

This correction is permitted because:

- the paper split was already frozen;
- canonical inputs were already SHA-locked;
- the failure occurred before corpus construction;
- no MetricDefinition, Comparison, TrendEvidence, TrendPrecision, or
  CrossContext scientific outputs existed or were inspected;
- no scientific semantic or acceptance threshold changes.

Existing completed projection stages remain hash-verified and are skipped on
resume.

Run:

```bash
python -m scripts.run_sers_alpha4c4d2_trend_holdout
```
