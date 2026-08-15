# alpha4c.4d.1 — Frozen Trend Holdout v2 Split

## Preconditions

The first unseen v1 holdout was consumed by a genuine MetricDefinition
scientific-contract failure and is permanently seen regression material.

The repair replay subsequently passed calibration, prior seen regression, and
the consumed v1 suite. The v2 split is therefore drawn only from the untouched
22-paper alpha4c.4a reserve.

## Source pool

```text
Kiwook_SERS_26
Kiwook_SERS_15
Kiwook_SERS_14
Kiwook_SERS_11
Kiwook_SERS_38
Kiwook_SERS_20
Kiwook_SERS_24
Kiwook_SERS_23
Kiwook_SERS_9
Kiwook_SERS_36
Kiwook_SERS_33
Kiwook_SERS_32
Kiwook_SERS_29
Kiwook_SERS_27
Kiwook_SERS_12
Kiwook_SERS_3
Kiwook_SERS_22
Kiwook_SERS_7
Kiwook_SERS_31
Kiwook_SERS_17
Kiwook_SERS_21
Kiwook_SERS_28
```

No paper outside this pool may enter v2.

## Deterministic split

Selection semantics:

```text
trend_holdout_epoch_split_v1_alpha4c4d1
```

Namespace:

```text
sers-alpha4c4-v2
```

Ranking:

```text
SHA256("sers-alpha4c4-v2|" + paper_id)
```

Only `paper_id` is an input. Scientific content, metadata, Trend output,
relation overlap, expected direction, and expected yield are not inspected.

The first 8 ranked papers are frozen v2 holdout:

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

The remaining 14 stay untouched for a possible v3 epoch:

```text
Kiwook_SERS_36
Kiwook_SERS_32
Kiwook_SERS_7
Kiwook_SERS_20
Kiwook_SERS_3
Kiwook_SERS_15
Kiwook_SERS_24
Kiwook_SERS_29
Kiwook_SERS_33
Kiwook_SERS_27
Kiwook_SERS_26
Kiwook_SERS_31
Kiwook_SERS_14
Kiwook_SERS_9
```

Split SHA256:

```text
6eebae74732070408e920154ba898841c879a8e97cb866a8971cb9feef526966
```

## Scientific freeze

v2 freezes the repaired MetricDefinition semantics:

```text
sers_au_ag_metric_definition_v3_alpha4c4c1
```

and the existing Comparison, Trend, TrendPrecision, CrossContext and
MeasurementResultIdentity semantics.

The generic MetricDefinition contract remains unchanged.

## Zero-yield rule frozen before evaluation

A v2 paper set may legitimately produce:

```text
TrendEvidence = 0
local TrendResults = 0
cross-paper pairs = 0
```

When local TrendResults are zero, CrossContext and Assessment are terminally
not applicable:

```text
not_applicable_zero_local_results
```

The v2 evaluation runner must not invoke the non-empty CrossContext source
constructor in that state.

This is not an output-count success target.

## Failure / future epoch policy

If v2 exposes another genuine generic or domain-adapter scientific-contract
bug and code is changed after inspection:

1. all 8 v2 papers become seen regression;
2. the fix is replayed on existing seen suites;
3. a v3 blind holdout may be selected only from the untouched 14-paper
   reserve;
4. v2 is never resumed as blind evaluation.

Low or zero Trend yield is not by itself a bug.
