# alpha4c.4c.1 — MetricDefinition Unknown-Evidence Invariant Fix

The first real alpha4c.4 unseen holdout reached MetricDefinition and failed
because an `unknown` MetricDefinitionContext carried interpreted definition
evidence.

The generic MetricDefinitionContext contract remains unchanged.

The SERS adapter is repaired conservatively:

```text
status != unknown
    interpreted fields unchanged

status == unknown
    criterion = ""
    formula_text = ""
    non-sentinel normalization/reference -> "unspecified"
    raw source_expression preserved
    source/provenance IDs preserved
```

No unknown context is promoted to partial/known.

MetricDefinition semantics is bumped from:

```text
sers_au_ag_metric_definition_v2_alpha4b3b4b1
```

to:

```text
sers_au_ag_metric_definition_v3_alpha4c4c1
```

The consumed v1 holdout papers are retired as blind holdout and become seen
regression material:

```text
Kiwook_SERS_16
Kiwook_SERS_35
Kiwook_SERS_34
Kiwook_SERS_19
Kiwook_SERS_13
Kiwook_SERS_37
Kiwook_SERS_30
Kiwook_SERS_25
Kiwook_SERS_4
Kiwook_SERS_18
```

The untouched 22-paper reserve remains the only source for a future blind
holdout. Do not select v2 until calibration + seen + consumed-v1 replay passes.
