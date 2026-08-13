# alpha4b.3b.4c — Quality-aware Numeric Comparison Gate

This phase binds frozen Comparison/MethodContext semantics to the frozen
MetricDefinitionContext sidecar.

## Core rule

Numeric ranking is allowed only when every previously required condition is
already satisfied **and**, for observables registered by the domain's
MetricDefinition adapter, the metric definitions are safely comparable.

The quality gate does not alter ordinary comparison compatibility. A pair may
remain scientifically comparable or mechanistically useful while numeric
ranking is blocked.

## Metric-definition compatibility

For a registered observable:

- both definitions must be `known`;
- the canonical definition signature must match:
  - definition family,
  - normalization basis,
  - reference basis,
  - criterion;
- aggregation scope must be explicit and equal.

`aggregation_scope=unspecified` blocks numeric ranking even if the formula
family is known.

States:

- `same_definition`
- `different_definition`
- `unknown`
- `not_applicable`

`not_applicable` is used only for observables not registered by the
MetricDefinition adapter and never blocks their existing ranking policy.

## Sidecar binding

The comparison builder now requires `--metric-definition-id` when the active
domain profile has a metric-definition adapter.

The frozen sidecar is accepted only when all of these match:

- domain profile,
- corpus ID,
- corpus mode,
- metric-definition semantics ID,
- canonical graph SHA-256 values.

The sidecar is structurally re-audited before use.

## ReproducibilityEvidence

ReproducibilityEvidence remains quality metadata. It does not open or close
the numeric ranking gate in alpha4b.3b.4c. In particular, strong
reproducibility must never make incompatible protocols or metric definitions
rankable.

## Semantics

Quality gate:

`quality_aware_numeric_gate_v1_alpha4b3b4c`

Frozen inputs:

- comparison: `sers_au_ag_comparison_v7_alpha4b3b321`
- method: `sers_au_ag_method_v4_alpha4b3b321`
- metric definition: `sers_au_ag_metric_definition_v2_alpha4b3b4b1`
