# alpha4b.3b.4c.1 — Metric Compatibility Precedence & Ranking-Relevance Diagnostics

This is a narrow semantic precision patch on top of alpha4b.3b.4c.

## 1. Known mismatch precedence

Metric-definition compatibility now follows this precedence:

1. an unknown definition status remains `unknown`;
2. when both definition statuses are known, an explicit definition-signature
   mismatch is `different_definition`;
3. only if the known definition signature matches do unknown aggregation
   scopes reduce the result to `unknown`;
4. explicit aggregation mismatch is `different_definition`;
5. otherwise the result is `same_definition`.

Therefore a known definition contradiction cannot be hidden by a missing
secondary aggregation field.

Example:

- left EF: `molecule_normalized_intensity_ratio`
- right EF: `concentration_normalized_intensity_ratio`
- both aggregation scopes: `unspecified`

Result:

`different_definition`, gate closed.

The unknown aggregation is still preserved as a secondary diagnostic reason.

## 2. Component-level mismatch reasons

Known signature mismatches now retain precise causes:

- `metric_definition_family_mismatch`
- `metric_normalization_basis_mismatch`
- `metric_reference_basis_mismatch`
- `metric_criterion_mismatch`

Aggregation reasons remain:

- `metric_aggregation_scope_unknown`
- `metric_aggregation_scope_mismatch`

## 3. Ranking-relevance diagnostics

The existing metric-definition gate counts remain unchanged in meaning.
Additional counters separate all registered metric-definition assessments from
the subset whose observable policy actually permits numeric ranking:

- `metric_definition_ranking_relevant_assessment_count`
- `metric_definition_ranking_relevant_gate_pass_count`
- `metric_definition_ranking_relevant_gate_blocked_count`

An observable whose `numeric_ranking_mode` is `disabled` can therefore have a
scientifically `same_definition` pair without being reported as a
ranking-relevant gate pass.

## Semantics

Quality gate:

`quality_aware_numeric_gate_v2_alpha4b3b4c1`

Frozen inputs remain unchanged:

- comparison: `sers_au_ag_comparison_v7_alpha4b3b321`
- method: `sers_au_ag_method_v4_alpha4b3b321`
- metric definition: `sers_au_ag_metric_definition_v2_alpha4b3b4b1`

No protocol, ComparisonContext, MethodContext, MetricDefinitionContext, or
observable ranking policy semantics are changed.
