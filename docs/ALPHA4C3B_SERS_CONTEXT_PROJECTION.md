# alpha4c.3b — SERS Context Projection

## Scope

alpha4c.3b projects frozen measurement/protocol context onto the already
frozen `PaperLocalTrendResult` layer.

It still does **not**:

- extract or rewrite TrendEvidence;
- change alpha4c.2 trend/precision semantics;
- build cross-paper pairwise contrasts;
- assign repeated/context-specific/reversed/insufficient status;
- reuse numeric-ranking compatibility as trend policy;
- infer causal relations.

Output is only one `TrendContextProfile` per `PaperLocalTrendResult`.

## Active semantics

Generic contract:

```text
cross_context_trend_contract_v1_alpha4c3a
```

SERS context projection:

```text
sers_au_ag_trend_context_v1_alpha4c3b
```

Frozen upstream trend/precision remain:

```text
sers_au_ag_trend_v5_alpha4c2121
sers_au_ag_trend_precision_v5_alpha4c21211
```

## Context dimensions

The SERS adapter exposes:

```text
analyte
reporter
analyte_concentration
excitation_wavelength
laser_power
integration_time
raman_peak
sample_preparation
preparation_medium
measurement_environment
sample_state
substrate_condition
```

The existing `MethodContext` is the source of truth for all dimensions except
`raman_peak`, which comes from `ComparisonContext`.

No new chemical/measurement parser is introduced.

## Provenance linkage

A paper-local trend may consume context only when it preserves an explicit
`source_measurement_id`.

Identity-aware comparison/method sidecars preserve original Measurement source
mentions in `source_node_ids`. Therefore a source Measurement can resolve to a
representative scientific-result context in either of two equivalent ways:

```text
source_measurement_id == sidecar.measurement_id

or

source_measurement_id is explicitly preserved in
sidecar.source_node_ids
```

For every source Measurement, exactly one `ComparisonContext` and exactly one
`MethodContext` must resolve. Ambiguous or missing resolution fails closed.

The linked `ComparisonContext.method_context_id` must agree with the resolved
`MethodContext`.

## No paper-global fallback

If:

```text
PaperLocalTrendResult.source_measurement_ids == ()
```

then the profile receives **no** ComparisonContext or MethodContext merely
because another measurement exists in the same paper.

Its context dimensions remain:

```text
unknown
```

except for an explicit `varied_control` marker.

This is a structural audit invariant, not a best-effort policy.

## Varied-control masking

Only an independent variable that is itself a measurement-context dimension is
masked:

```text
analyte_concentration -> analyte_concentration
excitation_wavelength -> excitation_wavelength
laser_power            -> laser_power
integration_time       -> integration_time
```

A historical `concentration` alias also maps to `analyte_concentration`.

Structural, composition, and synthesis controls do not mask broad
`substrate_condition`.

In particular:

```text
spr_excitation_detuning
```

does **not** mask `excitation_wavelength`: the laser wavelength is context for
the spectral-detuning relation and may remain fixed/known.

## Multi-measurement aggregation

When one paper-local numeric trend has several direct source Measurements:

- all known and same normalized value -> `known`;
- known values disagree -> `ambiguous`;
- any source dimension is explicitly ambiguous -> `ambiguous`;
- known + unknown coverage -> `unknown` (partial evidence is preserved in
  `source_values`, but no trend-wide value is asserted);
- all unknown -> `unknown`.

This prevents a known value from one point in a series from being silently
promoted to the whole trend.

## Output

The builder derives the frozen ComparisonContext binding from the parent trend
summary; users do not supply a new comparison ID.

```text
trend/<trend-id>/
  precision/<precision-id>/
    cross_context/<context-id>/
      context_profiles.jsonl
      audit.json
      summary.json
```

The summary explicitly records:

```text
pairwise_contrasts_built = false
cross_context_assessments_built = false
numeric_ranking_reused_as_trend_policy = false
paper_global_context_fallback_used = false
```

## Structural audit

The SERS projection audit verifies:

1. one profile per paper-local result;
2. original result provenance is not dropped;
3. exact varied-control masking;
4. every direct source Measurement resolves to the frozen sidecars;
5. selected sidecar IDs exactly match the direct provenance;
6. no context IDs are attached to a result without a direct Measurement;
7. no known/ambiguous context leaks into a result without a direct
   Measurement.

Unknown context is valid and does not fail the gate.

## Calibration / seen regression

`run_sers_alpha4c3b_projection_regression` reuses the already-frozen
alpha4c21211 calibration and seen outputs. It does not run an LLM or rebuild
alpha4c.2.

Key regressions include:

- one profile per local result;
- no paper-global leakage;
- identity-aware direct Measurement provenance resolves;
- analyte-concentration trends mark concentration as `varied_control`;
- spectral detuning does not mask excitation wavelength;
- shell thickness is not misclassified as a generic context mask.

## Next

alpha4c.3c will consume these `TrendContextProfile` rows and implement:

```text
same relation grouping
-> cross-paper PairwiseTrendContrast
-> deterministic repeated/context_specific/reversed/insufficient assessment
```

with the alpha4c.3a no-majority-vote invariant enforced by the generic audit.
