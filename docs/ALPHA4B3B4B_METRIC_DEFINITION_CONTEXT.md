# alpha4b.3b.4b — MetricDefinitionContext

This phase adds a domain-owned sidecar that records **what a reported metric
means**, separately from the metric value and separately from MethodContext.

The canonical scientific graph is not modified.

## SERS v1 scope

Only two observables are registered in this first SERS adapter:

- `sers_enhancement_factor`
- `detection_limit`

Every registered Measurement receives exactly one `MetricDefinitionContext`,
even when its definition is unknown.

## Definition status

- `known`: local provenance identifies the metric definition/criterion.
- `partial`: the source establishes that the metric was calculated/estimated,
  but does not expose enough information to identify the normalization or
  criterion.
- `unknown`: only the reported metric/value is available.

Unknown is a valid scientific state and must not be upgraded by inference.

## EF semantics

Known families in v1:

- `molecule_normalized_intensity_ratio`
- `concentration_normalized_intensity_ratio`

Otherwise the family is `reported_ef_unspecified`.

`aggregation_scope` is deliberately orthogonal to the EF formula. Examples:

- `population_mean`
- `population_distribution`
- `lower_bound`
- `maximum`
- `single_particle`

Knowing that a result is a population mean does **not** make the EF definition
known.

## LOD semantics

Known families in v1:

- `calibration_curve_statistical`
- `lowest_detected_concentration`

Otherwise the family is `reported_lod_unspecified`.

The adapter never converts a lowest tested concentration into a statistical
LOD and never repairs suspicious reported concentration values.

## Provenance boundary

Definition evidence may come only from the Measurement and its directly
connected:

- `Calculation --HAS_MEASUREMENT--> Measurement`
- `Experiment --HAS_MEASUREMENT--> Measurement`
- `Measurement --IN_MEASUREMENT_GROUP--> MeasurementGroup`

Unconnected calculations or other papers cannot supply a metric definition.

## Not yet implemented in this phase

This phase does **not** change numeric ranking. The later alpha4b.3b.4c gate
will combine:

1. observable policy,
2. MethodContext / ProtocolAssessment,
3. MetricDefinitionContext,
4. numeric/unit compatibility.

Thus this phase is diagnostic/context construction only.
