# alpha4b.2c.3 — SERS grounding completeness

Real SERS_1/5/8 projection calibration identified confirmed/exploratory Bridge
isolates with exactly two strict-anchor shapes.

1. `Precursor`
   - example: `AgNO3`
   - canonical incoming edge:
     `SynthesisMethod --USES_PRECURSOR--> Precursor`

2. `MeasurementGroup`
   - examples: nanogap/particle-size/comparative SERS sweeps
   - canonical incoming edges:
     `Measurement --IN_MEASUREMENT_GROUP--> MeasurementGroup`
   - the Measurement can then backtrace through existing
     `HAS_MEASUREMENT`, `TESTED_IN`, or `MEASURED_FOR` rules.

The projection algorithm itself is unchanged. This phase only completes the
SERS profile's direction-aware relation semantics by adding:

- incoming `IN_MEASUREMENT_GROUP`
- incoming `USES_PRECURSOR`

`Precursor` and `MeasurementGroup` are deliberately not promoted to
mechanism-origin node types. They remain evidence/context nodes and are lifted
to the nearest scientific origin.

The SERS projection semantics id is bumped to
`sers_au_ag_projection_v2_alpha4b2c3`.

No fuzzy Bridge merging, corpus logic, Bridge policy, HER logic, or hypothesis
logic is changed.
