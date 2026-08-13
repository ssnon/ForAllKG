# alpha4b.3b.3 — SERS MethodContext + measurement-local provenance

## Motivation

SERS methods are heterogeneous across papers. Numeric values can therefore be
poorly comparable even when mechanistic trends remain scientifically useful.

alpha4b.3b.3 makes method heterogeneity first-class data instead of trying to
normalize it away.

The immediate calibration defect was a provenance leak: a shared `Analyte`
node carried a single concentration attribute and that value propagated into
many unrelated measurements. Measurement-local source expressions, however,
contained the actual per-measurement concentrations.

## Generic MethodContext

Every measurement may now link to a `MethodContext`.

Tracked SERS method dimensions:

- analyte
- reporter
- analyte concentration
- excitation wavelength
- laser power
- integration time
- medium
- sample preparation
- substrate state

Each known/ambiguous method dimension records explicit provenance scopes and
source node IDs.

## Measurement-local concentration

Concentration is resolved conservatively in this priority order:

1. explicit Measurement concentration attributes;
2. probe-scoped concentration in the Measurement `source_expression`;
3. explicitly named analyte/reporter concentration attributes on
   Experiment/MeasurementGroup;
4. probe-scoped MeasurementGroup text.

A global concentration attribute on an Analyte or RamanReporter node is never
consumed as measurement context.

For concentration-valued output metrics such as `detection_limit`, the output
value is never recycled as context concentration.

Generic AgNO3 or precursor concentration sweeps are not accepted as analyte
concentration unless the value is explicitly probe-scoped.

## Protocol comparability

Cross-paper measurement pairs receive a separate `ProtocolAssessment`:

- `same_protocol`
- `harmonized_protocol`
- `partially_matched`
- `different_protocol`
- `unknown`

This is intentionally separate from observable compatibility and from future
mechanistic trend aggregation.

Critical SERS protocol dimensions are analyte, reporter, and excitation
wavelength. A mismatch in a known critical dimension yields
`different_protocol`. Matching critical dimensions with missing noncritical
metadata may yield `harmonized_protocol`; mixed partial evidence yields
`partially_matched`.

Direct numeric ranking is additionally protocol-gated and is allowed only when
the observable policy allows it and the protocol status is `same_protocol`.

## Sidecars

The comparison CLI additionally writes:

- `method_contexts.jsonl`
- `protocol_assessments.jsonl`

`contexts.jsonl` links each measurement comparison context to its
`method_context_id`.

## Calibration

```bash
python -m scripts.build_comparison_contexts \
  --domain-profile sers_au_ag \
  --data-root data_sers \
  --corpus-id sers_alpha4b3a_calibration \
  --mode exploratory \
  --comparison-id sers_alpha4b3b3_calibration
```

Expected safety behavior:

- the previous global `"7 M"` analyte concentration must disappear from SERS_8
  EF measurement contexts;
- measurement-local strings such as `MB (10^-5 M)` may become `1e-5 M`;
- LOD output values do not become context concentration;
- precursor/synthesis sweeps do not become analyte concentration;
- protocol assessments preserve heterogeneous experimental settings rather
  than forcing comparability;
- numeric ranking may remain zero.
