# alpha4c.2.1 — Trend Evidence Kind & Paper-local Identity Precision

Semantics:

- generic contract remains frozen: `trend_evidence_contract_v1_alpha4c1`
- active SERS raw trend semantics: `sers_au_ag_trend_v2_alpha4c21`
- precision sidecar: `sers_au_ag_trend_precision_v1_alpha4c21`

## Scope

This patch fixes semantic precision issues found by reviewing the alpha4c.2
seen regression. It does not implement cross-paper repeated/reversed voting.
That remains alpha4c.3 and must consume paper-local results, not raw Claim
mentions.

## Evidence-kind precision

Raw `TrendEvidence` stays contract-compatible. A separate annotation classifies
each raw item as `experimental_numeric`, `calculated_numeric`, or
`reported_claim`. Explicit Calculation lineage wins; a grounded lineage with an
explicit DDA/simulation/calculation marker is also classified as calculated.

This prevents the SERS_2 DDA-derived values (2 nm / 8 nm) from being treated as
empirical SERS EF evidence.

## Observable precision

Directional signal/intensity language takes precedence over a later
baseline-relative fold detail. Therefore a sentence saying SERS intensity rises
with shell thickness and later says `5.8 relative to Au nanocubes` remains a
Raman/SERS intensity trend. The `5.8` is not promoted to formal EF.

A pure baseline-relative response can use `relative_sers_intensity_ratio`;
formal `sers_enhancement_factor` remains distinct.

## Control registry

Structural: shell thickness, nanogap size, particle size.
Composition: Ag/Au ratio, Au content.
Concentration: analyte concentration, particle concentration.
Synthesis: gold precursor amount, AgNO3 concentration.
Measurement: laser power, excitation wavelength, integration time.

Numeric controls can be grounded in Measurement, MeasurementGroup,
producer/Experiment, measured subject, or sidecar-provenance nodes. Structured
conditions are preferred. Text fallback requires both an explicit control
phrase and numeric unit. Unknown context is not treated as compatible; explicit
non-varied method mismatches still block a numeric trend. Only the dimension
being deliberately swept is exempted from the method-equality guard.

## Ratio orientation

Canonical composition ratio is Ag/Au. A source `Au:Ag = 10:7` records source
`10:7`, canonical `0.7`, and transform `au_ag_to_ag_over_au`.

## Paper-local identity

Raw Claim nodes are preserved. Claim evidence is consolidated only inside one
paper when control, observable, direction, shape, semantic family, and
normalized subject family agree. Numeric and claim lanes never merge. Numeric
lineages remain separate. Thus repeated wording in one paper cannot become
multiple independent votes in alpha4c.3.

For numeric evidence, subjects directly reached by `MEASURED_FOR` are trend
subjects; extra comparison subjects are reference subjects.

## Runs

Calibration (SERS_1/5/8):

```bash
python -m scripts.run_sers_alpha4c21_calibration
```

Seen regression (SERS_2/6/10):

```bash
python -m scripts.run_sers_alpha4c21_seen_regression
```

Both are LLM-free. No evidence-count target is encoded. Review raw evidence,
annotations, and local results before freezing alpha4c.2.1.
