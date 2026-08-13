# alpha4b.3b.2 — Observable-specific ComparisonApplicability

## Motivation

SERS_1/5/8 calibration contains heterogeneous observables:

- SERS performance (`sers_enhancement_factor`, `raman_intensity`)
- Raman spectral positions
- analytical detection/quality metrics
- optical spectral peaks
- structural metrics
- composition metrics
- simulations and qualitative claims

Applying all nine SERS experimental dimensions to every observable produced
scientifically irrelevant `unknown`/`incompatible` decisions. In particular,
`raman_peak_position` was using the Raman peak itself as a context dimension,
and optical peak observables were being conditioned on analyte/Raman context.

## Contract

`ObservableComparisonPolicy` is domain-owned and exact-key based.

Each policy declares:

- family
- applicable dimensions
- dimensions required for numeric ranking
- whether numeric ranking is enabled
- ranking direction when enabled

Unseen observables are **not guessed into a family**. They fail closed with
`observable_policy_id = "unregistered"` and `compatibility = "unknown"`.

## SERS v1 applicability

Direct numeric ranking is enabled only for:

- `sers_enhancement_factor`
- `raman_intensity`

and remains conservative: every ranking-required dimension declared by its
policy must explicitly match, both values must be numeric, and units must
explicitly match.

`detection_limit`, reproducibility/stability statistics, spectral positions,
structural metrics, composition metrics, simulations, and calibration
statistics are descriptive/comparison-only in this phase. Their numeric
ranking is disabled even if values and units match.

This avoids encoding an unjustified "larger is better" / "smaller is better"
rule for observables whose protocol or scientific directionality is not yet
represented.

## Calibration-only exact keys

Policies were derived only from observables present in SERS_1/5/8. Hold-out
papers remain untouched. Unregistered keys in hold-out data will remain
unknown until a later explicit policy review; they do not trigger automatic
family inference.

## Unchanged

- strict canonical graph
- Bridge extraction/policy
- ProjectionSemantics
- CorpusSemantics
- concentration precision from alpha4b.3b.1
- unknown != compatible
- missing context is not quarantine
