# alpha4b.3b.1 — SERS ComparisonContext semantic precision

Calibration of SERS_1/5/8 exposed two semantic precision problems in the v1
comparison provider.

## 1. Concentration scope contamination

The v1 provider scanned broad local context text for any molarity token. That
can incorrectly treat synthesis or morphology sweep concentrations such as
AgNO3 50 mM / 300 mM as analyte/reporter comparison context.

v2 accepts concentration only from:

- Analyte/RamanReporter concentration attributes or text;
- explicitly named `analyte_concentration` / `reporter_concentration`
  attributes on Experiment, Measurement, or MeasurementGroup.

Detection-limit values and generic MeasurementGroup precursor sweeps are not
reused as context concentration.

## 2. Unit-aware physical normalization

Generic lowercasing made values such as `300 mM` appear as `300 mm` and did not
equate physically identical spellings such as `100 nM` and `1e-7 M`.

The SERS provider now canonicalizes:

- concentration -> M
- excitation wavelength -> nm
- laser power -> mW
- integration time -> s
- Raman peak -> cm^-1

No unit is guessed and no dimension-changing conversion is performed.

## Entity aliases

Only narrow, domain-owned normalization is used. Explicit parenthetical
abbreviations may be stripped, and `MB` / `Methylene blue (MB)` normalize to
`methylene blue`. Bare `ATP` is not expanded to `4-aminothiophenol`; missing or
ambiguous identity remains conservative.

## Unchanged safety rules

- unknown != compatible
- missing context is not quarantine
- all nine SERS dimensions remain required for direct numeric ranking
- medium/substrate_state are not inferred
- no graph topology is changed
- no Bridge/projection/corpus policy is changed
