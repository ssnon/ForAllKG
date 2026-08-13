# alpha4b.3b.3.1 — Conservative SERS MethodContext Harvesting

## Goal

Preserve SERS protocol heterogeneity as evidence instead of forcing cross-paper
measurements into a common experimental setting.

This patch extends the frozen alpha4b.3b.3 MethodContext contract with
conservative harvesting of method metadata that is already explicit in the
canonical graph:

- structured `conditions_json`;
- measurement-local `source_expression`;
- the producing physical Experiment's `raw_method_name` / `description`;
- measurement/Experiment protocol attributes.

It does **not** use an LLM, invent missing settings, infer a dry state from
adsorption alone, or treat simulation dielectric media as sample media.

## Semantics IDs

- comparison: `sers_au_ag_comparison_v5_alpha4b3b31`
- method: `sers_au_ag_method_v2_alpha4b3b31`

## Controlled method dimensions

### Medium

Only explicit physical measurement media are normalized:

- water / aqueous / deionized water -> `aqueous`
- ethanol / EtOH -> `ethanol`
- methanol -> `methanol`
- PBS -> `pbs`

A `Calculation` or calculation-like legacy `Experiment` does not contribute
sample medium. For example, a simulated water-filled nanogap remains unknown
for the SERS sample-medium dimension.

### Sample preparation

Explicit events are accumulated into one deterministic composite tag:

- `incubation`
- `adsorption`
- `drop_cast`
- `deposition`
- `immobilization`
- `mixing`
- `drying`

Thus `drop-cast ... and dried` becomes `drop_cast+drying`; the events are not
treated as competing values and therefore do not become `ambiguous`.

### Substrate/sample state

Only explicit local state evidence is accepted:

- `as_prepared`
- `solution`
- `solid`
- `dry`
- `stored`
- `aged`
- `oxidized`

Subject-global `substrate_state` attributes are deliberately ignored because
one substrate identity may be measured fresh, dry, in solution, or after
storage in different experiments.

## Structured conditions

Exact condition names are used for optical/protocol scalar dimensions:

- excitation/laser wavelength
- laser power
- acquisition/integration/exposure time

This increases coverage without fuzzy matching arbitrary condition names such
as precursor concentration, particle concentration, illumination duration, or
simulation dielectric parameters.

## Audit additions

`summary.json` and `audit.json` now expose:

- `method_dimension_status_counts`
- `method_provenance_scope_counts`
- `protocol_matched_dimension_counts`
- `protocol_mismatched_dimension_counts`
- `protocol_pairs_with_any_match`

The structural audit also fails on:

- global entity concentration leakage;
- method protocol dimensions sourced from non Measurement/Experiment nodes;
- simulation/calculation medium leakage.

## Unchanged safety rules

- missing context remains `unknown`, not compatible;
- no graph nodes/edges are invented;
- numeric ranking still requires both observable and protocol gates;
- Bridge, ProjectionSemantics, and CorpusSemantics are unchanged;
- holdout papers are not used for calibration or policy tuning.
