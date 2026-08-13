# alpha4b.3b.3.2 — Protocol Role Semantics Precision

## Purpose

alpha4b.3b.3.1 recovered more explicit SERS method metadata, but calibration
showed two semantic problems:

1. `partially_matched` could be emitted when `matched_dimensions` was empty.
2. the legacy `medium` / `substrate_state` slots conflated several distinct
   scientific roles.

This patch changes classification and role assignment only. It does not add
LLM inference, fuzzy completion, cross-paper merging, or hold-out tuning.

## Protocol comparability

`partially_matched` now requires at least one explicit matched dimension.

- critical mismatch -> `different_protocol`
- every method dimension known and matched -> `same_protocol`
- at least one explicit match, no critical mismatch -> `partially_matched`
- mismatch evidence with zero matches -> `different_protocol`
- neither match nor mismatch evidence -> `unknown`

`harmonized_protocol` remains a reserved state. It is not automatically
produced from exact critical matches; a future explicit equivalence/harmonizing
contract would be required.

## Method role split

The old method dimensions:

- `medium`
- `substrate_state`

are replaced by:

- `sample_preparation`
- `preparation_medium`
- `measurement_environment`
- `sample_state`
- `substrate_condition`

Examples:

- `aqueous MB solution ... dried on target`
  -> preparation medium `aqueous`; sample state `dry`
- `solution-based SERS`
  -> measurement environment `solution`
- `MB solid on glass`
  -> sample state `solid`
- `as-synthesized nanoparticles`
  -> substrate condition `as_prepared`
- `as-prepared and stored substrates were compared`
  -> substrate condition `ambiguous`, not a fabricated composite state

Multiple preparation actions such as `drop_cast+drying` may remain a known
sequence. Multiple mutually exclusive sample states or substrate conditions
fail closed as ambiguous.

## Comparison role split

SERS `ComparisonContext` no longer contains `medium` or `substrate_state`.
Observable policies use the role-specific comparison dimensions:

- `measurement_environment`
- `sample_state`
- `substrate_condition`

Preparation history and preparation medium remain MethodContext/protocol
dimensions and are enforced by the protocol gate rather than duplicated into
every observable comparison policy.

## Safety

- missing metadata remains `unknown`
- global analyte/reporter concentration is still prohibited
- simulation medium is not a physical SERS medium
- subject-global state is not propagated into measurement context
- numeric ranking still requires both observable and `same_protocol` gates
- no Bridge, extraction, projection, or corpus topology changes
- calibration remains SERS_1 / SERS_5 / SERS_8 only
