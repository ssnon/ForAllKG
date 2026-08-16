# alpha4c.5g.2 — Candidate Trend v6 Development Regression

This phase does not activate a new production Trend adapter.

It adds a candidate semantics:

`sers_au_ag_trend_v6_alpha4c5g2`

and evaluates it only on the already-open 53-paper development partition.

## Fix 1: quantitative nanogap control precedence

alpha4c211 introduced both:

- `nanogap_size`
- `nanogap_presence`

A claim containing an explicit dimensional phrase such as `interior nanogap
size` can therefore match both families and fail the generalized control
selector even though the historical v1 parser admitted the same grounded
claim.

The candidate gives explicit dimensional wording (`size`, `width`,
`distance`) precedence over the categorical presence family.

This is intentionally limited to the demonstrated nanogap regression.

## Fix 2: explicit pair comparative direction

A grounded statement such as:

`SERS EF is greater for the 2-nm gap than for the 8-nm gap`

contains enough information to derive a direction in the canonical
increasing-gap frame:

`gap increases -> EF decreases`

The candidate accepts only explicit numeric length pairs with a clear
greater/higher/stronger or lower/weaker relation.

## Fix 3: measurement-local method context precedence

Comparison/MethodContext remains frozen and unchanged.

For Trend compatibility only, an ambiguous excitation-wavelength MethodContext
is locally resolved when all ComparisonContexts using that exact method
context point to Measurements that each explicitly declare exactly one
excitation wavelength and all those local values agree.

Thus a Measurement that explicitly says `532 nm` is not blocked merely
because a broader experiment optical context also contains `633 nm`.

If any local measurement is missing an explicit wavelength, contains multiple
local values, or local values disagree, the ambiguity remains.

## Development regression

The regression requires:

- exact 53 development papers;
- current registered semantics remains v5;
- no v5 scientific evidence signature is removed;
- candidate TrendEvidence structural gate passes;
- the four audited nanogap claims are recovered;
- Reserve A is not read;
- Reserve B is not read and remains sealed;
- zero LLM calls.

A PASS does not activate v6. Activation/freeze is a later protocol epoch.
