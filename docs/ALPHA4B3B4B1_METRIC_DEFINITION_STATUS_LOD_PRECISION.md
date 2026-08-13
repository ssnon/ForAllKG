# alpha4b.3b.4b.1 — Metric Definition Status & LOD Criterion Precision

This is a narrow semantic precision patch on top of alpha4b.3b.4b.

## EF status rule

A reported/calculated/estimated EF value does not reveal how EF was defined.

- explicit complete normalization/reference evidence -> `known`
- explicit but incomplete definition component -> `partial`
- merely "calculated EF", "estimated EF", or an EF number -> `unknown`

`partial` therefore means partial *definition evidence*, not partial confidence
that an EF value exists.

## LOD rule

A lowest concentration is interpreted as a detection criterion only when the
source explicitly ties it to detection/identification/observation.

Accepted examples:

- `lowest concentration ... that can be detected`
- `signal could be detected at ...`
- `detection level was ...`

Rejected as definition evidence:

- `lowest concentration ... adsorbed on the substrate`
- `lowest concentration tested`
- `theoretical LOD = ...` without an explicit criterion

No statistical criterion is inferred from the words `theoretical LOD` or
`calculated LOD`.

## Semantics

`sers_au_ag_metric_definition_v2_alpha4b3b4b1`

The canonical graph, comparison layer, MethodContext, ProtocolAssessment and
ReproducibilityEvidence are unchanged.
