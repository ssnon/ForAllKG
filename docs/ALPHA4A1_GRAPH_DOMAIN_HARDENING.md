# v2.9.0 alpha4a.1 — graph-domain hardening

This patch closes four domain-leaky seams exposed by the first real Au–Ag SERS
strict/evidence smoke test.

## Included

- `GraphDomainAdapter` with a fail-closed adapter registry.
- DAC-HER keeps the existing catalyst-role inference.
- SERS performs no electrocatalysis-specific role coercion.
- Strict recovery re-applies the extraction-domain vocabulary gate after
  semantic-patch normalization and micro-reextract normalization.
- Paper-level resolution is scoped by the active `ScientificDomainProfile`.
- Reviewed resolution decisions are validated against the active profile's
  resolvable node types.
- Paper-graph metadata records the active graph adapter.

## Deliberately deferred

- SERS relation domain/range constraints.
- ChemicalSpecies role-model migration.
- SERS claim-overlap lexical semantics.
- SERS Bridge adapter/policy.
- SERS feasibility adapter.
