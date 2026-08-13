# alpha4b.2b — SERS Bridge semantics

This phase registers `sers_au_ag` as the second Bridge domain on the reusable
alpha4b.2a capability boundary.

## Added

- SERS-specific Bridge extraction prompt and candidate-recovery prompt.
- SERS scientific strict-node catalog: Au/Ag metal identity, architecture,
  morphology, structural motifs, and compact measurement context.
- Conservative hard anchor validation: explicit disjoint metal identity only;
  missing detail is not treated as contradiction.
- Shared plugin policy lane runtime (`bridge_policy_runtime.py`).
- SERS policy semantics for reusable relation patterns vs scalar/numeric strict
  evidence.
- SERS Bridge adapter registration and profile activation.

## Epistemic rules

- Zero Bridge concepts is valid.
- Raw EF/AEF/LOD/intensity values are not Bridge concepts.
- Explicit/grounded relations may be retained as patterns.
- Relation-cue ambiguity is a semantic candidate, not silently accepted.
- The Bridge layer never performs cross-paper numeric ranking.
- Measurement-context compatibility is deferred to alpha4b.3b.

## Not changed

- DAC-HER Bridge policy/prompt/signatures.
- Bridge schemas.
- strict SERS extraction/graph semantics.
- GraphAgents projection semantics (alpha4b.2c).
- corpus alignment semantics (alpha4b.3a).
