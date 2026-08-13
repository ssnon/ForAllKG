# alpha4b.2a — reusable Bridge capability contracts + HER parity

This phase is architecture-only. It does not register SERS Bridge semantics.

## Goals

- Keep mature DAC-HER Bridge behavior and direct Python callers intact.
- Make signature, validation, policy, and implementation-file ownership explicit.
- Let future domain adapters own domain-specific fingerprint files.
- Inject domain-specific anchor-context checks into shared Bridge validation.
- Preserve frozen HER fingerprint behavior while separating non-HER domain IDs.

## Non-goals

No SERS Bridge adapter, SERS prompt/policy, Bridge schema changes, GraphAgents
projection changes, corpus changes, strict extraction changes, or hypothesis
changes are introduced here.

The next phase is alpha4b.2b: SERS Bridge semantics on this capability boundary.
