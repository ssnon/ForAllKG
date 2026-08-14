# alpha4c.2.1.2.1.1 — Precision Semantic Reconciliation

This patch fixes a precision-side stale-semantics bug discovered after
alpha4c.2.1.2.1 successfully re-grounded the SERS_8 spectral relation.

## Observed state

Trend evidence was correct:

```text
spr_excitation_detuning
-> sers_enhancement_factor
negative
```

The active annotation was also correct:

```text
control_family = optical_alignment
observable_semantics = formal_sers_enhancement_factor
```

However the historical consolidation ancestry materialized the corresponding
PaperLocalTrendResult with:

```text
control_family = other
```

The precision audit therefore correctly raised:

```text
local_result_control_family_mismatch
```

and failed the structural gate.

## Cause

alpha4c.2.1.2.1 introduced a new trend axis after the historical precision
consolidation rules were written. The active annotation understands the new
axis, but an inner historical consolidator can still emit its older fallback
semantic label.

## Repair

Trend semantics are unchanged.

A new precision epoch wraps alpha4c2121 and reconciles exactly two semantic
fields from the active member annotations after historical consolidation:

- `control_family`
- `observable_semantics`

All member annotations in one PaperLocalTrendResult must agree. If they do not,
the patch fails closed rather than choosing a value.

No relation, direction, evidence kind, subject identity, provenance, member
identity, or support count is changed by this reconciliation.

## Semantics

Trend remains:

```text
sers_au_ag_trend_v5_alpha4c2121
```

Precision becomes:

```text
sers_au_ag_trend_precision_v5_alpha4c21211
```

## Preserved invariants

- canonical Claim re-grounding unchanged
- structural-family consolidation unchanged
- numeric / claim lane separation unchanged
- DDA calculated numeric classification unchanged
- Au:Ag -> Ag/Au normalization unchanged
- no cross-paper numeric ranking
- no majority-vote synthesis
