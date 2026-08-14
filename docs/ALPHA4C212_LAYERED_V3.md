# alpha4c.2.1.2 — layered v4

v4 addresses the final focused-test failure from v3.

## What v3 revealed

The alpha4c212 wrapper originally refined only TrendEvidence already emitted by
the alpha4c211 adapter.

The synthetic source claim:

```text
The closer the SPR ... to the exciting laser wavelength,
the higher the enhancement factor will be.
```

is a valid explicit spectral-alignment relation, but alpha4c211 did not emit a
TrendEvidence row for that synthetic wording. Therefore the alpha4c212
detuning refinement never ran.

This was not a required-input problem anymore; the base extractor simply
returned no matching evidence.

## v4 solution: narrow supplemental claim lane

The active alpha4c212 extractor now has two stages:

```text
alpha4c211 extraction
        |
        v
refine any emitted spectral-alignment claim
        |
        +---- scan explicit Claim nodes not already emitted
                    |
                    | only if all hold:
                    | - SPR/LSPR/plasmon resonance named
                    | - excitation/laser named
                    | - matching/proximity named
                    | - signed closer/lower-detuning -> higher relation
                    | - formal enhancement factor explicitly named
                    v
             supplemental TrendEvidence
```

The supplemental lane is intentionally narrow.

It does **not** emit a trend for a statement such as:

```text
SPR matching is important for SERS enhancement.
```

because that sentence does not state a signed relationship.

It also does not replace the historical parser for generic optical claims.

## Deduplication

If alpha4c211 already emitted evidence from the same source Claim node,
alpha4c212 does not add a supplemental duplicate.

After refinement/supplementation, exact `trend_id` duplicates must be identical;
conflicting rows raise an error.

## Causality

The matching relation is represented as:

```text
spr_excitation_detuning increases
-> SERS enhancement factor decreases
```

with:

```text
evidence_basis = reported_directional_claim
causal_status = not_asserted
```

The system does not convert a source association into a causal claim.

## Structural consolidation

The landmark-aware one-row-before-regroup implementation from v3 is retained
unchanged.
