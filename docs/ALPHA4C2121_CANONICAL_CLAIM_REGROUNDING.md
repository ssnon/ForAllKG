# alpha4c.2.1.2.1 — Canonical Claim Re-grounding

This is a narrow follow-up to alpha4c.2.1.2.

## Observed real-data failure

The alpha4c212 calibration still represented the grounded SERS_8 claim

```text
mech_lspr_laser_matching_increases_ef
```

as:

```text
excitation_wavelength -> raman_intensity
positive
```

even though its canonical Claim node stores:

```text
statement:
Ag–Au SERS enhancement increases when the
surface-plasmon-resonance wavelength is closer to the
excitation laser wavelength.

description:
... associated with higher enhancement factor.
```

The failure had two causes:

1. the spectral matcher did not treat the hyphenated canonical phrase
   `surface-plasmon-resonance` as equivalent to `surface plasmon resonance`;
2. alpha4c212 primarily refined the already-emitted evidence text rather than
   re-reading the grounded canonical Claim node that the evidence referenced.

## Repair

alpha4c2121 does not add another broad extraction lane.

Instead:

```text
alpha4c212 TrendEvidence
        |
        | source_claim_ids
        v
canonical Claim node
(statement + description + label + source expression)
        |
        | matching-only normalization
        v
semantic re-grounding
```

If the canonical Claim explicitly supports SPR/excitation proximity and a
signed closer/higher relation, the independent variable is normalized to:

```text
spr_excitation_detuning
```

Direction is defined with respect to increasing detuning:

```text
detuning increases -> enhancement decreases
direction = negative
shape = monotonic
```

The dependent observable is promoted to formal SERS enhancement factor only
when the canonical Claim text itself preserves explicit `enhancement factor`
wording.

## Provenance rule

Canonical Claim text is used for semantic matching only.

The existing `source_expression` and `source_expressions` in TrendEvidence are
not silently rewritten. The original `source_claim_ids` continue to provide
the provenance link to the canonical Claim node.

## Causality

The re-grounded relation uses:

```text
causal_status = not_asserted
```

No proximity/correlation statement is promoted to causation.

## Preserved invariants

- alpha4c.1 generic TrendEvidence contract unchanged
- alpha4c212 historical modules unchanged
- alpha4c212 structural-family consolidation unchanged
- DDA calculated-vs-experimental classification unchanged
- Au:Ag -> Ag/Au normalization unchanged
- numeric and claim lanes remain separate
- no cross-paper numeric ranking
- no majority-vote trend synthesis

## Expected calibration repair

For Kiwook_SERS_8:

```text
BEFORE:
excitation_wavelength -> raman_intensity
positive / measurement

AFTER:
spr_excitation_detuning -> sers_enhancement_factor
negative / optical_alignment
formal_sers_enhancement_factor
```

The seen 2/6/10 regression must remain unchanged and PASS.
