# alpha4c.5a — Trend-to-Hypothesis Grounding Contract

## Purpose

alpha4c.5a does not modify `HypothesisContext` and does not call an LLM.

It creates a deterministic sidecar between the frozen empirical Trend stack and
future hypothesis synthesis:

```text
TrendEvidence
  -> PaperLocalTrendResult
  -> TrendContextProfile
  -> CrossContextTrendAssessment
  -> HypothesisTrendGroundingBundle
```

The existing `HypothesisContext` remains Explorer-report grounded. This avoids
silently pretending that Trend results are Explorer statements.

## Mapping

### Paper-local Trend result

May provide a scoped empirical premise in a later integration phase.

It is never by itself cross-paper replication or a universal relation.

### repeated

May provide cross-paper empirical support.

A directional cross-paper premise is allowed only if the member results expose
one resolved direction. Even then, universal wording remains forbidden.

### context_specific

Does not become generic replicated support.

It may seed a `context_dependency` hypothesis signal and must preserve the
known differentiating dimensions. Unknown context remains unknown.

### reversed

Never collapses by majority vote.

Both directions/source identities remain in the grounding. It may seed a
boundary-condition/context-dependency hypothesis and requires explicit
counterevidence treatment.

### insufficient

Never becomes positive cross-paper support and never means negative evidence.

It may only contribute a replication/verification gap while the underlying
paper-local Trend result remains locally scoped empirical support.

## Causality

Trend grounding never authorizes causal claims, including when source text
uses causal language. `reported_correlation` is separately tracked as
association-only.

Mechanism evidence stays in the existing mechanism/evidence lanes and cannot
be promoted into empirical Trend support.

## Zero yield

Zero Trend/local-result yield creates an empty valid grounding bundle. It does
not fabricate a research gap.

## Development data

The completed alpha4c.4d.2 v2 result is now seen and is used only as a
regression fixture for alpha4c.5a. The 14-paper v3 reserve remains untouched.

Expected v2 behavior:

```text
particle_size -> sers_performance
paper-local empirical support: allowed
cross-context status: insufficient
cross-paper replicated premise: forbidden
replication gap signal: allowed
causal claim: forbidden
universal claim: forbidden
```

The next integration phase must explicitly consume this sidecar; alpha4c.5a
alone cannot add new `premise_statement_ids` to Hypothesis Maker.
