# alpha4c.5b — Explicit Trend-aware Hypothesis Input Contract

## Scope

alpha4c.5b creates the first explicit integration envelope between:

1. the existing Explorer-grounded `HypothesisContext`, and
2. the frozen alpha4c.5a `HypothesisTrendGroundingBundle`.

It does **not** change Hypothesis Maker, prompt, compiler, validator, runtime,
or `premise_statement_ids`.

```text
HypothesisContext -----------------------┐
                                        ├─> TrendAwareHypothesisInput
HypothesisTrendGroundingBundle ----------┘
```

## Why a new envelope

The current Hypothesis Maker contract uses Explorer statement IDs as positive
premises. Trend grounding IDs are a different provenance namespace and must
not be inserted into `premise_statement_ids` or `gap_statement_ids`.

alpha4c.5b therefore projects Trend grounding into explicit role lanes while
keeping those IDs separate.

## Corpus binding

The 5a bundle SHA-locks its source artifacts. alpha4c.5b:

1. verifies the 5a bundle SHA,
2. verifies every source-artifact SHA,
3. reads the uniquely locked `trend_summary`,
4. takes `corpus_id`, domain, paper IDs and Trend semantics from that summary,
5. requires `HypothesisContext.corpus_id` to equal that locked corpus ID.

No caller-supplied corpus assertion can override this binding.

## Role lanes

A single relation grounding may appear in more than one lane.

| 5a capability | 5b lane |
|---|---|
| local empirical premise allowed | `local_empirical_support` |
| cross-context replicated premise allowed | `cross_paper_replicated_support` |
| context-dependency premise allowed | `context_dependency_signal` |
| reversal counterevidence required | `reversal_boundary` |
| replication-gap signal allowed | `replication_gap` |

For the completed v2 seen fixture:

```text
status = insufficient

local_empirical_support = 1
replication_gap = 1

cross_paper_replicated_support = 0
context_dependency_signal = 0
reversal_boundary = 0
```

This preserves the paper-local empirical result while refusing to manufacture
cross-paper replication.

## Maker boundary

Every Trend input view has:

```text
maker_selectable = false
causal_use_allowed = false
universal_use_allowed = false
```

This is deliberate. alpha4c.5b is an integration/lineage contract only.

A future phase must explicitly extend the hypothesis draft/compiler/validator
with a separate Trend-ID namespace before an LLM may select these views.

## Zero yield

A zero-yield 5a bundle creates:

```text
trend_views = []
lane_counts = {}
```

It does not fabricate a research gap.

## Holdout policy

alpha4c.5b uses only the completed v2 result as a seen regression fixture.
The untouched 14-paper v3 reserve is not consumed.
