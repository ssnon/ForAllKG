# alpha4c.5c — Separate Trend-ID Draft / Compiler / Validator Contract

## Scope

alpha4c.5c is the first phase in which a hypothesis draft may explicitly
reference frozen Trend input views.

It still does not modify or invoke the existing Hypothesis Maker prompt/runtime.

The legacy Explorer-only stack remains intact.

```text
legacy:
HypothesisContext
  -> HypothesisPortfolioDraft
  -> HypothesisCompiler
  -> HypothesisValidator

alpha4c.5c:
TrendAwareHypothesisInput
  -> TrendAwareHypothesisPortfolioDraft
  -> TrendAwareHypothesisCompiler
  -> TrendAwareHypothesisValidator
```

## Separate namespaces

Explorer provenance remains:

```text
premise_statement_ids
gap_statement_ids
```

Trend provenance uses:

```text
trend_references[].view_id
trend_references[].use_role
```

A Trend view ID is never reinterpreted as an Explorer statement ID.

## Trend use roles

```text
positive_empirical_support
    -> local_empirical_support

cross_paper_empirical_support
    -> cross_paper_replicated_support

context_qualification
    -> context_dependency_signal

counterevidence_boundary
    -> reversal_boundary

replication_gap
    -> replication_gap
```

Any use/lane mismatch fails compilation.

## Positive support

A new trend-aware draft may have zero Explorer positive premises if it has at
least one positive Trend reference.

Therefore a Trend-only hypothesis is structurally allowed.

However gaps, context qualifications and reversal boundaries cannot satisfy the
positive-support requirement.

## Cross-context limitation preservation

Selecting positive Trend support triggers mandatory companion references:

```text
status = insufficient
    -> replication_gap required

status = context_specific
    -> context_qualification required

status = reversed
    -> context_qualification
       + counterevidence_boundary required

status = repeated
    -> no limitation companion required
```

This prevents a hypothesis from citing a local Trend while silently dropping
the exact cross-context limitation discovered by alpha4c.3.

## Cross-paper support

Only a view from:

```text
cross_paper_replicated_support
```

may be used as:

```text
cross_paper_empirical_support
```

and the view must contain at least two source papers.

A paper-local Trend plus a replication gap does not set
`cross_paper_synthesis=true`.

## Causality and universality

Every compiled Trend reference and card carries:

```text
trend_causal_authorization = false
trend_universal_authorization = false
```

These fields mean the Trend evidence does not establish causality or a
universal relation.

They do not forbid a generated hypothesis from proposing a causal mechanism as
an explicit inferential bridge in a later Maker phase.

## Numeric values

5b Trend input views do not expose raw numeric values. Consequently 5c does not
allow Trend references to license generated numerical predictions.

Exact numeric generation remains licensed only by selected Explorer positive
premise text.

## v2 seen regression

The completed v2 Trend result is used with a synthetic empty Explorer context.

The deterministic regression deliberately produces a Trend-only draft:

```text
Explorer positive premises = 0

Trend references:
  local_empirical_support -> positive_empirical_support
  replication_gap        -> replication_gap
```

Expected compiled semantics:

```text
trend positive support count = 1
trend gap count = 1
cross_paper_synthesis = false
trend causal authorization = false
trend universal authorization = false
validation = PASS
```

The synthetic context is a contract fixture, not a scientific result.

## Next phase

alpha4c.5d may connect the Hypothesis Maker prompt/runtime to this frozen
contract. That phase must expose Trend view IDs and allowed use roles explicitly
to the LLM instead of reusing `premise_statement_ids`.

The 14-paper v3 reserve remains untouched.
