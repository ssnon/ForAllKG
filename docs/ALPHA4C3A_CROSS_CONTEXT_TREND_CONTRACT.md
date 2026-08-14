# alpha4c.3a — Generic CrossContextTrend Contract

## Scope

alpha4c.3a introduces the domain-independent contract for comparing already
frozen paper-local scientific trends across papers.

It does **not**:

- extract new TrendEvidence;
- change alpha4c.2 trend or precision semantics;
- implement SERS context projection;
- infer real-data cross-context statuses;
- use majority voting;
- create causal claims.

The only scientific input unit is `PaperLocalTrendResult`.

## Flow

```text
PaperLocalTrendResult
        |
        v
TrendContextProfile
        |
        v
PairwiseTrendContrast
        |
        v
CrossContextTrendAssessment
```

A `TrendContextProfile` is one-to-one with a `PaperLocalTrendResult`.

## Relation identity

The relation key is:

```text
independent_variable_key
+ dependent_observable_key
+ control_family
+ observable_semantics
```

Direction, shape, paper, and evidence kind are intentionally excluded.

That permits positive and negative observations of the same scientific
control-response relation to meet at the cross-context layer, where a reversal
can be detected without collapsing either observation.

## Context states

The generic dimension contract supports:

- `known`
- `unknown`
- `ambiguous`
- `varied_control`
- `not_applicable`

Unknown context is an information state, not structural failure.

`varied_control` explicitly reserves the dimension corresponding to the
independent variable so alpha4c.3b/3c will not treat the experimental control
being varied as an ordinary context mismatch.

## Pairwise constraints

A `PairwiseTrendContrast`:

- is cross-paper only;
- must stay inside one relation ID;
- preserves direction, shape, evidence-kind, and context relations separately;
- partitions context dimensions into disjoint matched / mismatched / unknown /
  ambiguous / varied-control / not-applicable buckets;
- does not inherit numeric-ranking decisions as trend semantics.

Strict direction reversal is only:

```text
positive <-> negative
```

`positive <-> non_monotonic` is represented separately as
`monotonic_vs_non_monotonic`.

## Final status vocabulary

The generic contract reserves:

- `repeated`
- `context_specific`
- `reversed`
- `insufficient`

alpha4c.3a does not generate these statuses from real data. alpha4c.3c will
implement deterministic assessment.

The contract already enforces the non-majority-vote invariant:

> if an assessment contains any strict positive/negative reversal pair, the
> assessment status must be `reversed`.

Thus:

```text
positive + positive + negative
```

cannot be collapsed to `repeated`.

## Evidence modalities

These remain separate through the final assessment object:

- `experimental_numeric`
- `calculated_numeric`
- `reported_claim`

Cross-kind directional concordance is not represented as multiple
experimental replications.

## Provenance

`TrendContextProfile` can preserve:

- member TrendEvidence IDs;
- ComparisonContext IDs;
- MethodContext IDs;
- Claim IDs;
- Measurement IDs;
- MeasurementResult IDs;
- Calculation IDs;
- source node IDs.

alpha4c.3b must populate context only through explicit provenance linkage.
Paper-global context fallback is outside the contract.

## Adapter boundary

`CrossContextTrendAdapter` is a **domain context-projection adapter**, not an
assessment policy.

Domain-specific responsibilities are:

- define context dimensions;
- project sidecar context to exactly one profile per paper-local trend;
- explicitly mark the independent-variable dimension as `varied_control`.

Pairwise comparison and status policy remain generic/deterministic and will be
implemented in alpha4c.3c.

The alpha4c.3a registry is intentionally empty. alpha4c.3b will register the
SERS adapter under the already-frozen `trend_adapter_id`, so no new field is
added to `ScientificDomainProfile`.

## Structural audit

`audit_cross_context_trends` checks:

1. exactly one context profile per `PaperLocalTrendResult`;
2. profile relation fields, direction/shape, members, and evidence kinds match
   the local result;
3. pairwise contrasts are cross-paper and relation-consistent;
4. pairwise direction/shape/evidence-kind labels agree with their profiles;
5. every local result belongs to exactly one relation assessment;
6. every pairwise contrast belongs to exactly one assessment;
7. direction buckets exactly reflect local-result directions;
8. evidence-kind buckets preserve modality;
9. reversal pair IDs exactly match positive/negative contrasts;
10. any reversal pair forces `status="reversed"`;
11. repeated status requires independent support from at least two papers.

## Next

alpha4c.3b will implement SERS context projection from the already-bound
ComparisonContext and MethodContext sidecars with:

- explicit measurement/provenance linkage;
- no paper-global fallback;
- unknown/ambiguous preservation;
- independent-variable -> `varied_control` masking.
