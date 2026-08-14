# alpha4c.3c — Deterministic Pairwise Contrast & Assessment

## Scope

alpha4c.3c consumes the frozen `TrendContextProfile` rows produced by
alpha4c.3b and deterministically builds:

```text
TrendContextProfile
        |
        v
complete same-relation cross-paper pairs
        |
        v
PairwiseTrendContrast
        |
        v
one CrossContextTrendAssessment per relation
```

It does not:

- extract or rewrite TrendEvidence;
- re-run context projection;
- use ComparisonContext numeric-ranking decisions as trend policy;
- perform LLM reasoning;
- promote causal status;
- merge different relation IDs to manufacture overlap.

## Semantics

```text
contract:
cross_context_trend_contract_v1_alpha4c3a

SERS context:
sers_au_ag_trend_context_v1_alpha4c3b

assessment:
cross_context_trend_assessment_v1_alpha4c3c
```

The alpha4c.2 trend/precision semantics remain frozen.

## Complete pair generation

Profiles are grouped by the stable relation identity defined in alpha4c.3a:

```text
independent_variable_key
+ dependent_observable_key
+ control_family
+ observable_semantics
```

Every unordered pair of profiles in one relation is generated **only if the
papers differ**.

Same-paper numeric + reported-claim support never forms a pair.

The deterministic audit independently reconstructs the expected complete pair
set and fails if any cross-paper same-relation pair is missing or extra.

## Context comparison

For every shared context dimension:

```text
both varied_control       -> varied_control
both not_applicable       -> not_applicable
either not_applicable     -> ambiguous
either ambiguous          -> ambiguous
either unknown            -> unknown
known == known            -> matched
known != known            -> mismatched
```

A varied-control disagreement within the same relation fails closed because
the independent-variable mask should be relation-consistent.

The overall pair context relation is:

```text
known mismatch exists
    -> context_different

no mismatch + unresolved dimensions + at least one match
    -> context_partially_known

no mismatch + no resolved matching background context
    -> context_unknown

all relevant resolved context matches
    -> same_context
```

`varied_control` never becomes a context mismatch.

## Direction relation and pair role

The generic alpha4c.3a direction classifier remains authoritative.

Pair roles are deterministic:

```text
positive <-> negative
    -> reversal

same_direction
same_non_monotonic
    -> repeated

monotonic_vs_non_monotonic
or unchanged_contrast
+ at least one known context mismatch
    -> context_specific

otherwise
    -> unresolved
```

Shape differences are preserved in the contrast and reason codes but do not by
themselves turn directional agreement into context-specific evidence.

This avoids interpreting, for example, positive monotonic and positive
saturating as contradictory trends.

## Assessment priority

One assessment is built per relation with strict priority:

```text
1. any reversal pair
      -> reversed

2. otherwise any context_specific pair
      -> context_specific

3. otherwise any repeated pair
      -> repeated

4. otherwise
      -> insufficient
```

This is deliberately non-majoritarian.

Example:

```text
Paper A: positive
Paper B: positive
Paper C: negative
```

contains:

```text
A-B repeated
A-C reversal
B-C reversal
```

and the final relation status is always:

```text
reversed
```

never `repeated`.

## Independent support

A relation represented by several results from only one paper remains:

```text
insufficient
```

even if those results include different evidence modalities.

For example:

```text
same paper:
experimental numeric positive
reported claim positive
```

is not two-paper replication.

## Evidence modalities

Assessment continues to preserve:

```text
experimental_numeric_result_ids
calculated_numeric_result_ids
reported_claim_result_ids
```

Pairwise evidence-kind relation remains:

```text
same_kind
cross_kind
mixed_kind
unresolved
```

No scalar support count replaces these identities.

## Differentiating and unresolved dimensions

For a reversal or context-specific trend-character difference, known mismatched
context dimensions are recorded as `differentiating_dimensions`.

This records co-occurring context differences; it does **not** assert that
those dimensions caused the reversal.

Unknown and ambiguous pair dimensions are unioned into
`unresolved_dimensions` in the relation assessment.

## Structural audit

The alpha4c.3c audit runs the generic alpha4c.3a audit and additionally checks:

1. the complete expected cross-paper pair set;
2. deterministic pair content, including context partition and pair role;
3. exactly one assessment per relation;
4. exact relation membership;
5. deterministic assessment status and buckets.

Therefore hand-editing an assessment to majority-vote a reversal, omitting a
cross-paper pair, or counting same-paper support as replication fails the
structural gate.

## Builder

```text
scripts/build_cross_context_assessments.py
```

reads only the frozen precision and alpha4c.3b context sidecars.

It refuses a context source that reports:

```text
pairwise_contrasts_built != false
cross_context_assessments_built != false
paper_global_context_fallback_used != false
numeric_ranking_reused_as_trend_policy != false
```

Output:

```text
cross_context/<context-id>/
  assessment/<assessment-id>/
    pairwise_contrasts.jsonl
    assessments.jsonl
    audit.json
    summary.json
```

## Current SERS calibration / seen expectation

The current 1/5/8 and 2/6/10 sets need not contain cross-paper overlap for a
relation. Many or all assessments being `insufficient` is valid.

In particular, the two SERS_10 `shell_thickness -> Raman intensity` local
results originate from the same paper. They must remain in one relation
assessment but create no pair and must not be labeled repeated.

Synthetic tests, rather than output-count tuning, exercise the positive,
reversal, context-specific, unresolved, cross-kind, and no-majority-vote
cases.

## Next

After alpha4c.3c passes the frozen calibration/seen regression, alpha4c.3 is a
complete deterministic context-aware trend-assessment layer.

A genuinely unseen trend holdout should then be reserved for alpha4c.4 rather
than further tuning 1/5/8 or 2/6/10.
