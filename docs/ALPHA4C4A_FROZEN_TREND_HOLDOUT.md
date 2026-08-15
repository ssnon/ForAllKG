# alpha4c.4a — Frozen Curated-Corpus Trend Holdout Split

## Scope

alpha4c.4a freezes the genuinely unseen Trend-semantics evaluation split from
the pre-existing curated `Kiwook_SERS_1..38` corpus.

This phase does **not**:

- inspect or extract TrendEvidence from holdout papers;
- run Strict, Bridge, Comparison, Trend, precision, or cross-context builders;
- use the production acquisition snapshot;
- change alpha4c.1–4c.3 semantics;
- select papers by scientific content, trend presence, trend direction, or
  expected overlap.

The production acquisition snapshot remains reserved for later full E2E
validation from literature discovery through hypothesis generation.

## Development contamination registry

Only the six papers used during alpha4c development are excluded:

```text
calibration
  Kiwook_SERS_1
  Kiwook_SERS_5
  Kiwook_SERS_8

seen regression
  Kiwook_SERS_2
  Kiwook_SERS_6
  Kiwook_SERS_10
```

The remaining 32 papers form the clean candidate pool.

## Deterministic split

Selection semantics:

```text
trend_holdout_split_v1_alpha4c4a
```

Algorithm:

```text
sha256_namespace_paper_id_rank_v1
```

Namespace:

```text
sers-alpha4c4-v1
```

For every candidate paper:

```text
rank = SHA256("sers-alpha4c4-v1|" + paper_id)
```

Candidates are sorted lexicographically by `(rank, paper_id)`.

The first 10 papers become the frozen alpha4c.4 holdout; the remaining 22 are
kept untouched as future reserve.

Frozen holdout:

```text
Kiwook_SERS_16
Kiwook_SERS_35
Kiwook_SERS_34
Kiwook_SERS_19
Kiwook_SERS_13
Kiwook_SERS_37
Kiwook_SERS_30
Kiwook_SERS_25
Kiwook_SERS_4
Kiwook_SERS_18
```

Future reserve:

```text
Kiwook_SERS_26
Kiwook_SERS_15
Kiwook_SERS_14
Kiwook_SERS_11
Kiwook_SERS_38
Kiwook_SERS_20
Kiwook_SERS_24
Kiwook_SERS_23
Kiwook_SERS_9
Kiwook_SERS_36
Kiwook_SERS_33
Kiwook_SERS_32
Kiwook_SERS_29
Kiwook_SERS_27
Kiwook_SERS_12
Kiwook_SERS_3
Kiwook_SERS_22
Kiwook_SERS_7
Kiwook_SERS_31
Kiwook_SERS_17
Kiwook_SERS_21
Kiwook_SERS_28
```

This selection uses only the paper ID. No title, abstract, paper text,
extraction result, TrendEvidence, direction, shape, or cross-context overlap
enters the split.

## Frozen scientific semantics

```text
MeasurementResultIdentity
  measurement_result_identity_v1_alpha4b4a1

Comparison
  sers_au_ag_comparison_v7_alpha4b3b321

MethodContext
  sers_au_ag_method_v4_alpha4b3b321

Trend contract
  trend_evidence_contract_v1_alpha4c1

SERS Trend
  sers_au_ag_trend_v5_alpha4c2121

Trend precision
  sers_au_ag_trend_precision_v5_alpha4c21211

Cross-context contract
  cross_context_trend_contract_v1_alpha4c3a

SERS context projection
  sers_au_ag_trend_context_v1_alpha4c3b

Cross-context assessment
  cross_context_trend_assessment_v1_alpha4c3c
```

The split is bound to the reviewed branch state at base commit:

```text
0498b065f290281d0d63949f598be0a4ae6c7dc4
```

and to per-file Git blob hashes in the protocol. Unrelated later commits do not
silently change the holdout scientific implementation; the verifier checks the
actual working-tree blobs.

## No output-count success targets

alpha4c.4b must not require any minimum number of:

- TrendEvidence rows;
- cross-paper same-relation pairs;
- repeated assessments;
- reversed assessments;
- context-specific assessments.

Nor may it impose a maximum number of `insufficient` assessments.

Zero usable trends or all-insufficient assessments are valid holdout outcomes
if the structural/provenance/safety contracts pass.

## Epoch policy

If a holdout paper exposes a genuine generic implementation or contract bug and
the implementation is changed after inspecting that holdout:

1. the current 10 holdout papers become **seen regression** material;
2. the generic fix is replayed on prior calibration/seen sets;
3. a new holdout epoch must be selected only from the untouched 22-paper reserve;
4. the failed holdout split is never resumed as a blind evaluation.

Trend-distribution differences alone are not grounds for a fix.

## Verification

After installing:

```bash
python -m scripts.verify_sers_alpha4c4a_holdout_protocol
```

alpha4c.4b should consume the frozen holdout list from the protocol rather than
accepting an arbitrary paper list from the command line.
