# alpha4c.5g.1 — Trend Candidate Rejection Attribution

## Purpose

alpha4c.5g showed that the current Trend lane yields 9 local results on the
53-paper development partition, while a deliberately broad diagnostic census
found 206 non-admitted claim candidates and 109 non-admitted numeric-series
candidates.

Those broad candidates are **not ground truth**. alpha4c.5g.1 therefore does
not change Trend semantics. It attributes each miss to the current frozen
local Trend implementation before any semantic decision is made.

## Frozen local implementation

The completed alpha4c.5g summary records the exact implementation SHA256
values that produced the diagnostic. alpha4c.5g.1 verifies those local bytes
before attribution.

This matters because a remote GitHub checkout may not reflect unpushed local
changes. The 5g summary is the source of truth for the implementation that
actually generated the 9 TrendEvidence records.

## Claim attribution

For every one of the 206 broad claim misses, the current Trend helper stages
are evaluated in order:

```text
claim candidate
   ↓
current _claim_control()
   ├─ fail → claim_control_not_normalized
   ↓
current _claim_response()
   ├─ fail → claim_response_not_normalized
   ↓
current _claim_direction_shape()
   ├─ fail → claim_direction_not_normalized
   ↓
all inspected normalizers pass but claim absent
   → claim_unexplained_post_normalization_miss
```

The unexplained bucket is deliberate. The diagnostic never invents a reason.

## Numeric attribution

For each of the 109 broad numeric misses:

```text
observable supported?
   ↓
condition name normalized by current control parser?
   ↓
identity / comparison / method bindings unique enough?
   ↓
current per-measurement control resolved?
   ↓
same current control family?
   ↓
current control actually varies?
   ↓
current lineage resolved and shared?
   ↓
current method compatibility?
   ↓
otherwise unexplained_post_gate_miss
```

A special diagnostic bucket is emitted when the frozen method guard rejects a
series even though the 5g census found all *other* method dimensions
compatible and the conflicting dimension is exactly the intentionally varied
dimension:

`numeric_varied_dimension_blocked_by_method_guard`

That bucket is particularly important because it can identify a generic
implementation problem without weakening unrelated method/provenance gates.

## Human adjudication sample

The phase writes all attributions plus a deterministic SHA-selected sample of
up to 8 cases per reason bucket.

The sample contains blank fields:

- `human_adjudication`
- `human_reason`
- `recommended_action`

No semantic modification should be made before reviewing representative
samples from the major buckets.

## Safety

- Development 53 only.
- Reserve A scientific artifacts are not read.
- Reserve B is not read and remains sealed.
- No LLM calls.
- No source canonical graph mutation.
- No Trend/Comparison/Precision code mutation.
- No acceptance threshold.
