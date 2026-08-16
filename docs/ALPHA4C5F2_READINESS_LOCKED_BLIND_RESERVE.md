# alpha4c.5f.2 — Readiness-Locked Blind Reserve E2E

## Purpose

alpha4c.5f.2 is an orchestration-only epoch.

It does **not** change Comparison, TrendEvidence, TrendPrecision,
CrossContext, Trend→Hypothesis grounding, Trend-aware Maker semantics,
compiler/validator semantics, or the independent alpha4c.5e evaluator.

It corrects the reserve-consumption boundary exposed by alpha4c.5f/5f.1 and
applies that corrected boundary to a genuinely new SERS paper pool.

## New pool and blind split

The candidate pool must contain exactly 103 unique SERS paper IDs.

The split is deterministic and consumes **paper_id only**:

```text
score = SHA256(
    "sers_alpha4c5f2_blind_split_v1" + "\0" + paper_id
)

ascending score:
  ranks 1..53    -> development
  ranks 54..78   -> Reserve A
  ranks 79..103  -> Reserve B
```

Scientific graph content, extraction completeness, Measurement values,
Comparison compatibility, Trend yield, CrossContext status, Explorer output,
and Maker output are forbidden split inputs.

Reserve B is recorded but remains sealed for future confirmation.

## Reserve A flow

```text
new 103-paper pool
        ↓
pool_manifest.json
        ↓
blind_split.json
        ↓
5e Reserve-A registration
        ↓
Canonical Readiness preparation
        ↓
canonical_readiness_lock.json
        ↓
protocol freeze
        ↓
5f.2 preflight
        ↓
immediate readiness revalidation
        ↓
guarded_write_consumption_marker()
        ↓
=========== CONSUMED ===========
        ↓
isolated canonical copy
        ↓
Evidence Projection
        ↓
Evidence Corpus
        ↓
MeasurementResultIdentity
        ↓
MetricDefinition
        ↓
Comparison
        ↓
TrendEvidence
        ↓
TrendPrecision
        ↓
CrossContext
        ↓
5a + Explorer
        ↓
5b
        ↓
5d.1 Maker
        ↓
5c.1 Compiler/Validator
        ↓
5e independent evaluation
        ↓
PASS / FAIL
```

The downstream scientific sequence is inherited from the already-frozen
alpha4c.5f runner. The historical alpha4c.5f runner is not modified.

## Canonical readiness

The existing alpha4c.5f.1 gate is reused.

Allowed deterministic canonical migration remains limited to:

- canonical missing,
- stale Measurement merge invariant,
- Measurement numeric/text XOR violation.

No extraction LLM is called during readiness preparation.

The readiness lock is revalidated **immediately before** the real reserve
consumption marker. The new runner is forbidden from directly writing its
consumption marker.

## Acceptance semantics

No distribution/count target is introduced.

These remain valid outcomes:

- zero Trend yield,
- zero hypotheses / valid abstention,
- many unknown/different protocols,
- insufficient/context-specific/reversed Trend evidence,
- zero numeric ranking eligibility.

Reserve A fails only according to already-frozen structural/provenance/
contract/evaluation semantics.

A failed Reserve-A campaign remains consumed. It must not be patched and
rerun. Diagnose and regress only on the 53-paper development partition, then
open a new epoch against sealed Reserve B if needed.

## Installation does not consume the reserve

After installing this patch, perform pool registration, Reserve-A
registration, readiness preparation, protocol freeze, and preflight first.
Stop at preflight PASS before the real execution command.
