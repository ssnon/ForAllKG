# alpha4c.5g — Trend Yield / Recall Diagnostic

## Goal

Reserve A passed alpha4c.5f.2, but its Trend lane was sparse. That result must
not be tuned on Reserve A.

alpha4c.5g therefore uses **only the 53-paper development partition** to
diagnose whether low Trend yield is primarily caused by:

- **A — no broad directional candidate:** the current extracted KG does not
  expose a broad directional SERS/Raman claim candidate or a broad numeric
  varied-condition series candidate;
- **B — claim adapter miss candidate:** a broad directional claim candidate is
  present in the KG but no current claim TrendEvidence references that claim;
- **C — numeric pipeline block candidate:** a broad within-lineage numeric
  varied-condition series candidate exists but current numeric TrendEvidence
  does not admit it;
- **D — current Trend yield:** a local result survives TrendPrecision;
- **P — precision filter:** raw TrendEvidence exists but no local result
  survives TrendPrecision.

B and C are diagnostic flags, **not automatic proof of a bug**. They may flag
cases that the frozen Trend semantics correctly reject.

## Safety

This phase:

- never reads Reserve A scientific artifacts;
- never reads Reserve B scientific artifacts;
- uses the 53 development IDs from the frozen blind split;
- does not call extraction, Explorer, or Maker LLMs;
- does not change Comparison/Trend/Precision semantics;
- does not modify source canonical graphs;
- refuses non-ready development canonical graphs rather than refreezing them;
- uses no count threshold for acceptance.

## Pipeline

```text
53-paper development partition
        ↓
canonical readiness assertion
        ↓
isolated canonical copies
        ↓
Evidence projection
        ↓
Evidence corpus
        ↓
MeasurementResultIdentity
        ↓
MetricDefinition
        ↓
Comparison
        ↓
TrendEvidence      ← unchanged implementation
        ↓
TrendPrecision     ← unchanged implementation
        ↓
read-only diagnostic census
    ├─ broad claim candidates
    ├─ broad numeric series candidates
    ├─ current Trend admissions
    └─ A/B/C/D/P classification
```

No CrossContext, Graph Explorer, or Hypothesis Maker is needed to answer the
yield/recall question.

## Interpretation

A large **A** share suggests scarcity is upstream: extraction/KG simply does
not expose enough explicit directional/series evidence.

A large **B** share suggests claim-level recall should be inspected on the
development partition: control/response/direction normalization may be too
narrow, but every candidate must be manually audited before changing
semantics.

A large **C** share suggests numeric series are present but are being rejected
by control normalization, lineage/grouping, method compatibility, repeated-x,
or related fail-closed gates. The diagnostic records candidate groups and
compatibility reasons.

If development evidence supports a generic implementation improvement, make
and regression-test that change on development only, freeze a new epoch, and
leave Reserve B sealed until final confirmation.
