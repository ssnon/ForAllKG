# alpha4c.4b — Frozen Unseen SERS Trend Holdout Runner

## Scope

alpha4c.4b executes the already-frozen alpha4c.4a split. It does **not**
select papers and exposes no arbitrary paper-list CLI.

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

The 22-paper future reserve remains untouched.

## Why `evidence` projection mode

The alpha4c Trend stack consumes:

```text
canonical graph
MeasurementResultIdentity
MethodContext
ComparisonContext
```

and does not use Bridge edges as Trend evidence.

Therefore this holdout deliberately builds an `evidence` GraphAgents
projection from each frozen canonical graph and excludes Bridge from the
evaluation path. The existing projection builder requires Bridge only for
`mechanism` and `exploratory` modes; `evidence` mode is canonical-only.

This removes stochastic Bridge generation as an unrelated variable in a
Trend-semantics holdout.

## Upstream boundary

The runner **does not invoke Strict extraction or any LLM**.

Before a holdout run, all ten frozen papers must already have:

```text
data_sers/extracted/<paper>/<paper>.graphml
```

and every graph must satisfy:

```text
domain_profile = sers_au_ag
measurement_merge_invariant_id =
  measurement_payload_isolation_v1_alpha4b4a
Measurement value_numeric/value_text XOR
no manual resolution decisions
```

If one is missing, `--preflight-only` fails and reports the exact paper IDs.
Prepare those canonical graphs with the existing frozen Strict/paper-graph
workflow, without modifying Trend semantics, then rerun the preflight.

This separation matters: upstream KG preparation and Trend holdout evaluation
must not silently become one adaptive process.

## Frozen pipeline

```text
10 frozen canonical graphs
        |
        v
evidence projection (no Bridge)
        |
        v
campaign-specific 10-paper corpus
        |
        v
MeasurementResultIdentity
        |
        v
MetricDefinition
        |
        v
Comparison + MethodContext
        |
        v
TrendEvidence
        |
        v
TrendPrecision / PaperLocalTrendResult
        |
        v
TrendContextProfile
        |
        v
PairwiseTrendContrast
        |
        v
CrossContextTrendAssessment
        |
        v
invariant-only holdout report
```

Artifact IDs are fixed in the execution protocol; the runner does not accept
custom scientific IDs.

## Frozen semantic IDs

```text
projection:
  sers_au_ag_projection_v2_alpha4b2c3

corpus:
  sers_au_ag_corpus_v1_alpha4b3a

measurement merge invariant:
  measurement_payload_isolation_v1_alpha4b4a

MeasurementResultIdentity:
  measurement_result_identity_v1_alpha4b4a1

MetricDefinition:
  sers_au_ag_metric_definition_v2_alpha4b3b4b1

Comparison:
  sers_au_ag_comparison_v7_alpha4b3b321

MethodContext:
  sers_au_ag_method_v4_alpha4b3b321

quality gate:
  quality_aware_numeric_gate_v2_alpha4b3b4c1

Trend contract:
  trend_evidence_contract_v1_alpha4c1

SERS Trend:
  sers_au_ag_trend_v5_alpha4c2121

Trend precision:
  sers_au_ag_trend_precision_v5_alpha4c21211

cross-context contract:
  cross_context_trend_contract_v1_alpha4c3a

SERS trend context:
  sers_au_ag_trend_context_v1_alpha4c3b

cross-context assessment:
  cross_context_trend_assessment_v1_alpha4c3c
```

## Acceptance is invariant-only

The following are observations, not pass/fail targets:

```text
TrendEvidence count
numeric/claim trend count
paper-local result count
cross-paper same-relation pair count
repeated count
reversed count
context_specific count
insufficient count
```

Therefore these are all valid distributions:

```text
TrendEvidence = 0
cross-paper pairs = 0
repeated = 0
reversed = 0
all assessments = insufficient
```

A failure is limited to the frozen 4c.4a policy:

- semantic or implementation drift;
- ungrounded TrendEvidence / failed Trend audit;
- cross-paper numeric-series construction;
- MeasurementResult provenance/binding loss;
- paper-global context leakage;
- numeric-ranking policy reused as Trend policy;
- same-paper support counted as replication;
- incomplete cross-paper pair generation;
- majority vote overriding a reversal;
- correlation promoted to causation;
- relation/assessment structural failure.

## Commands

After installing the patch, first check only the protocol:

```bash
python -m scripts.run_sers_alpha4c4b_trend_holdout --protocol-only
```

Then check canonical KG readiness **without producing Trend outputs**:

```bash
python -m scripts.run_sers_alpha4c4b_trend_holdout --preflight-only
```

Optional command-plan smoke test:

```bash
python -m scripts.run_sers_alpha4c4b_trend_holdout --dry-run
```

Use a separate dry-run manifest from the real run by deleting the dry-run
evaluation directory before the real run, or simply skip `--dry-run` once
preflight passes.

Real holdout:

```bash
python -m scripts.run_sers_alpha4c4b_trend_holdout
```

The real run writes:

```text
evaluation/sers_alpha4c4b/holdout_v1/manifest.json
evaluation/sers_alpha4c4b/holdout_v1/holdout_report.json
```

and campaign-specific sidecars under:

```text
data_sers/corpus/sers_alpha4c4b_holdout_v1_corpus/evidence/
```

## Epoch rule

If inspecting these ten papers exposes a genuine generic bug and scientific
implementation code is changed, this v1 blind holdout is retired. These ten
papers become seen regression material and a new epoch must draw only from the
22 untouched reserve papers.

A low trend count, missing cross-paper overlap, or an all-insufficient result
is not a bug by itself.
