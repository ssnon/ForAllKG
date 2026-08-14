# alpha4b.4a11 — Refrozen SERS Holdout Reopen

## Why this is a new epoch

The original alpha4b.4 campaign exposed a generic paper-graph merge invariant
bug on a held-out SERS paper. The old campaign must never be resumed.

The permitted invariant-fix sequence was:

1. `measurement_payload_isolation_v1_alpha4b4a`
2. `measurement_result_identity_v1_alpha4b4a1`
3. `identity_aware_domain_reconstruction_v1_alpha4b4a11`

Calibration SERS_1/5/8 was replayed from the same frozen strict runs with no LLM
calls. The final replay restored all previously frozen Comparison,
MetricDefinition, MethodContext, ProtocolAssessment, Reproducibility, and
quality-gate behavior while preserving two source mentions as one scientific
ATP-LOD result.

The old `sers_alpha4b4_protocol.json` intentionally remains paused. This patch
creates a new protocol and runner rather than re-enabling the retired epoch.

## Calibration refreeze

Refrozen calibration replay:

`sers_alpha4b4a11_method_reconstruction_calibration_replay_v1`

Frozen checks include:

- source mentions / scientific results: `189 / 188`
- consolidated exact results: `1`
- unresolved same-lineage groups: `0`
- MetricDefinition contexts: `44`
- Comparison contexts / assessments: `188 / 598`
- sample_preparation: known `13`, ambiguous `0`, unknown `175`
- sample_preparation mismatches: `6`
- ranking-relevant metric gate passes: `0 / 248`
- numeric ranking allowed: `0`
- all structural gates: `True`

These are calibration regression checks only. They are **not** holdout output
targets.

## Holdout acceptance

Holdout papers remain exactly:

- `Kiwook_SERS_2`
- `Kiwook_SERS_6`
- `Kiwook_SERS_10`

The following are valid non-fatal holdout observations:

- more unknown/ambiguous method context
- different protocols
- different metric definitions
- unresolved same-lineage identity candidates
- zero rankable numeric pairs
- new/unregistered content requiring review

Failures are restricted to frozen-contract, structural, provenance, or safety
violations, including canonical Measurement XOR failure, semantic/hash drift,
destructive corpus merging, global concentration leakage, missing-context
quarantine, identity structural failure, or numeric ranking of detection-limit
comparisons.

## Canonical input refreeze

The Measurement merge bug lived upstream of Comparison, so the held-out
canonical GraphML must be rematerialized from the **same frozen strict
extraction runs** before the new campaign starts.

This is explicit and LLM-free:

```bash
python -m scripts.prepare_sers_alpha4b4a11_holdout_inputs
```

The script snapshots the pre-refreeze canonical files, re-runs
`build_paper_graph` with the existing frozen strict `run_id`, verifies
`measurement_payload_isolation_v1_alpha4b4a`, verifies numeric/text XOR, checks
that strict-run metadata did not change, and writes:

`evaluation/sers_alpha4b4a11/input_refreeze/sers_alpha4b4a11_holdout_input_refreeze_v1/holdout_input_refreeze_report.json`

The new holdout runner refuses to start without this report and re-verifies the
reported canonical hashes.

## Run

Dry-run and real run must use different campaign IDs:

```bash
python -m scripts.run_sers_alpha4b4a11_holdout \
  --campaign-id sers_alpha4b4a11_holdout_dryrun_v1 \
  --adopt-existing-bridge \
  --dry-run
```

Then:

```bash
python -m scripts.run_sers_alpha4b4a11_holdout \
  --campaign-id sers_alpha4b4a11_holdout_real_v1 \
  --adopt-existing-bridge
```

`--adopt-existing-bridge` is recommended here because the retired campaign
already produced complete Bridge pairs before the downstream invariant failure.
Explicit adoption prevents unnecessary stochastic Bridge regeneration while
the new campaign re-hashes and locks those artifacts. If those Bridge pairs are
missing or intentionally discarded, omit the flag and provide the frozen
OpenRouter Bridge environment instead.

## New pipeline

```text
frozen strict run
  -> refrozen canonical graph under Measurement payload isolation
  -> Bridge
  -> projections
  -> corpus
  -> MeasurementResultIdentity
  -> ReproducibilityEvidence
  -> MetricDefinitionContext (identity-aware)
  -> Method/Comparison re-extraction on transient identity interpretation graph
  -> quality-aware numeric gate
  -> holdout invariant report
```
