# alpha4b.4.1.1.1 — Identity-aware MethodContext Reconstruction

## Narrow precision fix

alpha4b.4.1.1 correctly identified two SERS_1 ATP-LOD source mentions as one
scientific result, but the first implementation then generically merged the
already-normalized MethodDimension values. That turned compatible preparation
steps (`drop_cast` and `mixing`) into an artificial `ambiguous` value.

The raw provenance showed that the old domain extractor already interpreted
both source experiments together as:

`sample_preparation = drop_cast+mixing`

The ambiguity was therefore introduced by the identity overlay, not discovered
in the paper.

## Fix

The scientific-result identity decision remains unchanged. Instead of merging
MethodContext and ComparisonContext dimensions after extraction, this patch
builds a **transient identity interpretation graph**:

1. canonical source-mention graph remains untouched;
2. each `consolidated_exact` identity keeps its representative Measurement;
3. incident provenance edges from all source mentions are temporarily re-homed
   onto that representative;
4. non-representative source-mention nodes are removed only from the transient
   graph;
5. the existing domain adapter re-extracts MethodContext and ComparisonContext
   from the unioned grounded provenance;
6. source-mention IDs are re-attached to output provenance.

No Measurement value attributes are field-wise merged, so the alpha4b.4.1 XOR
invariant is preserved.

Diagnostic ID:

`identity_aware_domain_reconstruction_v1_alpha4b4a11`

This is not a new Method/Comparison semantics version. The frozen IDs remain:

- `sers_au_ag_method_v4_alpha4b3b321`
- `sers_au_ag_comparison_v7_alpha4b3b321`
- `sers_au_ag_metric_definition_v2_alpha4b3b4b1`
- `quality_aware_numeric_gate_v2_alpha4b3b4c1`

## Calibration replay

Keep holdout paused and rerun the existing replay script under a new ID:

```bash
python -m scripts.replay_sers_alpha4b4_calibration_after_identity_fix \
  --replay-id sers_alpha4b4a11_method_reconstruction_calibration_replay_v1
```

Expected regression direction for the already-inspected ATP-LOD case:

- source mentions / scientific results: `189 / 188`
- consolidated exact results: `1`
- unresolved same-lineage groups: `0`
- MetricDefinition contexts: `44`
- Comparison contexts / assessments: `188 / 598`
- sample_preparation: `known=13, ambiguous=0, unknown=175`
- sample_preparation protocol mismatches: `6`
- numeric ranking allowed: `0`
- all structural gates: `True`

These counts are regression expectations for the known calibration corpus, not
acceptance thresholds for future holdout papers.
