# alpha4b.3b.4a.1 — Reproducibility Result Identity & Kind Precision

This patch calibrates the first SERS_1/5/8 reproducibility sidecar without
broadening scientific extraction.

## 1. Evidence-kind precision

A graph `Measurement` labeled with `metric_id=relative_standard_deviation`
does not by itself prove an RSD result.

- explicit numeric dispersion -> `relative_standard_deviation`
- explicit textual numeric dispersion such as `14.2% deviation` -> RSD kind,
  but no numeric value is invented
- qualitative `reproducible/repeatable` language without dispersion ->
  `repeatability_statement`

The original Measurement remains untouched. Only the reproducibility sidecar
classification is corrected.

## 2. Scientific result identity

The sidecar now distinguishes source mentions from scientific results.

Each evidence row records:

- `result_identity_status`
- `source_mention_node_ids`
- `source_expressions`
- `source_mention_count`

Exact consolidation is deliberately narrow. Multiple mentions are
automatically consolidated only when they share an explicit Experiment or
MeasurementGroup lineage, have compatible scientific subjects, scope and
metadata, and represent the same result payload. A qualitative repeatability
mention can be subsumed by an RSD mention only under that same exact-lineage
gate.

Same paper + same value is never sufficient for auto-merge.

## 3. Possible duplicate diagnostics

When two results have the same paper, scope, scientific target/context, and
the same quantitative result (or qualitative/quantitative reproducibility
pair) but do not share explicit lineage, they remain separate evidence rows.

The audit reports them as non-fatal `possible_duplicate_results`.

These diagnostics do **not** fail the structural gate and must not be used as
an automatic merge instruction.

## Semantics

`sers_au_ag_reproducibility_v2_alpha4b3b4a1`

## Calibration goal

The calibration is successful when:

- qualitative reproducibility is no longer mislabeled as RSD;
- no numeric value is created from qualitative prose;
- exact shared-lineage duplicates consolidate with all provenance preserved;
- uncertain duplicate candidates remain separate and visible in audit;
- `Structural gate: True` remains true.

Evidence count is not a target.
