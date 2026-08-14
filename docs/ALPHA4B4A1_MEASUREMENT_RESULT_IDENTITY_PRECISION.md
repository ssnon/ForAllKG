# alpha4b.4.1.1 — Measurement Result Identity Precision

## Problem exposed by the calibration replay

alpha4b.4.1 correctly stopped destructive cross-chunk value merging. It
preserved two source mentions of the SERS_1 ATP LOD:

- `meas_lod_atp`
- `meas_lod_atp__mention_measurement_fb404d02e10a`

Both report the same scientific result: ATP detection limit = 2.4 nM for the
SiO2@Au@Ag substrate. Treating the two mentions as two independent
Measurements inflated downstream counts.

The correct abstraction is:

```
source mention A ─┐
                  ├─ MeasurementResultIdentity ─→ one scientific result
source mention B ─┘
```

The canonical graph remains non-destructive. No source node or provenance edge
is deleted.

## Exact consolidation contract

Automatic `consolidated_exact` requires all of the following:

1. same original local Measurement lineage (`source_local_id`, or the original
   node ID for the first mention);
2. same `metric_id`;
3. same result representation and same result value;
4. compatible explicit numeric unit;
5. no explicit `basis` or `qualifier` contradiction;
6. no contradictory overlapping Measurement conditions;
7. compatible subject identity.

Missing conditions do not count as contradictions. For example, one mention
may state the 1575 cm^-1 Raman peak while another shorter mention omits it.

Subject identity is compatible when the canonical subject is the same, the
subject label is the same, or the subjects have the same composition with no
explicit structural-role contradiction. Thus an abbreviated subject mention
may consolidate, but an explicit `alloy` versus `core/shell` distinction does
not.

**Same numeric value alone never merges.**

## Downstream behavior

MetricDefinition and Comparison build scripts accept:

```
--measurement-result-identity-id <ID>
```

When a canonical graph contains `measurement_payload_conflict` mentions, the
scripts fail closed if the identity sidecar is omitted.

The identity layer:

- keeps one representative scientific result context;
- unions source-mention provenance;
- merges Method/Comparison dimensions conservatively;
- keeps explicit conflicting dimensions ambiguous;
- does not change Comparison, Method, MetricDefinition, Reproducibility, or
  quality-gate semantics IDs.

Semantics:

`measurement_result_identity_v1_alpha4b4a1`

## Calibration replay

After installation run:

```bash
python -m scripts.replay_sers_alpha4b4_calibration_after_identity_fix \
  --replay-id sers_alpha4b4a1_identity_calibration_replay_v1
```

No LLM calls are made. Existing post-alpha4b.4.1 canonical graphs and frozen
Bridge materializations are reused. Projections/corpus/identity sidecar and all
downstream calibration sidecars are rebuilt under new IDs.

The holdout remains paused until this replay is reviewed and a new frozen
protocol is issued.
