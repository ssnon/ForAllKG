# alpha4b.3b.4a — ReproducibilityEvidence contract + SERS adapter

## Goal

Add a domain-owned, fail-closed reproducibility-quality sidecar without changing
canonical graphs, Bridge semantics, projection semantics, corpus topology, or the
frozen Comparison/MethodContext semantics.

This layer answers a narrower question than ComparisonContext:

> What reproducibility, repeatability, or sampling evidence is explicitly
> reported for a scientific measurement or experiment?

It does **not** decide whether two numeric measurements are comparable. That
binding is deferred to alpha4b.3b.4c.

## Shared contract

`ReproducibilityEvidence` preserves grounded provenance to canonical graph nodes
and supports these cross-domain evidence kinds:

- `relative_standard_deviation`
- `repeatability_statement`
- `spatial_sampling`
- `population_sampling`

A domain adapter owns its admissible reproducibility scopes. The SERS Au/Ag
adapter uses:

- `spot_to_spot`
- `substrate_to_substrate`
- `batch_to_batch`
- `particle_to_particle`
- `replicate`
- `unknown`

`unknown` is a valid state. Missing counts or scope are never inferred merely to
make a quality record look complete.

## SERS extraction rules

The SERS provider is deterministic and does not call an LLM.

1. A `Measurement` with metric `relative_standard_deviation` becomes direct
   quantitative reproducibility evidence. Connected producer `Experiment` and
   `MeasurementGroup` nodes may contribute scope/count provenance.
2. A non-RSD `Measurement` is admitted only when its local source text explicitly
   reports reproducibility/repeatability.
3. Experiment-level evidence is admitted only for SERS/Raman experiments and only
   when local text explicitly reports reproducibility/repeatability, spatial
   replicate averaging, or a sampled single-particle population.
4. A standalone `MeasurementGroup` may preserve explicit group-level
   reproducibility when it is not already represented by direct RSD evidence.
5. `uniform` by itself is insufficient to create reproducibility evidence.
6. Mapping area and internal-standard fields are harvested only from explicit
   structured attributes. They are not guessed from free text.

The provider deliberately does not create new Measurement, Experiment, or Group
nodes and does not repair malformed scientific values.

## Output

The CLI binds to an already-built non-destructive corpus and writes:

```text
data_sers/
  corpus/
    <corpus-id>/
      exploratory/
        reproducibility/
          <reproducibility-id>/
            evidence.jsonl
            summary.json
            audit.json
```

The audit verifies that every provenance node exists in the corresponding
canonical graph and is one of `Measurement`, `MeasurementGroup`, or `Experiment`.

## Calibration

Use only the frozen SERS_1/5/8 calibration corpus:

```bash
python -m scripts.build_reproducibility_evidence \
  --domain-profile sers_au_ag \
  --data-root data_sers \
  --corpus-id sers_alpha4b3a_calibration \
  --mode exploratory \
  --reproducibility-id sers_alpha4b3b4a_calibration
```

Do not tune this phase on SERS_2/6/10. Those remain frozen holdout papers for
alpha4b.4.

A low evidence count is not a failure. The hard requirements are grounded
provenance, no invented metadata, deterministic extraction, and a passing
structural audit.
