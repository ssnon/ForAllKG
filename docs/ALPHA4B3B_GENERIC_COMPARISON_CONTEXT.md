# alpha4b.3b — Generic ComparisonContext + SERS provider

## Goal

Add a fail-closed cross-paper comparison layer without changing canonical
graphs, Bridge extraction/policy, GraphAgents projections, or corpus topology.

Comparison metadata is materialized as a sidecar derived from strict canonical
graphs because mechanism/exploratory projections intentionally remove
Measurement, Experiment, MeasurementGroup, Analyte, RamanReporter, and
OpticalCondition context nodes.

## Shared contract

Compatibility states:

- `compatible`
- `partially_compatible`
- `incompatible`
- `unknown`

`unknown != compatible`.

Missing context never quarantines a source paper or measurement. It only blocks
or limits direct numeric ranking.

`numeric_ranking_allowed` is true only when:

1. all domain-required comparison dimensions are explicitly known and equal;
2. both measurements are numeric;
3. both measurement units are explicitly present and exactly normalized-equal.

No unit conversion or missing-value imputation is performed.

## SERS dimensions

The SERS Au/Ag adapter owns these dimensions:

- analyte
- reporter
- concentration
- excitation wavelength
- laser power
- integration time
- Raman peak
- medium
- substrate state

All nine are required for direct numeric ranking in alpha4b.3b v1. This is
intentionally conservative.

The provider uses only explicit canonical graph nodes, relations, attributes,
and source expressions. Multiple conflicting explicit values become
`ambiguous`; missing values remain `unknown`.

## Output

Comparison sidecars are attached to an already-built non-destructive corpus:

```text
data_sers/
  corpus/
    <corpus-id>/
      exploratory/
        manifest.json
        comparison/
          <comparison-id>/
            contexts.jsonl
            assessments.jsonl
            summary.json
            audit.json
```

The CLI validates the corpus domain and CorpusSemantics identity before reading
the strict canonical graphs for the same paper IDs.

## Calibration

After installation:

```bash
python -m scripts.build_comparison_contexts \
  --domain-profile sers_au_ag \
  --data-root data_sers \
  --corpus-id sers_alpha4b3a_calibration \
  --mode exploratory \
  --comparison-id sers_alpha4b3b_calibration
```

Only SERS_1/5/8 should be present because the comparison paper set is inherited
from the frozen alpha4b.3a calibration corpus.

Inspect `summary.json`, `audit.json`, and a sample of `contexts.jsonl`.
A low or zero `numeric_ranking_allowed_count` is not a failure when the source
papers omit comparison-critical experimental conditions.
