# alpha4c.2 — SERS Trend Adapter

## Scope

alpha4c.2 activates the first domain-specific implementation of the generic
alpha4c.1 `TrendEvidence` contract for `sers_au_ag`.

Semantics ID:

`SERS_AU_AG_TREND_SEMANTICS_ID = sers_au_ag_trend_v1_alpha4c2`

This phase remains **paper-local**. It does not implement `repeated`,
`context_specific`, `reversed`, or `insufficient`; those belong to alpha4c.3.
It also does not modify HypothesisContext.

## Required frozen inputs

The SERS adapter requires all four source lanes:

- canonical graph;
- MeasurementResultIdentity;
- identity-aware MethodContext;
- identity-aware ComparisonContext.

The builder verifies corpus/profile/semantics bindings and canonical graph
hashes before extraction. This prevents a trend sidecar from silently mixing
contexts derived from different canonical graph epochs.

## Controlled numeric lane

Numeric trend extraction is deliberately conservative.

Eligible dependent observables in alpha4c.2:

- `raman_intensity`
- `sers_enhancement_factor`

Eligible structural controls:

- `shell_thickness` (canonical unit: nm)
- `nanogap_size` (canonical unit: nm)
- `ag_to_au_ratio` (canonical scalar: Ag/Au)

A numeric trend is emitted only when:

1. every point is a scientific MeasurementResult already represented by the
   Comparison sidecar;
2. a single explicit control value is found in measurement-local structured
   `conditions_json`;
3. all points share explicit MeasurementGroup lineage, otherwise explicit
   Experiment lineage;
4. all explicitly known non-varied MethodContext dimensions are compatible;
5. dependent units are identical;
6. control values are distinct;
7. at least two points remain.

Repeated control values are not averaged. Explicit method mismatch blocks the
numeric trend. Unknown method context is not treated as a mismatch, but it is
preserved in the source MethodContext references.

Numeric direction/shape is inferred without fitted thresholds:

- monotone nondecreasing -> positive/monotonic
- monotone nonincreasing -> negative/monotonic
- constant -> unchanged/unspecified
- one increase-to-decrease transition -> non_monotonic/single_optimum
- one decrease-to-increase transition -> non_monotonic/u_shaped
- otherwise -> non_monotonic/unspecified

The numeric lane never asserts causality and never combines values across
papers.

## Reported-claim lane

Claim extraction is source-text-bound and only emits a trend when one supported
control and one SERS response are both explicit and a directional pattern is
recognized.

Calibrated claim families in alpha4c.2:

- shell thickness -> SERS/Raman response
- nanogap size -> SERS/Raman response
- Au:Ag ratio optimum -> SERS/Raman response

Examples of intended semantics:

- "SERS increases as interior gap decreases" ->
  `nanogap_size`, negative, monotonic
- "Raman intensity increases with Ag shell thickness and approaches a maximum"
  -> `shell_thickness`, positive, saturating
- "among tested Au:Ag ratios, 10:7 gives the strongest SERRS signal" ->
  `ag_to_au_ratio`, non_monotonic, single_optimum

A reported correlation is never upgraded to causal evidence. Directional claims
use `causal_status=source_asserted` only when explicit causal language is in the
same source claim.

## Calibration

Use only the already-seen calibration papers SERS_1/5/8:

```bash
python -m scripts.run_sers_alpha4c2_calibration
```

This performs no LLM calls and writes the trend sidecar under the frozen
alpha4b calibration corpus. No evidence-count target is encoded. Review the
actual `evidence.jsonl`, especially control normalization, source claims,
lineage, MethodContext compatibility, and direction/shape before freezing
alpha4c.2.

SERS_2/6/10 are no longer blind for Trend semantics because their contents were
inspected during the alpha4b holdout. They may later be used as seen regression
material, not as the final alpha4c trend holdout.
