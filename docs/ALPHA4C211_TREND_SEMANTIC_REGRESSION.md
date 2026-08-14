# alpha4c.2.1.1 — Trend Semantic Regression Precision

Semantics:

- generic TrendEvidence contract remains frozen:
  `trend_evidence_contract_v1_alpha4c1`
- active SERS raw trend:
  `sers_au_ag_trend_v3_alpha4c211`
- active SERS precision:
  `sers_au_ag_trend_precision_v2_alpha4c211`

This patch is a narrow precision pass over alpha4c.2.1. It does not add
cross-paper aggregation. `alpha4c.3` remains downstream.

## Seven fixes

1. **Rise → peak → fall is non-monotonic.**
   `increased ... highest at X ... decreased ...` is represented as
   `non_monotonic / single_optimum`. An increase that approaches a plateau
   remains `positive / saturating`.

2. **Nanogap presence is not nanogap size.**
   Continuous `nanogap_size` requires explicit size/width/distance language.
   Presence/absence comparisons use `nanogap_presence` and never invent a
   numeric gap coordinate.

3. **Formal EF directional precedence.**
   When an explicit `enhancement factor`, `SERS EF`, or `EF coefficient`
   participates in the directional clause, the observable is formal SERS EF.
   A trailing baseline-relative `factor of 5.8` does not steal an otherwise
   explicit Raman-intensity trend.

4. **`ratio` and `ratios` are both recognized.**
   Tested `Au-Ag ratios` with a strongest 10:7 member recover
   `ag_to_au_ratio`, with canonical `Ag/Au = 0.7` and transform provenance.

5. **Raman peak intensity syntax is supported.**
   Phrases such as `Raman peak intensity increasing with ATP concentration`
   are recognized as measured signal-intensity trends.

6. **Paper-local subject structural-family normalization is stronger.**
   Materials, morphology, and architecture are canonicalized using both
   subject-node metadata and the grounded claim text. Numeric composition or
   thickness suffixes do not prevent restatements such as `Au@Ag10.0` and
   `Au55@Ag8.4 nanocubes` from collapsing when both claims explicitly concern
   the same Au-Ag nanocube family.

7. **Calculated numeric EF is model-derived semantics.**
   `calculated_numeric + sers_enhancement_factor` is annotated as
   `model_derived_sers_enhancement_factor`, never
   `formal_sers_enhancement_factor`.

## Exact regression fixtures

The unit regression suite embeds the previously observed claim language for:

- `Kiwook_SERS_1 / claim_optimal_agno3`
- `Kiwook_SERS_1 / claim_atp_detection`
- `Kiwook_SERS_5 / claim_interior_nanogap_enhancement`
- `Kiwook_SERS_6 / claim_ratio_10_7_highest_sers`
- `Kiwook_SERS_8 / claim_ef_increases_gold` source wording
- `Kiwook_SERS_10 / claim_shell_thickness_trend`
- `Kiwook_SERS_10 / claim_sers_shell_thickness`

The real-corpus runners additionally inspect named claim IDs and the DDA
calculation lineage. These are semantic regression assertions, not evidence
count targets.

## Rerun

Calibration:

```bash
python -m scripts.run_sers_alpha4c211_calibration
```

Already-seen 2/6/10 regression:

```bash
python -m scripts.run_sers_alpha4c211_seen_regression
```

The 2/6/10 run is explicitly a seen regression, not a new blind holdout.

Only after both structural and semantic-regression gates pass, followed by
manual evidence review, should the alpha4c.2 family be considered for freeze.
