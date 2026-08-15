# alpha4c.5c.1 + alpha4c.5d.1 — Directional Prediction Precision

## Why this patch exists

The first live alpha4c.5d seen-fixture smoke exposed a semantic inversion that
the 5c compiler/validator could not detect:

- frozen Trend input: `particle_size`, `direction=positive`,
- alpha4c direction convention: independent variable **increase** implies
  dependent observable **increase**,
- generated text: **decreasing** particle size was predicted to improve SERS.

The old portfolio still validated because 5c checked Trend provenance and
`expected_direction` but did not structurally bind the prediction to the Trend
sign.

## Architectural choice

This patch is additive. It does **not** rewrite frozen 5a, 5b, 5c, or 5d
modules. Instead:

```text
5b TrendAwareHypothesisInput
        |
        v
5d TrendMakerExposure                    [unchanged]
        |
        v
5d.1 DirectionalTrendMakerExposure       [new]
        |
        v
5d.1 direction-aware prompt / LLM
        |
        v
5c.1 DirectionAwareTrendHypothesisDraft  [new]
        |
        v
5c.1 directional compiler
        |
        +--> frozen 5c compiler
        |
        v
5c.1 directional validator
        |
        +--> frozen 5c validator
        |
        v
DirectionAwareTrendHypothesisPortfolio
```

## Canonical sign contract

All Trend directions are interpreted in one frame:

`independent_change = increase`.

Mappings:

- `positive` -> dependent `increase`
- `negative` -> dependent `decrease`
- `unchanged` -> dependent `unchanged`
- `non_monotonic` -> dependent `non_monotonic`
- mixed / ambiguous / unspecified -> `unspecified`

Every selected positive Trend support view must be attached to at least one
prediction using a structured `trend_direction_bindings` row.

The compiler rejects:

- missing direction binding,
- binding to a non-positive Trend reference,
- positive Trend bound as dependent decrease,
- negative Trend bound as dependent increase,
- a prediction `expected_direction` inconsistent with its bindings.

## Textual inversion guard

Structured binding alone is not enough if generated prose says the opposite.
The 5c.1 validator therefore requires the canonical independent-variable
**increase** frame in Trend-grounded title / hypothesis / bridge / prediction
text.

Decrease-frame wording near the bound independent variable, including forms
such as `decreasing`, `smaller`, `lower`, and `reduced`, is rejected with:

`NONCANONICAL_TREND_DIRECTION_FRAME`.

This is intentionally conservative. Equivalent decrease-frame reformulations
are disallowed so that the LLM cannot silently invert a frozen Trend sign.

## Frozen semantics preserved

The patch does not mutate:

- alpha4c.5a grounding,
- alpha4c.5b input views or `maker_selectable=False`,
- alpha4c.5c draft/compiler/validator files,
- alpha4c.5d exposure/prompt/runtime files,
- causal or universal authorization.

The existing 5c and 5d modules remain useful historical contracts. The new
direction-aware runtime must be used for subsequent Trend-aware Maker
evaluation.

## Evaluation order

1. install patch;
2. run deterministic v2-seen direction regression;
3. dry-run the new 5d.1 prompt;
4. rerun exactly one bounded v2-seen live smoke;
5. inspect structured direction bindings and generated prose;
6. only then consider an unseen reserve evaluation.

No v3 reserve is consumed during installation or deterministic regression.
