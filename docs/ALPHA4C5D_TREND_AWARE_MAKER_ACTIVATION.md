# alpha4c.5d — Trend-aware Maker Prompt / Runtime Activation

## Scope

alpha4c.5d is the first phase that lets the LLM consume the frozen Trend-aware
hypothesis contract built in alpha4c.5a–5c.

It deliberately does **not** edit the legacy Explorer-only Hypothesis Maker and
it does **not** mutate the alpha4c.5b input contract.

```text
TrendAwareHypothesisInput (5b; frozen)
        |
        v
TrendMakerExposure (5d; deterministic activation)
        |
        v
TrendAwareHypothesisPrompt
        |
        v
LLM -> TrendAwareHypothesisPortfolioDraft
        |
        v
TrendAwareHypothesisCompiler (5c; frozen)
        |
        v
TrendAwareHypothesisValidator (5c; frozen)
```

## Why a separate activation layer

Every alpha4c.5b `HypothesisTrendInputView` intentionally carries:

```text
maker_selectable = false
causal_use_allowed = false
universal_use_allowed = false
```

Those values describe the frozen 5b contract and are not rewritten by 5d.
Instead, 5d derives a separate `TrendMakerExposure` that exposes only exact 5b
view IDs and maps every lane to the single use role authorized by alpha4c.5c.

```text
local_empirical_support
  -> positive_empirical_support

cross_paper_replicated_support
  -> cross_paper_empirical_support

context_dependency_signal
  -> context_qualification

reversal_boundary
  -> counterevidence_boundary

replication_gap
  -> replication_gap
```

The activation object SHA-binds itself to the source Trend-aware input SHA.

## Limitation companions

5d does not ask the LLM to rediscover the 5c limitation logic. It projects the
required companion view IDs directly into the prompt.

```text
insufficient positive support
  -> replication_gap companion

context_specific positive support
  -> context_qualification companion

reversed positive support
  -> context_qualification + counterevidence_boundary companions

repeated
  -> no mandatory limitation companion
```

The 5c compiler and validator remain the final deterministic enforcement layer.
The 5d prompt is guidance, not authorization.

## Provenance namespaces

Explorer and Trend IDs remain separate in the LLM output contract.

```text
Explorer positive evidence
  -> premise_statement_ids

Explorer gaps
  -> gap_statement_ids

Trend evidence / limitations
  -> trend_references[].view_id + use_role
```

Trend-only hypotheses remain allowed when at least one positive Trend reference
is selected. Gap/context/counterevidence views cannot satisfy positive support
on their own.

`cross_paper_empirical_support` is the only Trend use role that explicitly means
replicated Trend support. The compiled card field `cross_paper_synthesis` is
broader: it records whether positive support spans at least two papers and must
not be reinterpreted as a Trend-replication flag.

## Causality, universality, numeric values

5d does not grant new epistemic authority.

- Trend exposure cannot authorize causal evidence claims.
- Trend exposure cannot authorize universal relations.
- Association-only results remain association-only.
- Unknown context is not filled.
- Reversal is not majority-voted away.
- Trend exposure contains no raw numeric values.
- Generated numeric values remain licensed only by selected Explorer positive
  premise text and are rechecked by the frozen 5c validator.
- External novelty claims and experimental protocols remain forbidden.

A causal mechanism may still be proposed as an explicit *hypothesis/inferential
bridge*; it is not represented as something established by Trend provenance.

## Runtime

`TrendAwareHypothesisMakerAgentRuntime` mirrors the existing bounded Maker
runtime but uses the 5c Trend-aware compiler and validator.

```text
generation -> compile -> validate
                 |          |
                 +-- at most one contract repair --+
```

Repair feedback is limited to deterministic compile/validation failures. It is
not an open-ended hypothesis-evolution loop.

## Legacy isolation

alpha4c.5d adds new modules and does not modify:

```text
dac_her/hypothesis_prompt.py
dac_her/hypothesis_llm.py
dac_her/hypothesis_runtime.py
dac_her/hypothesis_maker.py
scripts/run_hypothesis_maker.py
```

The Explorer-only Maker therefore remains behaviorally unchanged.

## Deterministic v2 seen regression

The existing alpha4c.5c v2 seen `trend_input.json` can be replayed without an
LLM:

```bash
python -m scripts.run_hypothesis_trend_maker_activation_regression \
  --input evaluation/sers_alpha4c5c/v2_seen/trend_input.json \
  --output-dir evaluation/sers_alpha4c5d/v2_seen
```

Expected semantics are structural, not count-tuning targets:

```text
5b maker_selectable remains false
5d exposes the exact local-support and replication-gap views
local positive support carries the mandatory gap companion
Trend-only draft compiles and validates
cross_paper_synthesis = false
Trend causal authorization = false
Trend universal authorization = false
LLM calls = 0
v3 reserve consumed = false
```

## Prompt dry run

```bash
python -m scripts.run_trend_aware_hypothesis_maker \
  --input evaluation/sers_alpha4c5c/v2_seen/trend_input.json \
  --dry-run-prompt \
  --save-prompt
```

This writes the deterministic 5d exposure and prompt without a model call.

## First live seen-fixture smoke

After inspecting the prompt, a single seen-fixture model smoke may be run with
the normal OpenAI-compatible environment configuration:

```bash
python -m scripts.run_trend_aware_hypothesis_maker \
  --input evaluation/sers_alpha4c5c/v2_seen/trend_input.json \
  --save-prompt \
  --max-hypotheses 1
```

The v2 fixture contains a synthetic Explorer context and must remain labelled as
a contract smoke, not a new scientific result. The 14-paper v3 reserve is not
needed for alpha4c.5d installation or deterministic regression.
