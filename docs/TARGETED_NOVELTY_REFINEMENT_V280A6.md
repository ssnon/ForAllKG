# GraphAgentsDAC v2.8.0-alpha6 — Targeted Novelty Refinement

## Purpose

Alpha5.1 answers a bounded prior-art question. Alpha6 uses that report to decide
whether a hypothesis should be kept, searched more narrowly, refined once, or
rejected. It is deliberately **not** an unconstrained "make this more novel"
loop.

## Epistemic invariants

1. External prior art is exclusion/boundary evidence only.
2. External papers never become `premise_statement_ids`.
3. A refined hypothesis must keep exactly the original premise IDs, gap IDs,
   and hypothesis type.
4. The original discovery axis is re-checked after refinement.
5. Corpus-internal canonical reconstruction is still rejected.
6. The refined hypothesis receives a fresh external novelty search.
7. Each hypothesis receives at most one regeneration attempt.
8. Novelty optimization never loops indefinitely.

## Flow

```text
alpha4 portfolio
      +
alpha5.1 external report
      |
      v
NoveltyGapAnalyzer
      |
      +-- keep ------------------------------+
      |                                     |
      +-- unresolved / adjacent / conflict  |
               |                            |
               v                            |
        targeted query delta                |
               |                            |
        targeted retrieval                  |
               |                            |
      reassess original hypothesis           |
               |                            |
       if search resolves uncertainty        |
               +---------- keep ------------+
               |
               v
       ONE bounded refinement
               |
      exact grounding-lineage gate
               |
      discovery-axis fidelity gate
               |
      corpus-internal novelty gate
               |
      FRESH external search
               |
      reject direct/conflicting prior art
               |
               v
         final portfolio
```

## Why targeted search precedes regeneration

For `INSUFFICIENT_SEARCH_EVIDENCE`, the correct first action is more evidence,
not more creativity. If targeted search changes the result to
`NEW_COMBINATION_OF_KNOWN_EFFECTS` or `PLAUSIBLY_NOVEL`, the original hypothesis
is retained and no regeneration occurs.

For `LITERATURE_SUPPORTED_EXTENSION`, targeted search focuses on the least
resolved/core differentiating claim. The model is then asked to sharpen only
that boundary, not to invent a new scientific domain.

## Files

- `dac_her/novelty_refinement_contracts.py`
- `dac_her/novelty_gap_analysis.py`
- `dac_her/targeted_novelty_retrieval.py`
- `dac_her/novelty_refinement_prompt.py`
- `dac_her/novelty_refinement_runtime.py`
- `scripts/run_novelty_refinement.py`
- focused tests

## Run on the current benchmark

```bash
RUN=runs/e2e/expanded_coordination_discovery_001

python -m scripts.run_novelty_refinement \
  --dual-context "$RUN/hypothesis.dual_context.a3.json" \
  --axis-plan "$RUN/hypothesis_axis_a4.axis_plan.json" \
  --portfolio "$RUN/hypothesis_axis_a4.portfolio.json" \
  --lineage "$RUN/hypothesis_axis_a4.lineage.json" \
  --external-report "$RUN/external_novelty_a51.report.json" \
  --external-query-plan "$RUN/external_novelty_a51.claims_queries.json" \
  --external-prior-art "$RUN/external_novelty_a51.prior_art.json" \
  --model "$OPENROUTER_AGENT_MODEL" \
  --critic-model "$OPENROUTER_CRITIC_MODEL" \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key-env OPENROUTER_API_KEY \
  --output-prefix "$RUN/novelty_refinement_a6"
```

Dry-run the gap plan first:

```bash
python -m scripts.run_novelty_refinement \
  --dual-context "$RUN/hypothesis.dual_context.a3.json" \
  --axis-plan "$RUN/hypothesis_axis_a4.axis_plan.json" \
  --portfolio "$RUN/hypothesis_axis_a4.portfolio.json" \
  --lineage "$RUN/hypothesis_axis_a4.lineage.json" \
  --external-report "$RUN/external_novelty_a51.report.json" \
  --external-query-plan "$RUN/external_novelty_a51.claims_queries.json" \
  --external-prior-art "$RUN/external_novelty_a51.prior_art.json" \
  --output-prefix "$RUN/novelty_refinement_a6" \
  --dry-run-gap-plan
```

Expected outputs:

- `novelty_refinement_a6.gap_plan.json`
- `novelty_refinement_a6.portfolio.json`
- `novelty_refinement_a6.report.json`
- `novelty_refinement_a6.external/targeted_*.{claims_queries,prior_art,report}.json`
- `novelty_refinement_a6.external/final_*.{claims_queries,prior_art,report}.json`

## Decision semantics

- `kept_original`: no refinement was required, or additional search resolved
  uncertainty without regeneration.
- `accepted_refinement`: one regenerated hypothesis preserved grounding and axis
  scope, survived internal novelty checks, and avoided direct/conflicting
  external prior art in a fresh search.
- `grounding_drift_rejected`: evidence IDs or hypothesis type changed.
- `axis_fidelity_rejected`: the refinement escaped its assigned discovery axis.
- `internal_novelty_rejected`: refinement collapsed back to a corpus claim/chain.
- `external_novelty_rejected`: fresh search found `WELL_ESTABLISHED` or
  `CONFLICTING_PRIOR_ART`.
- `abstained`: no safe refinement was available.

Under-filling is intentional.
