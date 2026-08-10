# GraphAgentsDAC v2.8.0-alpha6.1 — Non-destructive novelty refinement

Alpha6.1 fixes a failure mode observed in a live Dual-Atom HER run: five alpha4
hypotheses were all `INSUFFICIENT_SEARCH_EVIDENCE`, refinement drafts then failed
compilation, and the original grounded hypotheses were incorrectly discarded.

## Changes

1. **Deterministic provenance lock**
   - The LLM still proposes scientific wording, predictions, falsifiers, and assumptions.
   - `premise_statement_ids`, `gap_statement_ids`, and `hypothesis_type` are overwritten
     from the original grounded hypothesis before compilation.
   - Model-generated evidence identifiers are therefore non-authoritative.

2. **Non-destructive optional refinement**
   - If targeted reassessment of the original is not `WELL_ESTABLISHED` or
     `CONFLICTING_PRIOR_ART`, a failed refinement does not delete the original.
   - The original is kept with the targeted external-novelty status and an explicit
     `non_destructive_original_fallback` reason code.
   - No novelty upgrade is claimed by fallback.

3. **Destructive refinement only for strong negative prior-art states**
   - If targeted reassessment says `WELL_ESTABLISHED` or `CONFLICTING_PRIOR_ART`, the
     original may still be removed when a safe refinement cannot be produced.

4. **Failure taxonomy**
   - An alpha6 empty portfolio caused exclusively by compile/validation/provenance
     failures is now treated as a degraded pipeline state by the E2E runner.
   - It is no longer printed as a valid scientific fail-closed result.

5. **Debug visibility**
   - `scripts.run_novelty_refinement` prints `reason_codes` for every attempt.

## Rerun from stage 11 using the existing alpha5.2 artifacts

For the metal-pair run:

```bash
RUN=runs/e2e/dac_her_metal_pair_coordination_002

python -m scripts.run_novelty_refinement   --dual-context "$RUN/hypothesis.dual_context.a3.json"   --axis-plan "$RUN/hypothesis_axis_a4.axis_plan.json"   --portfolio "$RUN/hypothesis_axis_a4.portfolio.json"   --lineage "$RUN/hypothesis_axis_a4.lineage.json"   --external-report "$RUN/external_novelty_a52.report.json"   --external-query-plan "$RUN/external_novelty_a52.claims_queries.json"   --external-prior-art "$RUN/external_novelty_a52.prior_art.json"   --model "$OPENROUTER_AGENT_MODEL"   --critic-model "$OPENROUTER_CRITIC_MODEL"   --base-url "https://openrouter.ai/api/v1"   --api-key-env OPENROUTER_API_KEY   --providers semantic_scholar,crossref   --results-per-query 12   --output-prefix "$RUN/novelty_refinement_a61"
```

Expected behavior for the observed case: the five original hypotheses should no
longer disappear merely because the optional refinement draft has malformed
provenance IDs. They should either be accepted as refinements or retained as
originals with unresolved/extension external-novelty status.
