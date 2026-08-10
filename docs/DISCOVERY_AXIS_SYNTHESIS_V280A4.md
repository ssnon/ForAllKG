# GraphAgentsDAC v2.8.0-alpha4 — Discovery-Axis Hypothesis Synthesis

## Why alpha4 exists

Alpha3 solved the retrieval-side failure: candidate concepts are treated as multi-anchor `CandidateUnit`s, candidate-bearing routes are recovered, and the `DiscoveryBundle` can surface discovery-distinct mechanistic axes. The remaining failure is synthesis collapse: a portfolio-level LLM can ignore those inspirations and safely regenerate the canonical grounded chain.

Alpha4 changes only the **DiscoveryBundle → Hypothesis Maker interface**. It does not change KG construction, candidate-unit traversal, the grounded `HypothesisContext`, the deterministic compiler, or the existing hypothesis validator.

## Epistemic invariant

Two channels remain separate:

- `premise_statement_ids`: grounded evidence only.
- discovery lineage (`axis_id`, `inspiration_id`, `candidate_unit_id`): creativity provenance only.

A candidate unit is never promoted to evidence by alpha4. The final standard `HypothesisPortfolio` remains compatible with the existing semantic critic and feasibility pipeline. Discovery lineage is stored in a separate deterministic synthesis report.

## Components

### `DiscoveryAxisPlanner`

Reads the already fail-closed alpha3 `DiscoveryBundle` and deterministically selects up to `max_axes` candidate-unit inspirations. It filters low unit score and reaction-domain-switch-heavy routes and assigns stable `axis_id`s.

### Per-axis generation

Each axis is a separate generation task. The LLM must return exactly one hypothesis for that axis or abstain. The prompt explicitly requires the axis to create a mediator, moderator, pathway competition, descriptor interaction, conditional dependency, or another additional scientific dependency.

This removes the old failure mode:

```
8 inspirations -> one portfolio prompt -> LLM selects two safe canonical hypotheses
```

and replaces it with:

```
axis 1 -> one hypothesis or abstain
axis 2 -> one hypothesis or abstain
...
```

### Discovery lineage

The LLM is **not** trusted to emit discovery IDs. The orchestrator assigns lineage from the task that produced the accepted hypothesis. This prevents fabricated inspiration IDs and keeps discovery IDs out of `premise_statement_ids` by construction.

### `DiscoveryAxisFidelityCritic`

A deterministic guard combines:

- semantic similarity between the axis and inferential bridge,
- semantic similarity between the axis and hypothesis/predictions,
- coverage of axis-distinctive terms.

It is not a truth or novelty judge. Its purpose is to catch decorative lineage: if the assigned axis can be removed without materially changing the proposal, the proposal is repaired once or rejected.

### Corpus-internal novelty repair

Each accepted per-axis proposal is immediately assessed with the existing `InternalNoveltyAssessor`.

By default these statuses trigger one bounded repair:

- `reconstructs_existing_corpus_claim`
- `reconstructs_existing_corpus_chain`

If the repaired proposal still reconstructs prior corpus content, that axis is rejected rather than falling back to a canonical hypothesis.

`corpus_supported_extension`, `new_combination_within_corpus`, and `corpus_distinct_candidate` are retained. External novelty remains `not_assessed`.

## Files

New alpha4 modules:

- `dac_her/discovery_axis_contracts.py`
- `dac_her/discovery_axis_planner.py`
- `dac_her/discovery_axis_prompt.py`
- `dac_her/discovery_axis_fidelity.py`
- `dac_her/discovery_axis_runtime.py`
- `scripts/run_discovery_axis_hypothesis_maker.py`
- `tests/test_discovery_axis_planner.py`
- `tests/test_discovery_axis_fidelity.py`

## Run on the current alpha3 result

Apply this patch **after v2.8.0-alpha3**.

```bash
cd ~/GraphAgentsDAC

git apply --check /path/to/GraphAgentsDAC_discovery_v280a4_from_a3.patch
git apply /path/to/GraphAgentsDAC_discovery_v280a4_from_a3.patch
```

Focused tests:

```bash
python -m pytest -q \
  tests/test_discovery_axis_planner.py \
  tests/test_discovery_axis_fidelity.py
```

Run alpha4 synthesis:

```bash
RUN=runs/e2e/expanded_coordination_discovery_001

python -m scripts.run_discovery_axis_hypothesis_maker \
  --dual-context "$RUN/hypothesis.dual_context.a3.json" \
  --model "$OPENROUTER_AGENT_MODEL" \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key-env OPENROUTER_API_KEY \
  --max-axes 5 \
  --output-prefix "$RUN/hypothesis_axis_a4" \
  --save-prompts
```

The mechanism node index is inferred as:

```
data_dac/corpus/<corpus_id>/mechanism/navigation/node_index
```

Override it with `--index-dir` if needed.

Outputs:

- `hypothesis_axis_a4.axis_plan.json`
- `hypothesis_axis_a4.draft.json`
- `hypothesis_axis_a4.portfolio.json`
- `hypothesis_axis_a4.lineage.json`
- `hypothesis_axis_a4.internal_novelty.json`
- `hypothesis_axis_a4.prompts/axis_XX.prompt.txt` when `--save-prompts` is used.

The portfolio is an ordinary `HypothesisPortfolio`, so the existing semantic critic can consume it directly:

```bash
GROUND_RUN=runs/e2e/expanded_coordination_topn_002

python -m scripts.run_hypothesis_semantic_critic \
  --context "$GROUND_RUN/hypothesis.context.json" \
  --portfolio "$RUN/hypothesis_axis_a4.portfolio.json" \
  --model "$OPENROUTER_CRITIC_MODEL" \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key-env OPENROUTER_API_KEY \
  --output-prefix "$RUN/semantic_axis_a4"
```

## Interpretation

A successful alpha4 run should no longer silently regenerate the old canonical H1/H2 merely because they are easy to ground. For example, if an axis is `nitrogen-coordination–charge-donation correlation`, a proposal that only says `coordination -> ΔG_H -> HER volcano` should fail axis fidelity and/or corpus-internal novelty control. A surviving proposal should make charge donation part of the actual inferential bridge and at least one prediction/falsifier.

Under-filling is allowed. If only two of five discovery axes can support grounded, axis-faithful, non-reconstructive hypotheses, the final portfolio should contain two hypotheses rather than generating canonical filler.
