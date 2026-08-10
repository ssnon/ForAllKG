# GraphAgentsDAC v2.8.0-alpha1: Discovery Bundle + Dual Context + Internal Novelty

This bundle adds the first three novelty-oriented layers without weakening the v2.7.1 evidence contract.

## Design invariant

- `HypothesisContext` remains the only source of positive premise IDs.
- `DiscoveryBundle` contains exploration-oriented graph routes and **never** becomes evidence automatically.
- `DualHypothesisContext` is an envelope that keeps those two surfaces separate.
- The existing hypothesis compiler/validator still receives the original grounded `HypothesisContext`.
- `InternalNoveltyAssessor` evaluates only overlap with the frozen corpus. It always leaves external novelty as `not_assessed`.

## 1. Emit full traversal candidates

Apply the patch first. Then rerun traversal with `--include-candidate-paths`:

```bash
RUN=runs/e2e/expanded_coordination_discovery_001
rm -rf "$RUN"
mkdir -p "$RUN"

python -m scripts.run_graph_traversal \
  --corpus-id "$CORPUS" \
  --mode mechanism \
  --algorithm top_n \
  --source "nitrogen coordination" \
  --target "hydrogen evolution activity" \
  --max-depth 12 \
  --top-k 8 \
  --include-candidate-paths \
  --output "$RUN/traversal.json"
```

The normal returned bundle remains unchanged, but the JSON additionally contains `candidate_paths`.

## 2. Build DiscoveryBundle

```bash
python -m scripts.build_discovery_bundle \
  --traversal "$RUN/traversal.json" \
  --top-k 8 \
  --output "$RUN/discovery.bundle.json"
```

If you later have an exploratory corpus/traversal, pass both files:

```bash
python -m scripts.build_discovery_bundle \
  --traversal "$RUN/mechanism.traversal.json" \
  --traversal "$RUN/exploratory.traversal.json" \
  --top-k 10 \
  --output "$RUN/discovery.bundle.json"
```

The builder automatically loads each traversal's matching navigation graph from:

`data_dac/corpus/<corpus_id>/<mode>/navigation/graph.graphml`

The score is an **exploration score, not a scientific novelty score**. It rewards mechanistic content, cross-paper span, graph-community span, rare relation patterns, and exploratory projection membership; it penalizes overlap with the grounding paths, navigation burden, and reverse-edge burden. A small reserve prevents scarce `CROSS_PAPER_MECHANISTIC` paths from being crowded out by generic shared-entity routes.

## 3. Build the ordinary grounded context

Use the existing pipeline exactly as before:

```bash
python -m scripts.build_explorer_packet \
  --traversal-result "$RUN/traversal.json" \
  --question "How may nitrogen coordination influence hydrogen adsorption and HER activity?" \
  --objective explain_connection \
  --output "$RUN/explorer.packet.json"

python -m scripts.run_graph_explorer \
  --packet "$RUN/explorer.packet.json" \
  --model "$OPENROUTER_AGENT_MODEL" \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key-env OPENROUTER_API_KEY \
  --output-prefix "$RUN/explorer"

python -m scripts.build_hypothesis_context \
  --packet "$RUN/explorer.packet.json" \
  --report "$RUN/explorer.report.json" \
  --output "$RUN/hypothesis.context.json"
```

## 4. Create the dual context

```bash
python -m scripts.build_dual_hypothesis_context \
  --context "$RUN/hypothesis.context.json" \
  --discovery-bundle "$RUN/discovery.bundle.json" \
  --output "$RUN/hypothesis.dual_context.json"
```

The resulting envelope contains:

- `grounded_context`: existing v2.6.x `HypothesisContext`
- `discovery_bundle`: inspiration-only routes

No discovery path/node/edge ID can pass the existing hypothesis compiler as a premise statement ID.

## 5. Run discovery-aware Hypothesis Maker

```bash
python -m scripts.run_discovery_hypothesis_maker \
  --dual-context "$RUN/hypothesis.dual_context.json" \
  --model "$OPENROUTER_AGENT_MODEL" \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key-env OPENROUTER_API_KEY \
  --output-prefix "$RUN/hypothesis_discovery" \
  --max-hypotheses 5 \
  --save-prompt
```

This uses the existing compiler and validator. The new prompt explicitly tells the LLM that discovery paths may inspire `inferential_bridge`, but are not evidence and cannot be cited as positive premises.

Portfolio output:

`$RUN/hypothesis_discovery.portfolio.json`

## 6. Internal novelty assessment

Reuse the already-built node index:

```bash
python -m scripts.assess_internal_novelty \
  --dual-context "$RUN/hypothesis.dual_context.json" \
  --portfolio "$RUN/hypothesis_discovery.portfolio.json" \
  --index-dir "data_dac/corpus/$CORPUS/mechanism/navigation/node_index" \
  --output "$RUN/internal_novelty.report.json"
```

Statuses are calibrated to the frozen corpus only:

- `reconstructs_existing_corpus_claim`
- `reconstructs_existing_corpus_chain`
- `corpus_supported_extension`
- `new_combination_within_corpus`
- `corpus_distinct_candidate`
- `insufficient_internal_evidence`

Every report keeps `external_novelty_status = not_assessed`.

## Recommended first comparison

Run the same scientific question twice:

1. Existing `run_hypothesis_maker` on the grounded context.
2. `run_discovery_hypothesis_maker` on the dual context.

Compare:

- number of hypotheses,
- source-paper count,
- internal novelty status,
- semantic critic verdicts,
- whether the generated hypotheses remain falsifiable and bounded.

Do not tune the discovery weights from one example alone. First collect several queries and inspect whether the selected paths are scientifically useful or merely structurally unusual.
