# GraphAgentsDAC v2.8.0-alpha3 — Candidate-unit-aware discovery traversal

## Why alpha3 exists

The 31-paper nitrogen-coordination benchmark exposed a representation/traversal mismatch:

- exploratory NavigationGraph: 1,306 semantic-candidate navigation edges;
- ordinary exploratory `top_n`: zero candidate-bearing returned paths;
- candidate topology: 653 original grounded-anchor -> candidate edges plus 653 synthesized reverse-navigation edges;
- 437 reconstructed candidate concepts with grounding anchors;
- 73 candidate concepts have at least two distinct grounded anchors;
- a diagnostic virtual candidate-unit traversal produced 1,788 valid source -> unit -> target routes for the benchmark query.

A single provisional scientific bridge is therefore **not** one navigation edge. It is a semantic-candidate concept plus its grounded anchors. Traversing from anchor A through that concept to a distinct anchor B uses two navigation edges but consumes one epistemic candidate unit.

Alpha3 makes that unit explicit.

## New architecture

```text
Grounded source matches
        |
        v
CandidateUnitBuilder
  candidate concept
  + original (non-reverse) grounding anchors
  + proposed subject / relation / object
  + provenance
        |
        v
CandidateUnitSelector
  ranks (source, unit, target)
  with confirmed-only prefix/suffix
        |
        v
Candidate-unit traversal
  confirmed prefix
  + exactly one candidate unit
  + confirmed suffix
        |
        v
DiscoveryBundle v3
  candidate-core semantic redundancy
  candidate exploration reserve
  reaction-domain switch penalty
  existing generic/registry/grounding gates
        |
        v
Discovery-aware Hypothesis Maker a3
  candidate remains inspiration_only
```

## Epistemic semantics

For a candidate concept C grounded by anchors A and B, the NavigationGraph contains the original edges:

```text
A -> C
B -> C
```

and synthesized reverse-navigation edges:

```text
C -> A
C -> B
```

The discovery traversal may navigate:

```text
A -> C -> B
```

but **C -> B is not interpreted as a causal/scientific direction**. The candidate unit is represented separately as one unverified proposed scientific bridge grounded by A and B.

`A -> C -> A` is rejected as a trivial excursion because entry and exit anchors must be distinct.

## Files

New:

- `dac_her/candidate_units.py`
- `pipeline_core/discovery/candidate_unit_selection.py`
- `scripts/run_candidate_unit_traversal.py`
- `tests/test_candidate_units.py`
- `tests/test_candidate_unit_selection.py`
- `tests/test_discovery_bundle_candidate_unit.py`

Modified:

- `dac_her/discovery_contracts.py`
- `pipeline_core/discovery/discovery_bundle.py`
- `dac_her/discovery_hypothesis_prompt.py`
- `scripts/build_discovery_bundle.py`

## Candidate-unit selector dimensions

Positive:

- endpoint relevance
- candidate-unit semantic relevance to the source/target query
- mechanistic continuity around the candidate unit
- confirmed scientific-content density
- cross-paper span

Penalties:

- generic-entity burden
- alignment/registry burden
- confirmed reverse-navigation burden
- unrelated reaction-domain switching
- path length

The selector score is an **exploration heuristic**, not novelty, confidence, or evidentiary strength.

## DiscoveryBundle alpha3 changes

1. Candidate-unit routes use **candidate-core semantics** for semantic redundancy:

```text
entry anchor + candidate concept + exit anchor + proposed S/R/O
```

rather than mean-pooling the entire confirmed prefix/suffix. This prevents a genuinely new provisional bridge from being rejected merely because its surrounding grounding route resembles a canonical mechanism path.

2. `CANDIDATE_EXPLORATION` paths may receive reserved slots only after passing the existing alpha2.1 quality gates.

3. Candidate-unit lineage is preserved in each `DiscoveryInspiration`:

- candidate unit ID/label
- entry/exit anchors
- proposed subject/relation/object
- candidate-unit selector score
- reaction-domain switch penalty

4. Prompt version becomes:

```text
hypothesis-maker-discovery-prompt-v2.8.0-a3
```

The prompt explicitly states that the candidate relation is not a positive premise and that candidate->exit navigation direction is not a causal claim.

## Apply on top of alpha2.1

```bash
cd ~/GraphAgentsDAC

git apply --check /path/to/GraphAgentsDAC_discovery_v280a3_from_a21.patch
git apply /path/to/GraphAgentsDAC_discovery_v280a3_from_a21.patch
```

Tests:

```bash
python -m pytest -q \
  tests/test_candidate_units.py \
  tests/test_candidate_unit_selection.py \
  tests/test_discovery_bundle_candidate_unit.py \
  tests/test_discovery_bundle.py \
  tests/test_dual_hypothesis_context.py \
  tests/test_internal_novelty.py
```

## Benchmark run

The existing exploratory graph/index can be reused; no projection or index rebuild is required if they have not changed.

```bash
CORPUS=dac_her_expanded_v1
RUN=runs/e2e/expanded_coordination_discovery_001

python -m scripts.run_candidate_unit_traversal \
  --corpus-id "$CORPUS" \
  --source "nitrogen coordination" \
  --target "hydrogen evolution activity" \
  --node-map-k 20 \
  --max-depth 12 \
  --top-k 12 \
  --include-candidate-paths \
  --output "$RUN/candidate_unit.traversal.a3.json"
```

Important output fields:

```text
Bridge-capable candidate units
Valid source→unit→target routes
Returned paths
unit score
unit relevance
mechanistic continuity
reaction-domain switch penalty
entry / exit anchors
```

The exact number of valid routes can differ from the earlier diagnostic because alpha3 uses hop-budget-aware confirmed prefix/suffix search and deterministic route-quality selection.

## Build the dual-lane DiscoveryBundle

Use the mechanism traversal as grounding reference and the candidate-unit traversal as the discovery lane. The old exploratory `top_n` traversal is not required for this benchmark because it replayed the mechanism paths.

```bash
python -m scripts.build_discovery_bundle \
  --traversal "$RUN/traversal.json" \
  --traversal "$RUN/candidate_unit.traversal.a3.json" \
  --top-k 8 \
  --output "$RUN/discovery.bundle.a3.json"
```

A good result is quality-limited, not quota-limited. Two to four strong candidate-unit inspirations are preferable to eight weak ones.

## Continue to hypothesis generation

```bash
OLD_RUN=runs/e2e/expanded_coordination_topn_002

python -m scripts.build_dual_hypothesis_context \
  --context "$OLD_RUN/hypothesis.context.json" \
  --discovery-bundle "$RUN/discovery.bundle.a3.json" \
  --output "$RUN/hypothesis.dual_context.a3.json"

python -m scripts.run_discovery_hypothesis_maker \
  --dual-context "$RUN/hypothesis.dual_context.a3.json" \
  --model "$OPENROUTER_AGENT_MODEL" \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key-env OPENROUTER_API_KEY \
  --output-prefix "$RUN/hypothesis_discovery_a3" \
  --max-hypotheses 5 \
  --save-prompt
```

Then run the existing Semantic Critic and internal novelty assessor unchanged.

## Regression expectations for the nitrogen-coordination benchmark

Routes analogous to these should be favored:

- coordination geometry -> formation energy / symmetry / M-M distance / orbital hybridization;
- electrolyte-dependent activity selection;
- adjacent-site dependence of Tafel/Heyrovsky HER steps;
- nitrogen coordination -> charge donation -> Volmer behavior;
- DAC-vs-SAC HER/Volmer contrasts.

Routes dominated by these patterns should be suppressed:

- CO2RR/ORR detours unrelated to the requested HER mechanism;
- long metal-registry/entity hopping;
- comparator-only bridges;
- multiple semantically equivalent ΔG_H/HER candidate units consuming separate slots.
