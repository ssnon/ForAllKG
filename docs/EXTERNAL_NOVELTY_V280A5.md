# GraphAgentsDAC v2.8.0-alpha5 — External Novelty Agent

## Purpose

Alpha5 implements stage 4 of the novelty-aware discovery pipeline: a **search-bounded external prior-art assessment** for accepted alpha4 hypotheses.

It does **not** claim literature-wide novelty and it does **not** turn newly retrieved papers into scientific positive premises.

The epistemic boundary is explicit:

```text
HypothesisPortfolio
      ↓
NoveltyClaimDecomposer
      ↓
LiteratureQueryPlanner
      ↓
Semantic Scholar + Crossref metadata retrieval
      ↓
PriorArtPacket                  [prior-art only]
      ↓
claim-level semantic ranking
      ↓
LLM relationship review        [bounded to supplied title/abstract]
      ↓
ExternalNoveltyAssessor        [deterministic policy]
```

External prior-art records may be used to answer **"what is already known?"**. If a newly found paper is later needed as a positive scientific premise, it must enter through the normal ingestion → extraction → KG/evidence-context pipeline first.

## Overall statuses

- `WELL_ESTABLISHED`
- `LITERATURE_SUPPORTED_EXTENSION`
- `NEW_COMBINATION_OF_KNOWN_EFFECTS`
- `PLAUSIBLY_NOVEL`
- `CONFLICTING_PRIOR_ART`
- `INSUFFICIENT_SEARCH_EVIDENCE`

There is intentionally no unconditional `NOVEL` status.

`PLAUSIBLY_NOVEL` means only that no direct match was found for the core decomposed claims **under the recorded search coverage**. It is not proof of absence from the literature.

## Claim-level assessment

Each hypothesis is decomposed into up to four distinguishing claims such as:

- mediator,
- moderator / interaction,
- context condition,
- pathway competition,
- descriptor interaction,
- distinctive prediction,
- mechanistic link.

Retrieved records are reviewed as:

- `DIRECT_PRIOR_ART`
- `PARTIAL_PRIOR_ART`
- `COMPONENT_ONLY`
- `CONFLICTING_PRIOR_ART`
- `UNRELATED`
- `INSUFFICIENT_METADATA`

Strong direct/conflicting judgments require an abstract by default. A title-only direct judgment from the LLM is deterministically downgraded.

## Retrieval providers

Alpha5 contains adapters for:

1. Semantic Scholar Academic Graph relevance search.
   - API key is optional.
   - If present, `SEMANTIC_SCHOLAR_API_KEY` is sent as `x-api-key`.
2. Crossref REST `/works` bibliographic search.
   - Optional `CROSSREF_MAILTO` enables Crossref's polite-pool identification.

Provider failures are recorded per query and do not silently become evidence of absence.

## Files

New modules:

```text
dac_her/external_novelty_contracts.py
dac_her/novelty_claim_decomposition.py
dac_her/literature_retrieval.py
dac_her/prior_art_matching.py
dac_her/external_novelty_llm.py
dac_her/external_novelty.py
scripts/run_external_novelty.py
```

Tests:

```text
tests/test_external_novelty_policy.py
tests/test_prior_art_compiler.py
tests/test_literature_retrieval.py
```

## Apply

The patch is intended for:

```text
feat/novelty_agent-v2.8.0
```

Apply:

```bash
cd ~/GraphAgentsDAC

git apply --check /path/to/GraphAgentsDAC_external_novelty_v280a5_from_novelty_agent.patch
git apply /path/to/GraphAgentsDAC_external_novelty_v280a5_from_novelty_agent.patch
```

Focused tests:

```bash
python -m pytest -q \
  tests/test_external_novelty_policy.py \
  tests/test_prior_art_compiler.py \
  tests/test_literature_retrieval.py
```

## Run on the current alpha4 H1-H4

```bash
RUN=runs/e2e/expanded_coordination_discovery_001

python -m scripts.run_external_novelty \
  --portfolio "$RUN/hypothesis_axis_a4.portfolio.json" \
  --lineage "$RUN/hypothesis_axis_a4.lineage.json" \
  --model "$OPENROUTER_CRITIC_MODEL" \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key-env OPENROUTER_API_KEY \
  --providers semantic_scholar,crossref \
  --results-per-query 12 \
  --output-prefix "$RUN/external_novelty_a5" \
  --save-prompts
```

Optional:

```bash
export SEMANTIC_SCHOLAR_API_KEY=...
export CROSSREF_MAILTO=you@example.com
```

The API-key variables are not required by the alpha5 contracts; they only improve provider identification/rate limits where supported.

## Outputs

```text
external_novelty_a5.claims_queries.json
external_novelty_a5.prior_art.json
external_novelty_a5.report.json
external_novelty_a5.prompts/
```

The prior-art packet stores normalized metadata, query provenance, provider successes/failures, and the explicit epistemic label:

```text
prior_art_only_not_positive_premise
```

## Reproducible re-assessment

Retrieval is time-dependent. To re-run only matching/assessment against the same frozen prior-art packet:

```bash
python -m scripts.run_external_novelty \
  --portfolio "$RUN/hypothesis_axis_a4.portfolio.json" \
  --lineage "$RUN/hypothesis_axis_a4.lineage.json" \
  --model "$OPENROUTER_CRITIC_MODEL" \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key-env OPENROUTER_API_KEY \
  --reuse-prior-art "$RUN/external_novelty_a5.prior_art.json" \
  --output-prefix "$RUN/external_novelty_a5_recheck"
```

Note: claim decomposition/query generation is also model-generated. `--reuse-prior-art` therefore requires the regenerated query plan to have the same stable plan ID. For strict replay, preserve the saved query plan and prompts along with the packet; future alpha6 can add a direct `--query-plan` replay path.

## Fail-closed absence policy

Positive prior-art matches can support `WELL_ESTABLISHED` or `LITERATURE_SUPPORTED_EXTENSION` even under modest coverage because they are presence claims.

Absence-dependent categories (`NEW_COMBINATION_OF_KNOWN_EFFECTS`, `PLAUSIBLY_NOVEL`) require minimum coverage. Defaults:

```text
unique works >= 10
works with abstracts >= 5
abstract-bearing candidates per core claim >= 3
successful queries >= 2
```

If those conditions are not met, the status is `INSUFFICIENT_SEARCH_EVIDENCE` rather than a novelty-positive label.
