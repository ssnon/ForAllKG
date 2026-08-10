# GraphAgentsDAC v2.8.0-alpha5.1 — External Novelty Calibration

## Purpose

Alpha5.1 is a calibration hotfix for the alpha5 External Novelty Assessor. It does not change the epistemic boundary: retrieved external papers remain **prior-art evidence only** and are never promoted into `premise_statement_ids`.

The hotfix is motivated by the first real alpha5 benchmark. The architecture worked, but four calibration defects were visible in the output:

- a general AuPt-alloy d-band counterexample could elevate an N-coordination-specific H3 to hypothesis-level `CONFLICTING_PRIOR_ART`;
- title-only neighbors could remain `PARTIAL_PRIOR_ART` even when no abstract was available;
- ACS supporting-information DOI variants and provider/title duplicates inflated the effective work count;
- lexical coverage compared whole multi-word `search_concepts`, so it was usually 0.0 and ranking was dominated by semantic similarity.

## 1. Scope-aware conflict gate

`CONFLICTING_PRIOR_ART` is now a strong relation. The deterministic compiler requires:

- overlapping reaction domain;
- sufficient catalyst/site-scope overlap;
- preservation of critical structural qualifiers such as nitrogen coordination and dual/diatomic sites when those qualifiers are present in the claim.

If an LLM proposes `CONFLICTING_PRIOR_ART` but the record is scientifically broader/different in scope, the compiler rewrites it to:

```text
CONTEXTUAL_CONFLICT
```

A contextual conflict is retained in the report because it can challenge a descriptor assumption, but it cannot by itself force the hypothesis status to `CONFLICTING_PRIOR_ART`.

Example:

```text
claim: N-coordination × d-band moderation of HER
paper: AuPt-alloy HER is not well described by d-band properties

alpha5:   CONFLICTING_PRIOR_ART
alpha5.1: CONTEXTUAL_CONFLICT
```

The paper remains scientifically useful, but it is not treated as a direct contradiction of the N-coordination-specific claim.

## 2. Abstract-backed partial prior art

Alpha5.1 distinguishes:

```text
PARTIAL_PRIOR_ART
```

from:

```text
TITLE_ONLY_NEIGHBOR
```

By default, `PARTIAL_PRIOR_ART` now requires an abstract. A title-only record can still be retained and displayed, but it no longer counts as substantive partial prior art in hypothesis-level status aggregation.

`DIRECT_PRIOR_ART` and `CONFLICTING_PRIOR_ART` continue to require abstracts under the default policy.

This is intentionally asymmetric: positive evidence that prior art exists may be established from adequate metadata, while absence/distinctness claims remain fail-closed.

## 3. Work canonicalization and supplementary DOI collapsing

The retriever canonicalizes records in two stages:

1. DOI-family merge, including suffixes such as `.s001`, `.s002`, `.s003`;
2. exact normalized-title merge to collapse provider records where one source has a DOI and another does not.

`PriorArtPacket` now records:

```text
raw_work_count
canonical_work_count
deduplicated_work_count
supplementary_records_collapsed
```

When an old alpha5 `prior_art.json` is reused, it is re-canonicalized locally before ranking. This allows a controlled alpha5 → alpha5.1 A/B benchmark without repeating network retrieval.

On the first uploaded alpha5 packet used to design this hotfix, local re-canonicalization reduced 396 stored records to 350 canonical records. This number is a diagnostic for that packet only, not a general expected ratio.

## 4. Lexical and domain-aware ranking

Alpha5 compared every `search_concepts` item as a whole phrase. Long phrases almost never occurred verbatim, so `lexical_coverage` was commonly 0.0.

Alpha5.1 calculates lexical coverage over distinctive tokens from the search concepts and combines four signals:

```text
0.62 semantic similarity
+ 0.18 token/concept coverage
+ 0.12 reaction-domain relevance
+ 0.08 catalyst/site-scope relevance
```

This is a retrieval-ranking signal only; it is not a novelty judgment.

For an HER claim, an ORR/CO2RR/OER record is therefore demoted relative to a similarly embedded HER record. Photocatalytic hydrogen-evolution records are also downweighted when the claim is explicitly electrocatalytic/exchange-current based.

## New/changed output fields

`RankedPriorArtWork` adds:

```text
reaction_domain_relevance
catalyst_scope_relevance
```

`PriorArtMatch` adds:

```text
reaction_domain_relevance
catalyst_scope_relevance
scope_compatible_for_conflict
scope_reason_codes
```

`ExternalNoveltyCard` adds:

```text
contextual_conflict_work_ids
```

The hypothesis-level external novelty categories remain unchanged:

```text
WELL_ESTABLISHED
LITERATURE_SUPPORTED_EXTENSION
NEW_COMBINATION_OF_KNOWN_EFFECTS
PLAUSIBLY_NOVEL
CONFLICTING_PRIOR_ART
INSUFFICIENT_SEARCH_EVIDENCE
```

No unconditional `NOVEL` category is introduced.

## Apply

```bash
cd ~/GraphAgentsDAC

git apply --check /path/to/GraphAgentsDAC_external_novelty_v280a51_from_a5.patch
git apply /path/to/GraphAgentsDAC_external_novelty_v280a51_from_a5.patch
```

Focused tests:

```bash
python -m pytest -q \
  tests/test_external_novelty_policy.py \
  tests/test_literature_retrieval.py \
  tests/test_prior_art_compiler.py \
  tests/test_prior_art_ranking.py
```

## Recommended exact A/B rerun on the existing alpha5 benchmark

Reuse the exact alpha5 claim/query plan and prior-art packet. This isolates the calibration changes from search/model decomposition drift:

```bash
RUN=runs/e2e/expanded_coordination_discovery_001

python -m scripts.run_external_novelty \
  --portfolio "$RUN/hypothesis_axis_a4.portfolio.json" \
  --lineage "$RUN/hypothesis_axis_a4.lineage.json" \
  --model "$OPENROUTER_CRITIC_MODEL" \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key-env OPENROUTER_API_KEY \
  --reuse-query-plan "$RUN/external_novelty_a5.claims_queries.json" \
  --reuse-prior-art "$RUN/external_novelty_a5.prior_art.json" \
  --output-prefix "$RUN/external_novelty_a51" \
  --save-prompts
```

No Semantic Scholar/Crossref request is needed in this A/B mode. The existing packet is re-canonicalized and the claim-level reviews are rerun with the calibrated prompt/compiler/ranker.

After the controlled rerun, a fresh-retrieval run can be used to verify the new canonicalization at ingestion time:

```bash
python -m scripts.run_external_novelty \
  --portfolio "$RUN/hypothesis_axis_a4.portfolio.json" \
  --lineage "$RUN/hypothesis_axis_a4.lineage.json" \
  --model "$OPENROUTER_CRITIC_MODEL" \
  --base-url "https://openrouter.ai/api/v1" \
  --api-key-env OPENROUTER_API_KEY \
  --providers semantic_scholar,crossref \
  --results-per-query 12 \
  --output-prefix "$RUN/external_novelty_a51_fresh" \
  --save-prompts
```

## Expected interpretation

Alpha5.1 is deliberately more conservative. A hypothesis may move from `LITERATURE_SUPPORTED_EXTENSION` to `INSUFFICIENT_SEARCH_EVIDENCE` if its previous status depended on title-only core matches. That is not a regression: it means the current metadata does not support a stronger external-prior-art statement.

Likewise, a scope-mismatched d-band counterexample should remain visible as `CONTEXTUAL_CONFLICT` without automatically converting the whole hypothesis to `CONFLICTING_PRIOR_ART`.
