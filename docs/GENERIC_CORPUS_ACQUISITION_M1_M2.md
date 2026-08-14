# Generic Corpus Acquisition M1+M2

## Scope

This patch adds a new, domain-independent literature acquisition lane without
changing the existing external-novelty lane.

Existing:

```text
Hypothesis -> LiteratureQueryPlan -> PriorArtPacket
```

New:

```text
AcquisitionProfile
  -> neutral CatalogQuery
  -> Semantic Scholar / Crossref
  -> LiteratureCatalogPacket
  -> deterministic CandidateAssessment
  -> quota-aware SelectedCorpusWork
```

M1/M2 is intentionally **metadata/abstract only**. It does not download PDF/SI,
does not materialize `normalized.md`, and does not promote external literature
into positive KG evidence.

That promotion boundary remains:

```text
selected metadata
  -> M3 access resolution / source acquisition
  -> M4 document materialization
  -> existing extract_paper / provenance gates
  -> canonical KG evidence
```

## Epistemic separation

`LiteratureCatalogPacket.epistemic_usage` is frozen to:

`candidate_source_only_not_positive_premise`

Candidate scoring can identify that a paper appears to study an axis such as
`nanogap_size`, but it never writes an effect direction such as
`smaller gap -> stronger SERS`.

## M1 outputs

`discover_literature_catalog.py` writes:

- `catalog.json`
- `queries.jsonl`
- `candidates.jsonl`
- `discovery_report.json`

Semantic Scholar and Crossref are implemented as neutral catalog providers.
Provider rows are canonicalized by DOI family and exact normalized title while
preserving provider/query/axis provenance.

## M2 outputs

`select_corpus_candidates.py` writes:

- `assessments.jsonl`
- `selected_works.jsonl`
- `selection_report.json`

Selection uses:

1. hard eligibility rules,
2. deterministic lexical/scalar scoring,
3. per-axis primary quotas,
4. global score fill for remaining capacity.

One work can match multiple axes, but is charged to at most one primary quota
axis. This prevents a single broad paper from satisfying many diversity quotas.

## First profile

`configs/acquisition/sers_au_ag_v1.yaml` is the first domain-specific
configuration for the generic engine. It contains SERS search axes and quotas;
there is no SERS branch in the acquisition Python code.

## Example

```bash
python -m scripts.discover_literature_catalog \
  --profile configs/acquisition/sers_au_ag_v1.yaml \
  --output-dir data_acquisition/sers_au_ag_v1/m1
```

Then:

```bash
python -m scripts.select_corpus_candidates \
  --profile configs/acquisition/sers_au_ag_v1.yaml \
  --catalog data_acquisition/sers_au_ag_v1/m1/catalog.json \
  --output-dir data_acquisition/sers_au_ag_v1/m2
```

Optional environment variables:

```text
SEMANTIC_SCHOLAR_API_KEY
CROSSREF_MAILTO
```

## What to audit before M3

Before automatically downloading full text, inspect:

- provider-query failures,
- excluded/manual-review counts,
- selected work titles,
- axis candidate counts,
- unfilled axis quotas,
- OA availability rate,
- obvious scope leakage.

M3 should consume `selected_works.jsonl`, not the entire discovery pool.


## Progress output

Large discovery and selection runs print deterministic `N/NNN` progress.

M1 counts provider-query executions:

```text
[M1 01/32] semantic_scholar  axis=nanogap query="..."
[M1 01/32] ok   results= 50 elapsed=0.74s
[M1 02/32] crossref          axis=nanogap query="..."
```

M2 counts every candidate assessment and then every selected work:

```text
[M2 assess 001/527] eligible      score=  8.50 ...
[M2 assess 527/527] excluded      score=  1.00 ...
[M2 select 001/100] phase=quota axis=nanogap work=...
[M2 select 100/100] phase=global_fill axis=- work=...
```

The progress counters are display-only and are not used as acceptance targets.
