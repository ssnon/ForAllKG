# alpha4b.3a — Reusable CorpusSemantics

## Goal

Move cross-paper scientific policy out of `dac_her/corpus_graph.py` and into
`ScientificDomainProfile`, while keeping corpus algorithms shared and
non-destructive.

The invariant is:

> algorithm = shared; scientific meaning = domain-owned.

## Contract

`CorpusSemantics` owns:

- the node types eligible for cross-paper exact-label review candidates;
- whether confirmed exact Bridge-pattern alignment is enabled;
- an optional high-priority review override for legacy compatibility.

Registry-safe alignment types are **not duplicated** in `CorpusSemantics`.
They are read from `profile.resolution.auto_merge_types`, so resolution and
corpus alignment have one source of truth.

## Domain policies

### DAC-HER

- registry alignment: `Metal`, `Reaction`
- review candidates: legacy corpus set
- confirmed exact Bridge-pattern alignment: enabled
- legacy direct `build_corpus_graph(...)` callers remain DAC-HER compatible

### SERS Au/Ag

- registry alignment: `Metal` only
- review candidates:
  `PlasmonicSubstrate`, `Nanostructure`, `Support`, `Material`,
  `StructuralMotif`, `Morphology`, `Analyte`, `RamanReporter`
- confirmed exact Bridge-pattern alignment: enabled
- scientific text normalization is explicitly bound to the SERS profile

### Broad catalysis

- registry alignment: `Metal`, `Reaction`
- review behavior preserves the pre-alpha4b.3a corpus default
- Bridge-pattern alignment capability: disabled

## Safety invariants

1. Corpus construction never performs destructive cross-paper merges.
2. Explicit domain builds fail closed when any projection bundle has a missing
   or mismatched `domain_profile_id`.
3. A semantic Bridge candidate never becomes a `CorpusPattern`, even when a
   stale/legacy node also has `retention_lane=accepted_pattern`.
4. Confirmed pattern alignment remains exact normalized
   `(subject, relation, object)` only.
5. Registry alignment remains exact normalized signature only and is limited
   to the domain profile's `resolution.auto_merge_types`.
6. Missing scientific detail is not converted into an invented merge.
7. Legacy direct callers without an explicit profile preserve DAC-HER behavior.

## Provenance metadata

Corpus GraphML graph attributes now include:

- `domain_profile_id`
- `corpus_semantics_id`

Corpus manifest also records:

- `domain_profile_id`
- `corpus_semantics_id`
- `registry_alignment_types`
- `review_candidate_types`
- `high_priority_review_types`
- `pattern_alignment_mode`
- `pattern_alignment_enabled`
- `destructive_cross_paper_merges`

## Calibration

After installation, run the SERS calibration set:

```bash
python -m scripts.build_corpus_graph \
  --domain-profile sers_au_ag \
  --data-root data_sers \
  --corpus-id sers_alpha4b3a_calibration \
  --mode exploratory \
  --paper-ids \
    Kiwook_SERS_1 \
    Kiwook_SERS_5 \
    Kiwook_SERS_8
```

Check:

- structural audit passes;
- source projection node/edge counts are preserved;
- destructive merges = 0;
- every registry hub has `entity_type == Metal`;
- semantic-candidate Bridge nodes create no corpus pattern hub;
- review candidates use only the SERS review-candidate types;
- confirmed cross-paper Bridge patterns align only on exact normalized triples.

alpha4b.3b (ComparisonContext) is intentionally out of scope.
