# PR E — Corpus lifecycle + production publication

PR E closes the boundary between the acquisition/Strict/Bridge pipeline and the
existing Graph Explorer runtime.

## 1. Complete selected-work lifecycle accounting

`dac_her.corpus_publication.build_paper_lifecycle(...)` treats the latest
`selected_works.jsonl` as the accounting authority and joins:

```text
M3 selected work
  -> main SourceArtifact
  -> M4 materialization/paper_id
  -> M4.5 identity + full-text suitability gate
  -> Strict outcome
  -> Bridge outcome
  -> projection
  -> corpus eligibility
```

Outputs:

```text
<mode-root>/publication/paper_lifecycle.jsonl
<mode-root>/publication/funnel_summary.json
```

A selected work never disappears merely because a later stage did not emit a
row.  The lifecycle records a terminal stage/reason instead.

`funnel_summary.json` also reports selected-vs-corpus axis counts so downstream
attrition can be inspected before adding knowledge-level axis-aware backfill.

## 2. Traversal-ready production publish

`scripts.publish_strict_bridge_corpus` verifies:

- corpus structural audit passes;
- requested knowledge target is reached;
- `CORPUS_ELIGIBLE >= target_count` for a production target;
- all selected works are lifecycle-accounted.

It then builds/reuses:

```text
corpus/<id>/<mode>/graph.graphml
  -> navigation/graph.graphml
  -> navigation/node_index/manifest.json
```

and writes:

```text
<mode-root>/publication/corpus_publish_manifest.json
```

The manifest binds the corpus graph SHA-256 to the navigation graph and the
navigation graph + corpus `node_text.jsonl` to the embedding-index manifest.
Stale downstream artifacts are therefore not accepted as a current publish.

## 3. SERS/custom data roots

The existing navigation and node-index CLIs previously defaulted directly to
`data_dac`.  PR E adds `--data-root` while preserving `data_dac` as the default,
so Strict/Bridge corpora under `data_sers` publish into the same Explorer layout.

## 4. PR D integration

`run_strict_bridge_backfill` gains `--publish-on-success`.  When the knowledge
backfill returns `target_reached` or `target_already_satisfied`, the latest M3.2
selection is published without rerunning unchanged Strict/Bridge work.

Example:

```bash
python -m scripts.run_strict_bridge_backfill \
  ... existing PR D arguments ... \
  --target-count 20 \
  --target-status BRIDGE_USEFUL \
  --publish-on-success
```

For a publication-only rerun:

```bash
python -m scripts.publish_strict_bridge_corpus \
  --corpus-id sers_strict_bridge_pilot_20 \
  --domain-profile sers_au_ag \
  --data-root data_sers \
  --mode mechanism \
  --selected-works <latest-m3.2>/selected_works.jsonl \
  --m3-dir <latest-m3.2> \
  --m4-dir data_acquisition/sers_au_ag_v1/m4 \
  --m4-5-dir data_acquisition/sers_au_ag_v1/m4_5 \
  --target-count 20 \
  --target-status BRIDGE_USEFUL
```

Use `--skip-node-index` for a graph-only publication smoke test.  Node-index
resume validity is checked against the persisted navigation-graph and node-text
hashes, so an unchanged corpus does not re-embed nodes.

## Canonical M4/M4.5 supersets

M4 and M4.5 are canonical resumable acquisition/materialization directories and
may retain records from a broader historical selection than the latest active
M3.2 lifecycle snapshot. Publication therefore joins those stages by `work_id`
for the active selection and records extra rows as `allowed_superset_record_counts`.
They do not block publication. Active M3 artifacts and Strict/Bridge outcomes
remain exact-snapshot checks, and the final corpus manifest paper set must still
match lifecycle `CORPUS_ELIGIBLE` papers exactly.
