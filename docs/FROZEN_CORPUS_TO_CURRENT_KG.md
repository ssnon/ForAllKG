# Frozen Drive Corpus -> Current GraphAgentsDAC KG Pipeline (v0.2)

This adapter targets the actual `feat/feasibility-v2.7.0` pipeline contract.

## Why v0.1 flat staging was replaced

The current repository does **not** consume an arbitrary flat directory of Markdown files.
`scripts.extract_paper` resolves a paper through a `papers.yaml` document configuration, then
loads each Markdown as a **document package**. The package directory is used to find Marker
images and preserve figure/page provenance. Therefore flattening only `normalized.md` files
can detach relative image assets from the Markdown.

Ingestion `normalized.md` also contains operational YAML frontmatter that did not exist in the
original Marker documents. The scientific extractor should receive the raw Marker Markdown,
not this frontmatter.

v0.2 therefore:

1. freezes/deduplicates the ingestion manifest;
2. verifies frozen `normalized.md` hashes;
3. recovers the exact sibling raw Marker Markdown by content match;
4. generates a separate version-3 `papers.yaml` pointing to the **original Marker package
   directories**;
5. runs the repository's existing pipeline scripts without replacing their scientific logic.

## Actual current-repository flow

```text
frozen ingestion manifest
        |
        v
generated papers.yaml
        |
        +--> scripts.extract_paper
        |       |
        |       v
        |   strict chunk extraction
        |       |
        +--> scripts.build_paper_graph
        |       |
        |       v
        |   canonical per-paper GraphML
        |
        +--> [mechanism/exploratory only]
        |    scripts.extract_bridge_graph
        |
        +--> scripts.build_graphagents_projection
                |
                v
        per-paper GraphAgents projection
                |
                v
        scripts.build_corpus_graph
                |
                v
        scripts.build_navigation_graph
                |
                v
        scripts.build_node_index
```

## 1. Freeze the ingestion corpus

```bash
python -m scripts.freeze_ingestion_corpus \
  --manifest data_dac/ingestion/corpora/dac_her_drive_v1/manifest.json
```

Expected for the supplied 134-document manifest:

```text
source documents:          134
eligible documents:        134
exact duplicates removed:    1
frozen documents:          133
```

Output:

```text
data_dac/frozen_corpora/dac_her_drive_v1/manifest.json
```

## 2. Generate a current-repo-compatible papers.yaml

```bash
python -m scripts.generate_frozen_paper_config \
  --frozen-manifest data_dac/frozen_corpora/dac_her_drive_v1/manifest.json \
  --validate-with-repo
```

Output:

```text
data_dac/generated_configs/dac_her_drive_v1/papers.yaml
data_dac/generated_configs/dac_her_drive_v1/papers.adapter.json
```

The generated config is separate from the hand-maintained `configs/papers.yaml`.

A generated main document looks conceptually like:

```yaml
document_id: main
role: main
package_dir: data_dac/ingestion/markdown/<paper>/main/<marker-package>
markdown_file: <raw-marker-output>.md
selection:
  mode: whole_document
figure_processing:
  mode: caption_first
  vision_assets: []
```

SI documents use:

```yaml
selection:
  mode: referenced_blocks
  fallback: skip
  reference_scope: whole_main
```

No `normalized.md` is passed to the scientific extractor.

## 3. No-cost command preflight

First test two papers. Use a separate corpus ID for a subset.

```bash
python -m scripts.run_frozen_corpus_pipeline \
  --frozen-manifest data_dac/frozen_corpora/dac_her_drive_v1/manifest.json \
  --corpus-id dac_her_drive_smoke2 \
  --mode evidence \
  --paper-id '정아현_1' \
  --paper-id '주예준_3' \
  --skip-node-index \
  --dry-run
```

This only prints the exact existing-repository commands that would run.

## 4. Two-paper E2E smoke test

`evidence` mode is the safest first E2E because it tests strict extraction, canonical paper
GraphML, GraphAgents projection, corpus construction, and navigation without running the
Bridge LLM stage.

```bash
python -m scripts.run_frozen_corpus_pipeline \
  --frozen-manifest data_dac/frozen_corpora/dac_her_drive_v1/manifest.json \
  --corpus-id dac_her_drive_smoke2 \
  --mode evidence \
  --paper-id '정아현_1' \
  --paper-id '주예준_3' \
  --skip-node-index \
  --heartbeat-seconds 30
```

The runner invokes, for each selected paper:

```text
scripts.extract_paper
scripts.build_paper_graph
scripts.build_graphagents_projection --mode evidence
```

Then globally:

```text
scripts.build_corpus_graph
scripts.build_navigation_graph
```

## 5. Inspect smoke outputs

Expected high-level paths:

```text
data_dac/extracted/<paper>/latest_run.json
data_dac/extracted/<paper>/<paper>.graphml
data_dac/extracted/<paper>/graphagents/evidence/graph.graphml

data_dac/corpus/dac_her_drive_smoke2/evidence/graph.graphml
data_dac/corpus/dac_her_drive_smoke2/evidence/navigation/graph.graphml
```

Logs:

```text
data_dac/pipeline_logs/dac_her_drive_smoke2/evidence/
```

Resume state:

```text
data_dac/pipeline_state/dac_her_drive_smoke2/evidence.json
```

Successful stages are skipped on rerun when the state fingerprint and expected output still
match.

## 6. Full 133-paper evidence corpus

Once the smoke corpus passes:

```bash
python -m scripts.run_frozen_corpus_pipeline \
  --frozen-manifest data_dac/frozen_corpora/dac_her_drive_v1/manifest.json \
  --corpus-id dac_her_drive_v1_evidence \
  --mode evidence \
  --skip-node-index \
  --heartbeat-seconds 30
```

Build the node index after the structural corpus/navigation outputs have been reviewed:

```bash
python -m scripts.build_node_index \
  --corpus-id dac_her_drive_v1_evidence \
  --mode evidence
```

## 7. Mechanism / exploratory mode

After strict evidence extraction is stable, mechanism mode adds Bridge extraction per paper:

```bash
python -m scripts.run_frozen_corpus_pipeline \
  --frozen-manifest data_dac/frozen_corpora/dac_her_drive_v1/manifest.json \
  --corpus-id dac_her_drive_v1_mechanism \
  --mode mechanism \
  --skip-node-index
```

Exploratory mode additionally requires the candidate Bridge materialization expected by the
current `build_graphagents_projection` implementation:

```bash
python -m scripts.run_frozen_corpus_pipeline \
  --frozen-manifest data_dac/frozen_corpora/dac_her_drive_v1/manifest.json \
  --corpus-id dac_her_drive_v1_exploratory \
  --mode exploratory \
  --skip-node-index
```

Because Bridge extraction requires additional LLM work for every strict-valid chunk, do not
start with the full 133-paper exploratory run. Smoke-test the same one or two papers first.

## Operational flags

Run first N configured papers:

```bash
--limit 2
```

Select explicit papers:

```bash
--paper-id ID1 --paper-id ID2
```

Internal extraction concurrency:

```bash
--extract-concurrency 4
--bridge-concurrency 4
```

Progress heartbeat:

```bash
--heartbeat-seconds 30
```

Disable heartbeat:

```bash
--heartbeat-seconds 0
```

Restart stages even when runner state says passed:

```bash
--no-resume
```

Force strict/Bridge cache regeneration:

```bash
--force-extract
--force-bridge
```

Allow critical partial extraction only when explicitly intended:

```bash
--allow-partial
```

## Important invariants

- PDF fingerprint duplicate removal occurs before KG extraction.
- same-title-only duplicates are not automatically deleted.
- frozen Markdown hashes are verified before generating the KG input config.
- scientific extraction consumes raw Marker Markdown, not ingestion frontmatter.
- Marker package directories remain intact so relative figure/image provenance is preserved.
- main and SI remain documents of one paper ID; SI is not treated as an independent paper.
- existing extraction, graph normalization, Bridge, projection, corpus, navigation, and node
  indexing logic is reused rather than reimplemented.
