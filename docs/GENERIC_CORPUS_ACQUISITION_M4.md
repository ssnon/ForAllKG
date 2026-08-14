# Generic Corpus Acquisition M4 — DocumentPackage Materialization + Extraction Handoff

## Boundary

M4 is the last acquisition-stage step.

```text
M3 main artifacts
 + optional M3.1 SI artifacts
       ↓
document materializers
       ↓
normalized.md + metadata.json + assets
       ↓
generated version-3 paper config
       ↓
extraction_plan.jsonl
       ↓
EXISTING scripts.extract_paper
```

M4 deliberately does **not** run `extract_paper`. Extraction may invoke LLMs,
create strict-run state, and incur cost; that transition remains explicit.

## Existing GraphAgentsDAC compatibility

The generated paper config uses the repository's existing schema:

Main:

```yaml
document_id: main
role: main
package_dir: ...
markdown_file: normalized.md
metadata_file: metadata.json
selection:
  mode: whole_document
```

SI:

```yaml
document_id: si1
role: supporting_information
package_dir: ...
markdown_file: normalized.md
metadata_file: metadata.json
selection:
  mode: referenced_blocks
  fallback: skip
  reference_scope: whole_main
```

Figure handling defaults to `caption_first`.

The generated config is round-tripped through the existing
`load_paper_configs()` parser before M4 reports success.

## Materializers

### PDF

Default: the repository-pinned `marker-pdf` CLI via:

```text
marker_single <source.pdf> --output_dir <temporary-output>
```

M4 locates the generated Markdown, preserves the Marker result directory and
assets, writes `normalized.md`, and writes a provenance `metadata.json`.

The command and extra CLI arguments are policy-configurable; the generic core
does not depend on SERS.

### TXT / CSV / XLSX / DOCX / PPTX

These are converted deterministically with existing dependencies:

- TXT: UTF-8 replacement-decoding
- CSV: Markdown table
- XLSX: `openpyxl` read-only table extraction
- DOCX: `mammoth` -> HTML -> `markdownify`
- PPTX: `python-pptx` text extraction

### Deferred formats

M4 v1 does not silently unpack arbitrary ZIP bundles and does not parse legacy
`.xls`/`.doc` binaries. They remain source artifacts and appear as
`unsupported`.

This is deliberate: archive expansion needs its own zip-slip / decompression
budget and multi-document provenance contract rather than ad-hoc extraction.

## Stable paper IDs

IDs are deterministic hashes scoped by a caller-supplied prefix:

```text
SERS_API_<12 hex>
HER_API_<12 hex>
...
```

Changing corpus ordering therefore does not renumber existing papers.

## Extraction readiness

A paper is added to generated config only if its **main document** was
successfully materialized.

SI failure/absence does not make the main paper unusable; successfully
materialized SI documents are included individually.

## Progress

```text
[M4 001/100] paper=SERS_API_... main=1 si=2 ...
[M4 001/100] main:materialized si1:materialized si2:unsupported
```

## Run after M3 + M3.1

```bash
python -m scripts.materialize_corpus_documents \
  --profile-id sers_au_ag_corpus_acquisition_v1 \
  --domain-profile-id sers_au_ag \
  --data-root data_sers \
  --catalog data_acquisition/sers_au_ag_v1/m1/catalog.json \
  --selected-works data_acquisition/sers_au_ag_v1/m2/selected_works.jsonl \
  --selection-report data_acquisition/sers_au_ag_v1/m2/selection_report.json \
  --m3-dir data_acquisition/sers_au_ag_v1/m3 \
  --m3-1-dir data_acquisition/sers_au_ag_v1/m3_1 \
  --materialization-policy configs/acquisition/materialization_default_v1.yaml \
  --output-dir data_acquisition/sers_au_ag_v1/m4 \
  --generated-config configs/generated/sers_au_ag_api_v1.yaml \
  --materialization-id sers_au_ag_materialization_v1 \
  --paper-id-prefix SERS_API
```

If M3.1 is not available, omit `--m3-1-dir` to produce a main-only corpus.

## Outputs

```text
m4/
  materialization_report.json
  materialized_documents.jsonl
  paper_materialization_records.jsonl
  paper_map.jsonl
  extraction_plan.jsonl
  state/
  packages/
    SERS_API_<hash>/
      main/main/
        normalized.md
        metadata.json
        marker_original.md
        <Marker assets...>
      si1/si_1/
        normalized.md
        metadata.json
```

Generated config is written separately, for example:

```text
configs/generated/sers_au_ag_api_v1.yaml
```

## Next step

Inspect `materialization_report.json`, then run only extraction-ready entries
from `extraction_plan.jsonl`.

At that point the existing strict extraction/provenance/semantic machinery
takes over. M4 itself performs no LLM call and no positive-evidence promotion.
