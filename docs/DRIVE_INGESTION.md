# GraphAgentsDAC Drive ingestion v0.1

This layer is intentionally placed **before** the existing KG/explorer/hypothesis pipeline.
It treats Drive PDFs as immutable sources, Marker output as reproducible derived artifacts, and the registry as the canonical mapping between them.

## Stages implemented

1. **Discovery** — recursively list the configured Drive folder and read `Article_lists`.
2. **Matching** — join `File_Name` to the main PDF and tolerate both `_SI1.pdf` and `_SI_1.pdf`.
3. **Registry** — store stable Drive IDs/fingerprints and paper metadata.
4. **Source sync** — download only papers that need processing.
5. **Marker conversion** — call the existing `marker_single` CLI with Markdown output and page pagination.
6. **Normalization** — prepend provenance frontmatter without rewriting scientific content.
7. **QC** — block missing main/SI sources and conversion failures; warn on weak title/short output.
8. **Incremental sync** — skip unchanged source fingerprints when Marker version and QC state also match.
9. **Corpus manifest** — emit a stable JSON adapter for the next KG-building stage.

## Installation

Keep your working Marker installation. Add only the Google API dependencies:

```bash
python -m pip install -r requirements-ingestion.txt
python -m pip install -e . --no-deps
```

Confirm the local environment first:

```bash
python -m scripts.check_ingestion_env
```

You can also inspect Marker directly with `marker_single --help`. The wrapper checks that the installed CLI supports `--output_dir`, `--output_format`, and `--paginate_output` before conversion.

## Google service account

Create/read a service-account JSON credential and share both the Drive folder and `Article_lists` spreadsheet with the service-account email as **Viewer**.

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/service-account.json
export DAC_DRIVE_FOLDER_ID=1md4l30lERPwg7VCA274EN-c6YohwBWRD
export DAC_ARTICLE_SHEET_ID=1WAAKtdcXe9hmULmD0X3dUoWmUNbv1p54X-ppRmi9X5I
export DAC_ARTICLE_SHEET_RANGE='Sheet1!A:H'
```

The pipeline never writes to Drive or the spreadsheet.

## Optional annotator aliases

Copy the example:

```bash
cp configs/ingestion/annotator_aliases.example.json configs/ingestion/annotator_aliases.json
```

Example:

```json
{
  "홍기욱": "Kiwook"
}
```

With this alias, `홍기욱_1.pdf` becomes `paper_id=Kiwook_1`. Without an alias, the stable ID remains `홍기욱_1`.

## Stage 1 — metadata dry run

Run this before downloading anything:

```bash
python -m scripts.sync_drive_corpus --dry-run
```

Inspect:

```text
data_dac/ingestion/runs/latest.json
```

Important issue codes include:

- `missing_main_file`
- `duplicate_main_file`
- `si_count_mismatch`
- `unregistered_pdf`
- `duplicate_article_row`

A missing expected SI is blocking. Extra SI is a warning so it can be reviewed without hiding data.

## Stage 2 — download only

```bash
python -m scripts.sync_drive_corpus --download-only
```

Sources are cached as:

```text
data_dac/ingestion/sources/<paper_id>/main.pdf
data_dac/ingestion/sources/<paper_id>/si_1.pdf
...
```

## Stage 3 — download + Marker conversion + QC

```bash
python -m scripts.sync_drive_corpus --corpus-id dac_her_drive_v1
```

Marker outputs are isolated per document, and a `normalized.md` is written next to Marker output so image-relative paths continue to work.

```text
data_dac/ingestion/markdown/<paper_id>/main/**/normalized.md
data_dac/ingestion/markdown/<paper_id>/si_1/**/normalized.md
```

The Markdown body is Marker output. The pipeline only prepends frontmatter such as source Drive ID, source filename, paper/document IDs, role, Marker version, and parent SI relationship.

## Stage 4 — incremental re-run

Run the same command again. A paper is skipped only when all of these remain true:

- main/SI Drive fingerprints are unchanged;
- Marker version is unchanged;
- prior QC was passed or passed-with-warnings;
- normalized Markdown outputs still exist.

Force a rebuild after changing Marker settings/version expectations:

```bash
python -m scripts.sync_drive_corpus --force-reconvert --corpus-id dac_her_drive_v1
```

Extra Marker flags can be passed one at a time:

```bash
python -m scripts.sync_drive_corpus \
  --marker-arg=--force_ocr \
  --corpus-id dac_her_drive_v1
```

## Stage 5 — ingestion corpus manifest

Successful papers are exported to:

```text
data_dac/ingestion/corpora/<corpus_id>/manifest.json
```

Schema:

```json
{
  "schema_version": "graphagentsdac-ingestion-corpus-v01",
  "corpus_id": "dac_her_drive_v1",
  "documents": [
    {
      "paper_id": "Kiwook_1",
      "main_markdown": ".../normalized.md",
      "supporting_markdown": [".../normalized.md"],
      "source_fingerprint": {},
      "marker_version": "...",
      "qc_status": "passed"
    }
  ]
}
```

This manifest is deliberately **pre-KG**. The current graph traversal consumes already-built `data_dac/corpus/<corpus_id>/<mode>/navigation/graph.graphml` and node indexes, so v0.1 does not overwrite that contract. The next adapter should consume this ingestion manifest and invoke the repository's existing extraction/KG build path.

## Tests

```bash
python -m pytest -q \
  tests/test_ingestion_naming.py \
  tests/test_ingestion_discovery.py \
  tests/test_ingestion_registry_manifest.py \
  tests/test_ingestion_marker_runner.py
```

The tests do not require real Google credentials or Marker models; the Marker wrapper test uses a fake CLI.

## Recommended Git policy

For a public repository, do not commit publisher PDFs or generated full-text Markdown unless licensing permits it. A safe default is to ignore:

```gitignore
data_dac/ingestion/sources/
data_dac/ingestion/markdown/
data_dac/ingestion/runs/
```

Keep code, registry schema/configuration, and non-copyrighted manifests/metadata according to your project policy.

## Progress reporting

Long runs print progress by default. Typical output:

```text
[ingestion] Drive scan complete: 184 files discovered
[ingestion] [ 37/128] Yejun_18 | marker start: main.pdf
[ingestion] [ 37/128] Yejun_18 | marker still running: main.pdf (elapsed 60s)
[ingestion] [ 37/128] Yejun_18 | marker finished: main.pdf (exit=0, elapsed=83.4s)
[ingestion] [ 37/128] Yejun_18 | passed | passed=31, unchanged=5, metadata_blocked=1
```

`marker_single` heartbeat messages are emitted every 30 seconds by default. Change the interval with:

```bash
python -m scripts.sync_drive_corpus --heartbeat-seconds 15
```

Use `--heartbeat-seconds 0` to disable Marker heartbeat messages, or `--quiet` to disable terminal progress output entirely.

A machine-readable checkpoint is updated after every paper:

```text
data_dac/ingestion/runs/latest_progress.json
```

From another terminal, display it once:

```bash
python -m scripts.show_ingestion_progress
```

or watch it continuously:

```bash
python -m scripts.show_ingestion_progress --watch
```
