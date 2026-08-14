# Generic Corpus Acquisition M3.1 — Supplementary Artifact Discovery

## Goal

M3.1 discovers supporting-information artifacts without hard-coding ACS,
Nature, RSC, Elsevier, or other publisher URL templates.

It uses two generic evidence sources:

1. explicit Crossref typed relationships such as `is-supplemented-by`;
2. supplementary links that are actually visible in publicly retrievable
   landing-page HTML.

No authentication, cookies, browser automation, paywall bypass, or guessed
publisher URL patterns are used.

## Confidence

High confidence:

- explicit `is-supplemented-by` relation,
- public anchor text such as `Supporting Information` or
  `Supplementary Information` pointing to a file-like URL.

Medium confidence:

- broader Crossref related-material/part relationships,
- public file-like links whose URL contains a supplementary-specific token.

Only **high-confidence direct files** are auto-downloaded by default.

A DOI related by `is-supplemented-by` is preserved as metadata but not
automatically dereferenced/downloaded as if it were a file.

## Artifact validation

Supported supplementary payload families include:

- PDF
- ZIP
- XLS/XLSX
- DOC/DOCX
- CSV/TXT
- PPTX

HTML responses are rejected. PDF and ZIP-family formats receive magic-byte
validation. Files are size-limited, SHA-256 hashed, and written atomically.

## Inputs

M3.1 binds to:

- the exact M1 catalog,
- the exact M2 selected set/report,
- the completed M3 acquisition report and access resolutions.

It refuses to start before M3 writes `acquisition_report.json`.

## Progress

```text
[M3.1 001/100] discover doi=...
[M3.1 001/100] status=direct_file_candidates candidates=2
[M3.1 001/100] supp=downloaded:1 failed:0 not_attempted:1
```

Per-work state makes reruns resumable.

## Outputs

```text
<output-dir>/
  supplementary_discoveries.jsonl
  supplementary_artifacts.jsonl
  supplementary_acquisition_report.json
  state/
  artifacts/
    <work>/
      supplementary_<stable-id>.pdf
      supplementary_<stable-id>.zip
      ...
```

## Run

Wait for M3 to finish, then:

```bash
python -m scripts.discover_supplementary_artifacts \
  --profile-id sers_au_ag_corpus_acquisition_v1 \
  --catalog data_acquisition/sers_au_ag_v1/m1/catalog.json \
  --selected-works data_acquisition/sers_au_ag_v1/m2/selected_works.jsonl \
  --selection-report data_acquisition/sers_au_ag_v1/m2/selection_report.json \
  --m3-dir data_acquisition/sers_au_ag_v1/m3 \
  --supplementary-policy configs/acquisition/supplementary_default_v1.yaml \
  --output-dir data_acquisition/sers_au_ag_v1/m3_1 \
  --acquisition-id sers_au_ag_supplementary_acquisition_v1
```

## Epistemic boundary

A downloaded SI file is still only a source artifact.

```text
supplementary artifact
  != positive KG evidence
```

M4 must materialize main/SI into `DocumentPackage`, after which the existing
strict extraction and provenance gates decide what can enter the canonical KG.
