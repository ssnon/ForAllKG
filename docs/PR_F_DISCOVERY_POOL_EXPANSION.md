# PR F — Append-only discovery pool expansion

## Goal

Expand a scientifically fixed acquisition profile after the current quality-pass
reserve is exhausted by public-OA acquisition, without invalidating already
verified downloaded PDFs or existing downstream Strict/Bridge artifacts.

## Pipeline

```text
frozen base M1 catalog
        +
100/query × Semantic Scholar + Crossref + OpenAlex
        ↓
append-only catalog merge
(base CatalogWork rows remain byte-for-byte unchanged)
        ↓
expanded M1 catalog / new catalog_id
        ↓
M2 deterministic rescore
        ↓
M2.1 deterministic quality gate
        ↓
M3 lineage rebase
(copy + SHA/PDF-magic verify old downloaded PDFs; NO network)
        ↓
expanded M3-compatible starting snapshot
        ↓
existing PR D/D.2 knowledge-aware backfill
(new quality-pass candidates are the only new acquisition work)
```

## Important invariants

- OpenAlex discovery is metadata-only.
- Only explicit OpenAlex `pdf_url` values become `CatalogWork.open_access_url`.
  Landing pages and generic `oa_url` values are not promoted to direct-PDF hints.
- Every base `CatalogWork` row is frozen byte-for-byte and remains the prefix of
  the expanded packet.
- The expanded packet receives a new `catalog_id`; lineage is recorded in
  `expansion_report.json` rather than lying about catalog identity.
- M2/M2.1 are rerun because they are deterministic and cheap.
- M3 rebase performs zero resolver/download requests. Existing PDFs must pass
  PDF magic and SHA verification before being copied into the new snapshot.
- A retained old paper must still be M2 eligible and M2.1 `pass` in the expanded
  catalog; otherwise it is not silently carried forward.
- No positive-evidence promotion and no paywall bypass occur in this PR.

## Recommended SERS run

```bash
python -m scripts.run_discovery_pool_expansion \
  --profile configs/acquisition/sers_au_ag_v1.yaml \
  --quality-policy configs/acquisition/sers_au_ag_quality_v1.yaml \
  --base-catalog data_acquisition/sers_au_ag_v1/m1/catalog.json \
  --source-m3-dir data_acquisition/sers_au_ag_v1/m3_2 \
  --output-root data_acquisition/sers_au_ag_v1/discovery_expansion/openalex_depth100_v1 \
  --expansion-id sers_au_ag_openalex_depth100_v1 \
  --providers semantic_scholar,crossref,openalex \
  --results-per-query 100
```

Expected outputs:

```text
.../openalex_depth100_v1/
├── run.json
├── m1/
│   ├── catalog.json
│   ├── incoming_catalog.json
│   ├── candidates.jsonl
│   └── expansion_report.json
├── m2/
│   ├── assessments.jsonl
│   ├── selected_works.jsonl
│   └── selection_report.json
├── m2_1/
│   ├── quality_assessments.jsonl
│   ├── selected_works.jsonl
│   └── quality_gate_report.json
└── m3_rebased/
    ├── selected_works.jsonl
    ├── selection_report.json
    ├── access_resolutions.jsonl
    ├── artifacts.jsonl
    ├── acquisition_report.json
    ├── rebase_report.json
    ├── state/
    └── artifacts/
```

After this succeeds, use the expanded M1/M2/M2.1 files and `m3_rebased` as the
inputs to `scripts.run_strict_bridge_backfill`. Keep PR D.2 recovery options on;
only new expanded-catalog reserve candidates should require new acquisition.
