# Generic Corpus Acquisition M4.5 — Strict pre-extraction gate

M4.5 is a deterministic, no-LLM boundary between source materialization and
strict positive-evidence extraction.

It performs two checks on every M4 extraction-ready main document:

1. **Bibliographic identity** — verify that the materialized document belongs to
   the catalog work using an exact DOI match when available, otherwise a strong
   front-matter title match.
2. **Full-text bridge suitability** — verify that at least one evidence-grounded
   selected acquisition axis is present in the materialized body and occurs in
   local context with a deterministic relation/evidence signal.

M4.5 never infers scientific effect direction, causality, or a positive KG
premise. It only decides whether automatic strict extraction is allowed.

## Why this is separate from M4

M4 remains a pure source-materialization stage. Its generated config is an
artifact of what was successfully materialized, not an authorization to promote
that paper into the strict evidence lane.

M4.5 consumes the M4 config and emits a **filtered** v3 papers config plus a new
extraction plan. The M4.5 config/plan are the authoritative inputs for the next
strict extraction runner.

## SERS Au/Ag example

Run after M4:

```bash
python -m scripts.apply_pre_extraction_gate \
  --acquisition-profile configs/acquisition/sers_au_ag_v1.yaml \
  --gate-policy configs/acquisition/sers_au_ag_pre_extraction_gate_v1.yaml \
  --catalog data_acquisition/sers_au_ag_v1/m1/catalog.json \
  --selected-works data_acquisition/sers_au_ag_v1/m3_2/selected_works.jsonl \
  --m4-dir data_acquisition/sers_au_ag_v1/m4 \
  --input-config configs/generated/sers_au_ag_api_v1.yaml \
  --output-dir data_acquisition/sers_au_ag_v1/m4_5 \
  --output-config configs/generated/sers_au_ag_strict_ready_v1.yaml \
  --domain-profile-id sers_au_ag \
  --data-root data_sers
```

Use the selected-works file that actually fed the M4 run. If M3.2 backfill was
used, that is normally the M3.2 compatibility `selected_works.jsonl`.

## Outputs

```text
m4_5/
  pre_extraction_gate_assessments.jsonl
  pre_extraction_gate_summary.jsonl
  pre_extraction_gate_report.json
  extraction_plan.jsonl

configs/generated/
  sers_au_ag_strict_ready_v1.yaml
```

`pre_extraction_gate_assessments.jsonl` records the complete deterministic
basis for each decision, including observed DOI candidates, title similarity,
axis-term hits, local relation-context counts, and the final automatic-extraction
boolean.

## Identity states

- `verified` — exact DOI or strong title match; eligible for automatic extraction.
- `weak_match` — plausible title match but below the strict threshold; manual review.
- `mismatch` — conflicting front-matter DOI plus weak title match; blocked.
- `unverifiable` — insufficient bibliographic evidence; manual review / blocked.

The default SERS policy only auto-allows `verified`.

## Suitability states

- `suitable` — a selected evidence axis is found in body text with local
  relation/evidence context.
- `manual_review` — the axis is present but relation context is insufficient, or
  the materialized text is abnormally short.
- `unsuitable` — selected axis terms are absent from the materialized body.
- `unavailable` — no readable materialized main Markdown is available.

The default SERS policy only auto-allows `suitable`.

Title-like front-matter blocks are excluded from suitability evidence so that a
paper title such as "Nanogap Dependence of ... Enhancement" cannot pass the gate
without supporting body/abstract context.

## Epistemic boundary

M4.5 is deliberately conservative. A blocked paper is not declared false or
scientifically irrelevant. It is merely not authorized for unattended strict
positive-evidence extraction. Manual review or later backfill can recover it.
