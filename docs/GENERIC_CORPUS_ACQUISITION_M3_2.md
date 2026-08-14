# Generic Corpus Acquisition M3.2 — Acquisition-Aware Backfill

## Purpose

M3.2 addresses the low `main=0` rate without weakening scientific selection.

It does **not** mean:

```text
OA paper -> scientifically preferred
```

It means:

```text
M2 eligible
  -> M2.1 quality=pass
  -> attempt legal/public main acquisition
  -> only verified downloaded main PDF can occupy final usable corpus slot
```

OA metadata is not added to scientific score. At most, an OA hint breaks a tie
between candidates with the exact same scientific `total_score`.

## Inputs

M3.2 uses:

- M1 catalog,
- original M2 candidate assessments,
- M2.1 quality assessments and report,
- M2.1 selected set/report,
- completed robust M3 output.

It reuses every verified M3 `downloaded` main PDF.

A work from the original M2.1 selected 100 whose main is `download_failed` or
`not_attempted` is treated as an exhausted slot. M3.2 then searches the
remaining `quality=pass` pool.

## Quota algorithm

1. Count usable M3 downloads by their original `primary_quota_axis`.
2. Compute acquisition-adjusted quota deficits.
3. Pick the currently most constrained axis: fewest remaining quality-pass
   candidates per missing slot.
4. Try the highest scientific-score candidate for that axis.
5. On acquisition failure, continue to the next candidate.
6. On successful verified PDF acquisition, occupy that quota slot.
7. After quota attempts, optionally use remaining quality-pass candidates for
   global fill to maximize usable corpus size. These do **not** pretend to
   satisfy an unfilled axis quota.
8. Stop at target size, configured attempt limit, or pool exhaustion.

## Fail-closed behavior

If the quality-pass pool cannot provide 100 downloadable main PDFs, M3.2 reports
the smaller corpus. It does not automatically add:

- manual-review works,
- quality-excluded works,
- retracted/review papers,
- subscription/paywalled content,
- unverifiable HTML masquerading as PDF.

The appropriate next action is then discovery expansion (additional M1
queries/providers), not quality weakening.

## Runtime state

New backfill acquisition attempts are stored under:

```text
m3_2/state/
m3_2/artifacts/
```

Reruns reuse candidate state. `--retry-failed` retries only M3.2
`download_failed` states.

Existing successful M3 artifacts remain referenced at their original paths;
they are not duplicated.

## Outputs

```text
m3_2/
  backfill_report.json
  backfill_attempts.jsonl
  backfill_selected_works.jsonl

  # Existing downstream-compatible interfaces:
  selected_works.jsonl
  selection_report.json
  access_resolutions.jsonl
  artifacts.jsonl
  acquisition_report.json

  state/
  artifacts/
```

`backfill_selected_works.jsonl` explicitly marks every work as
`downloaded_main`.

The compatibility `selected_works.jsonl` retains the old
`SelectedCorpusWork.acquisition_status=selected_metadata_only` literal only
because M3.1/M4 currently parse that contract. The M3.2 report/artifact files
are the authoritative acquisition state.

## Recommended run

```bash
python -m scripts.backfill_acquisition_ready_corpus \
  --profile configs/acquisition/sers_au_ag_v1.yaml \
  --backfill-policy configs/acquisition/backfill_default_v1.yaml \
  --source-policy configs/acquisition/source_access_default_v1.yaml \
  --catalog data_acquisition/sers_au_ag_v1/m1/catalog.json \
  --m2-assessments data_acquisition/sers_au_ag_v1/m2/assessments.jsonl \
  --quality-assessments data_acquisition/sers_au_ag_v1/m2_1/quality_assessments.jsonl \
  --quality-gate-report data_acquisition/sers_au_ag_v1/m2_1/quality_gate_report.json \
  --m2-1-selected-works data_acquisition/sers_au_ag_v1/m2_1/selected_works.jsonl \
  --m2-1-selection-report data_acquisition/sers_au_ag_v1/m2_1/selection_report.json \
  --m3-dir data_acquisition/sers_au_ag_v1/m3 \
  --output-dir data_acquisition/sers_au_ag_v1/m3_2 \
  --backfill-id sers_au_ag_acquisition_backfill_v1
```

## Downstream

After M3.2:

```text
M3.2 selected_works/report + M3.2 acquisition dir
  -> fresh M3.1 supplementary run
  -> fresh M4 materialization run
```

Using fresh downstream output directories is recommended to keep the original
100-paper experiment auditable.
