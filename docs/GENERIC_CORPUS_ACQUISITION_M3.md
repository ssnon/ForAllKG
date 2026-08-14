# Generic Corpus Acquisition M3 — OA Access Resolution + Source Artifacts

## Scope

M3 consumes the exact M1 catalog and M2 selected-work set.

```text
M1 catalog
 + M2 selection
      ↓
OA access resolution
      ↓
verified main PDF acquisition
      ↓
SourceArtifact manifest
```

M3 does **not**:

- bypass paywalls,
- authenticate to subscription services,
- scrape restricted publisher content,
- infer scientific results,
- promote downloaded files into positive KG evidence,
- materialize `normalized.md`,
- automatically guess publisher-specific supplementary-information URLs.

Supporting information discovery is intentionally deferred to M3.1 because SI
URL/metadata conventions vary substantially by publisher and repository.

## Resolver policy

The default policy tries:

1. Unpaywall DOI lookup, when a DOI and email are available.
2. The existing catalog `open_access_url` as a fallback candidate.

The Unpaywall response is reduced to auditable access locations including:

- direct PDF URL,
- landing URL,
- host type,
- version,
- license,
- best-location flag.

Only direct PDF candidates are automatically downloaded. Landing-only locations
are preserved but not scraped.

## PDF safety

A downloaded artifact is accepted only after:

- normal public HTTP(S) retrieval,
- size-limit enforcement,
- `%PDF-` magic validation,
- SHA-256 computation,
- atomic finalization.

HTML login/paywall pages therefore become `download_failed`, not PDFs.

## Resumability

Every selected work gets a per-work state file under:

`<output-dir>/state/`

Completed work is reused on subsequent runs. `--retry-failed` retries only
previous `download_failed` states.

## Outputs

```text
<output-dir>/
  access_resolutions.jsonl
  artifacts.jsonl
  acquisition_report.json
  state/
  artifacts/
    <stable-work-dir>/
      main.pdf
```

## Run

Use a real email address for Unpaywall:

```bash
export UNPAYWALL_EMAIL="you@example.com"
```

Then:

```bash
python -m scripts.acquire_corpus_sources \
  --profile-id sers_au_ag_corpus_acquisition_v1 \
  --catalog data_acquisition/sers_au_ag_v1/m1/catalog.json \
  --selected-works data_acquisition/sers_au_ag_v1/m2/selected_works.jsonl \
  --selection-report data_acquisition/sers_au_ag_v1/m2/selection_report.json \
  --source-policy configs/acquisition/source_access_default_v1.yaml \
  --output-dir data_acquisition/sers_au_ag_v1/m3 \
  --acquisition-id sers_au_ag_source_acquisition_v1
```

Progress is shown as:

```text
[M3 001/100] resolve doi=...
[M3 001/100] access=resolved_direct_pdf locations=2
[M3 001/100] artifact=downloaded bytes=...
```

## Upstream coverage

M3 reports the M1 provider-query success ratio but does not use it as an M3
acceptance threshold. A partial M1 search can still yield a valid selected set;
the coverage warning remains visible for later discovery expansion.

## Next stage

After auditing main-PDF acquisition:

- M3.1: generic supplementary-artifact discovery adapters
- M4: PDF/XML -> `DocumentPackage` / `normalized.md` materialization
