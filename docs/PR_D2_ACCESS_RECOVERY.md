# PR D.2 — Access-state recovery

This patch makes knowledge-aware backfill recoverable when access capability or
cached OA state changes, without weakening scientific selection and without
bypassing publisher access controls.

## Recovery stage

Before each M3.2 round the coordinator runs `scripts.prepare_access_recovery`.
It:

- propagates candidate state from the prior M3.2 round so failures are not
  forgotten between rounds;
- preserves already-downloaded PDFs exactly as-is;
- re-resolves cached `unresolved` / `resolved_landing_only` candidates when
  `--retry-access-misses` is requested;
- re-resolves failed downloads when `--retry-failed-acquisition` is requested;
- automatically invalidates non-downloaded states after a *known* resolver
  capability or source-policy fingerprint change;
- never stores API keys or e-mail values, only secret-free capability booleans;
- writes `access_recovery_context.json`,
  `access_recovery_bad_locations.json`, and `access_recovery_report.json`.

Legacy states with no D.2 capability context remain resume-safe by default.
Use an explicit retry flag once to refresh those states and establish the
context baseline.

## Failure classes

Hard endpoint failures are learned per recovery generation:

- HTTP 401 / 403 / 404 / 410
- a response that is not a PDF (`not_pdf`)

Those concrete URLs are suppressed for the current recovery generation while
other Unpaywall/OpenAlex/catalog OA locations remain eligible. Timeout, 5xx,
empty responses, and other transient failures are not added to the hard
suppression ledger.

A resolver-capability or source-policy change starts a new recovery generation,
so old hard endpoints may be tried once again under the new capability set.

## CLI

Typical recovery after enabling Unpaywall/OpenAlex credentials:

```bash
python -m scripts.run_strict_bridge_backfill \
  ... \
  --retry-failed-acquisition \
  --retry-access-misses
```

The existing guarantees remain unchanged:

- scientific quality gate is not weakened;
- OA availability does not become scientific score;
- paywall bypass is not attempted;
- downloaded verified PDFs remain immutable/resume-safe.
