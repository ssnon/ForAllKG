# Generic Corpus Acquisition M3.3a — OpenAlex access adapter

This patch connects the repository's existing `OpenAlexProvider` to the generic
M3 access-resolution lane without duplicating its HTTP transport, retry, or
backoff implementation.

## Scope

- Add identifier-targeted `OpenAlexProvider.get_work()` using the same request
  machinery as discovery search.
- Collect `best_oa_location`, `primary_location`, and every explicitly OA row in
  OpenAlex `locations`.
- Convert those rows into the existing `AccessLocation` contract.
- Preserve license, version, OpenAlex source id/name/type, and role provenance.
- Keep existing Unpaywall priority; place OpenAlex direct-PDF candidates before
  the weaker catalog OA URL fallback.
- Reuse the existing multi-location downloader and `%PDF-` validation.
- Add `--retry-access-misses` so older `unresolved` / `landing_only` states can
  be re-resolved after installing a new resolver without redownloading already
  downloaded artifacts.
- Add OpenAlex-specific report counters so incremental coverage can be measured.

## Safety invariants

The adapter only uses public HTTP(S) OA locations explicitly marked `is_oa=true`
by OpenAlex. It does not authenticate to publishers, automate browser/login
flows, bypass paywalls, or promote catalog/access metadata to positive scientific
KG evidence.

## Environment

The default source policy enables OpenAlex and requires `OPENALEX_API_KEY` for
that lane. If it is absent, only the OpenAlex resolver attempt is marked skipped;
Unpaywall and catalog fallback continue normally. `OPENALEX_MAILTO` is optional
and falls back to the configured generic contact email environment variable.

## Recommended replay after installation

Run the existing `scripts.acquire_corpus_sources` command with both:

```text
--retry-failed --retry-access-misses
```

`--retry-failed` re-evaluates prior failed direct-PDF downloads. The new
`--retry-access-misses` re-evaluates prior unresolved/landing-only states whose
artifact was `not_attempted`. Existing downloaded artifacts remain resumable and
are not deliberately redownloaded.
