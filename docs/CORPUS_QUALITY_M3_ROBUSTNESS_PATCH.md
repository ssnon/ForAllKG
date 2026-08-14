# Corpus Quality + Robust OA Acquisition Patch

This patch addresses two failures observed in the first 100-paper SERS run.

## M2.1 quality gate

M2.1 evaluates the **entire M2 candidate pool**, not only the already-selected
100 works. It then re-runs the existing quota-aware selector so a rejected work
can be replaced by another quality-passing candidate from the same discovery
substrate.

The generic engine supports policy-driven:

- hard title exclusions,
- manual-review title signals,
- publication-type exclusions/review,
- preprint/manual-review DOI prefixes,
- primary-topic lexical grounding,
- title-context grounding.

The first SERS policy automatically excludes retracted/review records and keeps
review-like/preprint/weak-title-grounding records out of auto-selection while
preserving them for manual review.

No scientific effect direction is inferred.

## M3.0.1 multi-location fallback

The old downloader could resolve several OA locations but only attempt the one
selected location. M3.0.1 orders all direct public PDF candidates and tries them
sequentially.

For each HTTP attempt it records:

- location ID,
- URL and host,
- success/failure,
- elapsed time,
- resolved URL,
- content type,
- bytes,
- stable failure code.

A failure such as HTML returned from the first candidate therefore does not
prevent a second OA location from succeeding.

The final M3 report includes:

- total download-location attempts,
- multi-location recovery count,
- failure-reason histogram,
- per-host attempts/successes/failures.

No paywall/login/browser automation is added.

## M3.1 upstream binding

M3.1 per-work state is now bound to a SHA-256 of the corresponding M3
`AccessResolution`.

If robust M3 changes which OA/landing locations are available, M3.1 automatically
re-discovers supplementary material for that work instead of silently reusing
stale supplementary state.

## Recommended rerun

1. M2.1 from the existing M1/M2 outputs.
2. Robust M3 using the M2.1 selected set/report. Reuse the existing M3 output
   directory and pass `--retry-failed` to retain verified downloads while
   retrying prior failures.
3. M3.1 using the M2.1 selected set/report and the refreshed M3 outputs.
   Access-hash binding refreshes only stale work states.
4. M4 using the final quality-gated set.
