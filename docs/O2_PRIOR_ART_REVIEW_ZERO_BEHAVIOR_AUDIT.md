# O2 — Prior-art review zero-behavior-change audit

## Why this audit exists

Billing-aware E2E telemetry showed that `external_novelty:prior_art_review`
dominates query cost. Raw request fingerprints also suggested many exact repeats.

That does **not** by itself authorize caching or skipping calls.

The scientific safety question is narrower:

> When the model receives the exact same claim + ranked prior-art prompt again,
> is the repeated call providing new evidence or a deliberately aggregated
> judgment?

The current pipeline does not vote, average, ensemble, or estimate uncertainty
across repeated prior-art review outputs. Nevertheless, model outputs can vary
even at temperature 0, so reuse must not be introduced before measuring that
variation and tracing where repeats arise.

## What this patch changes

Only observability.

It does not change:
- retrieval;
- ranking;
- ranked-work limits;
- prompts;
- model;
- schema;
- temperature;
- retries;
- compilation;
- external novelty status policy;
- alpha6 branching;
- any LLM call count.

Every prior-art review LLM call still executes.

When the environment variable below is set, one append-only JSONL audit row is
written **after** each successful review call:

```bash
export GRAPHAGENTS_PRIOR_ART_REVIEW_AUDIT_PATH=/absolute/path/review_audit.jsonl
```

Each row includes:
- exact semantic request fingerprint over model/mode/temperature/system/user/schema;
- response fingerprint and response payload;
- claim/hypothesis IDs;
- ranked candidate work IDs;
- assessment context;
- PR-O1 provider cost/token fields when available.

Audit write failures are warning-only so observability cannot change a
scientific pipeline result.

## Assessment labels

The patch marks:
- `alpha5_initial`
- `alpha6_targeted_reassessment`
- `alpha6_fresh_final`

For alpha6 targeted reassessment it also records the focal hypothesis and gap.
This lets the report distinguish focal re-evaluation from whole-portfolio
non-focal repetition.

## Summary

```bash
python -m scripts.summarize_prior_art_review_audit \
  data_dac/telemetry/<RUN>.prior_art_review_audit.jsonl \
  --output data_dac/telemetry/<RUN>.prior_art_review_audit.summary.json
```

Key fields:
- `duplicate_calls`
- `duplicate_cost_fraction_of_observed_review_cost`
- `response_stable_duplicate_groups`
- `response_divergent_duplicate_groups`
- `targeted_reassessment_non_focal_duplicate_calls`
- `duplicate_transitions`

## Safety rule

No cache/reuse optimization is authorized by an audit PASS.

If exact-repeat responses diverge, inspect the paired response payloads and the
downstream compiled/status effects before any reuse experiment.

If exact repeats are stable and the majority of repeats are non-focal calls
whose results are not consumed by the current alpha6 branch, that establishes a
strong candidate for a separate, explicitly tested reuse optimization.
