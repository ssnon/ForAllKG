# alpha4b.4 — Frozen SERS Holdout Validation

This phase validates the frozen SERS evidence substrate on `Kiwook_SERS_2`,
`Kiwook_SERS_6`, and `Kiwook_SERS_10`.

## Scientific rule

SERS protocol heterogeneity is first-class data, not noise to be tuned away.
The holdout therefore has **no target count** for:

- numeric-rankable pairs;
- same-protocol pairs;
- unknown MethodContext dimensions;
- known MetricDefinition contexts;
- different-protocol or different-definition pairs.

A holdout may legitimately produce more unknowns, new observable distributions,
or zero rankable pairs.

## Scope

This is an **evidence-substrate holdout**. It deliberately adopts the already
existing strict-extraction/canonical graph for each held-out paper and hash-locks
it at campaign creation. It does not regenerate strict extraction after the
holdout starts.

From that frozen substrate it validates:

1. canonical graph audit;
2. frozen SERS Bridge materialization (or explicit adoption of an existing
   confirmed/candidate Bridge pair);
3. evidence/mechanism/exploratory projections;
4. non-destructive corpus construction;
5. ReproducibilityEvidence;
6. MetricDefinitionContext;
7. Comparison + MethodContext + ProtocolAssessment + quality-aware numeric gate.

## Failure semantics

Only structural/provenance/frozen-contract invariants fail the holdout.
Examples:

- structural gate false;
- destructive cross-paper merge;
- global/shared concentration consumed as measurement-local context;
- missing context treated as quarantine;
- semantic ID or protected implementation drift;
- held-out canonical/strict-run input changes after campaign creation;
- manual entity-resolution decisions on held-out inputs.

The following are **not** failures:

- `different_protocol`;
- protocol `unknown`;
- MetricDefinition `unknown`;
- `different_definition`;
- zero numeric-ranking pairs;
- new/unsupported observable distributions;
- possible reproducibility duplicate candidates.

## Post-holdout change rule

If the holdout reveals a true domain-independent schema/provenance/invariant bug,
do not patch the running campaign. Fix the bug, rerun calibration from the
frozen calibration papers, then start a **new campaign ID** and rerun all three
holdout papers from the beginning.

Do not add a new protocol, observable distribution, or exception merely to make
`SERS_2/6/10` look cleaner.

## Commands

First install the patch, then dry-run the campaign:

```bash
python -m scripts.run_sers_alpha4b4_holdout \
  --campaign-id sers_alpha4b4_holdout_v1 \
  --dry-run
```

For a real run with fresh frozen Bridge materialization:

```bash
python -m scripts.run_sers_alpha4b4_holdout \
  --campaign-id sers_alpha4b4_holdout_v1
```

This requires `OPENROUTER_API_KEY` and either `OPENROUTER_BRIDGE_MODEL` or
`OPENROUTER_EXTRACT_MODEL`.

If a confirmed/candidate Bridge pair was already generated under the same frozen
setup and you explicitly want to adopt it:

```bash
python -m scripts.run_sers_alpha4b4_holdout \
  --campaign-id sers_alpha4b4_holdout_v1 \
  --adopt-existing-bridge
```

The campaign writes a persistent manifest and final report under:

`evaluation/sers_alpha4b4/<campaign-id>/`

After a real run, send back `holdout_report.json`, `manifest.json`, and the
holdout comparison/reproducibility/metric-definition summaries for review.
