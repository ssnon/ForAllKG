# GraphAgentsDAC v2.8.0-alpha5.2 robustness hotfix

This hotfix addresses three failures observed in the full Dual-Atom HER E2E run.

## 1. Reviewer work-ID hallucination / mutation

Before alpha5.2, `ClaimPriorArtCompiler` aborted the entire external-novelty run if
the LLM returned any work ID that was not in the bounded ranked candidate set.

Alpha5.2 keeps the strict epistemic boundary but changes failure handling:

- unknown/unranked IDs are **never fuzzy matched**;
- unknown IDs are **never promoted to prior art**;
- they are dropped from the evidence set;
- the exact dropped IDs are recorded in `reviewer_unknown_work_ids`;
- `reviewer_unknown_work_id_dropped` is recorded in `reason_codes`;
- if the reviewer returned only unusable IDs and there is no valid positive
  prior-art match, the claim becomes `INSUFFICIENT_METADATA` rather than
  `NO_DIRECT_MATCH_FOUND`.

This is fail-closed for novelty inference without allowing a malformed ID to
kill all hypotheses in the portfolio.

## 2. Stale external-novelty report protection

`run_external_novelty` now removes an existing `<prefix>.report.json` before a
new run starts. Query-plan and prior-art artifacts may be written before the LLM
review finishes, so an old report must not remain visible after a failed
assessment.

The new E2E runner adds a stronger run-level guard: it refuses to use a nonempty
run directory unless `--overwrite-run` is explicitly supplied. With
`--overwrite-run`, the entire named run directory is recreated before stage 1.

## 3. Fail-fast E2E + semantic-stop fallback

New script: `scripts/run_dac_discovery_e2e.py`

Properties:

- every subprocess runs with `check=True`;
- any failed stage stops downstream execution immediately;
- expected artifacts are checked after each stage;
- alpha4 zero-hypothesis output stops before external novelty;
- alpha6 zero-hypothesis output stops before final semantic/feasibility stages;
- external report `source_portfolio_id` is checked against the current alpha4
  portfolio before alpha6 starts;
- the run records `e2e_runner.manifest.json`;
- when a hard semantic waypoint returns zero paths, grounding automatically
  falls back from `semantic_stop` to ordinary `top_n` while the stop concept
  remains in the natural-language question.

The fallback policy is therefore:

```text
semantic_stop(source, stop, target)
        |
        +-- paths > 0 -> use semantic_stop artifact
        |
        +-- paths = 0 -> top_n(source, target)
                            |
                            +-- paths > 0 -> continue
                            +-- paths = 0 -> fail before hypothesis generation
```

## Apply

Apply this patch after alpha5.1. It is also compatible with a local checkout
that already has the additive alpha6 files because the hotfix does not replace
alpha6 modules.

```bash
git apply --check /path/to/GraphAgentsDAC_robustness_v280a52.patch
git apply /path/to/GraphAgentsDAC_robustness_v280a52.patch
```

Focused tests:

```bash
python -m pytest -q \
  tests/test_prior_art_reviewer_id_guard.py \
  tests/test_dac_discovery_e2e_runner.py
```

## Recommended rerun

Use new run directories rather than the old `_001` directories:

```bash
python -m scripts.run_dac_discovery_e2e \
  --corpus-id dac_her_expanded_v1 \
  --run-dir runs/e2e/dac_her_metal_pair_coordination_002 \
  --source "metal-pair identity and coordination environment" \
  --stop "electronic structure" \
  --target "hydrogen evolution activity" \
  --question "How do metal-pair identity and local coordination environment regulate electronic structure and hydrogen adsorption to determine HER activity in dual-atom catalysts?" \
  --title "Dual-Atom HER — Metal Pair & Coordination Discovery"
```

and

```bash
python -m scripts.run_dac_discovery_e2e \
  --corpus-id dac_her_expanded_v1 \
  --run-dir runs/e2e/dac_her_ndoped_coordination_002 \
  --source "nitrogen coordination environment" \
  --stop "charge transfer" \
  --target "hydrogen evolution activity" \
  --question "How does the nitrogen coordination environment in N-doped graphene regulate charge redistribution and hydrogen adsorption, and thereby control the stability and HER activity of dual-atom sites?" \
  --title "N-doped Graphene DAC HER — Coordination Discovery"
```

The runner defaults to the OpenRouter environment variables used elsewhere in
the project and requires alpha6 (`scripts.run_novelty_refinement`) to already be
present for the full 13-stage pipeline.
