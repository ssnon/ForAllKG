# alpha4c.5f.1 — Canonical Readiness Gate

## Why this phase exists

The first alpha4c.5f v3 reserve campaign was correctly marked consumed before
scientific execution. It later failed at Comparison because a canonical
Measurement (`meas_bpe_lod`) violated the already-frozen numeric/text XOR
invariant. The failure is therefore treated as an orchestration/readiness bug,
not as permission to weaken Comparison or to rerun the same reserve.

alpha4c.5f.1 adds a structural canonical-readiness gate that must complete
before any future reserve-consumption marker may be written.

## Non-negotiable historical rule

The failed alpha4c.5f v3 campaign remains consumed/seen.

- `evaluation/sers_alpha4c5f/reserve_v1/consumption_started.json` is preserved.
- `evaluation/sers_alpha4c5f/reserve_v1/CAMPAIGN_FAIL.json` is preserved.
- the historical campaign `work_data_sers` is preserved.
- alpha4c.5f.1 never reuses that campaign for acceptance.
- any new blind holdout requires a new untouched reserve and a new protocol
  epoch.

The consumed 14-paper v3 reserve may now be used only as a debugging/regression
fixture.

## Gate semantics

For every paper, before reserve consumption, the gate verifies:

1. the frozen Strict source is eligible for positive evidence;
2. rejected Strict sources fail closed;
3. the canonical graph exists;
4. the canonical domain profile is exactly the expected domain;
5. `measurement_merge_invariant_id` equals the frozen invariant;
6. every Measurement satisfies numeric/text XOR;
7. manual resolution decisions remain unchanged if canonical migration is
   required;
8. canonical bytes and resolution-decision bytes are SHA-locked.

A blind readiness lock intentionally contains no Measurement values and no
XOR-offending node details. Pre-consumption output may disclose only structural
status/counts and generic readiness reason labels.

## Allowed correction

No extraction LLM call is allowed.

Canonical refreeze is allowed only when the current canonical representation is
missing/stale or violates the frozen Measurement merge/XOR invariant:

- `canonical_missing`
- `measurement_merge_invariant_mismatch`
- `measurement_numeric_text_xor_violation`

A domain-profile mismatch is not auto-repaired. Rejected/invalid Strict source
quality is not auto-repaired. Manual resolution decisions must be byte/semantic
stable across rebuild.

The deterministic rebuild reuses the existing Strict run/chunk outputs through
`scripts.build_paper_graph`; it does not perform new extraction.

## Consumption guard

`dac_her.canonical_readiness.guarded_write_consumption_marker(...)` revalidates
the readiness lock immediately before marker creation. The marker is written
only when:

- the exact paper list/order matches;
- every locked canonical SHA is unchanged;
- domain identity is unchanged;
- Measurement merge invariant is unchanged;
- Measurement XOR count is still zero;
- resolution-decision presence/SHA is unchanged;
- the lock payload SHA is valid.

Future blind reserve orchestrators must call this guard instead of writing the
consumption marker directly.

## Consumed-v3 audit

Because the original v3 reserve is already consumed, detailed issue disclosure
is now allowed for debugging only:

```bash
python -m scripts.audit_sers_alpha4c5f_consumed_reserve
```

This compares the immutable campaign-frozen canonical copies against the
current source canonicals and reports all Measurement XOR violations. It does
not refreeze, rerun, or modify the historical campaign.

## Consumed-v3 seen regression

After reviewing the audit, deterministic canonical migration and a regression
through Comparison may be run explicitly:

```bash
python -m scripts.run_sers_alpha4c5f1_seen_regression \
  --confirm-refreeze-consumed-seen
```

This may rebuild current `data_sers` canonical graphs from already-frozen Strict
chunk outputs when eligible, while preserving manual decisions and Strict input
hashes. It then copies the ready canonicals into a new derived root:

`evaluation/sers_alpha4c5f1/consumed_v3_seen/regression_v1/work_data_sers`

and reruns only:

```text
projection
  -> corpus
  -> MeasurementResultIdentity
  -> MetricDefinition
  -> Comparison
```

It does not run Trend, CrossContext, Explorer, Maker, or 5e acceptance. A PASS
means only that the generic readiness bug is removed at the previously failing
Comparison boundary.

## Future blind reserve usage

For a new frozen reserve protocol:

```bash
python -m scripts.prepare_sers_canonical_readiness \
  --protocol <new-reserve-protocol.json> \
  --preflight
```

If migration is required, before reserve consumption:

```bash
python -m scripts.prepare_sers_canonical_readiness \
  --protocol <new-reserve-protocol.json> \
  --prepare \
  --confirm-canonical-refreeze \
  --output <new-evaluation-root>/canonical_readiness_lock.json
```

The new reserve orchestrator must then use
`guarded_write_consumption_marker(...)` against that exact lock immediately
before consumption.

## Scientific policy

alpha4c.5f.1 changes no Comparison, Trend, CrossContext, hypothesis, novelty, or
acceptance semantics. It introduces no count threshold and does not convert an
`unknown`, incompatible context, zero Trend yield, abstention, or zero
hypotheses into failure merely to improve apparent performance.
