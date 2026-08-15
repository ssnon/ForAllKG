# alpha4c.5f — Frozen Reserve End-to-End Execution

## Role

alpha4c.5f is an orchestration layer, not a new scientific semantics layer.

It binds the already frozen:

- alpha4c v3 `reserved_future_v3` 14-paper split,
- alpha4c.5e evaluation protocol,
- alpha4c.5e reserve manifest,
- frozen Trend/Precision/CrossContext semantics,
- alpha4c.5a/5b/5c contracts,
- alpha4c.5d.1 Maker,
- and a pre-reserve frozen Graph Explorer policy

into one single-shot reserve campaign.

## Start boundary

The campaign starts from the existing strict canonical scientific graphs in
`data_sers/extracted/<paper>/<paper>.graphml`.

No new extraction LLM call is allowed by alpha4c.5f. At execution time, after
the reserve consumption marker is written, the exact canonical graph bytes are
copied into an isolated campaign data root and hash-locked.

The campaign does not overwrite the original `data_sers` corpus/projection
outputs.

## Exact reserve

The reserve is the existing untouched `reserved_future_v3` split:

- Kiwook_SERS_36
- Kiwook_SERS_32
- Kiwook_SERS_7
- Kiwook_SERS_20
- Kiwook_SERS_3
- Kiwook_SERS_15
- Kiwook_SERS_24
- Kiwook_SERS_29
- Kiwook_SERS_33
- Kiwook_SERS_27
- Kiwook_SERS_26
- Kiwook_SERS_31
- Kiwook_SERS_14
- Kiwook_SERS_9

The 5e manifest-sorted order is used downstream so that the Trend corpus paper
list exactly equals the reserve manifest paper list. No paper override exists.

## Scientific execution

```text
canonical source lock/copy
    -> evidence projections
    -> evidence corpus
    -> MeasurementResultIdentity
    -> MetricDefinition
    -> Comparison
    -> TrendEvidence
    -> TrendPrecision
    -> CrossContext profiles/assessments when local Trend results exist
    -> navigation graph + node index
    -> frozen evidence traversal
    -> GraphExplorerPacket
    -> Graph Explorer LLM
    -> HypothesisContext
    -> alpha4c.5a grounding
    -> alpha4c.5b Trend-aware input
    -> alpha4c.5d.1 direction-aware Maker LLM
    -> alpha4c.5e independent reserve evaluation
```

If TrendPrecision yields zero local results, CrossContext is skipped. Zero
Trend yield is not itself a campaign failure.

## Explorer policy

The Explorer is necessary because alpha4c.5b requires a validated
`HypothesisContext`; 5f does not fabricate a synthetic reserve context.

The Explorer policy is frozen before reserve execution:

- projection/traversal mode: `evidence`
- Bridge: not required
- traversal algorithm: `top_n`
- source query: `nanostructure design`
- target query: `SERS performance`
- research question:
  `How do Au/Ag nanostructure design variables and local experimental context relate to SERS performance?`
- objective: `map_evidence`
- max depth: 8
- top-k: 8
- endpoint map pool: 20
- reachable endpoint pair pool: 12
- Explorer temperature: 0
- Explorer max repair: 1
- Explorer model/provider settings mirror the already frozen 5e Maker settings.

The question/query policy is domain-generic for the registered SERS reserve and
is frozen before reserve content is semantically inspected.

## Reserve consumption

Installation and `--preflight` do not consume the reserve.

Real execution requires:

```bash
python -m scripts.run_sers_alpha4c5f_reserve \
  --execute-reserve \
  --confirm-consume-reserve
```

Before parsing/copying a canonical reserve graph, building a projection, or
calling either LLM, the runner writes:

`evaluation/sers_alpha4c5f/reserve_v1/consumption_started.json`

From that point onward the reserve is consumed even if a later stage fails.

The runner refuses a second execution if the marker exists. Scientific outputs
are not automatically rolled back.

A semantic patch after consumed failure requires a new protocol epoch and an
untouched reserve. A poor Trend distribution, zero hypotheses, abstention, or
insufficient evidence is not a reason to tune the frozen rules.

## Acceptance

alpha4c.5f has no independent success-count target. Final scientific
acceptance is delegated to the already frozen alpha4c.5e evaluation protocol.

`count_thresholds_used_for_acceptance = false`.



## Upstream freeze boundary

The historical alpha4c.4d.2 run config contains its frozen semantic IDs and
the config file itself is Git-blob locked, but it does not contain per-file
`frozen_implementation_blobs`. alpha4c.5f therefore does not fabricate such a
historical guarantee.

Instead, pre-reserve freeze performs two separate checks:

1. verify the exact historical alpha4c.4d.2 config Git blob and its recorded
   frozen semantic IDs; and
2. hash every current implementation component that 5f will execute and bind
   those SHA256 values into the new alpha4c.5f protocol.

Real reserve execution re-verifies those 5f component hashes before the
consumption marker is written. Thus code drift after the 5f freeze fails
closed without claiming file-level provenance that alpha4c.4d.2 never stored.
