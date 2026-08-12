# LLM telemetry

This repository can emit append-only JSONL usage events for LLM calls without
changing scientific prompts, validation policy, recovery budgets, or gate
ordering.

## Enable a ledger

Set one environment variable before running an existing pipeline:

```bash
export GRAPHAGENTS_LLM_TELEMETRY_PATH=data_broad/telemetry/broad_pilot_10.jsonl
```

Backends that expose an explicit `telemetry_path` argument may use that instead.
If neither is set, telemetry persistence is disabled and the scientific call
path continues normally.

Telemetry I/O is fail-soft: an unwritable ledger emits a warning but does not
change extraction, recovery, hypothesis generation, or validation decisions.

## Event semantics

`provider_input_tokens`, `provider_output_tokens`, and `provider_total_tokens`
come from the returned provider completion when available. They are kept
separate from local component estimates.

`estimated_components` is diagnostic. It records token estimates and SHA-256
fingerprints for serialized surfaces such as `system`, `schema`, `source`,
`vocabulary`, `current_draft`, `issues`, and `premises`. Raw prompt/source/model
text is not written to the ledger.

Only top-level non-overlapping surfaces have
`counted_in_estimated_sum=true`. Nested components such as `source` are views of
an already-counted user prompt and therefore are not added twice.

`provider_usage_scope` distinguishes direct provider calls from Instructor
calls where usage is taken from the returned raw completion. The latter must
not be interpreted as a guaranteed aggregate of every provider request that an
Instructor parse-retry may have made internally.

## Summarize a ledger

```bash
python -m scripts.summarize_llm_telemetry \
  data_broad/telemetry/broad_pilot_10.jsonl \
  --output data_broad/telemetry/broad_pilot_10.summary.json
```

The summary reports stage/pipeline token totals, component estimates,
source-scoped input overhead, provider-usage scopes, outcomes, token-estimate
gaps, and repeated serialization fingerprints.

The source overhead ratio uses only events that actually contain a `source`
component, so hypothesis/critic calls do not inflate an extraction-specific
ratio.

## Non-goals of this change

This telemetry layer does **not** shorten prompts or schemas, change recovery
budgets, reorder validators, slice eligible premises, stop retries based on
ROI/stagnation, or change any scientific acceptance policy. Those decisions
should be made only after observing real usage distributions.

## Telemetry v1.1: call outcome vs artifact outcome

Telemetry v1.1 separates provider/parse success from downstream scientific
artifact disposition.

Call records use `record_type=call` and `call_outcome`, for example `success`,
`validation_error`, or `error`. A successful call only means the provider call
returned and the immediate structured result parsed; it does not mean the paper
or hypothesis was accepted.

After extraction quality is known, the pipeline appends
`record_type=artifact_resolution` rows keyed by `call_id`. These carry:

- `artifact_outcome`: `accepted`, `accepted_after_repair`, `accepted_partial`,
  `quarantined`, `failed`, `rejected`, or `unknown`.
- `terminal_contribution`: `terminal`, `non_terminal`, `discarded`, or `unknown`.
- final materialization status and resolution reason.

The ledger remains append-only; summaries join the latest resolution for each
call ID. Existing v1 call rows remain readable.

The v1.1 summary also normalizes response-model labels such as
`KnowledgeGraphPatch` to the pipeline stage `semantic_patch`, while retaining
`response_model` separately. Source overhead is reported per stage, including a
`graph_generation_source_overhead` section so generation-only protocol overhead
is not diluted by repair calls.

### Backfill existing extraction telemetry

A telemetry file captured under v1 can be resolved without rerunning the LLM if
the corresponding extraction runs still contain `attempt_usages` with nested
telemetry call IDs:

```bash
python -m scripts.backfill_extraction_telemetry_resolutions \
  --data-root data_broad \
  --telemetry-path data_broad/telemetry/broad_pilot_10_fresh.jsonl \
  --paper-ids <paper-id-1> <paper-id-2> ...
```

The command only appends resolution rows for call IDs already present in the
specified telemetry file.
