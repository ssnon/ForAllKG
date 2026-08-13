# Broad compact schema experiment

This experiment changes only the structured-output schema used by the initial
`catalysis_mechanism` graph-generation call.

Unchanged surfaces:

- Broad system prompt
- vocabulary context
- model and temperature
- deterministic validators
- semantic patch schema and recovery budget
- graph projection and corpus audit

The compact schema keeps the same top-level graph shape but replaces the two
Broad-disabled measurement collections with trivial empty-array fields. The
parsed result is immediately expanded back to `KnowledgeGraphDraft` before any
domain gate or deterministic validation runs.

## Inspect the schema before spending API tokens

```bash
python -m scripts.inspect_broad_compact_schema
```

The output reports estimated full/compact schema tokens using the same local
telemetry estimator and confirms whether the heavy measurement definitions are
still serialized.

## Controlled 3-paper candidate run

Use papers that cover a simple success, a larger graph, and a prior repair case.

```bash
export GRAPHAGENTS_LLM_TELEMETRY_PATH="data_broad/telemetry/broad_compact_3.jsonl"
rm -f "$GRAPHAGENTS_LLM_TELEMETRY_PATH"

python -m scripts.run_broad_corpus_pilot \
  --config /path/to/papers.yaml \
  --corpus-id broad_compact_3 \
  --paper-id broad_12549367c327c283ccf7 \
  --paper-id broad_1e86b38c448a764b68a3 \
  --paper-id broad_241a667a6ef693f01c95 \
  --force-extract \
  --retry-rejected \
  --broad-compact-schema
```

Then summarize:

```bash
python -m scripts.summarize_llm_telemetry \
  data_broad/telemetry/broad_compact_3.jsonl \
  --output data_broad/telemetry/broad_compact_3.summary.json
```

Compare against the full-schema baseline on:

- graph-generation input tokens/call
- schema estimated tokens
- graph-generation source overhead ratio
- usable/rejected/repair rates
- terminal validation issues
- graph-semantics warnings
- mechanism-bearing fraction
- direct mechanism edges per usable paper

Use `--force-extract` for A/B runs because the Broad pilot's paper-level resume
logic can otherwise reuse an older successful extraction before a new run
fingerprint is computed.
