# Frozen domain-gate recovery replay harness

This harness exists to protect scientific quality while evaluating PR6.3.

## Non-negotiable policy

Token savings never justify any observed quality degradation.

The replay summary never emits an automatic adoption verdict. Even an automated
zero-loss PASS still requires manual scientific semantic comparison of paired
canonical outputs before changing a production default.

## Why replay is needed

Production domain-gate recovery is conditional. Initial LLM generation changes
whether recovery is reached and changes the rejected draft inserted into the
recovery prompt. Whole-pipeline A/B therefore confounds initial stochasticity,
recovery-entry probability, rejected-draft content, and response schema.

The fixture freezes the exact system prompt, user prompt, rejected draft,
domain error, source/context metadata, response-schema fingerprints,
model/provider, and completion-token budget.

Replay changes only the response schema:
- full: KnowledgeGraphDraft
- compact: adapter-owned compact recovery schema

## Capture

Whenever strict extraction reaches targeted domain-gate recovery it writes:

<attempt>/replay_fixtures/<safe_chunk_id>__domain_gate_recovery_0.fixture.json

Capture is control-plane I/O only. The exact same frozen prompt string is sent
to the production recovery call.

## Replay

python -m scripts.replay_domain_gate_recovery \
  --fixture <fixture1.json> \
  --fixture <fixture2.json> \
  --replicates 4 \
  --output-dir data_broad/replay/pr63_frozen_v1

Replicate order is counterbalanced:
r1 full -> compact
r2 compact -> full
r3 full -> compact
r4 compact -> full

Outputs:
- telemetry.jsonl
- results.jsonl
- summary.json
- canonical_outputs/...
- raw_invalid/...

## Zero-loss gate

Compact is blocked from adoption if it shows any observed decrease in LLM
structured-output success, domain-gate pass, strict-validation pass,
finalization success, connected mechanism output, mechanism-claim count, or
mechanism-incident-edge count; or any increase in measurement-related issues;
or any new validation issue family not observed under full schema.

Lower generic graph size triggers manual HOLD instead of automatic degradation
because omission of unsupported material can be correct. HOLD still blocks
default adoption.

Canonical outputs are always retained for scientific semantic review.
