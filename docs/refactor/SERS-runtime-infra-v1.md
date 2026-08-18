# SERS Runtime Infrastructure Refactor v1

## Objective

Refactor duplicated deterministic evaluation infrastructure while preserving
the completed SERS scientific campaign exactly.

This work is:

- a behavior-preserving implementation refactor
- a maintenance and deduplication campaign
- separate from scientific development
- separate from evaluation execution

This work is not:

- a new scientific development campaign
- a new Fresh-C epoch
- a new novelty evaluation
- authorization to rerun failed or consumed one-shot stages
- authorization to rewrite frozen artifacts

No scientific state may advance.

---

## Historical baseline

Original refactor baseline commit:

```text
9760966f93a6c2cdf440f538c43e3c1c97b2024d
```

Meaning:

- Fresh-C final campaign closeout was frozen.
- Campaign state was closed.
- No automatic next stage was authorized.

### F0 baseline environment

```text
Python: 3.11.15
pytest: 9.1.1
repository test files: 300
Fresh-C test files: 17
Fresh-C tests: 117
Fresh-C frozen evaluation files: 113
```

### F0 DEV verification

```text
117 / 117 PASS
frozen-tree hash diff: zero
working-tree mutation: zero
```

### F0 clean-room verification

```text
117 / 117 PASS
frozen-tree hash diff: zero
working-tree mutation: zero
```

F0 status:

```text
CLOSED
```

---

## Frozen scientific state

### R0

R0 is frozen.

Important invariant:

```text
targeted_search_only
    =>
pass_original_to_r2
R1 authorization = false
max refinements = 0
```

Do not alter this routing during early infrastructure refactoring.

### R1

R1 was not executed because no R0 outcome authorized it.

This is intentional.

Do not reinterpret this as missing execution or incomplete implementation.

### R2

Pre-Fresh-C frozen decisions:

```text
H1: KEEP_BOUNDED_EXTENSION
H2: REJECT_AS_FORMULATED
H3: KEEP_RELATIONAL_GAP_CANDIDATE
```

### I0

I0 is a frozen orchestration handoff.

It did not perform new scientific retrieval, ranking, rewriting, or
reassessment.

### Fresh Reserve C

Reserve C was eventually consumed irreversibly.

Historical failures remain part of the evidence lineage.

Important invariants:

- freshness was not restored after failure
- same-parent rerun was not authorized
- recovery did not create a new Fresh Reserve
- failed responses were not silently reused as successful evidence
- recovery lineage must remain preserved

Final scientific closeout states:

```text
H1: FRESH_C_PRESERVES_PRE_C_BOUNDED_EXTENSION
H2: REJECT_AS_FORMULATED
H3: FRESH_C_ERODES_PRE_C_RELATIONAL_GAP
```

Scientific accounting:

```text
accepted scientific outputs: 26
total scientific attempts including failed parent: 27
```

These values are frozen historical state, not refactor targets.

---

## Refactor strategy

Historical artifacts are immutable evidence.

Production implementation may be deduplicated only when exact behavior is
characterized first.

Operating sequence:

```text
characterize
    ->
extract
    ->
delegate
    ->
targeted regression
    ->
full Fresh-C regression
    ->
frozen-tree hash verification
    ->
review
    ->
commit
    ->
stop
```

Do not combine multiple scientific or lifecycle concerns into one checkpoint.

---

## Completed checkpoints

### F0 — Baseline characterization

Status:

```text
CLOSED
```

No production changes.

Verified:

- DEV reproducibility
- clean-room reproducibility
- 117 Fresh-C tests
- 113 frozen evaluation files unchanged

### F1.1 — Shared artifact primitives

Status:

```text
CLOSED
```

Commit:

```text
6a19248
refactor: add shared evaluation artifact primitives
```

Added:

```text
dac_her/evaluation_runtime/__init__.py
dac_her/evaluation_runtime/artifacts.py
tests/test_evaluation_runtime_artifacts.py
```

Shared primitives:

```text
canonical_json
sha256_json
sha256_file
sha256_json_without_fields
load_json_object
```

Characterization:

```text
10 / 10 PASS
```

Production migration performed:

```text
none
```

The shared artifact layer is intentionally free of:

- scientific semantics
- stage authorization
- protocol prefixes
- marker behavior
- provider behavior
- network behavior
- hypothesis logic

---

## Current checkpoint

# F1.2a — Fresh-C live-discovery hashing migration

Target production file:

```text
dac_her/fresh_c_live_discovery.py
```

Allowed production-file count:

```text
1
```

Allowed implementation change:

Delegate the existing implementations of:

```text
_payload_sha
_protocol_identity_sha
```

to:

```text
dac_her.evaluation_runtime.artifacts.sha256_json_without_fields
```

Compatibility requirement:

Keep the existing helper names and signatures.

Expected conceptual transformation:

```python
def _payload_sha(payload, field):
    return sha256_json_without_fields(payload, field)
```

and:

```python
def _protocol_identity_sha(payload):
    return sha256_json_without_fields(
        payload,
        "protocol_id",
        "protocol_sha256",
    )
```

Do not remove these wrapper functions in F1.2a.

Do not migrate additional modules in F1.2a.

## F1.2a forbidden changes

Do not change:

- `expected_protocol_id`
- protocol prefix
- protocol schema/model
- protocol error messages
- broad queries
- provider set
- search depth
- target acquired-paper count
- historical exclusion behavior
- identity projection
- blind ordering
- scientific-field exclusion
- STARTED marker ordering
- failure semantics
- Fresh-C consumption semantics
- network behavior
- automatic-next-stage behavior

Do not modify:

```text
evaluation/sers_fresh_c/
```

Do not modify any frozen protocol JSON.

## F1.2a targeted validation

At minimum run:

```text
tests/test_evaluation_runtime_artifacts.py
tests/test_sers_fresh_c_live_discovery_v1.py
```

Then run the complete Fresh-C characterization suite.

Expected baseline Fresh-C suite:

```text
117 tests
```

Frozen evaluation tree must remain byte-identical.

---

## Planned later checkpoints

These are plans only.

Do not automatically execute them.

### F1.2b

Tentative target:

```text
dac_her/fresh_c_content_acquisition_v1.py
```

Likely migration:

- shared payload hashing
- shared protocol-identity hashing primitive
- shared JSON-object loader

Only begin after F1.2a is reviewed and closed.

### F1.2c

Tentative target:

selected Fresh-C recovery modules.

Migrate in small groups or one module at a time.

### F1.3

Tentative target:

freeze / run / verify script infrastructure.

Do not mix this with production-module migration unless explicitly planned.

### F2

Tentative target:

protocol-validation infrastructure.

Potential areas:

- shared strict protocol loading
- shared ID/SHA validation mechanics

Stage-specific prefixes, schemas, and error semantics remain stage-owned.

### F3

Tentative target:

stage lifecycle and marker infrastructure.

Potential concepts:

```text
STARTED
FAILED
COMPLETE
CONSUMED
FREEZE_READY
```

Do not generalize lifecycle behavior before existing stage semantics are fully
characterized.

### R0

R0 scientific routing is not an early refactor target.

Do not refactor `r0_runtime.py` merely because infrastructure duplication is
visible elsewhere.

---

## Architectural direction

Desired dependency direction:

```text
scientific policy
        |
        v
stage-specific protocol
        |
        v
evaluation infrastructure
        |
        v
pure artifact primitives
```

The infrastructure layer must not know:

- final H1/H2/H3 outcomes
- scientific novelty labels
- hypothesis-specific rules
- R0/R2 scientific decisions
- provider scientific relevance
- Fresh-C adjudication outcomes

---

## Canonical artifact contract

Canonical JSON must preserve the historical repository convention:

```python
json.dumps(
    value,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
)
```

If the input supports:

```python
model_dump(mode="json")
```

that conversion must occur before serialization.

Any change to canonical JSON representation is a behavior change and is outside
this refactor.

Hashing must remain deterministic and byte-compatible with historical artifact
identity.

---

## Clean-room policy

Development workspace:

```text
~/GraphAgentsDAC-refactor
```

Independent verification workspace:

```text
~/GraphAgentsDAC-refactor-check
```

Codex must not modify the clean-room checkout.

The clean-room checkout must consume committed state only.

Important committed checkpoints should be independently validated there before
being considered fully closed.

---

## Checkpoint completion rule

A checkpoint is not complete merely because targeted tests pass.

A checkpoint is complete only when all required conditions hold:

```text
requested source scope respected
py_compile PASS
targeted tests PASS
Fresh-C characterization PASS
frozen-tree hash diff = zero
git diff --check PASS
complete diff reviewed
no protected artifacts changed
no scientific semantics changed
```

After reporting these results:

```text
STOP
```

Do not begin the next checkpoint automatically.
