# Mid-Runtime Extraction Boundary v1

## Scope

This checkpoint characterizes the next layer above the already extracted shared
interface/contract modules.

It focuses on three current runtime modules:

- `dac_her/graphagents_adapter.py`
- `dac_her/corpus_pipeline.py`
- `dac_her/strict_bridge_corpus_pipeline.py`

These modules are not pure contracts. They are runtime/service layers that sit
between the extracted core interfaces and the CLI scripts.

Baseline HEAD during characterization:

```text
f3b94e8756c1f4a8ce335f1e7f2f2cf5cc40c2be
```

## High-level conclusion

These three modules do not belong in the same next-step bucket.

They separate naturally into:

1. a projection service layer
2. a thin frozen-corpus orchestration layer
3. a heavier strict/resumable orchestration layer

The safest next extraction target is the projection service layer in
`graphagents_adapter.py`.

The corpus pipeline runners should be deferred until after the projection
runtime boundary is stabilized.

## Importer map

Each module currently has exactly one direct script-facing importer.

| Runtime module | Direct importer |
|---|---|
| `dac_her.graphagents_adapter` | `scripts.build_graphagents_projection` |
| `dac_her.corpus_pipeline` | `scripts.run_frozen_corpus_pipeline` |
| `dac_her.strict_bridge_corpus_pipeline` | `scripts.run_strict_bridge_corpus` |

Interpretation:

- these modules are already script-facing services
- they are not broad utility modules imported all over the codebase
- they can be extracted with compatibility shims more easily than a highly
  entangled internal helper layer

## Module characterization

### 1. `dac_her.graphagents_adapter`

Role:

- projection runtime/service
- converts canonical graph + Bridge graph into evidence/mechanism/exploratory
  projection outputs
- emits graph plus JSONL support artifacts

Current dependency shape:

- directly depends only on `dac_her.domain_profile`
- now indirectly benefits from the already extracted `pipeline_core`
  `domain_profile` shim path

Internal structure:

- mostly deterministic helper functions
- one main public service entrypoint:
  `build_graphagents_projection(...)`
- one output helper:
  `write_jsonl(...)`

Important caveat:

- it still carries `_LEGACY_DAC_HER_PROJECTION_SEMANTICS`
- therefore it is not yet fully domain-neutral in naming/history, even though
  the runtime surface is largely shared

Interpretation:

- this is the best next extraction candidate
- likely future home: shared projection runtime package
- rename can be deferred; extraction can happen first under a historical name

### 2. `dac_her.corpus_pipeline`

Role:

- thin orchestration runtime for frozen corpus processing
- shell-out runner around the existing CLI stages

Observed characteristics:

- little to no direct `dac_her` import coupling
- owns pipeline fingerprinting, state persistence, stage skipping, and
  subprocess orchestration
- calls:
  - `scripts.extract_paper`
  - `scripts.build_paper_graph`
  - `scripts.extract_bridge_graph`
  - `scripts.build_graphagents_projection`

Interpretation:

- this is a shared orchestration layer, not a domain adapter
- but it is more about CLI/runtime composition than reusable scientific logic
- it should probably move later than `graphagents_adapter`

### 3. `dac_her.strict_bridge_corpus_pipeline`

Role:

- heavier orchestration runtime for strict-ready corpora
- resumable extraction/Bridge/projection/corpus runner

Observed characteristics:

- depends on `dac_her.config` and `dac_her.run_state`
- fingerprints large parts of the implementation tree
- binds extraction/Bridge/projection/corpus stages into one resumable state
  machine
- mixes orchestration, provenance, and implementation-tree compatibility logic

Interpretation:

- this is not a good immediate extraction target
- it is still shared infrastructure in principle
- but it is the riskiest of the three because it couples runtime state, resume
  behavior, provenance, and source-tree hashing

## Boundary recommendation

The next extraction sequence should be:

1. extract `graphagents_adapter.py`
2. stabilize its shim/import behavior
3. only then evaluate `corpus_pipeline.py`
4. defer `strict_bridge_corpus_pipeline.py` until after the projection service
   and simpler corpus runner are already separated

## Why `graphagents_adapter.py` should go next

Reasons:

- narrower dependency surface than the pipeline runners
- deterministic service behavior
- already script-contained behind one CLI entrypoint
- naturally layered on top of the extracted domain-profile interfaces
- lower risk than moving a resumable pipeline state machine

It is effectively a “runtime core” module, not a campaign-specific file.

## Why the corpus runners should wait

### `corpus_pipeline.py`

This file is shared and probably belongs outside `dac_her` eventually, but it
is a subprocess orchestration layer rather than a domain interface.

Moving it before the projection runtime would invert the natural layering:

```text
projection service
    ->
corpus orchestration
```

So it should remain second, not first.

### `strict_bridge_corpus_pipeline.py`

This file adds extra concerns:

- implementation-tree fingerprinting
- resume compatibility
- stateful provenance
- strict-ready corpus assumptions

Those concerns make it a poor first runtime extraction candidate even though it
may later belong in a shared orchestration package.

## Proposed next extraction slice

Recommended next slice:

```text
Slice B0: shared projection runtime
```

Primary module:

- `dac_her/graphagents_adapter.py`

Likely migration pattern:

1. move implementation to `pipeline_core` under a temporary historical name
2. keep `dac_her.graphagents_adapter` as a re-export shim
3. do not rename `scripts.build_graphagents_projection` yet
4. preserve the current `build_graphagents_projection(...)` and `write_jsonl(...)`
   call surface exactly

## Deferred follow-up slices

After B0 stabilizes:

```text
Slice B1: simple corpus orchestration
```

Target:

- `dac_her/corpus_pipeline.py`

After B1 stabilizes:

```text
Slice B2: strict/resumable corpus orchestration
```

Target:

- `dac_her/strict_bridge_corpus_pipeline.py`

## Validation implications

The next actual runtime move after A0/A1 should still remain behavior-
preserving and should use the existing checkpoint-commit rule before file edits.

For the future B0 extraction, validation should prioritize:

- import compatibility for `dac_her.graphagents_adapter`
- smoke loading for `scripts.build_graphagents_projection`
- targeted projection tests
- Fresh-C characterization once a working pytest environment is available
