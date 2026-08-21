# First Shared Interface Extraction Slice v1

## Scope

This checkpoint defines the first actual code-extraction slice for the future
domain-agnostic pipeline core.

It does not move or rename code yet.

It identifies the smallest safe set of modules that can be extracted first
without dragging in campaign-specific SERS state or large runtime rewiring.

Baseline HEAD during characterization:

```text
fa5bd2b65e74f1d91181e5dd19110d377651847b
```

## Goal

Start the real repository split with modules that are:

- deterministic
- adapter/contract oriented
- low in dependency fan-in
- already shared by HER and SERS or clearly intended to be shared
- not coupled to `evaluation/sers_fresh_c/`
- not coupled to Fresh-C campaign state machines

## Dependency findings

### Candidate modules with no internal slice dependencies

These modules are effectively self-contained interface/contract layers:

- [dac_her/domain_profile.py](../../dac_her/domain_profile.py)
- [dac_her/bridge_domain.py](../../dac_her/bridge_domain.py)
- [dac_her/graph_domain.py](../../dac_her/graph_domain.py)
- [dac_her/reproducibility_domain.py](../../dac_her/reproducibility_domain.py)
- [dac_her/metric_definition_domain.py](../../dac_her/metric_definition_domain.py)
- [dac_her/trend_domain.py](../../dac_her/trend_domain.py)
- [dac_her/evaluation_runtime/artifacts.py](../../dac_her/evaluation_runtime/artifacts.py)

These are the cleanest first extraction candidates.

### Candidate modules with small dependency closure

- [dac_her/extraction_domain.py](../../dac_her/extraction_domain.py)
  depends on:
  - [dac_her/graph_domain.py](../../dac_her/graph_domain.py)
  - [dac_her/draft_schema.py](../../dac_her/draft_schema.py)

- [dac_her/comparison_domain.py](../../dac_her/comparison_domain.py)
  depends on:
  - [dac_her/method_context.py](../../dac_her/method_context.py)

- [dac_her/feasibility_domain.py](../../dac_her/feasibility_domain.py)
  depends on:
  - [dac_her/experimental_contracts.py](../../dac_her/experimental_contracts.py)
  - [dac_her/feasibility_contracts.py](../../dac_her/feasibility_contracts.py)
  - [dac_her/physics_contracts.py](../../dac_her/physics_contracts.py)
  - [dac_her/scope_contracts.py](../../dac_her/scope_contracts.py)
  - [dac_her/validation_contracts.py](../../dac_her/validation_contracts.py)

Interpretation:

- these are still good core candidates
- but they should not be part of the very first move if the goal is the
  smallest possible edit batch

## Recommended extraction staging

### Slice A0: pure shared interfaces

This should be the first real code-move checkpoint.

Modules:

- [dac_her/domain_profile.py](../../dac_her/domain_profile.py)
- [dac_her/bridge_domain.py](../../dac_her/bridge_domain.py)
- [dac_her/graph_domain.py](../../dac_her/graph_domain.py)
- [dac_her/reproducibility_domain.py](../../dac_her/reproducibility_domain.py)
- [dac_her/metric_definition_domain.py](../../dac_her/metric_definition_domain.py)
- [dac_her/trend_domain.py](../../dac_her/trend_domain.py)
- [dac_her/evaluation_runtime/artifacts.py](../../dac_her/evaluation_runtime/artifacts.py)

Why this slice first:

- no dependence on Fresh-C runtime state
- no dependence on HER-only scientific rules
- no dependence on SERS-only scientific rules
- no dependence on large orchestration modules
- easy to leave behind compatibility shims under `dac_her.*`

Expected migration pattern:

```text
new core package module
    ->
move implementation
    ->
leave dac_her shim that re-exports old symbol names
    ->
run targeted imports/tests
```

### Slice A1: interface closure completion

This should follow only after Slice A0 is stable.

Modules:

- [dac_her/draft_schema.py](../../dac_her/draft_schema.py)
- [dac_her/extraction_domain.py](../../dac_her/extraction_domain.py)
- [dac_her/method_context.py](../../dac_her/method_context.py)
- [dac_her/comparison_domain.py](../../dac_her/comparison_domain.py)
- [dac_her/experimental_contracts.py](../../dac_her/experimental_contracts.py)
- [dac_her/feasibility_contracts.py](../../dac_her/feasibility_contracts.py)
- [dac_her/physics_contracts.py](../../dac_her/physics_contracts.py)
- [dac_her/scope_contracts.py](../../dac_her/scope_contracts.py)
- [dac_her/validation_contracts.py](../../dac_her/validation_contracts.py)
- [dac_her/feasibility_domain.py](../../dac_her/feasibility_domain.py)

Why this slice second:

- still mostly contract/interface oriented
- extends the shared abstraction boundary meaningfully
- introduces modest dependency closure without yet touching campaign logic or
  large runtimes

## Why not start with larger “core-looking” files

The following files are important but should not be first-move targets:

- [dac_her/graphagents_adapter.py](../../dac_her/graphagents_adapter.py)
- [dac_her/corpus_pipeline.py](../../dac_her/corpus_pipeline.py)
- [dac_her/strict_bridge_corpus_pipeline.py](../../dac_her/strict_bridge_corpus_pipeline.py)
- [dac_her/hypothesis_runtime.py](../../dac_her/hypothesis_runtime.py)
- [pipeline_core/discovery/explorer_runtime.py](../../pipeline_core/discovery/explorer_runtime.py)

Reasons:

- larger import surfaces
- more runtime wiring
- more historical naming baggage
- greater risk of silently touching SERS evaluation behavior

These should be deferred until the interface and contract layers have already
been extracted behind stable compatibility shims.

## External importer pressure

The strongest outwardly referenced modules in this slice are:

- `dac_her.domain_profile`
- `dac_her.graph_domain`
- `dac_her.trend_domain`

This is acceptable for a first extraction because they are imported widely as
definitions rather than as stateful runtimes.

The recommended strategy is to preserve old import paths during the transition:

```text
dac_her.domain_profile
    ->
re-export from new core location
```

not:

```text
mass update every importer in one checkpoint
```

## Proposed future target layout

The A0/A1 slices point toward a future structure like:

```text
pipeline_core/
  domain_profile.py
  extraction_domain.py
  bridge_domain.py
  graph_domain.py
  comparison_domain.py
  reproducibility_domain.py
  metric_definition_domain.py
  trend_domain.py
  feasibility_domain.py
  method_context.py
  draft_schema.py
  evaluation_runtime/
    artifacts.py
  contracts/
    feasibility_contracts.py
    physics_contracts.py
    scope_contracts.py
    validation_contracts.py
    experimental_contracts.py
```

Exact naming can wait.

The main point is that these files form a coherent, domain-agnostic interface
layer and are a better first extraction target than any campaign-bound or
runtime-heavy module group.

## Recommended next checkpoint

The next execution checkpoint should be:

```text
perform Slice A0 extraction with compatibility re-export shims
```

Before that checkpoint begins:

1. record a checkpoint commit as required by `AGENTS.md`
2. choose the destination package name for the new shared core
3. keep old `dac_her.*` import paths working via thin re-export modules

## Validation expectation for Slice A0

Because Slice A0 changes import paths but not intended behavior, validation
should prioritize:

- `python -m py_compile` for changed files
- nearest targeted tests covering domain registries and adapter loading
- `tests/test_evaluation_runtime_artifacts.py`
- full Fresh-C characterization before accepting the checkpoint

This remains a behavior-preserving refactor, not a semantic rewrite.
