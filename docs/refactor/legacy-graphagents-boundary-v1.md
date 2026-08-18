# Legacy GraphAgents Boundary Characterization v1

## Scope

This checkpoint characterizes the safe isolation boundary for legacy upstream
GraphAgents material.

It does not move, rename, or delete files.

It does not change import paths.

It does not touch protected SERS evaluation state.

Baseline HEAD during characterization:

```text
fa5bd2b65e74f1d91181e5dd19110d377651847b
```

## Candidate legacy boundary

The following repository areas are the primary candidates for isolation into a
future `legacy_graphagents/` or similarly named boundary:

- `GraphReasoning/`
- `Notebooks/`
- `Experiments/`
- `README.md`
- `setup.py`
- `requirements.txt`

Rationale:

- `GraphReasoning/`, `Notebooks/`, and `Experiments/` align with the original
  upstream GraphAgents paper/demo structure.
- `README.md` documents the upstream GraphAgents workflow rather than the
  current DAC-HER/SERS runtime.
- `setup.py` still publishes the repository as `GraphReasoning`.
- `requirements.txt` still contains an external `GraphReasoning` package path.

## Reference map

### 1. `GraphReasoning/`

Classification:

- packaging-coupled legacy candidate

Observed live references:

- No imports from `dac_her/`, `scripts/`, `tests/`, `configs/`, or `docs/`
  into `GraphReasoning`.
- Repository-wide `GraphReasoning` imports are internal to the
  `GraphReasoning/` package itself.

Observed coupling:

- `setup.py` publishes the repository under `name='GraphReasoning'` and uses
  `packages=find_packages()`, so the legacy package is still part of the
  current distribution surface.
- `requirements.txt` includes
  `GraphReasoning @ file:///nfs/pool002/users/istewart/saintgobain/GraphReasoning_SG`.

Isolation implication:

- Runtime code does not appear to depend on `GraphReasoning` imports.
- Packaging and environment metadata still do.
- Packaging must be decoupled before or alongside any file move.

### 2. `Notebooks/`

Classification:

- notebook-only legacy candidate

Observed live references:

- No references from `dac_her/`, `scripts/`, `tests/`, or `configs/`.
- References appear only in `README.md`.

Isolation implication:

- Safe candidate for later relocation once documentation is updated.

### 3. `Experiments/`

Classification:

- artifact/demo-only legacy candidate

Observed live references:

- No references from `dac_her/`, `scripts/`, `tests/`, or `configs/`.
- References appear only in `README.md`.

Isolation implication:

- Safe candidate for later relocation once documentation is updated.

### 4. `README.md`

Classification:

- legacy-facing documentation candidate

Observed references:

- Describes the original GraphAgents install flow, notebooks, and
  `Experiments/` artifacts.
- Does not describe the current `dac_her` runtime as the repository primary
  entrypoint.

Isolation implication:

- A future restructuring should likely split upstream GraphAgents documentation
  from current pipeline/runtime documentation.

### 5. `setup.py`

Classification:

- legacy packaging surface

Observed coupling:

- Distribution name is `GraphReasoning`.
- `packages=find_packages()` will include both legacy and current packages in
  one distribution boundary.

Isolation implication:

- Current package metadata does not need to be migrated as-is.
- `setup.py` may be quarantined with the legacy GraphAgents material and
  replaced later by a new packaging definition once the refactored repository
  structure is stable.

### 6. `requirements.txt`

Classification:

- legacy environment surface

Observed coupling:

- Direct dependency on an external `GraphReasoning` filesystem package path.

Isolation implication:

- The file reflects an old GraphReasoning-centered environment model.
- It may be quarantined with legacy material and replaced later by a new
  environment definition generated from the actual refactored runtime needs.

## Important non-boundary findings

Many current runtime files still use the word `GraphAgents`, but these are not
legacy upstream dependencies by themselves.

Examples:

- `dac_her/graphagents_adapter.py`
- `scripts/build_graphagents_projection.py`
- `dac_her/corpus_pipeline.py`
- `dac_her/strict_bridge_corpus_pipeline.py`
- multiple SERS holdout and reserve scripts that call
  `scripts.build_graphagents_projection`

Interpretation:

- `GraphAgents` is still the historical name for one current projection layer
  inside the DAC-HER/SERS runtime.
- These files are live current infrastructure and must not be misclassified as
  removable upstream legacy material.

Therefore:

- `GraphReasoning/` is a legacy candidate.
- `graphagents_adapter.py` and `build_graphagents_projection.py` are current
  runtime code with historical naming.

## Safe isolation constraints

Before moving any legacy candidate directory:

1. Keep current `dac_her` runtime modules with historical `GraphAgents` naming
   in place until a later, explicitly scoped rename checkpoint.
2. Update repository documentation so the current pipeline and the upstream
   GraphAgents legacy material have separate entrypoints.
3. When the repository structure stabilizes, generate new packaging and
   environment entrypoints instead of porting the existing legacy
   `setup.py`/`requirements.txt` surfaces.

## Recommended next checkpoint

The next safe checkpoint is:

```text
design the legacy isolation layout and replacement entrypoints without moving
runtime code yet
```

That checkpoint should answer:

- what the new top-level package/distribution name should be
- whether `GraphReasoning/` remains vendored under a legacy path or is removed
  from the main runtime tree
- how `README.md` should be split between upstream legacy docs and current
  pipeline docs
- what new packaging/environment files should replace the legacy
  `setup.py`/`requirements.txt` surfaces after the refactor
