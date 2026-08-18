# GraphAgentsDAC Agent Instructions

## Purpose

This repository is currently undergoing a behavior-preserving
infrastructure refactor of the completed SERS evaluation campaign.

The primary goal is implementation deduplication and maintainability.

Scientific semantics, evaluation history, frozen artifacts, and campaign
state are not part of the refactor and must remain unchanged.

Before working on SERS runtime infrastructure, read:

- `docs/refactor/SERS-runtime-infra-v1.md`

## Operating principle

Make the smallest change that satisfies the explicitly requested task.

Do not broaden the scope because adjacent duplication or cleanup is visible.

One checkpoint should change one clearly bounded concern.

## Protected scientific state

Never modify, delete, regenerate, rename, or overwrite anything under:

- `evaluation/sers_fresh_c/`

Treat all historical success artifacts, failure artifacts, recovery lineage,
consumption markers, freeze manifests, and closeout artifacts as immutable
evidence.

Existing frozen SERS protocol JSON files are also immutable unless the task
explicitly states otherwise.

In particular, do not change existing files matching:

- `dac_her/sers_fresh_c_*protocol*.json`

## Scientific invariants

The refactor must not change:

- hypothesis scientific meaning
- R0/R1/R2 decisions
- Fresh Reserve C membership or ordering
- provider-selection scientific semantics
- search depth or query semantics
- protocol IDs
- protocol SHA derivation
- frozen artifact IDs
- frozen artifact hashes
- failure lineage
- irreversible consumption semantics
- rerun authorization semantics
- automatic-next-stage authorization
- scientific evidence interpretation
- LLM-call accounting
- network-call accounting
- final H1/H2/H3 states

Unknown evidence must never be converted into negative evidence.

External prior art must never become a positive generation premise merely
because infrastructure was refactored.

## Forbidden execution

During infrastructure refactoring, do not execute live or irreversible SERS
campaign stages.

Do not run:

- `scripts/run_sers_fresh_c_*`
- `scripts/freeze_sers_fresh_c_*`
- scientific adjudication runners
- live discovery runners
- Reserve-C consumption or materialization runners

Do not make network requests.

Do not invoke OpenAI, OpenAlex, Semantic Scholar, Crossref, Unpaywall,
or other external services.

Do not use API credentials during refactor validation.

## Git safety

Do not:

- push
- force-push
- rebase
- amend existing commits
- merge branches
- change branches
- delete branches
- run `git reset --hard`
- run `git clean -fdx`
- discard unrelated working-tree changes

Do not touch sibling repositories or worktrees.

In particular, never modify:

- `~/GraphAgentsDAC-refactor-check`

That repository is reserved for independent clean-room verification.

Do not commit unless the task explicitly authorizes a commit.

## Allowed refactor work

Allowed work includes:

- extracting pure deterministic utilities
- removing duplicated implementation
- adding characterization tests
- adding regression tests
- delegating existing helpers to shared infrastructure
- improving internal organization without changing behavior

Prefer delegation before deleting historical helper APIs.

Preserve existing public and private function names when compatibility is
uncertain.

## Repository restructuring track

In addition to the SERS runtime deduplication work, this repository may be
incrementally restructured to separate:

- legacy upstream GraphAgents material
- reusable domain-agnostic pipeline infrastructure
- domain packages such as DAC-HER and SERS
- frozen evaluation and campaign evidence

This restructuring must remain behavior-preserving for the completed SERS
campaign and must not modify protected scientific state.

Use the following staged sequence:

1. Establish a baseline and identify the legacy GraphAgents boundary.
2. Measure all live references from the current runtime into that legacy
   boundary.
3. Document the safe isolation strategy, including compatibility shims needed
   before any moves or renames.
4. Extract or delegate shared domain-agnostic infrastructure behind existing
   call surfaces.
5. Move domain-specific implementations only after the shared layer is proven
   by targeted regression and full Fresh-C characterization.

Do not start with broad package renames or large file moves.

Prefer compatibility shims and delegation layers before removing historical
paths.

Treat DAC-HER and SERS as current domain examples of a more general pipeline,
but do not assume a correct general abstraction without first characterizing
the existing shared behavior.

Legacy packaging and environment entrypoints may be quarantined instead of
migrated.

In particular, if `setup.py` and `requirements.txt` are determined to be legacy
GraphAgents surfaces, they do not need behavior-preserving line-by-line
porting. They may be isolated and later replaced with newly generated
repository-appropriate packaging and environment definitions after the core
refactor stabilizes.

When repository restructuring progresses from characterization into actual
large-scale file edits, record a checkpoint commit before each substantial edit
batch.

For this purpose, a substantial edit batch means any broad rename, file move,
package split, or multi-file structural refactor that would be difficult to
review or revert mentally as one uncommitted change.

This checkpoint-commit rule does not override the rule against making commits
without explicit user authorization. It means that once such a refactor step is
authorized, the work should be organized into commit-sized checkpoints and each
major edit phase should begin from a recorded commit boundary.

## Stage 1 checkpoint: legacy boundary characterization

The first repository-restructuring checkpoint is strictly limited to:

- identifying legacy GraphAgents candidate files and directories
- building a repository-wide reference map to those candidates
- classifying each candidate as live dependency, documentation-only, notebook-
  only, or currently unused
- documenting constraints that block immediate isolation

During this checkpoint:

- do not move files
- do not rename packages
- do not delete legacy material
- do not alter import paths
- do not touch `evaluation/sers_fresh_c/`
- do not touch existing SERS protocol JSON files

If isolation appears safe, record the required shim or delegation plan first.
Actual movement or renaming belongs to a later checkpoint.

## Required preflight

Before editing:

1. Run `git status --short`.
2. Record `git rev-parse HEAD`.
3. Confirm the requested files exist.
4. Inspect the exact implementation being replaced.
5. Confirm the requested change can be made without touching protected state.

If any precondition is unexpected, stop without writing.

## Required validation

After a source-code change:

1. Run `python -m py_compile` for changed Python files.
2. Run the nearest targeted tests.
3. Run `tests/test_evaluation_runtime_artifacts.py` when artifact primitives
   are involved.
4. Run the complete Fresh-C characterization suite.
5. Verify hashes under `evaluation/sers_fresh_c/` are identical before and
   after the tests.
6. Run `git diff --check`.
7. Inspect the complete diff.
8. Run `git status --short`.

Fresh-C characterization suite:

```bash
mapfile -t FRESH_TESTS < <(
  find tests -maxdepth 1 -type f \
    -name 'test_*fresh_c*.py' | sort
)

env \
  -u OPENAI_API_KEY \
  -u OPENALEX_API_KEY \
  -u SEMANTIC_SCHOLAR_API_KEY \
  -u S2_API_KEY \
  PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest \
    -q \
    -p no:cacheprovider \
    "${FRESH_TESTS[@]}"
```
