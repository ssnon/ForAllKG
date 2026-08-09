#!/usr/bin/env bash
set -euo pipefail

BASE="2c0620963adff2031e4347d070ec58086b82c75f"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$SCRIPT_DIR/GraphAgentsDAC_v270_to_v271_core.patch"

RESTORE_PATHS=(
  "run_hypothesis_v261_smokes.sh"
  "tests/_hypothesis_v260_fixtures.py"
  "tests/_hypothesis_v261_fixtures.py"
  "tests/test_hypothesis_benchmark_evaluator_v262.py"
  "tests/test_hypothesis_benchmark_suite_v262.py"
  "tests/test_hypothesis_compiler_v260.py"
  "tests/test_hypothesis_context_v260.py"
  "tests/test_hypothesis_e2_v262.py"
  "tests/test_hypothesis_gold_comparator_v262.py"
  "tests/test_hypothesis_llm_contract_v261.py"
  "tests/test_hypothesis_prompt_v261.py"
  "tests/test_hypothesis_real_gold_v262.py"
  "tests/test_hypothesis_runtime_v261.py"
  "tests/test_hypothesis_semantic_checks_v262.py"
  "tests/test_hypothesis_semantic_prompt_v262.py"
  "tests/test_hypothesis_semantic_runtime_v262.py"
  "tests/test_hypothesis_v262_fixture_isolation.py"
  "tests/test_hypothesis_validation_v260.py"
)

STALE_LAB_PATHS=(
  "dac_her/lab_profile.py"
  "configs/lab_profiles/local_lab.yaml"
  "configs/lab_profiles/prototype_lab.example.yaml"
  "tests/test_lab_profile_v0.py"
)

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: run this from inside the GraphAgentsDAC git repository." >&2
  exit 2
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if ! git cat-file -e "$BASE^{commit}" 2>/dev/null; then
  echo "ERROR: required v2.6.2 parent commit $BASE is not present locally." >&2
  echo "This bundle restores the deleted regression suite from that local ancestor." >&2
  exit 3
fi

# Do not overwrite local edits to files this cleanup owns.
OWNED_PATHS=(
  "dac_her/scope_compiler.py"
  "tests/test_scope_cleanup_v271.py"
  "tests/test_feasibility_real_e2e_regression_v271.py"
  "tests/fixtures/feasibility_v271_real_intake.json"
  "${STALE_LAB_PATHS[@]}"
  "${RESTORE_PATHS[@]}"
)
if ! git diff --quiet -- "${OWNED_PATHS[@]}" ||    ! git diff --cached --quiet -- "${OWNED_PATHS[@]}"; then
  echo "ERROR: local changes exist in v2.7.1-owned paths. Commit/stash them first." >&2
  exit 4
fi

git apply --check "$PATCH"
git apply "$PATCH"

# Restore the hypothesis/semantic regression suite exactly from the v2.6.2 parent.
git restore --source="$BASE" -- "${RESTORE_PATHS[@]}"

# Remove stale lab-specific remnants; core feasibility is laboratory-agnostic.
rm -f -- "${STALE_LAB_PATHS[@]}"
rmdir configs/lab_profiles 2>/dev/null || true

echo
echo "GraphAgentsDAC v2.7.1 cleanup applied."
echo "Recommended targeted regression:"
echo "  python -m pytest -q tests/test_scope_cleanup_v271.py tests/test_scope_and_validation_v02.py tests/test_feasibility_real_e2e_regression_v271.py"
echo
echo "Then run the complete suite:"
echo "  python -m pytest -q"

if [[ "${1:-}" == "--test" ]]; then
  python -m pytest -q
fi
