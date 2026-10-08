#!/usr/bin/env bash

# Deliberately NOT `set -e`:
# one failed scientific case must not prevent the remaining cases from running.
set -o pipefail

REPO="$HOME/FAKG/ForAllKG"
cd "$REPO" || exit 1

# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "graphAgent311" ]]; then
  conda activate graphAgent311 || exit 1
fi

# Enable undefined-variable checking only after conda initialization.
set -u

DATA_ROOT="${DATA_ROOT:-/home/rldnr7959/GraphAgentsDAC-corpora/sers_500_v1/overnight_acquisition_v1/discovery_runtime_v2}"
export DATA_ROOT

export PYTHONUNBUFFERED=1

export OPENROUTER_AGENT_MODEL="${OPENROUTER_AGENT_MODEL:-openai/gpt-5.6-luna}"
export OPENROUTER_CRITIC_MODEL="${OPENROUTER_CRITIC_MODEL:-$OPENROUTER_AGENT_MODEL}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"

MODEL="$OPENROUTER_AGENT_MODEL"
CRITIC_MODEL="$OPENROUTER_CRITIC_MODEL"
BASE_URL="$OPENAI_BASE_URL"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set."
  exit 1
fi

if [[ -z "${OPENALEX_API_KEY:-}" ]]; then
  echo "ERROR: OPENALEX_API_KEY is not set."
  exit 1
fi

# ----------------------------------------------------------------------
# Preflight: fail before going unattended if local v3.2 wiring is missing.
# ----------------------------------------------------------------------
echo "===== PREFLIGHT ====="

python -m scripts.discovery.run_dac_discovery_e2e --help >/dev/null || {
  echo "ERROR: main E2E runner unavailable"
  exit 1
}

python -m scripts.discovery.run_research_idea_e2e_search_v3_2 --help >/dev/null || {
  echo "ERROR: SIS-v3.2 E2E continuation runner unavailable"
  exit 1
}

python -m scripts.discovery.run_standard_portfolio_verification_shadow --help >/dev/null || {
  echo "ERROR: common verifier unavailable"
  exit 1
}

echo "Git commit: $(git rev-parse HEAD)"
echo "Model:      $MODEL"
echo "Critic:     $CRITIC_MODEL"
echo "Base URL:   $BASE_URL"
echo

# ----------------------------------------------------------------------
# Run root
# ----------------------------------------------------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"

BASE="$HOME/GraphAgentsDAC/evaluation/FULL_CURRENT_AI_SCIENTIST_V1_${STAMP}"
LOGS="$BASE/logs"
STATUS="$BASE/overnight_status.tsv"

mkdir -p "$LOGS"

printf "case\tstage\tstatus\trun_dir\n" > "$STATUS"

echo "======================================================================"
echo "FULL_CURRENT_AI_SCIENTIST_V1"
echo "BASE=$BASE"
echo "======================================================================"

# ----------------------------------------------------------------------
# One complete capability run
# ----------------------------------------------------------------------
run_case () {
  local LABEL="$1"
  local SOURCE="$2"
  local TARGET="$3"
  local QUESTION="$4"

  local RUN="$BASE/$LABEL"
  local MAIN_LOG="$LOGS/${LABEL}.01_main_e2e.log"
  local SIS_LOG="$LOGS/${LABEL}.02_sis_v3_1.log"
  local VERIFY_LOG="$LOGS/${LABEL}.03_verification.log"

  echo
  echo "######################################################################"
  echo "# $LABEL"
  echo "######################################################################"
  echo "SOURCE:   $SOURCE"
  echo "TARGET:   $TARGET"
  echo "QUESTION: $QUESTION"
  echo "RUN:      $RUN"
  echo

  # ------------------------------------------------------------------
  # Phase 1
  # Current integrated E2E idea-space construction.
  #
  # Effective scientific lane:
  # KG + open-world + Direct-RP + Direct-HO + HO
  #   -> Frontier
  #   -> Idea Evolution
  #   -> Scientific Portfolio
  #
  # We stop the unrelated legacy lane after its initial semantic stage.
  # ------------------------------------------------------------------

  echo "[$LABEL] PHASE 1: MAIN E2E"

  python -m scripts.discovery.run_dac_discovery_e2e \
    --corpus-id sers500_final_v2 \
    --data-root "$DATA_ROOT" \
    --domain-profile sers_au_ag \
    --run-dir "$RUN" \
    --source "$SOURCE" \
    --target "$TARGET" \
    --question "$QUESTION" \
    --objective explain_connection \
    --title "FULL_CURRENT_AI_SCIENTIST_V1 :: $LABEL" \
    --grounding-policy semantic_stop_fallback_top_n \
    --context-review-mode auto \
    --question-task-preservation-shadow \
    --open-world-discovery \
    --direct-relationpattern-task-shadow \
    --direct-relationpattern-top-k 20 \
    --direct-higher-order-shadow \
    --direct-higher-order-max-contexts 4 \
    --higher-order-shadow \
    --higher-order-max-contexts 12 \
    --frontier-idea-population-shadow \
    --idea-evolution-shadow \
    --idea-evolution-max-cross-source-outputs 6 \
    --idea-evolution-max-backbone-mutation-outputs 6 \
    --idea-evolution-max-candidate-interpretation-outputs 4 \
    --idea-evolution-max-candidate-parent-pool 8 \
    --scientific-portfolio-selection-shadow \
    --scientific-portfolio-max-evaluation-candidates 48 \
    --scientific-portfolio-max-retained 8 \
    --scientific-portfolio-max-retained-per-profile 2 \
    --post-generation-n10-authority-mode certification_only \
    --results-per-query 12 \
    --model "$MODEL" \
    --critic-model "$CRITIC_MODEL" \
    --base-url "$BASE_URL" \
    --api-key-env OPENROUTER_API_KEY \
    --stop-after-initial-semantic \
    2>&1 | tee "$MAIN_LOG"

  MAIN_RC=${PIPESTATUS[0]}

  if [[ "$MAIN_RC" -ne 0 ]]; then
    echo "[$LABEL] MAIN E2E FAILED rc=$MAIN_RC"
    printf "%s\tmain_e2e\tFAILED_%s\t%s\n" \
      "$LABEL" "$MAIN_RC" "$RUN" >> "$STATUS"
    return 0
  fi

  printf "%s\tmain_e2e\tPASS\t%s\n" \
    "$LABEL" "$RUN" >> "$STATUS"

  # Required Stage-7 -> SIS boundary.
  REQUIRED=(
    "$RUN/hypothesis.context.json"
    "$RUN/frontier_idea_population.shadow.json"
    "$RUN/idea_evolution.shadow.json"
    "$RUN/scientific_portfolio_shadow/candidate_pool.json"
    "$RUN/scientific_portfolio_shadow/selection.json"
    "$RUN/scientific_portfolio_shadow/materialized.shadow.portfolio.json"
  )

  for F in "${REQUIRED[@]}"; do
    if [[ ! -s "$F" ]]; then
      echo "[$LABEL] ERROR: required artifact missing: $F"
      printf "%s\tstage7_boundary\tMISSING_ARTIFACT\t%s\n" \
        "$LABEL" "$RUN" >> "$STATUS"
      return 0
    fi
  done

  printf "%s\tstage7_boundary\tPASS\t%s\n" \
    "$LABEL" "$RUN" >> "$STATUS"

  # ------------------------------------------------------------------
  # Phase 2
  # SIS-v3.1 becomes the actual continuation/search engine.
  #
  # 2 cycles:
  #   initial realization
  #   -> adaptive fertility/evolution
  #   -> realization
  #   -> adaptive fertility/evolution
  #   -> terminal realization
  #
  # Population growth remains the runner's conservative/default policy.
  # ------------------------------------------------------------------

  echo
  echo "[$LABEL] PHASE 2: SIS-v3.1, 2 CYCLES"

  python -m scripts.discovery.run_research_idea_e2e_search_v3_2 \
    --run-dir "$RUN" \
    --mode v3_1 \
    --cycles 2 \
    2>&1 | tee "$SIS_LOG"

  SIS_RC=${PIPESTATUS[0]}

  if [[ "$SIS_RC" -ne 0 ]]; then
    echo "[$LABEL] SIS-v3.1 FAILED rc=$SIS_RC"
    printf "%s\tsis_v3_1\tFAILED_%s\t%s\n" \
      "$LABEL" "$SIS_RC" "$RUN" >> "$STATUS"
    return 0
  fi

  FINAL="$RUN/scientific_portfolio_shadow/sis_v3_2_e2e/arms/v3_1/final.materialized.portfolio.json"

  if [[ ! -s "$FINAL" ]]; then
    echo "[$LABEL] ERROR: final SIS portfolio not found:"
    echo "  $FINAL"
    printf "%s\tsis_final\tMISSING_PORTFOLIO\t%s\n" \
      "$LABEL" "$RUN" >> "$STATUS"
    return 0
  fi

  printf "%s\tsis_v3_1\tPASS\t%s\n" \
    "$LABEL" "$RUN" >> "$STATUS"

  # Convenient top-level copy.
  cp "$FINAL" "$RUN/FINAL.full_current_ai_scientist_v1.portfolio.json"

  # ------------------------------------------------------------------
  # Phase 3
  # One common, non-authoritative evaluation stack.
  #
  # semantic -> external novelty -> N9
  # feasibility is capability-aware (SERS currently expected to skip).
  #
  # OpenAlex preflight is skipped because the current implementation
  # incorrectly treats exhausted free daily credits as exhausted even
  # when prepaid credits remain. Actual OpenAlex requests still enforce
  # the real server-side budget.
  # ------------------------------------------------------------------

  VERIFY_DIR="$RUN/full_current_verification"

  echo
  echo "[$LABEL] PHASE 3: COMMON VERIFICATION"

  python -m scripts.discovery.run_standard_portfolio_verification_shadow \
    --context "$RUN/hypothesis.context.json" \
    --portfolio "$FINAL" \
    --domain-profile sers_au_ag \
    --output-dir "$VERIFY_DIR" \
    --model "$CRITIC_MODEL" \
    --base-url "$BASE_URL" \
    --api-key-env OPENROUTER_API_KEY \
    --skip-provider-budget-preflight \
    2>&1 | tee "$VERIFY_LOG"

  VERIFY_RC=${PIPESTATUS[0]}

  if [[ "$VERIFY_RC" -ne 0 ]]; then
    echo "[$LABEL] VERIFICATION FAILED/PAUSED rc=$VERIFY_RC"
    printf "%s\tverification\tFAILED_OR_PAUSED_%s\t%s\n" \
      "$LABEL" "$VERIFY_RC" "$RUN" >> "$STATUS"
    return 0
  fi

  printf "%s\tverification\tPASS\t%s\n" \
    "$LABEL" "$RUN" >> "$STATUS"

  # ------------------------------------------------------------------
  # Small human-readable run record.
  # ------------------------------------------------------------------

  {
    echo "preset=FULL_CURRENT_AI_SCIENTIST_V1"
    echo "case=$LABEL"
    echo "source=$SOURCE"
    echo "target=$TARGET"
    echo "question=$QUESTION"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "model=$MODEL"
    echo "critic_model=$CRITIC_MODEL"
    echo "sis_cycles=2"
    echo "final_portfolio=$FINAL"
    echo "verification=$VERIFY_DIR/verification.summary.json"
  } > "$RUN/FULL_CURRENT_AI_SCIENTIST_V1.run.txt"

  echo
  echo "[$LABEL] COMPLETE"
  echo "Final portfolio:"
  echo "  $FINAL"
  echo "Verification:"
  echo "  $VERIFY_DIR/verification.summary.json"
}


# ======================================================================
# Q-A
#
# Tests:
#   regime boundary
#   adsorption/orientation mechanism
#   competing explanations
#   differential observation
# ======================================================================

run_case \
  "Q_A_orientation_breakdown" \
  "molecular orientation" \
  "Raman intensity" \
  "Under what conditions do molecular orientation and adsorption geometry reliably explain relative SERS band intensities on Au versus Ag surfaces, and under what conditions does that relationship break down? Identify plausible competing mechanisms and a measurable observation or experiment that could distinguish them."


# ======================================================================
# Q-B
#
# Tests:
#   proxy challenge
#   latent/effective mediator
#   nominal vs accessible hotspot structure
# ======================================================================

run_case \
  "Q_B_hotspot_proxy" \
  "hotspot density" \
  "Raman intensity" \
  "Under what conditions does increasing nominal hotspot density fail to improve quantitative SERS performance? Determine whether an effective or analyte-accessible hotspot variable better explains enhancement and reproducibility than nominal hotspot density, and propose a measurement that could distinguish the competing explanations."


# ======================================================================
# Q-C
#
# Tests:
#   surface-state mediator
#   temporal/aging regime
#   enhancement vs reproducibility tradeoff
# ======================================================================

run_case \
  "Q_C_architecture_aging" \
  "plasmonic architecture" \
  "Raman intensity" \
  "Why can Au and Ag plasmonic architectures with similar mean SERS enhancement exhibit different reproducibility and aging stability? Identify measurable surface-state or structural variables that could mediate this divergence and propose an experiment that distinguishes competing mechanisms."


echo
echo "======================================================================"
echo "OVERNIGHT RUN FINISHED"
echo "======================================================================"
echo "Base directory:"
echo "  $BASE"
echo
echo "Status:"
column -t -s $'\t' "$STATUS" 2>/dev/null || cat "$STATUS"
echo
echo "Final portfolio shortcuts, when successful:"
find "$BASE" -name 'FINAL.full_current_ai_scientist_v1.portfolio.json' -print
echo
echo "Verification summaries:"
find "$BASE" -path '*/full_current_verification/verification.summary.json' -print
