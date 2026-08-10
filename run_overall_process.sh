set -a
source .env
set +a

CORPUS=dac_her_expanded_v1

run_dac_discovery () {
    local RUN="$1"
    local SOURCE="$2"
    local STOP="$3"
    local TARGET="$4"
    local QUESTION="$5"
    local TITLE="$6"

    echo
    echo "################################################################"
    echo "# RUN: $RUN"
    echo "# QUESTION: $QUESTION"
    echo "################################################################"

    mkdir -p "$RUN"

    # ============================================================
    # 1. Grounding traversal
    #    mechanism graph + semantic-stop
    # ============================================================

    echo
    echo "===== [1/13] Grounding traversal ====="

    python -m scripts.run_graph_traversal \
        --corpus-id "$CORPUS" \
        --mode mechanism \
        --algorithm top_n \
        --source "$SOURCE" \
        --stop "$STOP" \
        --target "$TARGET" \
        --node-map-k 20 \
        --waypoint-k 12 \
        --endpoint-pair-k 12 \
        --semantic-stop-max-depth 12 \
        --top-k 8 \
        --include-candidate-paths \
        --output "$RUN/traversal.json"


    # ============================================================
    # 2. Build GraphExplorerPacket
    # ============================================================

    echo
    echo "===== [2/13] Explorer packet ====="

    python -m scripts.build_explorer_packet \
        --traversal-result "$RUN/traversal.json" \
        --question "$QUESTION" \
        --objective explain_connection \
        --output "$RUN/explorer.packet.json"


    # ============================================================
    # 3. Graph Explorer
    # ============================================================

    echo
    echo "===== [3/13] Graph Explorer ====="

    python -m scripts.run_graph_explorer \
        --packet "$RUN/explorer.packet.json" \
        --model "$OPENROUTER_AGENT_MODEL" \
        --base-url "https://openrouter.ai/api/v1" \
        --api-key-env OPENROUTER_API_KEY \
        --output-prefix "$RUN/explorer" \
        --save-prompt


    # ============================================================
    # 4. Grounded HypothesisContext
    # ============================================================

    echo
    echo "===== [4/13] Grounded hypothesis context ====="

    python -m scripts.build_hypothesis_context \
        --packet "$RUN/explorer.packet.json" \
        --report "$RUN/explorer.report.json" \
        --output "$RUN/hypothesis.context.json"


    # ============================================================
    # 5. Candidate-unit discovery traversal
    #
    # NOTE:
    # candidate-unit traversal은 현재 stop waypoint를 직접 받지 않는다.
    # source → candidate-unit → target discovery lane이다.
    # ============================================================

    echo
    echo "===== [5/13] Candidate-unit discovery ====="

    python -m scripts.run_candidate_unit_traversal \
        --corpus-id "$CORPUS" \
        --source "$SOURCE" \
        --target "$TARGET" \
        --node-map-k 20 \
        --max-depth 12 \
        --top-k 12 \
        --include-candidate-paths \
        --output "$RUN/candidate_unit.traversal.a3.json"


    # ============================================================
    # 6. DiscoveryBundle
    #
    # mechanism traversal = grounding/reference lane
    # candidate unit       = discovery lane
    # ============================================================

    echo
    echo "===== [6/13] Discovery bundle ====="

    python -m scripts.build_discovery_bundle \
        --traversal "$RUN/traversal.json" \
        --traversal "$RUN/candidate_unit.traversal.a3.json" \
        --top-k 8 \
        --output "$RUN/discovery.bundle.a3.json"


    # DiscoveryBundle이 비면 alpha4는 의도적으로 canonical fallback을
    # 하지 않으므로 여기서 빠르게 확인한다.
    python - "$RUN/discovery.bundle.a3.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
n = len(d.get("inspirations", []))
print(f"Discovery inspirations: {n}")
if n == 0:
    raise SystemExit(
        "ERROR: no discovery inspiration survived. "
        "Do not continue into canonical hypothesis synthesis."
    )
PY


    # ============================================================
    # 7. Grounding + Discovery dual context
    # ============================================================

    echo
    echo "===== [7/13] Dual hypothesis context ====="

    python -m scripts.build_dual_hypothesis_context \
        --context "$RUN/hypothesis.context.json" \
        --discovery-bundle "$RUN/discovery.bundle.a3.json" \
        --output "$RUN/hypothesis.dual_context.a3.json"


    # ============================================================
    # 8. Alpha4 discovery-axis hypothesis synthesis
    #    + axis fidelity
    #    + internal novelty repair
    # ============================================================

    echo
    echo "===== [8/13] Discovery-axis hypothesis synthesis ====="

    python -m scripts.run_discovery_axis_hypothesis_maker \
        --dual-context "$RUN/hypothesis.dual_context.a3.json" \
        --model "$OPENROUTER_AGENT_MODEL" \
        --base-url "https://openrouter.ai/api/v1" \
        --api-key-env OPENROUTER_API_KEY \
        --max-axes 5 \
        --output-prefix "$RUN/hypothesis_axis_a4" \
        --save-prompts


    # ============================================================
    # 9. Semantic critic — pre external-novelty
    # ============================================================

    echo
    echo "===== [9/13] Semantic critic (alpha4) ====="

    python -m scripts.run_hypothesis_semantic_critic \
        --context "$RUN/hypothesis.context.json" \
        --portfolio "$RUN/hypothesis_axis_a4.portfolio.json" \
        --model "$OPENROUTER_CRITIC_MODEL" \
        --base-url "https://openrouter.ai/api/v1" \
        --api-key-env OPENROUTER_API_KEY \
        --output-prefix "$RUN/semantic_axis_a4" \
        --save-prompt


    # ============================================================
    # 10. Alpha5.1 external novelty
    #
    # 새로운 query이므로 alpha5 결과 reuse가 아니라
    # Semantic Scholar + Crossref를 fresh search한다.
    # ============================================================

    echo
    echo "===== [10/13] External novelty alpha5.1 ====="

    python -m scripts.run_external_novelty \
        --portfolio "$RUN/hypothesis_axis_a4.portfolio.json" \
        --lineage "$RUN/hypothesis_axis_a4.lineage.json" \
        --model "$OPENROUTER_CRITIC_MODEL" \
        --base-url "https://openrouter.ai/api/v1" \
        --api-key-env OPENROUTER_API_KEY \
        --providers semantic_scholar,crossref \
        --results-per-query 12 \
        --output-prefix "$RUN/external_novelty_a51" \
        --save-prompts


    # ============================================================
    # 11. Alpha6 targeted novelty refinement
    #
    # gap analysis
    # → targeted literature search
    # → at most one refinement
    # → grounding preservation
    # → axis fidelity
    # → internal novelty
    # → fresh external novelty re-check
    # ============================================================

    echo
    echo "===== [11/13] Targeted novelty refinement alpha6 ====="

    python -m scripts.run_novelty_refinement \
        --dual-context "$RUN/hypothesis.dual_context.a3.json" \
        --axis-plan "$RUN/hypothesis_axis_a4.axis_plan.json" \
        --portfolio "$RUN/hypothesis_axis_a4.portfolio.json" \
        --lineage "$RUN/hypothesis_axis_a4.lineage.json" \
        --external-report "$RUN/external_novelty_a51.report.json" \
        --external-query-plan "$RUN/external_novelty_a51.claims_queries.json" \
        --external-prior-art "$RUN/external_novelty_a51.prior_art.json" \
        --model "$OPENROUTER_AGENT_MODEL" \
        --critic-model "$OPENROUTER_CRITIC_MODEL" \
        --base-url "https://openrouter.ai/api/v1" \
        --api-key-env OPENROUTER_API_KEY \
        --output-prefix "$RUN/novelty_refinement_a6"


    # ============================================================
    # 최종 portfolio가 비었는지 확인
    # ============================================================

    FINAL_N=$(
        python - "$RUN/novelty_refinement_a6.portfolio.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(len(d.get("hypotheses", [])))
PY
    )

    echo "Final surviving hypotheses: $FINAL_N"

    if [ "$FINAL_N" -eq 0 ]; then
        echo
        echo "============================================================"
        echo "No hypothesis survived alpha6."
        echo "This is a valid fail-closed result."
        echo "Skipping final semantic critic / feasibility / demo viewer."
        echo "============================================================"
        return 0
    fi


    # ============================================================
    # 12. 최종 refined portfolio semantic critic
    # ============================================================

    echo
    echo "===== [12/13] Final semantic critic ====="

    python -m scripts.run_hypothesis_semantic_critic \
        --context "$RUN/hypothesis.context.json" \
        --portfolio "$RUN/novelty_refinement_a6.portfolio.json" \
        --model "$OPENROUTER_CRITIC_MODEL" \
        --base-url "https://openrouter.ai/api/v1" \
        --api-key-env OPENROUTER_API_KEY \
        --output-prefix "$RUN/semantic_final" \
        --save-prompt


    # ============================================================
    # 13. Feasibility + final hypothesis viewer
    # ============================================================

    echo
    echo "===== [13/13] Feasibility ====="

    python -m scripts.run_feasibility_e2e \
        --context "$RUN/hypothesis.context.json" \
        --portfolio "$RUN/novelty_refinement_a6.portfolio.json" \
        --semantic-review "$RUN/semantic_final.review.json" \
        --output-dir "$RUN/feasibility_final"


    echo
    echo "===== Building final demo viewer ====="

    python -m scripts.build_demo_viewer \
        --run-dir "$RUN" \
        --feasibility-dir "$RUN/feasibility_final" \
        --title "$TITLE"


    echo
    echo "################################################################"
    echo "# DONE"
    echo "# $QUESTION"
    echo "#"
    echo "# Viewer:"
    echo "# $RUN/demo/index.html"
    echo "################################################################"
}