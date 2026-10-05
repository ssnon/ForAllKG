from pipeline_core.discovery.adaptive_yield_evaluation import (
    BASELINE_COMMIT,
    CONDITION_ORDER,
    AdaptiveYieldCaseAudit,
    AdaptiveYieldStageMetrics,
    build_blind_packet,
    normalize_card,
    scientific_fingerprint,
)


def _card(hid: str, statement: str):
    return {
        "hypothesis_id": hid,
        "source_context_id": "ctx",
        "source_context_sha256": "sha",
        "title": "A title",
        "hypothesis_statement": statement,
        "hypothesis_type": "context_dependency",
        "premise_statement_ids": ["s1"],
        "inferential_bridge": "A bounded bridge",
        "predicted_observations": [
            {
                "observable": "y",
                "expected_direction": "unspecified",
                "rationale": "test",
            }
        ],
        "falsification_criteria": [
            {
                "observable": "y",
                "falsifying_outcome": "no relation",
            }
        ],
        "assumptions": [],
        "source_paper_ids": ["p1"],
    }


def test_scientific_fingerprint_ignores_hypothesis_identity():
    a = _card("h1", "same science")
    b = _card("h2", "same science")
    assert scientific_fingerprint(a) == scientific_fingerprint(b)


def test_marginal_record_uses_scientific_fingerprint():
    card = _card("h2", "same science")
    fp = scientific_fingerprint(card)
    row = normalize_card(
        condition="LOCAL_ADAPTIVE_776",
        card=card,
        portfolio_path="/tmp/p.json",
        premise_text_by_id={"s1": "premise"},
        previous_fingerprints={fp},
    )
    assert row.marginal_from_previous_condition is False


def test_blind_packet_hides_condition_labels_from_candidate_surface():
    rows = []
    previous = set()
    for condition, statement in zip(CONDITION_ORDER, ["a", "b", "c"]):
        card = _card("h-" + condition, statement)
        row = normalize_card(
            condition=condition,
            card=card,
            portfolio_path=f"/tmp/{condition}.json",
            premise_text_by_id={"s1": "premise"},
            previous_fingerprints=previous,
        )
        rows.append(row)
        previous.add(row.scientific_fingerprint)

    stages = [
        AdaptiveYieldStageMetrics(
            condition=condition,
            status="COMPLETE",
            cumulative_effective_count=1,
            marginal_effective_count=1,
            unresolved_count=0,
            graph_handoff_count=0,
            graph_retraversal_count=0,
            new_context_count=0,
            structurally_new_premise_count=0,
            llm_call_artifact_count=0,
            retrieval_report_artifact_count=0,
        )
        for condition in CONDITION_ORDER
    ]
    audit = AdaptiveYieldCaseAudit(
        case_id="case",
        run_dir="/tmp/run",
        baseline_commit=BASELINE_COMMIT,
        question="question",
        domain_profile_id="sers_au_ag",
        stages=stages,
        candidates=rows,
    )
    packet, key = build_blind_packet(audit)
    text = packet.model_dump_json()
    assert "CLOSED_LOOP_775" not in text
    assert "LOCAL_ADAPTIVE_776" not in text
    assert "GRAPH_ADAPTIVE_777" not in text
    assert key.case_id == "case"


def test_no_quality_score_or_production_authority():
    stage = AdaptiveYieldStageMetrics(
        condition="CLOSED_LOOP_775",
        status="COMPLETE",
        cumulative_effective_count=0,
        marginal_effective_count=0,
        unresolved_count=0,
        graph_handoff_count=0,
        graph_retraversal_count=0,
        new_context_count=0,
        structurally_new_premise_count=0,
        llm_call_artifact_count=0,
        retrieval_report_artifact_count=0,
    )
    assert stage.composite_quality_score_computed is False
    assert stage.candidate_count_is_quality_signal is False
