from pipeline_core.discovery.reframing.evidence_tension import (
    EvidenceLevelTensionWitness,
    ScientificEvidenceTensionReport,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ConditionEvidenceExample,
    ReframeEvidenceStatement,
    ScientificReframeEvidencePacket,
)
from pipeline_core.discovery.reframing.review_evidence import (
    build_task_trigger_review_evidence,
    build_trigger_review_evidence_pack,
)
from pipeline_core.discovery.reframing.trigger_calibration import (
    TriggerReviewItem,
    TriggerReviewTemplate,
)


def _evidence():
    return ScientificReframeEvidencePacket(
        task_id="task:1",
        question="Why does A differ across contexts?",
        source_context_id="ctx:1",
        source_context_sha256="sha",
        premise_statements=[
            ReframeEvidenceStatement(
                statement_id="s1",
                text="A increases in context X.",
                epistemic_role="reported",
                claim_kind="observation",
                paper_ids=["p1"],
            ),
            ReframeEvidenceStatement(
                statement_id="s2",
                text="A decreases in context Y through mechanism M.",
                epistemic_role="reported",
                claim_kind="mechanism",
                paper_ids=["p2"],
            ),
        ],
        gap_statements=[
            ReframeEvidenceStatement(
                statement_id="g1",
                text="The controlling context variable is unresolved.",
                epistemic_role="unresolved",
                claim_kind="scope_limit",
                paper_ids=["p1", "p2"],
            )
        ],
        condition_examples=[
            ConditionEvidenceExample(
                object_kind="measurement",
                paper_id="p1",
                chunk_id="c1",
                object_id="m1",
                label="signal",
                conditions=[{"name": "temperature", "value": 20}],
            ),
            ConditionEvidenceExample(
                object_kind="experiment",
                paper_id="p2",
                chunk_id="c2",
                object_id="e1",
                label="experiment",
                conditions=[{"name": "temperature", "value": 40}],
            ),
        ],
        grounded_source_chunk_count=4,
        unresolved_grounded_node_count=0,
    )


def _tensions():
    return ScientificEvidenceTensionReport(
        report_id="tensions:1",
        source_task_id="task:1",
        source_context_id="ctx:1",
        source_context_sha256="sha",
        source_explorer_report_sha256="ersha",
        source_evidence_sha256="evsha",
        extraction_mode="packet_assisted",
        witnesses=[
            EvidenceLevelTensionWitness(
                witness_id="w1",
                source_tension_id="source-w1",
                source_tension_type="context_dependency",
                focal_statement_id="g1",
                side_a_statement_ids=["s1"],
                side_b_statement_ids=["s2"],
                grounded_statement_ids=["g1", "s1", "s2"],
                paper_ids=["p1", "p2"],
                claim_kinds=["mechanism", "observation", "scope_limit"],
                tension_types=["CONTEXT", "DIRECTIONAL", "MECHANISTIC"],
                classification_bases=["test basis"],
                relevant_condition_signature_count=2,
                relevant_condition_names=["temperature"],
                independence_basis="cross_paper",
                independent_family_signal=True,
            )
        ],
        tension_type_counts={"CONTEXT": 1, "DIRECTIONAL": 1, "MECHANISTIC": 1},
    )


def _rows():
    common = dict(
        task_key="K01/replicate_01",
        case_key="K01",
        question="Why does A differ across contexts?",
        readiness_status="ready_now",
        tension_types=["CONTEXT", "DIRECTIONAL", "MECHANISTIC"],
        tension_witness_count=1,
        condition_signature_count=2,
        grounded_source_chunk_count=4,
    )
    return [
        TriggerReviewItem(
            **common,
            operator_id="LATENT_VARIABLE",
            trigger_decision="triggered",
            trigger_signal=True,
            readiness_trigger_alignment="ready_and_triggered",
        ),
        TriggerReviewItem(
            **common,
            operator_id="REGIME_BOUNDARY",
            trigger_decision="insufficient_trigger_evidence",
            trigger_signal=False,
            readiness_trigger_alignment="ready_not_triggered",
        ),
    ]


def test_review_pack_preserves_full_grounded_statement_text():
    task = build_task_trigger_review_evidence(
        template_rows=_rows(), evidence=_evidence(), tensions=_tensions()
    )
    assert [row.text for row in task.premise_statements] == [
        "A increases in context X.",
        "A decreases in context Y through mechanism M.",
    ]
    assert task.gap_statements[0].text.startswith("The controlling")


def test_tension_sides_recover_exact_grounded_statements():
    task = build_task_trigger_review_evidence(
        template_rows=_rows(), evidence=_evidence(), tensions=_tensions()
    )
    witness = task.tension_witnesses[0]
    assert witness.side_a_statements[0].statement_id == "s1"
    assert witness.side_b_statements[0].statement_id == "s2"
    assert witness.focal_statement.statement_id == "g1"


def test_pair_local_condition_signal_is_exposed_without_regime_authority():
    task = build_task_trigger_review_evidence(
        template_rows=_rows(), evidence=_evidence(), tensions=_tensions()
    )
    witness = task.tension_witnesses[0]
    assert witness.relevant_condition_signature_count == 2
    assert witness.relevant_condition_names == ["temperature"]
    assert witness.regime_boundary_authority is False


def test_task_condition_summary_is_deterministic():
    task = build_task_trigger_review_evidence(
        template_rows=_rows(), evidence=_evidence(), tensions=_tensions()
    )
    assert task.condition_summary.example_count == 2
    assert task.condition_summary.distinct_signature_count == 2
    assert task.condition_summary.paper_count == 2
    assert task.condition_summary.condition_names == ["temperature"]


def test_operator_review_questions_are_operator_specific_and_unlabeled():
    task = build_task_trigger_review_evidence(
        template_rows=_rows(), evidence=_evidence(), tensions=_tensions()
    )
    by_id = {row.operator_id: row for row in task.operator_reviews}
    assert "hidden-variable" in by_id["LATENT_VARIABLE"].review_question
    assert "qualitative change" in by_id["REGIME_BOUNDARY"].review_question
    assert by_id["LATENT_VARIABLE"].expert_label is None
    assert by_id["REGIME_BOUNDARY"].expert_rationale is None


def test_current_trigger_is_explicitly_not_ground_truth():
    task = build_task_trigger_review_evidence(
        template_rows=_rows(), evidence=_evidence(), tensions=_tensions()
    )
    latent = {row.operator_id: row for row in task.operator_reviews}["LATENT_VARIABLE"]
    assert latent.current_trigger_signal is True
    assert latent.current_trigger_is_not_ground_truth is True
    assert latent.reviewer_should_ignore_current_decision_when_labeling is True


def test_pack_counts_tasks_and_operator_rows():
    template = TriggerReviewTemplate(source_audit_id="audit:1", rows=_rows())
    pack = build_trigger_review_evidence_pack(
        template=template,
        benchmark_root="/benchmark",
        extraction_roots=["/corpus"],
        task_evidence={"K01/replicate_01": (_evidence(), _tensions())},
    )
    assert pack.task_count == 1
    assert pack.operator_review_row_count == 2
    assert pack.labels_prefilled is False
    assert pack.current_trigger_is_ground_truth is False


def test_pack_rejects_missing_task_evidence():
    template = TriggerReviewTemplate(source_audit_id="audit:1", rows=_rows())
    try:
        build_trigger_review_evidence_pack(
            template=template,
            benchmark_root="/benchmark",
            extraction_roots=["/corpus"],
            task_evidence={},
        )
    except ValueError as exc:
        assert "missing grounded review evidence" in str(exc)
    else:
        raise AssertionError("missing task evidence must fail closed")
