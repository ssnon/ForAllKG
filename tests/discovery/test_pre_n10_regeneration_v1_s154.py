from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisPolicy,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftGeneration
from pipeline_core.discovery.pre_n10_regeneration_v1 import (
    execute_pre_n10_regeneration_v1,
)
from pipeline_core.discovery.pre_n10_source_alignment_primary_v1 import (
    PreN10SourceAlignmentHypothesisResultV1,
    PreN10SourceAlignmentPrimaryReportV1,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    build_regeneration_unit_v2_freeze,
)


def _sha(value):
    import hashlib
    import json
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _context() -> HypothesisContext:
    return HypothesisContext(
        context_id="context:1",
        context_sha256="a" * 64,
        source_packet_id="packet:1",
        source_packet_sha256="b" * 64,
        source_report_id="report:1",
        source_report_sha256="c" * 64,
        task_id="task:1",
        question="How are X and Y related?",
        corpus_id="corpus:1",
        domain_profile_id="domain:1",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="statement:1",
                text="X is associated with Y.",
                epistemic_role="reported",
                claim_kind="reported_relation",
                paper_ids=["paper:1"],
                eligible_as_premise=True,
            )
        ],
        policy=HypothesisPolicy(),
    )


def _draft() -> HypothesisPortfolioDraft:
    return HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id="h1",
                title="X-Y relation",
                hypothesis_statement="X modulates Y.",
                hypothesis_type="mechanistic_extension",
                premise_statement_ids=["statement:1"],
                inferential_bridge=(
                    "The reported association may reflect modulation."
                ),
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p1",
                        observable="Y response",
                        expected_direction="qualitative_change",
                        rationale="Tests the proposed relation.",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f1",
                        observable="Y response",
                        falsifying_outcome=(
                            "Y response is unchanged across X."
                        ),
                    )
                ],
            )
        ]
    )


class _Backend:
    backend_name = "test_backend"
    model_name = "test_model"

    def __init__(self, draft):
        self.draft = draft
        self.generate_calls = 0
        self.repair_calls = 0

    def generate(self, prompt):
        self.generate_calls += 1
        return HypothesisDraftGeneration(
            draft=self.draft,
            input_tokens=10,
            output_tokens=20,
            response_id="response:1",
        )

    def repair(self, prompt, previous_draft, feedback):
        self.repair_calls += 1
        raise AssertionError("pre-N10 regeneration must never repair")


def _primary(*, fallback: bool = True):
    hypothesis = PreN10SourceAlignmentHypothesisResultV1(
        hypothesis_id="hypothesis:source",
        source_contract_status="REQUIRES_PRE_N10_INTERVENTION",
        source_alignment_results=[],
        post_contract_status=(
            "REQUIRES_PRE_N10_INTERVENTION"
            if fallback
            else "READY_FOR_N10"
        ),
        recovered_for_n10=not fallback,
        regeneration_fallback_required=fallback,
    )
    body = {
        "schema_version":
            "pre-n10-source-alignment-primary-report-v1",
        "source_contract_report_id": "contract:1",
        "source_contract_report_sha256": "d" * 64,
        "source_query_plan_id": "plan:1",
        "source_query_plan_sha256": "e" * 64,
        "aligned_query_plan_id": "plan:2",
        "aligned_query_plan_sha256": "f" * 64,
        "post_contract_report_id": "contract:2",
        "post_contract_report_sha256": "1" * 64,
        "hypotheses": [hypothesis.model_dump(mode="json")],
        "hypothesis_count": 1,
        "materialized_alignment_count": 0,
        "unavailable_alignment_count": 0,
        "ambiguous_alignment_count": 0,
        "semantic_reject_count": 0,
        "audit_failure_count": 0,
        "recovered_for_n10_count": int(not fallback),
        "regeneration_fallback_required_count": int(fallback),
        "audit_llm_call_count": 0,
        "claim_generation_llm_calls": 0,
        "retrieval_performed": False,
        "novelty_assessment_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "regeneration_performed": False,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha(body)
    return PreN10SourceAlignmentPrimaryReportV1(
        **body,
        report_id=(
            "pre_n10_source_alignment_primary_v1:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


def test_pre_n10_regeneration_uses_single_existing_unit_and_no_repair(
    tmp_path: Path,
):
    backend = _Backend(_draft())
    freeze = build_regeneration_unit_v2_freeze(
        repository_head_sha="2" * 40,
        repository_tracked_worktree_dirty=False,
    )

    report, raw = execute_pre_n10_regeneration_v1(
        context=_context(),
        primary_report=_primary(),
        unit_freeze=freeze,
        backend_factory=lambda _source, _dir: backend,
        output_root=tmp_path / "regen",
    )

    assert report.regeneration_required_count == 1
    assert report.generation_call_count == 1
    assert report.repair_call_count == 0
    assert report.generated_and_compiled_count == 1
    assert backend.generate_calls == 1
    assert backend.repair_calls == 0
    assert raw["hypothesis:source"].status == "GENERATED_AND_COMPILED"
    assert report.previous_hypothesis_text_consumed is False
    assert report.n10_outcome_consumed is False


def test_pre_n10_regeneration_writes_separate_portfolio_for_downstream(
    tmp_path: Path,
):
    backend = _Backend(_draft())
    freeze = build_regeneration_unit_v2_freeze(
        repository_head_sha="2" * 40,
        repository_tracked_worktree_dirty=False,
    )

    report, _raw = execute_pre_n10_regeneration_v1(
        context=_context(),
        primary_report=_primary(),
        unit_freeze=freeze,
        backend_factory=lambda _source, _dir: backend,
        output_root=tmp_path / "regen",
    )

    row = report.lineages[0]
    assert row.regenerated_portfolio_path is not None
    assert Path(row.regenerated_portfolio_path).is_file()
    assert row.regenerated_hypothesis_count == 1
    assert report.semantic_critic_performed is False
    assert report.pre_n10_contract_reentry_performed is False


def test_pre_n10_regeneration_refuses_when_primary_recovered():
    freeze = build_regeneration_unit_v2_freeze(
        repository_head_sha="2" * 40,
        repository_tracked_worktree_dirty=False,
    )

    try:
        execute_pre_n10_regeneration_v1(
            context=_context(),
            primary_report=_primary(fallback=False),
            unit_freeze=freeze,
            backend_factory=lambda _source, _dir: _Backend(_draft()),
            output_root=Path("/tmp/unused-pre-n10-regeneration-test"),
        )
    except ValueError as exc:
        assert "at least one fallback lineage" in str(exc)
    else:
        raise AssertionError("expected fail-closed no-fallback rejection")
