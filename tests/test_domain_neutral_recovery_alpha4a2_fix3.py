from __future__ import annotations

from dac_her.micro_reextract_prompts import (
    build_domain_gate_recovery_prompt,
    build_micro_reextract_prompt,
)
from dac_her.recovery_policy import (
    RecoveryAction,
    decide_recovery,
    has_common_undefined_endpoint_cluster,
)
from dac_her.strict_recovery import _is_reserved_collection_domain_violation
from dac_her.validation_issues import (
    IssueCode,
    IssueStage,
    ValidationReport,
    issue,
)


def _undefined_source_report(count: int = 5) -> ValidationReport:
    issues = [
        issue(
            code=IssueCode.UNDEFINED_EDGE_SOURCE,
            stage=IssueStage.STRUCTURAL,
            message=f"Edge {index} has undefined source.",
            edge_index=index,
            source_id="synthesis_sio2_au_ag",
            target_id=f"target_{index}",
            relation="USES_MATERIAL",
        )
        for index in range(count)
    ]
    return ValidationReport.from_issues(issues)


def _decision(report: ValidationReport, *, micro_attempts: int):
    return decide_recovery(
        report=report,
        normalization_attempted=True,
        patch_attempts=2,
        micro_reextract_attempts=micro_attempts,
        post_micro_patch_attempts=0,
        split_depth=2,
        source_tokens=309,
        max_patch_attempts=2,
        max_micro_reextract_attempts=1,
        max_post_micro_patch_attempts=1,
        micro_reextract_max_source_tokens=900,
        max_split_depth=2,
        min_rechunk_source_tokens=400,
        isolated_rechunk_threshold=3,
        issue_family_rechunk_threshold=3,
        undefined_endpoint_rechunk_threshold=4,
    )


def test_reserved_collection_violation_detection():
    error = ValueError(
        "Extraction-domain vocabulary violation [sers_au_ag]: "
        "entity types outside domain vocabulary: Experiment; "
        "reserved structured node type(s) Experiment must use their "
        "dedicated top-level collection, not entities[]"
    )
    assert _is_reserved_collection_domain_violation(error)
    assert not _is_reserved_collection_domain_violation(
        ValueError(
            "Extraction-domain vocabulary violation [x]: "
            "relation types outside domain vocabulary: MADE_OF"
        )
    )


def test_domain_gate_recovery_prompt_is_targeted_and_non_speculative():
    prompt = build_domain_gate_recovery_prompt(
        paper_id="P",
        chunk_id="P:main:c",
        document_id="main",
        document_role="main",
        section="whole",
        page_ids=[],
        asset_ids=[],
        core_text="A SERS measurement was performed.",
        left_context="",
        right_context="",
        asset_context="",
        rejected_graph_payload={
            "entities": [{"id": "exp1", "type": "Experiment"}],
            "experiments": [],
            "edges": [],
        },
        domain_error=(
            "reserved structured node type(s) Experiment must use "
            "their dedicated top-level collection, not entities[]"
        ),
    )
    assert "DOMAIN_GATE_ERROR" in prompt
    assert "dedicated top-level collection" in prompt
    assert "do not invent" in prompt.lower()
    assert "PREVIOUS_DOMAIN_REJECTED_GRAPH_JSON" in prompt


def test_common_undefined_endpoint_cluster_detected():
    report = _undefined_source_report()
    assert has_common_undefined_endpoint_cluster(report)


def test_common_undefined_endpoint_gets_one_targeted_second_micro():
    decision = _decision(_undefined_source_report(), micro_attempts=1)
    assert decision.action == RecoveryAction.MICRO_REEXTRACT
    assert "common undefined endpoint" in decision.reason.lower()


def test_second_targeted_micro_is_bounded():
    decision = _decision(_undefined_source_report(), micro_attempts=2)
    assert decision.action == RecoveryAction.SEMANTIC_PATCH


def test_micro_prompt_surfaces_common_missing_endpoint_hint():
    report = _undefined_source_report()
    prompt = build_micro_reextract_prompt(
        paper_id="P",
        chunk_id="P:main:c",
        document_id="main",
        document_role="main",
        section="whole",
        page_ids=[],
        asset_ids=[],
        core_text="Gold chloride and AgNO3 were used in synthesis.",
        left_context="",
        right_context="",
        asset_context="",
        graph_payload={
            "entities": [],
            "experiments": [],
            "calculations": [],
            "measurements": [],
            "measurement_groups": [],
            "observation_claims": [],
            "mechanism_claims": [],
            "edges": [],
        },
        report=report,
    )
    assert "COMMON_UNDEFINED_ENDPOINT_RECOVERY_HINTS" in prompt
    assert "synthesis_sio2_au_ag" in prompt
    assert "Do not preserve dangling edges" in prompt
