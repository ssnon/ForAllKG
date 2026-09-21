from __future__ import annotations

from pipeline_core.discovery.reframing.response_semantics_probe import (
    build_response_semantics_probe,
)


def _statement(
    statement_id: str,
    text: str,
    *,
    role: str = "premise",
    frozen: bool = False,
    paper_ids: list[str] | None = None,
):
    return {
        "statement_id": statement_id,
        "role": role,
        "text": text,
        "paper_ids": paper_ids or ["P1"],
        "frozen_trigger_response_cue": {
            "matched": frozen,
            "matches": ["Raman"] if frozen else [],
        },
    }


def _payload(statements):
    return {
        "pack_id": "diag:1",
        "validation_domain_label": "dac_her",
        "tasks": [
            {
                "task_id": "task:1",
                "statement_traces": statements,
            }
        ],
    }


def test_generic_activity_is_response_bearing():
    report = build_response_semantics_probe(
        _payload([
            _statement(
                "s1",
                "The catalyst showed more than 50-fold greater hydrogen-evolution activity than Pt/C.",
            )
        ])
    )
    assert report.candidate_response_match_count == 1
    assert report.statements[0].trigger_eligible_response is True


def test_raman_alone_is_measurement_not_performance_for_dac_her():
    report = build_response_semantics_probe(
        _payload([
            _statement(
                "s1",
                "The proposal is based on Pt-H and Ru-O Raman signals.",
                frozen=True,
            )
        ])
    )
    row = report.statements[0]
    assert row.v1_frozen_response_matched is True
    assert row.trigger_eligible_response is False
    assert any(match.role == "measurement_observable" for match in row.response_matches)


def test_volcano_relation_is_direct_nonmonotonic_signal():
    report = build_response_semantics_probe(
        _payload([
            _statement(
                "s1",
                "HER activity follows a volcano relationship with the Gibbs free energy of hydrogen adsorption.",
            )
        ])
    )
    kinds = {signal.kind for signal in report.statements[0].candidate_response_law_signals}
    assert "VOLCANO_NON_MONOTONIC_RESPONSE" in kinds
    assert report.candidate_law_signal_counts["VOLCANO_NON_MONOTONIC_RESPONSE"] == 1


def test_domain_parameter_adapter_detects_hydrogen_adsorption_free_energy():
    report = build_response_semantics_probe(
        _payload([
            _statement(
                "s1",
                "HER activity follows a volcano relationship with the Gibbs free energy of hydrogen adsorption.",
            )
        ])
    )
    assert any(
        match.source == "domain_adapter"
        and match.role == "domain_parameter"
        for match in report.statements[0].parameter_matches
    )


def test_current_latent_floor_remains_structural_and_is_not_relaxed():
    report = build_response_semantics_probe(
        _payload([
            _statement("s1", "HER activity was greater than the reference.", paper_ids=["P1"]),
            _statement(
                "s2",
                "HER activity follows a volcano relationship with free energy.",
                paper_ids=["P2"],
            ),
            _statement(
                "g1",
                "The packet does not establish how interaction tuning jointly determines HER activity.",
                role="gap",
                paper_ids=["P1", "P2"],
            ),
        ])
    )
    assert report.candidate_latent_response_support_premise_count == 2
    assert report.candidate_latent_support_paper_count == 2
    assert report.candidate_latent_current_floor_met is False


def test_three_response_premises_from_two_papers_meet_probe_floor_only():
    report = build_response_semantics_probe(
        _payload([
            _statement("s1", "HER activity was greater than the reference.", paper_ids=["P1"]),
            _statement("s2", "Catalytic activity increased.", paper_ids=["P2"]),
            _statement("s3", "Performance improved.", paper_ids=["P2"]),
        ])
    )
    assert report.candidate_latent_current_floor_met is True
    assert report.actual_trigger_decision_changed is False


def test_report_marks_current_task_as_adaptation_not_validation():
    report = build_response_semantics_probe(
        _payload([_statement("s1", "HER activity increased.")])
    )
    assert report.adaptation_probe is True
    assert report.eligible_as_untouched_validation_after_design_use is False
    assert report.generalization_claim_established is False


def test_frozen_semantics_are_not_modified_by_probe_contract():
    report = build_response_semantics_probe(
        _payload([_statement("s1", "HER activity increased.")])
    )
    assert report.frozen_trigger_semantics_modified is False
    assert report.frozen_prompt_semantics_modified is False
    assert report.threshold_tuning_performed is False
    assert report.llm_calls_performed == 0
