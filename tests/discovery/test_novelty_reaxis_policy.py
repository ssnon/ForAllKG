from types import SimpleNamespace

from pipeline_core.discovery.novelty_refinement_contracts import (
    RefinementAttempt,
)
from pipeline_core.discovery.novelty_refinement_runtime import (
    TargetedNoveltyRefinementRuntime,
)


def _statement(
    statement_id: str,
    *,
    premise: bool = True,
    gap: bool = False,
    verify: bool = False,
    restrictions: list[str] | None = None,
    role: str = "reported",
):
    return SimpleNamespace(
        statement_id=statement_id,
        eligible_as_premise=premise,
        eligible_as_gap=gap,
        requires_verification=verify,
        premise_restrictions=list(restrictions or []),
        epistemic_role=role,
    )


def _dual(statements):
    return SimpleNamespace(
        grounded_context=SimpleNamespace(
            evidence_statements=list(statements),
        )
    )


def _original():
    return SimpleNamespace(
        premise_statement_ids=["stmt:p1"],
        gap_statement_ids=[],
        hypothesis_type="mechanistic_extension",
    )


def test_fresh_reaxis_safe_unused_premises_are_conservative():
    dual = _dual(
        [
            _statement("stmt:p1"),
            _statement("stmt:p2"),
            _statement(
                "stmt:verify",
                verify=True,
            ),
            _statement(
                "stmt:restricted",
                restrictions=["candidate_only"],
            ),
            _statement(
                "stmt:gap",
                premise=False,
                gap=True,
                role="unresolved",
            ),
            _statement(
                "stmt:nav",
                premise=True,
                role="navigation_note",
            ),
        ]
    )

    assert (
        TargetedNoveltyRefinementRuntime
        ._fresh_reaxis_safe_unused_premise_ids(
            dual,
            _original(),
        )
        == ["stmt:p2"]
    )


def test_fresh_reaxis_is_only_triggered_by_strong_known_axis_signal():
    runtime = TargetedNoveltyRefinementRuntime

    assert runtime._should_attempt_fresh_reaxis(
        "LITERATURE_SUPPORTED_EXTENSION",
        ["stmt:p2"],
    )

    assert runtime._should_attempt_fresh_reaxis(
        "WELL_ESTABLISHED",
        ["stmt:p2"],
    )

    assert not runtime._should_attempt_fresh_reaxis(
        "INSUFFICIENT_SEARCH_EVIDENCE",
        ["stmt:p2"],
    )

    assert not runtime._should_attempt_fresh_reaxis(
        "LITERATURE_SUPPORTED_EXTENSION",
        [],
    )


def test_fresh_reaxis_grounding_may_change_type_but_requires_unused_safe_premise():
    dual = _dual(
        [
            _statement("stmt:p1"),
            _statement("stmt:p2"),
            _statement(
                "stmt:g1",
                premise=False,
                gap=True,
                role="unresolved",
            ),
        ]
    )

    original = _original()

    valid = SimpleNamespace(
        premise_statement_ids=[
            "stmt:p1",
            "stmt:p2",
        ],
        gap_statement_ids=[],
        hypothesis_type="design_lever_interaction",
    )

    same_premises = SimpleNamespace(
        premise_statement_ids=[
            "stmt:p1",
        ],
        gap_statement_ids=[],
        hypothesis_type="context_dependency",
    )

    invented = SimpleNamespace(
        premise_statement_ids=[
            "stmt:p1",
            "stmt:invented",
        ],
        gap_statement_ids=[],
        hypothesis_type="context_dependency",
    )

    bad_gap = SimpleNamespace(
        premise_statement_ids=[
            "stmt:p1",
            "stmt:p2",
        ],
        gap_statement_ids=[
            "stmt:not_a_gap",
        ],
        hypothesis_type="context_dependency",
    )

    assert (
        TargetedNoveltyRefinementRuntime
        ._fresh_reaxis_grounding_valid(
            dual,
            original,
            valid,
        )
    )

    assert not (
        TargetedNoveltyRefinementRuntime
        ._fresh_reaxis_grounding_valid(
            dual,
            original,
            same_premises,
        )
    )

    assert not (
        TargetedNoveltyRefinementRuntime
        ._fresh_reaxis_grounding_valid(
            dual,
            original,
            invented,
        )
    )

    assert not (
        TargetedNoveltyRefinementRuntime
        ._fresh_reaxis_grounding_valid(
            dual,
            original,
            bad_gap,
        )
    )


def test_accepted_reaxis_has_explicit_generation_provenance():
    attempt = RefinementAttempt(
        original_hypothesis_id="hypothesis:old",
        candidate_hypothesis_id="hypothesis:new",
        gap_id="novelty_gap:g1",
        action="targeted_search_then_refine",
        decision="accepted_reaxis",
        original_external_status="LITERATURE_SUPPORTED_EXTENSION",
        targeted_external_status="LITERATURE_SUPPORTED_EXTENSION",
        final_external_status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        axis_fidelity_status="fresh_reaxis_context_bound",
        internal_novelty_status="corpus_supported_extension",
        grounding_preserved=False,
        refinement_generated=True,
        generation_mode="fresh_context_reaxis",
        context_grounding_valid=True,
        reason_codes=[
            "fresh_context_reaxis",
            "used_unused_eligible_premise",
        ],
        interpretation="accepted",
    )

    assert attempt.decision == "accepted_reaxis"
    assert attempt.generation_mode == "fresh_context_reaxis"
    assert attempt.context_grounding_valid is True
    assert attempt.grounding_preserved is False


def test_relational_gap_is_a_resolved_candidate_not_a_reaxis_trigger():
    runtime = TargetedNoveltyRefinementRuntime

    assert (
        "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
        in runtime.RESOLVED_CANDIDATE_EXTERNAL
    )

    assert (
        "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
        not in runtime.REAXIS_EXTERNAL
    )
