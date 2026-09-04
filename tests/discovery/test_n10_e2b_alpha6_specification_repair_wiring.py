from pathlib import Path

from pipeline_core.discovery.n10_alpha6_resolution_policy import (
    REFINE_NOVELTY_BEARING_SPECIFICATION,
    alpha6_resolution_directive_from_gate_row,
)


RUNTIME = Path(
    "pipeline_core/discovery/"
    "novelty_refinement_runtime.py"
)


def _directive(action=REFINE_NOVELTY_BEARING_SPECIFICATION):
    return alpha6_resolution_directive_from_gate_row(
        {
            "selection_class": "CONDITIONAL",
            "fallback_allowed": False,
            "positive_nonobviousness_authority": False,
            "base_aggregation_action": action,
        }
    )


def test_specification_repair_directive_is_active():
    directive = _directive()

    assert directive.force_bounded_refinement is True
    assert (
        directive.use_source_external_without_targeted_search
        is True
    )
    assert (
        directive.bypass_resolved_candidate_external_exit
        is True
    )


def test_other_actions_remain_inert():
    for action in [
        "RESOLVE_NOVELTY_BEARING_PRIOR_ART_RELATION",
        "RESOLVE_NOVELTY_BEARING_REFINEMENT_STATE",
        "RESOLVE_NOVELTY_BEARING_EVIDENCE",
        "REMOVE_OR_REAXIS_ROUTINE_NOVELTY_BRANCH",
    ]:
        assert (
            _directive(action).force_bounded_refinement
            is False
        )


def test_frozen_keep_branch_surface_is_preserved():
    text = RUNTIME.read_text(encoding="utf-8")

    assert 'if gap.action == "keep":' in text


def test_frozen_no_query_branch_surface_is_preserved():
    text = RUNTIME.read_text(encoding="utf-8")

    assert "if not gap.targeted_queries:" in text


def test_runtime_derives_directive_only_for_v2():
    text = RUNTIME.read_text(encoding="utf-8")

    pos = text.index(
        "n10_resolution_directive = ("
    )

    window = text[pos:pos + 1800]

    assert (
        "alpha6_resolution_directive_from_gate_row"
        in window
    )
    assert (
        "scientific-novelty-fallback-"
        in window
    )
    assert "gate-v2" in window
    assert (
        "else NO_N10_ALPHA6_OVERRIDE"
        in window
    )


def test_no_query_repair_reuses_source_external():
    text = RUNTIME.read_text(encoding="utf-8")

    pos = text.index(
        "targeted_card = source_external"
    )

    window = text[
        max(0, pos - 1200):
        pos + 500
    ]

    assert (
        "use_source_external_without_targeted_search"
        in window
    )

    assert (
        "does not upgrade novelty or add evidence"
        in window
    )


def test_real_targeted_retrieval_is_preserved():
    text = RUNTIME.read_text(encoding="utf-8")

    assert (
        "self.targeted_retriever.retrieve("
        in text
    )

    assert (
        "self.external_assessor.assess("
        in text
    )


def test_resolved_exit_has_n10_nested_guard():
    text = RUNTIME.read_text(encoding="utf-8")

    pos = text.index(
        "in self.RESOLVED_CANDIDATE_EXTERNAL"
    )

    window = text[pos:pos + 1000]

    assert (
        "bypass_resolved_candidate_external_exit"
        in window
    )

    assert (
        "force_bounded_refinement"
        in window
    )
