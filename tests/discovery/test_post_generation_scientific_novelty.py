from __future__ import annotations

from types import SimpleNamespace

import pipeline_core.discovery.post_generation_scientific_novelty as mod


class _FakeAnalyzer:
    def build(
        self,
        report,
        plan,
        packet,
    ):
        return SimpleNamespace(
            reviews=[
                SimpleNamespace(
                    hypothesis_id="hypothesis:test",
                )
            ],
        )


class _FakeAssembler:
    def build(
        self,
        scientific_review,
        external_card,
        packet,
    ):
        return SimpleNamespace(
            kind="normal",
        )

    def build_reference_validation_repair(
        self,
        *,
        original_prompt,
        previous_draft,
        issues,
    ):
        return SimpleNamespace(
            kind="repair",
        )


class _FakeBackend:
    backend_name = "fake-semantic"

    def __init__(
        self,
        tiers,
    ):
        self.tiers = list(tiers)

    def review(
        self,
        prompt,
        *,
        review_pass_index,
        debug_path=None,
    ):
        return SimpleNamespace(
            draft=SimpleNamespace(
                tier=self.tiers[
                    review_pass_index - 1
                ],
            ),
            requested_model="fake",
            served_model="fake",
        )


def _fake_compile(
    *,
    review_pass_index,
    draft,
    **kwargs,
):
    return SimpleNamespace(
        overall_tier=draft.tier,
        review_pass_index=review_pass_index,
    )


def _run(
    monkeypatch,
    *,
    external_status,
    tiers,
):
    monkeypatch.setattr(
        mod,
        "ScientificDistinctivenessAnalyzer",
        _FakeAnalyzer,
    )

    monkeypatch.setattr(
        mod,
        "SemanticDistinctivenessPromptAssembler",
        _FakeAssembler,
    )

    monkeypatch.setattr(
        mod,
        "compile_semantic_distinctiveness_review",
        _fake_compile,
    )

    report = SimpleNamespace(
        cards=[
            SimpleNamespace(
                hypothesis_id="hypothesis:test",
                status=external_status,
            )
        ],
    )

    return (
        mod.evaluate_post_generation_scientific_novelty(
            hypothesis_id="hypothesis:test",
            report=report,
            plan=SimpleNamespace(),
            packet=SimpleNamespace(),
            backend=_FakeBackend(tiers),
        )
    )


def test_indeterminate_insufficient_search_is_ineligible(
    monkeypatch,
):
    result = _run(
        monkeypatch,
        external_status=(
            "INSUFFICIENT_SEARCH_EVIDENCE"
        ),
        tiers=[
            "INDETERMINATE",
            "INDETERMINATE",
        ],
    )

    assert (
        result.action_decision.action
        == "EVIDENCE_REQUIRED"
    )

    assert (
        result.action_decision.selection_class
        == "INELIGIBLE"
    )


def test_high_relational_gap_is_eligible(
    monkeypatch,
):
    result = _run(
        monkeypatch,
        external_status=(
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
        ),
        tiers=[
            "HIGH",
            "HIGH",
        ],
    )

    assert (
        result.action_decision.action
        == "KEEP_ELIGIBLE"
    )

    assert (
        result.action_decision.selection_class
        == "ELIGIBLE"
    )
