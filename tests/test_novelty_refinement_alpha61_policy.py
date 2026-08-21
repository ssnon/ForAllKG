from types import SimpleNamespace

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from pipeline_core.discovery.novelty_refinement_runtime import TargetedNoveltyRefinementRuntime


def _draft():
    return HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id="h1",
                title="refined",
                hypothesis_statement="refined statement",
                hypothesis_type="context_dependency",
                premise_statement_ids=["hallucinated:p"],
                gap_statement_ids=["hallucinated:g"],
                inferential_bridge="bridge",
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p1",
                        observable="obs",
                        expected_direction="shift",
                        rationale="why",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f1",
                        observable="obs",
                        falsifying_outcome="no shift",
                    )
                ],
            )
        ],
        abstention_reason=None,
    )


def test_refinement_provenance_is_deterministically_locked():
    original = SimpleNamespace(
        premise_statement_ids=["stmt:p1", "stmt:p2"],
        gap_statement_ids=["stmt:g1"],
        hypothesis_type="mechanistic_extension",
    )
    locked = TargetedNoveltyRefinementRuntime._lock_refinement_provenance(
        original, _draft()
    )
    row = locked.hypotheses[0]
    assert row.premise_statement_ids == ["stmt:p1", "stmt:p2"]
    assert row.gap_statement_ids == ["stmt:g1"]
    assert row.hypothesis_type == "mechanistic_extension"


def test_optional_refinement_failure_is_non_destructive_for_unresolved_or_extension():
    assert TargetedNoveltyRefinementRuntime._original_fallback_allowed(
        "INSUFFICIENT_SEARCH_EVIDENCE"
    )
    assert TargetedNoveltyRefinementRuntime._original_fallback_allowed(
        "LITERATURE_SUPPORTED_EXTENSION"
    )
    assert not TargetedNoveltyRefinementRuntime._original_fallback_allowed(
        "WELL_ESTABLISHED"
    )
    assert not TargetedNoveltyRefinementRuntime._original_fallback_allowed(
        "CONFLICTING_PRIOR_ART"
    )
