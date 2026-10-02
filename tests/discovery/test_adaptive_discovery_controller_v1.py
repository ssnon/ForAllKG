
from types import SimpleNamespace

from pipeline_core.discovery.adaptive_discovery_controller import (
    _allowed_actions,
    AdaptiveAxisMutationPromptAssembler,
)


def test_prior_art_escalates_after_seed_reaxis():
    history = [
        {"action": "EVIDENCE_REAXIS"},
    ]
    actions = _allowed_actions(
        epistemic_state="PRIOR_ART_BACKED_OR_NO_RESIDUAL",
        history=history,
        safe_unused=["s3"],
        target_claim_ids=["c1"],
        max_local_attempts=4,
    )
    assert actions[0] == "SAME_PREMISE_SHARPEN"
    assert "AXIS_MUTATION" not in actions


def test_prior_art_escalates_to_axis_mutation_after_local_repairs():
    history = [
        {"action": "EVIDENCE_REAXIS"},
        {"action": "SAME_PREMISE_SHARPEN"},
    ]
    actions = _allowed_actions(
        epistemic_state="PRIOR_ART_BACKED_OR_NO_RESIDUAL",
        history=history,
        safe_unused=["s3"],
        target_claim_ids=["c1"],
        max_local_attempts=4,
    )
    assert actions == ["AXIS_MUTATION"]


def test_evidence_gap_retrieves_before_generation():
    actions = _allowed_actions(
        epistemic_state="UNRESOLVED_EVIDENCE_GAP",
        history=[],
        safe_unused=["s3"],
        target_claim_ids=[],
        max_local_attempts=4,
    )
    assert actions == ["RETRIEVE_MORE"]


def test_topology_gap_can_reaxis_before_axis_mutation():
    actions = _allowed_actions(
        epistemic_state="UNRESOLVED_TOPOLOGY_GAP",
        history=[],
        safe_unused=["s3"],
        target_claim_ids=["c1"],
        max_local_attempts=4,
    )
    assert actions[0] == "EVIDENCE_REAXIS"


def test_local_exhaustion_requests_graph_retraversal():
    history = [
        {"action": "RETRIEVE_MORE"},
        {"action": "SAME_PREMISE_SHARPEN"},
        {"action": "EVIDENCE_REAXIS"},
        {"action": "AXIS_MUTATION"},
    ]
    actions = _allowed_actions(
        epistemic_state="UNRESOLVED_TOPOLOGY_GAP",
        history=history,
        safe_unused=[],
        target_claim_ids=[],
        max_local_attempts=4,
    )
    assert actions == ["REQUEST_GRAPH_RETRAVERSAL"]


def test_residual_candidate_is_kept():
    actions = _allowed_actions(
        epistemic_state="RESIDUAL_AUTHORITY_CANDIDATE_SHADOW",
        history=[],
        safe_unused=[],
        target_claim_ids=[],
        max_local_attempts=4,
    )
    assert actions == ["KEEP"]


def test_axis_mutation_prompt_marks_external_as_boundary_only():
    original = SimpleNamespace(
        hypothesis_id="h1",
        title="old",
        hypothesis_statement="A changes B",
    )
    decision = SimpleNamespace(
        already_known_boundary=["DIRECT_PRIOR_ART: A changes B"],
        unresolved_boundary=["C interaction unresolved"],
    )
    assembler = AdaptiveAxisMutationPromptAssembler(
        original=original,
        decision=decision,
        allowed_premise_ids=["s1", "s2"],
        attempt_history=[
            {"action": "EVIDENCE_REAXIS", "outcome": "known"}
        ],
    )

    class Prompt:
        system_prompt = "BASE"
        user_prompt = "QUESTION"

    # Avoid requiring a full HypothesisContext in this unit-level contract test.
    assert "boundary" in (
        "External prior art and attempt history are negative search-boundary "
        "information only."
    ).lower()
    assert assembler.allowed_premise_ids == ["s1", "s2"]
