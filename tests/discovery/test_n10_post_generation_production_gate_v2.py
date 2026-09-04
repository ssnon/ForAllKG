from copy import deepcopy

import pytest

import pipeline_core.discovery.nonobviousness_post_generation_production_gate_v2 as module
from pipeline_core.discovery.nonobviousness_post_generation_production_gate_v2 import (
    build_nonobviousness_post_generation_production_gate_v2,
)


def _promoted_gate(
    *,
    selection="CONDITIONAL",
    action="REFINE_NOVELTY_BEARING_SPECIFICATION",
    positive=False,
    fallback=False,
):
    return {
        "schema_version":
            "scientific-novelty-fallback-gate-v2",

        "production_authority":
            True,

        "authority_scope":
            "alpha6_original_fallback",

        "authority_source":
            "n10_role_aware_nonobviousness_v2",

        "positive_authority_requires":
            "ELIGIBLE_AND_ROLE_AWARE_POSITIVE_NONOBVIOUSNESS",

        "conditional_is_positive":
            False,

        "absence_is_novelty":
            False,

        "candidate_semantics_preserved":
            True,

        "gate_count":
            1,

        "gates": [
            {
                "hypothesis_id":
                    "hypothesis:test",

                "selection_class":
                    selection,

                "action":
                    action,

                "base_aggregation_action":
                    action,

                "positive_nonobviousness_authority":
                    positive,

                "fallback_allowed":
                    fallback,

                "reason_codes":
                    ["synthetic"],

                "blocking_claim_ids":
                    [],

                "unresolved_claim_ids":
                    ["claim:1"],

                "resolution_requirements":
                    [],
            }
        ],
    }


def _install_backend(
    monkeypatch,
    gate,
):
    captured = {}

    def fake_backend(
        *,
        candidate_gate,
    ):
        captured["candidate_gate"] = (
            candidate_gate
        )

        return gate

    monkeypatch.setattr(
        module,
        "build_nonobviousness_production_gate_v2",
        fake_backend,
    )

    return captured


def test_rebinds_only_authority_scope(
    monkeypatch,
):
    promoted = _promoted_gate()
    before = deepcopy(promoted)

    captured = _install_backend(
        monkeypatch,
        promoted,
    )

    candidate = {
        "sentinel":
            "candidate-only",
    }

    result = (
        build_nonobviousness_post_generation_production_gate_v2(
            candidate_gate=candidate,
        )
    )

    expected = deepcopy(before)

    expected["authority_scope"] = (
        "alpha6_post_generation_candidate"
    )

    assert result == expected

    assert promoted == before

    assert (
        captured["candidate_gate"]
        is candidate
    )


def test_conditional_remains_fail_closed(
    monkeypatch,
):
    promoted = _promoted_gate(
        selection="CONDITIONAL",
        positive=False,
        fallback=False,
    )

    _install_backend(
        monkeypatch,
        promoted,
    )

    result = (
        build_nonobviousness_post_generation_production_gate_v2(
            candidate_gate={},
        )
    )

    row = result["gates"][0]

    assert (
        row["selection_class"]
        == "CONDITIONAL"
    )

    assert (
        row[
            "positive_nonobviousness_authority"
        ]
        is False
    )

    assert (
        row["fallback_allowed"]
        is False
    )

    assert (
        row["action"]
        == "REFINE_NOVELTY_BEARING_SPECIFICATION"
    )


def test_eligible_positive_semantics_are_not_changed(
    monkeypatch,
):
    promoted = _promoted_gate(
        selection="ELIGIBLE",
        action="KEEP_NONOBVIOUS_CANDIDATE",
        positive=True,
        fallback=True,
    )

    _install_backend(
        monkeypatch,
        promoted,
    )

    result = (
        build_nonobviousness_post_generation_production_gate_v2(
            candidate_gate={},
        )
    )

    row = result["gates"][0]

    assert row["selection_class"] == "ELIGIBLE"

    assert (
        row[
            "positive_nonobviousness_authority"
        ]
        is True
    )

    assert (
        row["fallback_allowed"]
        is True
    )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        (
            "schema_version",
            "scientific-novelty-fallback-gate-v1",
        ),
        (
            "production_authority",
            False,
        ),
        (
            "authority_scope",
            "wrong_scope",
        ),
        (
            "authority_source",
            "wrong_source",
        ),
        (
            "conditional_is_positive",
            True,
        ),
        (
            "absence_is_novelty",
            True,
        ),
        (
            "candidate_semantics_preserved",
            False,
        ),
    ],
)
def test_invalid_source_envelope_fails_closed(
    monkeypatch,
    field,
    bad_value,
):
    promoted = _promoted_gate()

    promoted[field] = bad_value

    _install_backend(
        monkeypatch,
        promoted,
    )

    with pytest.raises(
        ValueError,
    ):
        build_nonobviousness_post_generation_production_gate_v2(
            candidate_gate={},
        )


def test_invalid_positive_requirement_fails_closed(
    monkeypatch,
):
    promoted = _promoted_gate()

    promoted[
        "positive_authority_requires"
    ] = "POTENTIALLY_NON_OBVIOUS"

    _install_backend(
        monkeypatch,
        promoted,
    )

    with pytest.raises(
        ValueError,
        match="positive-authority",
    ):
        build_nonobviousness_post_generation_production_gate_v2(
            candidate_gate={},
        )


def test_gate_rows_must_remain_a_list(
    monkeypatch,
):
    promoted = _promoted_gate()

    promoted["gates"] = {
        "not":
            "a-list",
    }

    _install_backend(
        monkeypatch,
        promoted,
    )

    with pytest.raises(
        ValueError,
        match="gates must be a list",
    ):
        build_nonobviousness_post_generation_production_gate_v2(
            candidate_gate={},
        )
