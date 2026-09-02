from types import SimpleNamespace

from pipeline_core.discovery.nonobviousness_alternate_mechanism_semantic_cohort import (
    build_alternate_mechanism_semantic_cohort,
    build_semantic_prompt_for_case,
)


def candidate(
    *,
    candidate_id="supply:1",
    claim_node_id="claim:1",
    factor_node_id="factor:1",
    local_segments=None,
    local_scopes=None,
    claim_text=None,
):
    return SimpleNamespace(
        supply_candidate_id=
            candidate_id,

        claim_node_id=
            claim_node_id,

        factor_node_id=
            factor_node_id,

        source_paper_ids=[
            "paper:1"
        ],

        factor_local_text_segments=(
            local_segments
            or [
                (
                    "Smaller nanogaps increase "
                    "the local electric field."
                )
            ]
        ),

        mechanism_scope_features=(
            local_scopes
            or [
                "nanogap",
                "electromagnetic_enhancement",
            ]
        ),

        claim_text=(
            claim_text
            or (
                "Smaller nanogaps increase the local "
                "electric field."
            )
        ),
    )


def supply_result(
    candidates,
    *,
    status=(
        "FOUND_FACTOR_GROUNDED_MECHANISM_SUPPLY"
    ),
):
    return SimpleNamespace(
        status=status,
        candidates=candidates,
    )


def baseline():
    return [
        {
            "statement_id":
                "stmt:baseline",

            "text": (
                "Decreasing interparticle spacing is "
                "attributed to near-field coupling and "
                "plasmon hybridization."
            ),

            "paper_ids": [
                "paper:baseline"
            ],

            "claim_kind":
                "mechanism",

            "epistemic_role":
                "evidence_synthesis",
        }
    ]


def test_cohort_deduplicates_claim_node():
    result = (
        build_alternate_mechanism_semantic_cohort(
            supply_result(
                [
                    candidate(
                        candidate_id="supply:1",
                        factor_node_id="factor:1",
                    ),
                    candidate(
                        candidate_id="supply:2",
                        factor_node_id="factor:2",
                    ),
                ]
            )
        )
    )

    assert result.case_count == 1
    assert result.unique_claim_count == 1

    case = result.cases[0]

    assert case.source_supply_candidate_ids == [
        "supply:1",
        "supply:2",
    ]

    assert case.source_factor_node_ids == [
        "factor:1",
        "factor:2",
    ]


def test_factor_local_prompt_does_not_leak_whole_claim_ct():
    result = (
        build_alternate_mechanism_semantic_cohort(
            supply_result(
                [
                    candidate(
                        local_segments=[
                            (
                                "Dense nanogaps form "
                                "electromagnetic hotspots."
                            )
                        ],
                        claim_text=(
                            "Dense nanogaps form electromagnetic "
                            "hotspots, while charge transfer "
                            "further enhances SERS."
                        ),
                    )
                ]
            )
        )
    )

    prompt = build_semantic_prompt_for_case(
        case=result.cases[0],

        scientific_task=(
            "How does interparticle spacing relate "
            "to SERS enhancement behavior?"
        ),

        canonical_task_feature=(
            "interparticle spacing"
        ),

        baseline_mechanism_statements=
            baseline(),

        factor_node_text_by_id={
            "factor:1":
                "Dense nanogaps"
        },
    )

    assert (
        "charge transfer"
        not in prompt.user_prompt.lower()
    )

    assert (
        "electromagnetic hotspots"
        in prompt.user_prompt.lower()
    )


def test_prompt_preserves_exact_factor_local_scientific_step():
    segment = (
        "Decreasing nanogap width increases "
        "the local electromagnetic field."
    )

    result = (
        build_alternate_mechanism_semantic_cohort(
            supply_result(
                [
                    candidate(
                        local_segments=[
                            segment
                        ]
                    )
                ]
            )
        )
    )

    prompt = build_semantic_prompt_for_case(
        case=result.cases[0],

        scientific_task=(
            "How does interparticle spacing relate "
            "to SERS enhancement behavior?"
        ),

        canonical_task_feature=(
            "interparticle spacing"
        ),

        baseline_mechanism_statements=
            baseline(),

        factor_node_text_by_id={
            "factor:1":
                "Interparticle nanogap"
        },
    )

    assert segment in prompt.user_prompt

    assert (
        '"applies_to_edge_used_as_'
        'scientific_relation": false'
        in prompt.user_prompt.lower()
    )

    assert (
        '"canonical_task_feature": '
        '"interparticle spacing"'
        in prompt.user_prompt.lower()
    )


def test_noneligible_d3a_does_not_create_cases():
    result = (
        build_alternate_mechanism_semantic_cohort(
            supply_result(
                [
                    candidate()
                ],
                status=(
                    "ABSTAIN_NO_FACTOR_GROUNDED_"
                    "MECHANISM_SUPPLY"
                ),
            )
        )
    )

    assert (
        result.status
        == "NOT_ELIGIBLE_FROM_D3A"
    )

    assert result.case_count == 0


def test_semantic_prompt_v2_uses_governing_mechanism_identity():
    result = (
        build_alternate_mechanism_semantic_cohort(
            supply_result(
                [
                    candidate()
                ]
            )
        )
    )

    prompt = build_semantic_prompt_for_case(
        case=result.cases[0],

        scientific_task=(
            "How does interparticle spacing relate "
            "to SERS enhancement behavior?"
        ),

        canonical_task_feature=(
            "interparticle spacing"
        ),

        baseline_mechanism_statements=
            baseline(),

        factor_node_text_by_id={
            "factor:1":
                "Interparticle nanogap"
        },
    )

    assert (
        prompt.prompt_version
        == (
            "n11-mechanism-semantic-review-"
            "prompt-v2-governing-mechanism"
        )
    )

    assert (
        "GOVERNING-MECHANISM IDENTITY"
        in prompt.system_prompt
    )

    assert (
        "A different downstream consequence does NOT"
        in prompt.system_prompt
    )

    assert (
        "laser positioning"
        in prompt.system_prompt
    )

    assert (
        "grounded_factor_nodes"
        in prompt.system_prompt
    )
