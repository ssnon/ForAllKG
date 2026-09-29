import hashlib
import json

from pipeline_core.discovery.prospective_novelty_validation_cohort import (
    ProspectiveNoveltyValidationCohortSpec,
    build_prospective_novelty_validation_cohort_freeze,
)


def _spec():
    return ProspectiveNoveltyValidationCohortSpec.model_validate(
        {
            "cohort_name": "test",
            "cases": [
                {
                    "case_id": "SERS",
                    "case_role": "DIRECT_HO_REFERENCE",
                    "domain_profile_id": "sers_au_ag",
                    "corpus_id": "sers500_final_v2",
                    "source": "molecular orientation",
                    "target": "Raman intensity",
                    "question": "q1",
                    "evaluation_focus": ["conceptual_first_gap_level"],
                    "research_value_capability_expected": False,
                },
                {
                    "case_id": "DAC",
                    "case_role": "MULTIFACTOR_MECHANISM",
                    "domain_profile_id": "dac_her",
                    "corpus_id": "dac_her_expanded_v2",
                    "source": "metal-pair identity",
                    "stop": "charge transfer",
                    "target": "hydrogen evolution activity",
                    "question": "q2",
                    "evaluation_focus": ["research_value_shadow"],
                    "research_value_capability_expected": True,
                },
            ],
        }
    )


def test_freeze_is_deterministic_and_outcome_blind():
    spec = _spec()
    sha = hashlib.sha256(b"spec").hexdigest()

    left = build_prospective_novelty_validation_cohort_freeze(
        spec=spec,
        source_spec_sha256=sha,
    )
    right = build_prospective_novelty_validation_cohort_freeze(
        spec=spec,
        source_spec_sha256=sha,
    )

    assert left == right
    assert left.freeze_sha256 == right.freeze_sha256
    assert left.frozen_case_count == 2
    assert left.domain_counts == {
        "sers_au_ag": 1,
        "dac_her": 1,
    }

    assert left.prior_results_observed_for_selection is False
    assert left.external_novelty_outcomes_used_for_selection is False
    assert left.conceptual_knownness_outcomes_used_for_selection is False
    assert left.alpha6_recommendations_used_for_selection is False
    assert left.research_value_outcomes_used_for_selection is False
    assert left.post_freeze_case_mutation_allowed is False
    assert left.production_selection_authority is False

    assert all(
        row.scientific_outcome_expected == "UNSPECIFIED"
        for row in left.frozen_cases
    )


def test_duplicate_case_ids_are_rejected():
    payload = _spec().model_dump(mode="json")
    payload["cases"].append(dict(payload["cases"][0]))

    try:
        ProspectiveNoveltyValidationCohortSpec.model_validate(payload)
    except ValueError as exc:
        assert "duplicate prospective case_id" in str(exc)
    else:
        raise AssertionError("duplicate case IDs should fail")
