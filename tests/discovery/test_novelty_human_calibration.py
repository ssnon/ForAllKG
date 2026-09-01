from pipeline_core.discovery.novelty_calibration import (
    NoveltyCalibrationCase,
    calibration_bucket_from_structural_status,
    calibration_direction_is_safe,
)


def test_direct_prior_art_maps_to_directly_known():
    assert (
        calibration_bucket_from_structural_status(
            "DIRECTLY_KNOWN"
        )
        == "DIRECTLY_KNOWN"
    )


def test_routine_composition_maps_to_routine_extension():
    assert (
        calibration_bucket_from_structural_status(
            "ROUTINE_COMPOSITION"
        )
        == "ROUTINE_EXTENSION"
    )


def test_insufficient_closure_stays_unresolved():
    assert (
        calibration_bucket_from_structural_status(
            "INSUFFICIENT_CLOSURE"
        )
        == "UNRESOLVED"
    )


def test_interaction_leap_is_not_called_nonobvious():
    assert (
        calibration_bucket_from_structural_status(
            "INTERACTION_LEAP"
        )
        == "NONOBVIOUSNESS_CANDIDATE"
    )


def test_regime_leap_is_only_a_candidate():
    assert (
        calibration_bucket_from_structural_status(
            "REGIME_OR_THRESHOLD_LEAP"
        )
        == "NONOBVIOUSNESS_CANDIDATE"
    )


def test_potentially_nonobvious_human_case_accepts_candidate_only():
    case = NoveltyCalibrationCase(
        case_id="threshold-control",
        source="SYNTHETIC_CONTROL",
        description="threshold control",
        human_label="POTENTIALLY_NON_OBVIOUS",
        structural_status="REGIME_OR_THRESHOLD_LEAP",
        rationale="test",
    )

    assert calibration_direction_is_safe(case)
