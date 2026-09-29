from scripts.discovery.run_s227_3_external_novelty_control_replay import (
    classify_triplet,
)


def test_s227_3_fully_stable():
    assert (
        classify_triplet("A", "A", "A")
        == "FULLY_STABLE"
    )


def test_s227_3_control_stable_treatment_changed():
    assert (
        classify_triplet("A", "A", "B")
        == "CONTROL_STABLE_TREATMENT_CHANGED"
    )


def test_s227_3_control_drift_shared_by_treatment():
    assert (
        classify_triplet("A", "B", "B")
        == "CONTROL_DRIFT_SHARED_BY_TREATMENT"
    )


def test_s227_3_control_drift_treatment_returns_historical():
    assert (
        classify_triplet("A", "B", "A")
        == "CONTROL_DRIFT_TREATMENT_RETURNS_HISTORICAL"
    )


def test_s227_3_all_diverge():
    assert (
        classify_triplet("A", "B", "C")
        == "CONTROL_AND_TREATMENT_DIVERGE"
    )
