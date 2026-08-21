from scripts.discovery.run_dac_discovery_e2e import _alpha6_empty_is_degraded


def test_empty_alpha6_all_compile_failures_is_degraded():
    report = {
        "attempts": [
            {"decision": "compile_rejected"},
            {"decision": "validation_rejected"},
            {"decision": "grounding_drift_rejected"},
        ]
    }
    assert _alpha6_empty_is_degraded(report)


def test_empty_alpha6_scientific_rejection_is_not_mislabeled_degraded():
    report = {
        "attempts": [
            {"decision": "external_novelty_rejected"},
            {"decision": "internal_novelty_rejected"},
        ]
    }
    assert not _alpha6_empty_is_degraded(report)
