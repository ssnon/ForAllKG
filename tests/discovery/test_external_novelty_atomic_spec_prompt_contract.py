from pipeline_core.discovery.external_novelty_llm import (
    _DECOMPOSE_SYSTEM,
)


def test_atomic_specification_requires_self_contained_fields() -> None:
    assert (
        "SELF-CONTAINED ATOMIC SPECIFICATION CONTRACT"
        in _DECOMPOSE_SYSTEM
    )

    assert (
        'bare anaphoric expression such as "the interaction"'
        in _DECOMPOSE_SYSTEM
    )

    assert (
        "required_bridge remains stricter"
        in _DECOMPOSE_SYSTEM
    )


def test_operational_prediction_is_not_automatically_core() -> None:
    assert (
        "CORE-VERSUS-TESTING-PREDICTION CONTRACT"
        in _DECOMPOSE_SYSTEM
    )

    assert (
        "should normally be importance=supporting"
        in _DECOMPOSE_SYSTEM
    )

    assert (
        "interaction model outperforms an additive model"
        in _DECOMPOSE_SYSTEM
    )
