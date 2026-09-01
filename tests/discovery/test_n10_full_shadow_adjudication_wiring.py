from pathlib import Path


def test_full_shadow_runs_independent_adjudicator_for_ready_claims():
    text = Path(
        "scripts/discovery/"
        "run_nonobviousness_full_shadow.py"
    ).read_text()

    assert (
        "review_and_compile_nonobviousness_adjudication"
        in text
    )

    assert (
        '== "READY_FOR_NONOBVIOUSNESS_REVIEW"'
        in text
    )

    assert (
        '"INDEPENDENT_ADJUDICATION_COMPILED"'
        in text
    )

    assert (
        "independent_adjudication_performed"
        in text
    )


def test_e2e_help_no_longer_claims_ready_stays_pending():
    text = Path(
        "scripts/discovery/"
        "run_dac_discovery_e2e.py"
    ).read_text()

    assert (
        "READY candidates remain pending an independent adjudicator"
        not in text
    )
