from __future__ import annotations

from pathlib import Path


E2E = Path(
    "scripts/discovery/"
    "run_dac_discovery_e2e.py"
)


def _source() -> str:
    return E2E.read_text(
        encoding="utf-8"
    )


def test_v1_production_artifact_is_retained():
    text = _source()

    assert (
        "nonobviousness_v1_production_gate"
        in text
    )

    assert (
        "fallback_gate.production.json"
        in text
    )

    assert (
        '"build_nonobviousness_production_gate"'
        in text
    )


def test_dual_run_comparison_is_retained():
    text = _source()

    assert (
        "dual_run_comparison.shadow.json"
        in text
    )

    assert (
        '"build_nonobviousness_dual_run_comparison"'
        in text
    )


def test_v2_candidate_stage_is_present():
    text = _source()

    assert (
        "fallback_gate_v2.candidate.json"
        in text
    )

    assert (
        '"build_nonobviousness_production_gate_v2_candidate"'
        in text
    )

    assert (
        "--query-plan"
        in text
    )


def test_v2_authoritative_stage_is_present():
    text = _source()

    assert (
        "fallback_gate_v2.production.json"
        in text
    )

    assert (
        '"build_nonobviousness_production_gate_v2"'
        in text
    )

    assert (
        "--candidate-gate"
        in text
    )


def test_alpha6_scientific_gate_variable_is_v2_path():
    text = _source()

    assignment = (
        'scientific_novelty_gate = (\n'
        '            run\n'
        '            / "nonobviousness_n10."\n'
        '              "fallback_gate_v2.production.json"\n'
        '        )'
    )

    assert assignment in text

    assert (
        '"--scientific-novelty-gate",'
        in text
    )

    assert (
        "str(scientific_novelty_gate)"
        in text
    )


def test_stage_order_is_v1_compare_candidate_v2_alpha6():
    text = _source()

    v1 = text.index(
        '"build_nonobviousness_production_gate"'
    )

    comparison = text.index(
        '"build_nonobviousness_dual_run_comparison"',
        v1,
    )

    candidate = text.index(
        '"build_nonobviousness_production_gate_v2_candidate"',
        comparison,
    )

    v2 = text.index(
        '"build_nonobviousness_production_gate_v2"',
        candidate + 1,
    )

    alpha6 = text.index(
        '"scripts.discovery.run_novelty_refinement"',
        v2,
    )

    assert (
        v1
        < comparison
        < candidate
        < v2
        < alpha6
    )


def test_v1_gate_is_not_assigned_to_alpha6_gate_variable():
    text = _source()

    forbidden = (
        "scientific_novelty_gate = (\n"
        "            run\n"
        '            / "nonobviousness_n10."\n'
        '              "fallback_gate.production.json"'
    )

    assert forbidden not in text


def test_legacy_scientific_novelty_branch_remains_present():
    text = _source()

    assert (
        "scientific_novelty_fallback_gate_a10.production.json"
        in text
    )

    assert (
        '"build_scientific_novelty_production_gate"'
        in text
    )


def test_post_generation_contract_is_not_removed():
    text = _source()

    assert (
        "nonobviousness_post_generation_enforce"
        in text
    )

    assert (
        "N10 post-generation enforcement requires"
        in text
    )



def test_promoted_comparison_declares_v2_runtime_authority():
    text = _source()

    comparison = text.index(
        '"build_nonobviousness_dual_run_comparison"'
    )

    candidate = text.index(
        '"build_nonobviousness_production_gate_v2_candidate"',
        comparison,
    )

    block = text[
        comparison:
        candidate
    ]

    assert (
        '"--runtime-authority-policy"'
        in block
    )

    assert (
        '"v2_production"'
        in block
    )
