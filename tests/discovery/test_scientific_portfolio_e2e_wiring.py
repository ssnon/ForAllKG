import ast
from pathlib import Path


def _source():
    path = Path("scripts/discovery/run_dac_discovery_e2e.py")
    return path.read_text(encoding="utf-8")


def _strings(text):
    tree = ast.parse(text)
    return {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def test_scientific_portfolio_shadow_is_wired_before_stage8():
    text = _source()
    strings = _strings(text)
    assert "--scientific-portfolio-selection-shadow" in strings
    assert "[7.73/13] Scientific Portfolio Selection shadow" in strings
    assert "scripts.discovery.run_scientific_portfolio_selection_shadow" in strings
    assert text.index("[7.73/13]") < text.index("stage8_axis_plan_input = _stage8_axis_plan_input")


def test_optional_verification_is_shadow_only_and_does_not_change_stage8():
    text = _source()
    strings = _strings(text)
    assert "--scientific-portfolio-verification-shadow" in strings
    assert "[7.74/13] Scientific Portfolio downstream verification shadow" in strings
    assert "scripts.discovery.run_scientific_portfolio_verification_shadow" in strings
    assert '"stage8_input_changed": False' in text
    assert '"production_selection_authority": False' in text
