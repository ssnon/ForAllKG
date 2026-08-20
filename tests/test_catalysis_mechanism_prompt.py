from __future__ import annotations

from pathlib import Path

from domains.catalysis_mechanism.prompts import (
    CATALYSIS_MECHANISM_PROMPT_VERSION,
    CATALYSIS_MECHANISM_SYSTEM_PROMPT,
)


def test_abstract_prompt_preserves_epistemic_boundary():
    prompt = CATALYSIS_MECHANISM_SYSTEM_PROMPT

    assert CATALYSIS_MECHANISM_PROMPT_VERSION
    assert "Treat an abstract as a limited evidence source" in prompt
    assert "Prefer direct mechanism relations" in prompt
    assert "Never create a generic Experiment or Calculation" in prompt
    assert "Do not infer RDS from" in prompt
    assert "CORRELATES_WITH is non-causal" in prompt
    assert "source-explicit limitation or breakdown" in prompt
    assert "STRICT ENDPOINT MATRIX" in prompt
    assert "Catalyst/CatalystModel --HAS_ACTIVE_SITE--> ActiveSite" in prompt
    assert "Do NOT emit Measurement or MeasurementGroup nodes" in prompt
    assert "measurements[] and measurement_groups[] are empty" in prompt
    assert "NEVER emit Catalyst --MODEL_OF-->" in prompt
    assert "CHARACTERIZED_BY always points FROM" in prompt
    assert "remove any node with degree zero" in prompt


def test_registry_source_wires_broad_profile_and_adapter():
    root = Path(__file__).resolve().parents[1]
    profile_registry = (
        root / "domains" / "registry.py"
    ).read_text(encoding="utf-8")
    extraction_registry = (
        root / "domains" / "extraction_registry.py"
    ).read_text(encoding="utf-8")

    assert "CATALYSIS_MECHANISM_PROFILE" in profile_registry
    assert "'broad': 'catalysis_mechanism'" in profile_registry
    assert "CATALYSIS_MECHANISM_EXTRACTION_ADAPTER" in extraction_registry
