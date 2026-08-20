from __future__ import annotations

from pathlib import Path

from domains.sers.prompts import SERS_PROMPT_VERSION, SERS_SYSTEM_PROMPT
from pipeline_core.corpus.vocab_registry import load_default_registries


def test_prompt_version_alpha4a3_3():
    assert SERS_PROMPT_VERSION.startswith("sers-au-ag-extraction-v1-alpha4a")


def test_measurement_xor_rule_is_explicit():
    prompt = " ".join(SERS_SYSTEM_PROMPT.split())
    assert "exactly one of value_numeric or value_text" in prompt
    assert "source_expression" in prompt


def test_dls_uses_schema_compatible_family():
    experiments, _ = load_default_registries(Path.cwd())
    dls = experiments.resolve(None, "Dynamic light scattering")
    assert dls is not None
    assert dls.entry_id == "dynamic_light_scattering"
    assert dls.metadata.get("family") == "other"


def test_remaining_pilot_vocabulary_is_registered():
    experiments, metrics = load_default_registries(Path.cwd())

    eds = experiments.resolve(None, "Energy-dispersive X-ray spectroscopy")
    assert eds is not None
    assert eds.entry_id == "energy_dispersive_x_ray_spectroscopy"

    cases = {
        "Absorption-band wavelength": "absorption_band_wavelength",
        "Absorption shoulder wavelength": "absorption_shoulder_wavelength",
        "Concentration–SERS intensity correlation R²":
            "concentration_sers_intensity_correlation_r2",
    }
    for label, expected in cases.items():
        entry = metrics.resolve(None, label)
        assert entry is not None, label
        assert entry.entry_id == expected
