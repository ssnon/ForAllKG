from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from pipeline_core.corpus.figure_extraction import (
    analyze_figure,
    build_figure_system_prompt,
)


def test_figure_prompt_requires_explicit_domain_context():
    parameter = inspect.signature(
        analyze_figure
    ).parameters["domain_context"]

    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_figure_prompt_embeds_supplied_domain_context():
    context = (
        "Surface-enhanced Raman spectroscopy on Au/Ag "
        "nanostructured substrates."
    )

    prompt = build_figure_system_prompt(context)

    assert context in prompt
    assert "dual-atom electrocatalysts and HER" not in prompt
    assert "GraphAgents DAC-HER" not in prompt
    assert "data_dac" not in prompt


def test_figure_prompt_rejects_empty_domain_context():
    with pytest.raises(
        ValueError,
        match="domain_context must not be empty",
    ):
        build_figure_system_prompt("   ")


def test_generic_figure_module_has_no_dac_specific_literals():
    source = Path(
        "pipeline_core/corpus/figure_extraction.py"
    ).read_text(encoding="utf-8")

    assert "GraphAgents DAC-HER" not in source
    assert "data_dac" not in source
    assert "dual-atom electrocatalysts and HER" not in source
