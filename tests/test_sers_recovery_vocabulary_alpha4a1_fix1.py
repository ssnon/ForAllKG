from __future__ import annotations

from types import SimpleNamespace

import pytest

from dac_her.domains.extraction_registry import get_extraction_adapter


def _draft(*relations: str):
    return SimpleNamespace(
        entities=[SimpleNamespace(type="Analyte")],
        edges=[SimpleNamespace(relation=value) for value in relations],
    )


def test_sers_recovery_aliases_normalize_to_canonical_relations():
    adapter = get_extraction_adapter("sers_au_ag")
    draft = _draft(
        "COMPOSED_OF",
        "HAS_ANALYTE",
        "INVOLVES_ANALYTE",
        "EVALUATED_BY",
    )
    adapter.normalize_draft_vocabulary(draft)
    assert [edge.relation for edge in draft.edges] == [
        "HAS_COMPONENT",
        "USES_ANALYTE",
        "USES_ANALYTE",
        "TESTED_IN",
    ]
    adapter.validate_draft_vocabulary(draft)


def test_sers_observed_role_relations_are_official():
    adapter = get_extraction_adapter("sers_au_ag")
    draft = _draft(
        "HAS_MORPHOLOGY",
        "USES_ANALYTE",
        "USES_REPORTER",
        "USES_OPTICAL_CONDITION",
    )
    adapter.validate_draft_vocabulary(draft)


def test_sers_unsafe_synthesis_or_typing_relations_remain_rejected():
    adapter = get_extraction_adapter("sers_au_ag")
    with pytest.raises(ValueError, match="IS_MATERIAL"):
        adapter.validate_draft_vocabulary(_draft("IS_MATERIAL"))
    with pytest.raises(ValueError, match="USED_IN_SYNTHESIS_OF"):
        adapter.validate_draft_vocabulary(_draft("USED_IN_SYNTHESIS_OF"))


def test_reserved_experiment_type_is_not_allowed_inside_entities():
    adapter = get_extraction_adapter("sers_au_ag")
    bad = SimpleNamespace(
        entities=[SimpleNamespace(type="Experiment")],
        edges=[],
    )
    with pytest.raises(ValueError, match="dedicated top-level collection"):
        adapter.validate_draft_vocabulary(bad)
