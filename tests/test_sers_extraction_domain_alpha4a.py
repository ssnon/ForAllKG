from types import SimpleNamespace

import pytest

from dac_her.domains.extraction_registry import (
    available_extraction_adapters,
    get_extraction_adapter,
)
from dac_her.domains.feasibility_registry import get_feasibility_adapter
from dac_her.domains.registry import (
    available_domain_profiles,
    get_domain_profile,
)
from pipeline_core.corpus.schemas import EntityNode
from scripts.ingest_local_corpus import discover_local_papers


def test_sers_profile_and_extraction_adapter_are_registered():
    assert set(available_domain_profiles()) >= {"dac_her", "sers_au_ag"}
    assert set(available_extraction_adapters()) >= {"dac_her", "sers_au_ag"}
    profile = get_domain_profile("sers_au_ag")
    adapter = get_extraction_adapter("sers_au_ag")
    assert profile.extraction_adapter_id == "sers_au_ag"
    assert adapter.domain_profile_id == "sers_au_ag"
    assert adapter.default_data_root == "data_sers"


def test_sers_feasibility_still_fails_closed_in_alpha4a():
    with pytest.raises(ValueError, match="has no feasibility adapter"):
        get_feasibility_adapter("sers_au_ag")


def test_shared_schema_accepts_sers_entity_identifier():
    entity = EntityNode(
        id="substrate:1",
        type="PlasmonicSubstrate",
        label="Au@Ag nanocube substrate",
        description=None,
    )
    assert entity.type == "PlasmonicSubstrate"


def test_sers_adapter_rejects_her_scientific_vocab():
    adapter = get_extraction_adapter("sers_au_ag")
    draft = SimpleNamespace(
        entities=[SimpleNamespace(type="Catalyst")],
        edges=[SimpleNamespace(relation="CATALYZES")],
    )
    with pytest.raises(ValueError, match="vocabulary violation"):
        adapter.validate_draft_vocabulary(draft)


def test_sers_adapter_accepts_sers_scientific_vocab():
    adapter = get_extraction_adapter("sers_au_ag")
    draft = SimpleNamespace(
        entities=[
            SimpleNamespace(type="PlasmonicSubstrate"),
            SimpleNamespace(type="StructuralMotif"),
        ],
        edges=[SimpleNamespace(relation="HAS_STRUCTURAL_MOTIF")],
    )
    adapter.validate_draft_vocabulary(draft)


def test_local_ingestion_groups_main_and_si_and_detects_ids(tmp_path):
    (tmp_path / "Kiwook_SERS_1.pdf").write_bytes(b"main-one")
    (tmp_path / "Kiwook_SERS_1_SI1.pdf").write_bytes(b"si-one")
    (tmp_path / "Kiwook_SERS_2.pdf").write_bytes(b"main-two")
    papers = discover_local_papers(tmp_path)
    assert [paper.paper_id for paper in papers] == [
        "Kiwook_SERS_1", "Kiwook_SERS_2",
    ]
    assert len(papers[0].si_pdfs) == 1
    assert len(papers[1].si_pdfs) == 0


def test_local_ingestion_fails_closed_on_duplicate_main_content(tmp_path):
    (tmp_path / "Kiwook_SERS_1.pdf").write_bytes(b"same")
    (tmp_path / "Kiwook_SERS_2.pdf").write_bytes(b"same")
    with pytest.raises(ValueError, match="Duplicate main-PDF content"):
        discover_local_papers(tmp_path)
