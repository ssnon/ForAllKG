from __future__ import annotations

from pipeline_core.corpus.broad_compact_schema import BroadMechanismGraphDraft
from domains.extraction_registry import get_extraction_adapter
from pipeline_core.corpus.extraction.draft_schema import KnowledgeGraphDraft
from pipeline_core.corpus.graph.graph_validation import collect_graph_issues
from pipeline_core.corpus.schemas import EntityNode, KGEdge


def _draft_with_edge(
    source_type: str,
    relation: str,
    target_type: str,
) -> KnowledgeGraphDraft:
    source = EntityNode(
        id="source",
        type=source_type,
        label="source",
        description=None,
    )
    target = EntityNode(
        id="target",
        type=target_type,
        label="target",
        description=None,
    )
    edge = KGEdge.model_construct(
        source="source",
        relation=relation,
        target="target",
        evidence_type="author_interpretation",
        evidence_strength="interpretive",
        evidence_text="fixture",
        confidence="high",
        evidence_pointers=[],
        subsection=None,
    )
    return KnowledgeGraphDraft(
        paper_id="paper",
        chunk_id="chunk",
        section="abstract",
        document_id="abstract",
        document_role="main",
        page_ids=[],
        asset_ids=[],
        entities=[source, target],
        experiments=[],
        calculations=[],
        measurements=[],
        measurement_groups=[],
        observation_claims=[],
        mechanism_claims=[],
        edges=[edge],
    )


def test_compact_generation_schema_is_adapter_owned():
    broad = get_extraction_adapter("catalysis_mechanism")
    dac = get_extraction_adapter("dac_her")
    sers = get_extraction_adapter("sers_au_ag")

    assert broad.generation_response_model(compact=True) is BroadMechanismGraphDraft
    assert broad.generation_response_model(compact=False) is KnowledgeGraphDraft

    for adapter in (dac, sers):
        try:
            adapter.generation_response_model(compact=True)
        except ValueError as error:
            assert adapter.domain_profile_id in str(error)
        else:  # pragma: no cover
            raise AssertionError("non-Broad adapter accepted compact schema")


def test_canonical_generation_output_is_generic():
    adapter = get_extraction_adapter("catalysis_mechanism")
    canonical = KnowledgeGraphDraft(
        paper_id="paper",
        chunk_id="chunk",
        section="abstract",
        document_id="abstract",
        document_role="main",
        page_ids=[],
        asset_ids=[],
        entities=[],
        experiments=[],
        calculations=[],
        measurements=[],
        measurement_groups=[],
        observation_claims=[],
        mechanism_claims=[],
        edges=[],
    )
    assert adapter.canonicalize_generation_output(canonical) is canonical


def test_custom_empty_contract_does_not_fall_back_to_dac_semantics():
    draft = _draft_with_edge("Material", "CATALYZES", "Material")
    report = collect_graph_issues(
        draft,
        relation_constraints=(),
    )
    codes = {item.code.value for item in report.issues}
    assert "RELATION_SOURCE_TYPE_MISMATCH" not in codes
    assert "RELATION_TARGET_TYPE_MISMATCH" not in codes


def test_dac_contract_retains_legacy_catalyzes_semantics():
    adapter = get_extraction_adapter("dac_her")
    draft = _draft_with_edge("Material", "CATALYZES", "Material")
    report = collect_graph_issues(
        draft,
        relation_constraints=adapter.strict_relation_constraints,
    )
    codes = {item.code.value for item in report.issues}
    assert "RELATION_SOURCE_TYPE_MISMATCH" in codes
    assert "RELATION_TARGET_TYPE_MISMATCH" in codes


def test_sers_contract_does_not_inherit_dac_catalyzes_semantics():
    adapter = get_extraction_adapter("sers_au_ag")
    draft = _draft_with_edge("Material", "CATALYZES", "Material")
    report = collect_graph_issues(
        draft,
        relation_constraints=adapter.strict_relation_constraints,
    )
    codes = {item.code.value for item in report.issues}
    assert "RELATION_SOURCE_TYPE_MISMATCH" not in codes
    assert "RELATION_TARGET_TYPE_MISMATCH" not in codes
