from __future__ import annotations

from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
    RelationComponentProvenance,
    RelationComponentView,
)
from pipeline_core.discovery.task_backbone_chain import (
    build_task_endpoint_coverage_ledger,
)


def _known(
    component_id: str,
    subject: str,
    relation: str,
    object_: str,
) -> RelationComponentView:
    accepted_pattern_id = f"pattern:{component_id}"
    return RelationComponentView(
        component_id=component_id,
        label=component_id,
        subject=subject,
        relation=relation,
        object=object_,
        authority=RelationComponentAuthority.CONFIRMED_KNOWN,
        provenance=RelationComponentProvenance(
            source_kind="accepted_pattern",
            source_id=accepted_pattern_id,
            paper_id=f"paper:{component_id}",
            chunk_id=f"chunk:{component_id}",
            document_id="main",
        ),
        accepted_pattern_id=accepted_pattern_id,
    )


def test_compound_endpoint_exposes_partial_facet_and_qualifier_gap() -> None:
    components = (
        _known(
            "hotspot-intensity",
            "hotspot intensity",
            "IMPOSES_TRADEOFF",
            "hotspot size",
        ),
        _known(
            "sers-hotspot-intensity",
            "SERS hotspot density and intensity",
            "PROMOTES",
            "SERS effect",
        ),
    )

    ledger = build_task_endpoint_coverage_ledger(
        components=components,
        task_endpoint="electromagnetic hotspot location and intensity",
    )

    assert ledger.schema_version == "task-endpoint-coverage-ledger-v1"
    assert ledger.coordinated is True
    assert ledger.shared_qualifier_tokens == ["electromagnetic"]
    assert ledger.status == "partial_or_qualifier_unresolved"
    assert ledger.coverage_authority is False
    assert ledger.task_filter_relaxed is False
    assert ledger.positive_premise_authority_created is False
    assert ledger.novelty_authority_created is False

    by_core = {row.core_endpoint: row for row in ledger.facets}

    assert by_core["hotspot location"].status == "core_facet_unresolved"
    assert by_core["hotspot location"].core_binding_count == 0

    intensity = by_core["hotspot intensity"]
    assert intensity.core_binding_count == 2
    assert (
        intensity.status
        == "facet_core_supported_qualifier_unresolved"
    )
    assert intensity.task_slot_qualified_binding_count == 0
    assert intensity.component_qualified_binding_count == 0


def test_compound_endpoint_can_be_complete_without_shared_qualifier() -> None:
    components = (
        _known(
            "sers-enhancement",
            "SERS enhancement",
            "VARIES_WITH",
            "nanostructure geometry",
        ),
        _known(
            "sers-repeatability",
            "SERS sensitivity and signal repeatability",
            "VARIES_WITH",
            "substrate architecture",
        ),
    )

    ledger = build_task_endpoint_coverage_ledger(
        components=components,
        task_endpoint="SERS enhancement and repeatability",
    )

    assert ledger.coordinated is True
    assert ledger.shared_qualifier_tokens == []
    assert ledger.status == "complete"

    by_core = {row.core_endpoint: row for row in ledger.facets}
    assert (
        by_core["sers enhancement"].status
        == "facet_core_supported_no_qualifier_obligation"
    )
    assert (
        by_core["sers repeatability"].status
        == "facet_core_supported_no_qualifier_obligation"
    )


def test_noncoordinated_endpoint_is_not_applicable() -> None:
    ledger = build_task_endpoint_coverage_ledger(
        components=(),
        task_endpoint="broad plasmonic response",
    )

    assert ledger.coordinated is False
    assert ledger.status == "not_applicable"
    assert ledger.facets == []
    assert ledger.coverage_authority is False
