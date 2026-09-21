from __future__ import annotations

from pipeline_core.discovery.task_backbone_chain import (
    TaskEndpointCoverageLedger,
    TaskEndpointFacetCoverageView,
)
from pipeline_core.discovery.task_endpoint_coverage_critic import (
    critique_task_endpoint_coverage,
)


def _not_applicable(endpoint: str) -> TaskEndpointCoverageLedger:
    return TaskEndpointCoverageLedger(
        original_endpoint=endpoint,
        coordinated=False,
        status="not_applicable",
    )


def test_partial_compound_target_emits_nonblocking_diagnostics() -> None:
    target = TaskEndpointCoverageLedger(
        original_endpoint=(
            "electromagnetic hotspot location and intensity"
        ),
        coordinated=True,
        shared_qualifier_tokens=["electromagnetic"],
        head_tokens=["hotspot"],
        facet_tokens=[["location"], ["intensity"]],
        facets=[
            TaskEndpointFacetCoverageView(
                core_endpoint="hotspot location",
                facet_tokens=["location"],
                shared_qualifier_tokens=["electromagnetic"],
                status="core_facet_unresolved",
            ),
            TaskEndpointFacetCoverageView(
                core_endpoint="hotspot intensity",
                facet_tokens=["intensity"],
                shared_qualifier_tokens=["electromagnetic"],
                core_binding_count=4,
                status=(
                    "facet_core_supported_qualifier_unresolved"
                ),
            ),
        ],
        status="partial_or_qualifier_unresolved",
    )

    result = critique_task_endpoint_coverage(
        source=_not_applicable("nanostructure shape"),
        target=target,
    )

    assert result.issue_codes == [
        "COMPOUND_ENDPOINT_COVERAGE_INCOMPLETE",
        "COMPOUND_ENDPOINT_FACET_UNRESOLVED",
        "COMPOUND_ENDPOINT_QUALIFIER_UNRESOLVED",
    ]
    assert result.diagnostic_only is True
    assert result.blocking is False
    assert result.rejection_authority is False
    assert result.selection_authority is False
    assert result.task_filter_relaxed is False
    assert result.positive_premise_authority_created is False
    assert result.novelty_authority_created is False


def test_complete_compound_target_has_no_issue() -> None:
    target = TaskEndpointCoverageLedger(
        original_endpoint="SERS enhancement and repeatability",
        coordinated=True,
        head_tokens=["sers"],
        facet_tokens=[["enhancement"], ["repeatability"]],
        facets=[
            TaskEndpointFacetCoverageView(
                core_endpoint="sers enhancement",
                facet_tokens=["enhancement"],
                core_binding_count=10,
                task_slot_qualified_binding_count=10,
                component_qualified_binding_count=10,
                status=(
                    "facet_core_supported_no_qualifier_obligation"
                ),
            ),
            TaskEndpointFacetCoverageView(
                core_endpoint="sers repeatability",
                facet_tokens=["repeatability"],
                core_binding_count=2,
                task_slot_qualified_binding_count=2,
                component_qualified_binding_count=2,
                status=(
                    "facet_core_supported_no_qualifier_obligation"
                ),
            ),
        ],
        status="complete",
    )

    result = critique_task_endpoint_coverage(
        source=_not_applicable("SERS substrate design"),
        target=target,
    )

    assert result.issues == []
    assert result.issue_codes == []
    assert result.blocking is False


def test_noncoordinated_endpoints_are_silent() -> None:
    result = critique_task_endpoint_coverage(
        source=_not_applicable("broad plasmonic response"),
        target=_not_applicable("excitation wavelength"),
    )

    assert result.issues == []
    assert result.issue_codes == []
