from __future__ import annotations

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.source_bound_topology_completion import (
    complete_explicit_source_bound_topology_shadow,
)


def _claim(local_id: str, text: str, proposition: str):
    return NoveltyClaimDraft(
        local_id=local_id,
        kind="mechanistic_link",
        text=text,
        rationale="r",
        semantic_fidelity_binding=NoveltyClaimSemanticFidelityBindingDraft(
            proposition_basis=proposition,
        ),
    )


def _composite(local_id: str, basis: str, components=None):
    return NoveltyClaimDraft(
        local_id=local_id,
        kind="composite",
        text=basis,
        rationale="r",
        higher_order_relation_basis=[basis],
        higher_order_component_local_ids=list(components or []),
    )


def test_exact_source_containment_completes_empty_topology():
    draft = NoveltyClaimDecompositionDraft(
        claims=[
            _claim("a", "A changes B", "A changes B"),
            _claim("b", "B changes C", "B changes C"),
            _composite(
                "c",
                "A changes B, and B changes C under the same regime.",
            ),
        ]
    )
    result, records = complete_explicit_source_bound_topology_shadow(draft)
    comp = next(x for x in result.claims if x.local_id == "c")
    assert comp.higher_order_component_local_ids == ["a", "b"]
    assert records[-1]["status"] == "SOURCE_BOUND_TOPOLOGY_COMPLETED_SHADOW"


def test_no_semantic_inference_when_source_span_does_not_contain_component():
    draft = NoveltyClaimDecompositionDraft(
        claims=[
            _claim("a", "A changes B", "A affects B in another context"),
            _composite("c", "A changes B only under condition X."),
        ]
    )
    result, records = complete_explicit_source_bound_topology_shadow(draft)
    comp = next(x for x in result.claims if x.local_id == "c")
    assert comp.higher_order_component_local_ids == []
    assert records[-1]["status"] == "NO_SOURCE_BOUND_COMPONENT_TOPOLOGY"


def test_existing_explicit_topology_is_never_rewritten():
    draft = NoveltyClaimDecompositionDraft(
        claims=[
            _claim("a", "A changes B", "A changes B"),
            _claim("b", "B changes C", "B changes C"),
            _composite("c", "A changes B and B changes C", ["a"]),
        ]
    )
    result, records = complete_explicit_source_bound_topology_shadow(draft)
    comp = next(x for x in result.claims if x.local_id == "c")
    assert comp.higher_order_component_local_ids == ["a"]
    assert records[-1]["status"] == "EXPLICIT_TOPOLOGY_PRESERVED"
