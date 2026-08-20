"""Characterization tests for the graph/Bridge extraction compatibility slice."""

from __future__ import annotations

import pipeline_core.bridge_draft_schema as core_bridge_draft
import pipeline_core.bridge_schemas as core_bridge
import pipeline_core.graph_io as core_graph_io
import pipeline_core.graph_validation as core_graph_validation
import pipeline_core.node_references as core_node_references
import pipeline_core.discovery_semantics as core_discovery
import pipeline_core.asset_index as core_assets
import pipeline_core.locator_index as core_locators
import pipeline_core.extraction_policy as core_extraction_policy
import pipeline_core.explorer_text_safety as core_text_safety
import pipeline_core.markdown as core_markdown
import pipeline_core.traversal_runtime_policy as core_traversal_policy
import pipeline_core.validation as core_validation
import pipeline_core.validation_issues as core_issues




def test_node_reference_remapping_preserves_graphml_foreign_keys():
    payload = {
        "type": "Measurement",
        "subject_id": "entity-1",
        "group_id": "group-1",
        "unrelated": "keep",
    }
    assert core_node_references.remap_node_reference_attributes(
        payload,
        {"entity-1": "paper:entity-1", "group-1": "paper:group-1"},
    ) == {
        "type": "Measurement",
        "subject_id": "paper:entity-1",
        "group_id": "paper:group-1",
        "unrelated": "keep",
    }


def test_asset_index_preserves_asset_identity_and_package_scan(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "figure.png").write_bytes(b"figure-bytes")
    (package / "loose.jpg").write_bytes(b"loose-bytes")
    markdown = (
        "# Methods\n"
        "<span id=\"page-3-marker\"></span>\n"
        "![Au/Ag](figure.png)\n"
        "Figure 1. Au/Ag morphology.\n"
    )

    assets = core_assets.build_asset_index(
        paper_id="paper",
        document_id="main",
        document_role="main",
        package_dir=package,
        markdown=markdown,
    )
    assert [item.relative_path for item in assets] == ["figure.png", "loose.jpg"]
    linked, loose = assets
    assert linked.page_id == 3
    assert linked.caption == "Figure 1. Au/Ag morphology."
    assert linked.referenced_in_markdown is True
    assert linked.discovery_method == "markdown_image+package_scan"
    assert loose.referenced_in_markdown is False
    assert loose.discovery_method == "package_scan"
    assert linked.sha256 is not None
    assert core_assets.assets_by_id(assets)[linked.asset_id] is linked
    assert core_assets.asset_path_to_id(assets)["figure.png"] == linked.asset_id

    output = core_assets.write_assets_jsonl(tmp_path / "assets.jsonl", assets)
    assert output.read_text(encoding="utf-8").count("\n") == 2

    locators = core_locators.build_locator_index(
        document_id="main",
        document_role="main",
        markdown=markdown,
        assets=assets,
    )
    figure = next(item for item in locators if item.locator_key == "figure:1")
    assert set(figure.asset_ids) == {linked.asset_id, loose.asset_id}
    assert figure.mapping_method == "asset_caption_exact"
    assert figure.confidence == "high"
    assert figure.ambiguous is True
    locator_output = core_locators.write_locator_index_json(
        tmp_path / "locators.json",
        locators,
    )
    loaded = core_locators.load_locator_index(locator_output)
    assert any(row["locator_key"] == "figure:1" for row in loaded)


def test_traversal_policy_preserves_legacy_depth_and_budget_boundaries():
    assert core_traversal_policy.resolve_semantic_stop_max_depth(
        base_max_depth=4,
        semantic_stop_max_depth=None,
        base_max_depth_explicit=False,
    ) == 12
    assert core_traversal_policy.resolve_semantic_stop_max_depth(
        base_max_depth=4,
        semantic_stop_max_depth=None,
        base_max_depth_explicit=True,
    ) == 4
    assert core_traversal_policy.resolve_semantic_stop_max_depth(
        base_max_depth=4,
        semantic_stop_max_depth=0,
        base_max_depth_explicit=False,
    ) == 1

    sources, targets, diagnostic = (
        core_traversal_policy.guard_semantic_stop_ablation(
            [{"id": index} for index in range(5)],
            [{"id": index} for index in range(5)],
            waypoint_count=2,
            max_triples=8,
        )
    )
    assert len(sources) == len(targets) == 2
    assert diagnostic.applied is True
    assert diagnostic.traversal_triple_upper_bound == 8
