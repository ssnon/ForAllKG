from __future__ import annotations

from pathlib import Path


E2E = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "discovery"
    / "run_dac_discovery_e2e.py"
)


def _source() -> str:
    return E2E.read_text(encoding="utf-8")


def test_s197_cli_is_explicit_opt_in_with_bounded_top_k():
    text = _source()

    assert '"--direct-relationpattern-task-shadow"' in text
    assert '"--direct-relationpattern-top-k"' in text
    assert "default=20" in text


def test_s197_wires_dedicated_shadow_stage_after_task_plan():
    text = _source()

    task_stage = text.index(
        '"[7.5/13] Task-conditioned discovery-axis plan"'
    )
    shadow_stage = text.index(
        '"[7.52/13] Direct accepted RelationPattern "'
    )
    higher_order = text.index(
        "if args.higher_order_shadow:"
    )

    assert task_stage < shadow_stage < higher_order

    block = text[shadow_stage:higher_order]

    assert (
        '"scripts.discovery."\n'
        '                "run_direct_relationpattern_task_shadow"'
    ) in block
    assert '"--final-traversal"' in block
    assert '"--question"' in block
    assert '"--retrieval-source"' in block
    assert '"--retrieval-target"' in block
    assert '"--retrieval-top-k"' in block

    # Preserve the S18a authority boundary: requested-* is Stage 7.5 only.
    assert text.count('"--requested-source"') == 1
    assert text.count('"--requested-target"') == 1


def test_s197_shadow_cannot_change_production_inputs():
    text = _source()

    start = text.index(
        "if args.direct_relationpattern_task_shadow:"
    )
    end = text.index(
        "if args.higher_order_shadow:",
        start,
    )
    block = text[start:end]

    assert '"production_selection_changed": False' in block
    assert '"task_conditioned_axis_plan_changed": False' in block
    assert '"dual_context_changed_by_shadow": False' in block
    assert '"stage8_input_changed_by_shadow": False' in block
    assert '"novelty_authority_created": False' in block
    assert '"positive_premise_authority_created": False' in block

    assert "task_conditioned_axis_plan =" not in block
    assert "dual_context =" not in block
    assert "stage8_axis_plan_input =" not in block


def test_s197_manifest_separates_retrieval_anchors_from_task_semantics():
    text = _source()

    start = text.index(
        'runner.manifest[\n'
        '            "direct_relationpattern_task_shadow"'
    )
    end = text.index(
        "if args.higher_order_shadow:",
        start,
    )
    block = text[start:end]

    assert (
        '"query_contract": (\n'
        '                "GRAPH_RETRIEVAL_SOURCE_PLUS_TARGET"'
    ) in block
    assert '"retrieval_source": str(' in block
    assert '"retrieval_target": str(' in block
    assert (
        '"full_question_is_task_semantic_authority":\n'
        '                True'
    ) in block
