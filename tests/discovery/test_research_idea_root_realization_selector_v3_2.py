from types import SimpleNamespace

from pipeline_core.discovery.research_idea_offspring_execution import (
    select_offspring_for_realization,
)


def test_root_research_idea_uses_execution_parent_bucket_for_realization_quota():
    idea_id = "research_idea:root"
    node = SimpleNamespace(
        idea_id=idea_id,
        kernel_sha256="kernel-root",
        parent_idea_ids=[],
    )
    semantic = SimpleNamespace(
        idea_id=idea_id,
        channel="TRANSFORM",
        accepted_for_realization=True,
        parent_idea_ids=[idea_id],
        transition=SimpleNamespace(identity_relation="DIFFERENT_IDEA"),
    )
    execution = SimpleNamespace(
        offspring_nodes=[node],
        semantic_records=[semantic],
    )

    selected = select_offspring_for_realization(
        execution,
        max_realizations=1,
        max_per_parent=1,
    )

    assert node.parent_idea_ids == []
    assert selected == [idea_id]
