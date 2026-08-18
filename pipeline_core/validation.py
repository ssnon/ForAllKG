from __future__ import annotations

from dac_her.schemas import KnowledgeGraph


def validate_graph_provenance(
    graph: KnowledgeGraph,
    *,
    paper_id: str,
    chunk_id: str,
    section: str,
    document_id: str | None = None,
    document_role: str | None = None,
    page_ids: tuple[int, ...] | list[int] | None = None,
    asset_ids: tuple[str, ...] | list[str] | None = None,
) -> None:
    expected_document_id = document_id or graph.document_id
    expected_document_role = document_role or graph.document_role
    expected_page_ids = list(page_ids) if page_ids is not None else graph.page_ids
    expected_asset_ids = list(asset_ids) if asset_ids is not None else graph.asset_ids

    checks = (
        ("paper_id", graph.paper_id, paper_id),
        ("chunk_id", graph.chunk_id, chunk_id),
        ("section", graph.section, section),
        ("document_id", graph.document_id, expected_document_id),
        ("document_role", graph.document_role, expected_document_role),
    )
    messages = [
        f"Incorrect top-level {name}. Expected {expected!r}; received {actual!r}."
        for name, actual, expected in checks
        if actual != expected
    ]

    if graph.page_ids != expected_page_ids:
        messages.append(
            "Incorrect top-level page_ids. "
            f"Expected {expected_page_ids!r}; received {graph.page_ids!r}."
        )
    if graph.asset_ids != expected_asset_ids:
        messages.append(
            "Incorrect top-level asset_ids. "
            f"Expected {expected_asset_ids!r}; received {graph.asset_ids!r}."
        )
    if messages:
        raise ValueError("\n".join(messages))

    node_ids = graph.all_node_ids()
    missing_sources = sorted({
        edge.source for edge in graph.edges if edge.source not in node_ids
    })
    missing_targets = sorted({
        edge.target for edge in graph.edges if edge.target not in node_ids
    })
    if missing_sources or missing_targets:
        details = [
            *(f"missing source: {node_id}" for node_id in missing_sources),
            *(f"missing target: {node_id}" for node_id in missing_targets),
        ]
        raise ValueError(
            "Edges reference undefined nodes:\n" + "\n".join(details)
        )
