"""Application-side loading for discovery traversal artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import networkx as nx

from domains.extraction_registry import (
    get_extraction_adapter,
)
from domains.registry import get_domain_profile
from pipeline_core.domain.domain_profile import (
    ScientificDomainProfile,
)


def _resolve_traversal_data_root(
    payload: Mapping[str, Any],
    *,
    project_root: str | Path,
    domain_profile_id: str | None = None,
    data_root: str | Path | None = None,
) -> tuple[ScientificDomainProfile, Path]:
    traversal_domain = str(
        payload.get("domain_profile_id") or ""
    ).strip()

    requested_domain = str(
        domain_profile_id
        or traversal_domain
        or ""
    ).strip()

    if not requested_domain:
        raise ValueError(
            "domain_profile_id is required via the caller or traversal artifact."
        )

    if (
        traversal_domain
        and requested_domain != traversal_domain
    ):
        raise ValueError(
            "Requested discovery domain profile does not match "
            "traversal artifact: "
            f"requested={requested_domain!r}, "
            f"traversal={traversal_domain!r}"
        )

    profile = get_domain_profile(
        requested_domain
    )

    extraction_adapter = (
        get_extraction_adapter(
            profile.profile_id
        )
    )

    root = Path(
        data_root
        or payload.get("data_root")
        or extraction_adapter.default_data_root
    )

    if not root.is_absolute():
        root = (
            Path(project_root)
            / root
        )

    return profile, root


def load_traversal_with_graph(
    path: str | Path,
    *,
    project_root: str | Path = ".",
    domain_profile_id: str | None = None,
    data_root: str | Path | None = None,
) -> tuple[
    str,
    dict[str, Any],
    nx.DiGraph,
]:
    path = Path(path)

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"expected traversal JSON object: {path}"
        )

    corpus_id = str(
        payload.get(
            "corpus_id",
            "",
        )
    ).strip()

    mode = str(
        payload.get(
            "mode",
            "",
        )
    ).strip()

    if not corpus_id or not mode:
        raise ValueError(
            "traversal is missing corpus_id/mode: "
            f"{path}"
        )

    _, root = (
        _resolve_traversal_data_root(
            payload,
            project_root=project_root,
            domain_profile_id=(
                domain_profile_id
            ),
            data_root=data_root,
        )
    )

    graph_path = (
        root
        / "corpus"
        / corpus_id
        / mode
        / "navigation"
        / "graph.graphml"
    )

    graph = nx.read_graphml(
        graph_path
    )

    return (
        str(path),
        payload,
        graph,
    )


def load_semantic_index_for_traversal(
    payload: dict[str, Any],
    *,
    project_root: str | Path = ".",
    domain_profile_id: str | None = None,
    data_root: str | Path | None = None,
) -> Any | None:
    """Load an existing node index without instantiating a model."""

    corpus_id = str(
        payload.get(
            "corpus_id",
            "",
        )
    ).strip()

    mode = str(
        payload.get(
            "mode",
            "",
        )
    ).strip()

    if not corpus_id or not mode:
        return None

    _, root = (
        _resolve_traversal_data_root(
            payload,
            project_root=project_root,
            domain_profile_id=(
                domain_profile_id
            ),
            data_root=data_root,
        )
    )

    index_dir = (
        root
        / "corpus"
        / corpus_id
        / mode
        / "navigation"
        / "node_index"
    )

    if not (
        index_dir
        / "manifest.json"
    ).exists():
        return None

    try:
        from pipeline_core.discovery.node_mapping import (
            load_node_embedding_index,
        )

        return (
            load_node_embedding_index(
                index_dir
            )
        )

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ):
        return None
