#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pipeline_core.discovery.accepted_relationpattern_export import (
    export_accepted_relationpatterns_from_canonical_graph,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _graph_from_traversal(path: Path) -> Path:
    payload = _load_json(path)
    data_root = Path(str(payload["data_root"])).expanduser()
    if not data_root.is_absolute():
        data_root = (Path.cwd() / data_root).resolve()
    return (
        data_root
        / "corpus"
        / str(payload["corpus_id"])
        / str(payload["mode"])
        / "navigation"
        / "graph.graphml"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export validator-approved accepted RelationPattern nodes from "
            "the canonical discovery graph into the legacy CSV compatibility "
            "view required by the higher-order shadow lane. No scientific "
            "authority is created or changed."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--graph", type=Path)
    source.add_argument("--traversal", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    graph = (
        args.graph.expanduser().resolve()
        if args.graph is not None
        else _graph_from_traversal(args.traversal)
    )

    audit = export_accepted_relationpatterns_from_canonical_graph(
        graph_path=graph,
        output_csv=args.output,
    )
    _write_json(args.audit_output, audit)

    print("Canonical accepted RelationPattern export complete")
    print("graph:", audit.canonical_graph)
    print(
        "accepted RelationPattern nodes:",
        audit.accepted_relationpattern_node_count,
    )
    print("validated components:", audit.validated_component_count)
    print("validation rejections:", audit.validation_rejection_count)
    print("csv:", audit.output_csv)
    print("audit:", args.audit_output)
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
