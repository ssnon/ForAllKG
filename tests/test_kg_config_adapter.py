from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from dac_her.kg_config_adapter import (
    KGConfigAdapterError,
    build_generated_paper_config,
    resolve_raw_marker_markdown,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(raw_text: str, *, paper_id: str, role: str) -> str:
    return (
        "---\n"
        f'paper_id: "{paper_id}"\n'
        f'document_role: "{role}"\n'
        "---\n"
        + raw_text
    )


def test_generated_config_points_to_raw_marker_packages_not_normalized(tmp_path: Path):
    root = tmp_path
    main_pkg = root / "data_dac" / "ingestion" / "markdown" / "P1" / "main" / "main"
    si_pkg = root / "data_dac" / "ingestion" / "markdown" / "P1" / "si_1" / "si_1"
    main_pkg.mkdir(parents=True)
    si_pkg.mkdir(parents=True)

    main_text = "# Main paper\n\nText.\n\n![Figure](image_001.png)\n"
    si_text = "# Supporting Information\n\nSupplementary Figure S1.\n"
    main_raw = main_pkg / "P1.md"
    si_raw = si_pkg / "P1_SI1.md"
    main_raw.write_text(main_text, encoding="utf-8")
    si_raw.write_text(si_text, encoding="utf-8")
    (main_pkg / "image_001.png").write_bytes(b"fake")
    (main_pkg / "P1_meta.json").write_text("{}", encoding="utf-8")

    main_norm = main_pkg / "normalized.md"
    si_norm = si_pkg / "normalized.md"
    main_norm.write_text(_normalized(main_text, paper_id="P1", role="main"), encoding="utf-8")
    si_norm.write_text(
        _normalized(si_text, paper_id="P1", role="supporting_information"),
        encoding="utf-8",
    )

    frozen = {
        "schema_version": "graphagentsdac-frozen-corpus-v01",
        "corpus_id": "demo",
        "documents": [
            {
                "paper_id": "P1",
                "title": "Demo",
                "annotator": "tester",
                "qc_status": "passed",
                "alias_paper_ids": [],
                "main_markdown": str(main_norm.relative_to(root)),
                "supporting_markdown": [str(si_norm.relative_to(root))],
                "content_fingerprint": {
                    "main_markdown_sha256": _sha(main_norm),
                    "supporting_markdown": [
                        {
                            "path": str(si_norm.relative_to(root)),
                            "sha256": _sha(si_norm),
                        }
                    ],
                },
            }
        ],
    }

    output = root / "data_dac" / "generated_configs" / "demo" / "papers.yaml"
    result = build_generated_paper_config(
        frozen,
        frozen_manifest_path="frozen.json",
        output_yaml=output,
        project_root=root,
    )

    config = yaml.safe_load(result.papers_yaml.read_text(encoding="utf-8"))
    docs = config["papers"]["P1"]["documents"]
    main = docs[0]
    si = docs[1]

    assert main["markdown_file"] == "P1.md"
    assert main["package_dir"].endswith("/P1/main/main")
    assert main["selection"] == {"mode": "whole_document"}
    assert main["metadata_file"] == "P1_meta.json"
    assert "normalized.md" not in json.dumps(config, ensure_ascii=False)

    assert si["markdown_file"] == "P1_SI1.md"
    assert si["selection"] == {
        "mode": "referenced_blocks",
        "fallback": "skip",
        "reference_scope": "whole_main",
    }
    assert result.paper_ids == ("P1",)

    adapter = json.loads(result.adapter_manifest.read_text(encoding="utf-8"))
    assert adapter["papers"][0]["documents"][0]["raw_markdown"].endswith("P1.md")
    assert adapter["papers"][0]["documents"][0]["package_dir"].endswith("/P1/main/main")


def test_raw_marker_resolution_preserves_leading_newline(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    raw = pkg / "main.md"
    raw_text = "\n# A\n\nBody.\n"
    raw.write_text(raw_text, encoding="utf-8")
    normalized = pkg / "normalized.md"
    normalized.write_text(
        _normalized(raw_text, paper_id="P", role="main"),
        encoding="utf-8",
    )

    assert resolve_raw_marker_markdown(normalized) == raw


def test_raw_marker_resolution_rejects_changed_frozen_markdown(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    raw = pkg / "paper.md"
    raw.write_text("# A\n", encoding="utf-8")
    normalized = pkg / "normalized.md"
    normalized.write_text(_normalized("# A\n", paper_id="P", role="main"), encoding="utf-8")

    assert resolve_raw_marker_markdown(normalized) == raw

    frozen = {
        "schema_version": "graphagentsdac-frozen-corpus-v01",
        "corpus_id": "demo",
        "documents": [
            {
                "paper_id": "P",
                "title": "P",
                "annotator": "x",
                "qc_status": "passed",
                "main_markdown": str(normalized),
                "supporting_markdown": [],
                "content_fingerprint": {
                    "main_markdown_sha256": "0" * 64,
                    "supporting_markdown": [],
                },
            }
        ],
    }
    with pytest.raises(KGConfigAdapterError, match="Frozen Markdown changed"):
        build_generated_paper_config(
            frozen,
            frozen_manifest_path="frozen.json",
            output_yaml=tmp_path / "papers.yaml",
            project_root=tmp_path,
        )
