from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import dac_her.incremental_reconcile as legacy
import pipeline_core.reconcile_freshness as core


ROOT = Path(__file__).resolve().parents[1]


def test_freshness_engine_has_no_domain_dependency() -> None:
    path = (
        ROOT
        / "pipeline_core"
        / "reconcile_freshness.py"
    )

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    forbidden = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.ImportFrom,
        ):
            module = node.module or ""

            if module.startswith(
                (
                    "dac_her",
                    "domains",
                    "campaigns",
                )
            ):
                forbidden.append(
                    module
                )

        elif isinstance(
            node,
            ast.Import,
        ):
            for alias in node.names:
                if alias.name.startswith(
                    (
                        "dac_her",
                        "domains",
                        "campaigns",
                    )
                ):
                    forbidden.append(
                        alias.name
                    )

    assert forbidden == []

    assert "dac_her" not in source
    assert "data_dac" not in source
    assert "sers_au_ag" not in source


def test_legacy_normalizers_delegate_to_shared_engine() -> None:
    run_paper = {
        "paper_id": "P1",
        "enabled": True,
        "documents": [],
    }

    assert (
        legacy.IncrementalCorpusReconciler
        ._semantic_paper_payload_from_run(
            run_paper
        )
        == core.semantic_paper_payload_from_run(
            run_paper
        )
    )

    policy = {
        "logical_batch_size": 2,
        "concurrency": 3,
        "max_api_retries": 4,
        "scientific_flag": True,
    }

    assert (
        legacy.IncrementalCorpusReconciler
        ._semantic_policy_payload(
            policy
        )
        == core.semantic_policy_payload(
            policy
        )
    )


def test_full_freshness_tracks_implementation_file(
    tmp_path: Path,
) -> None:
    impl = tmp_path / "impl.py"

    impl.write_text(
        "x = 1\n",
        encoding="utf-8",
    )

    digest = hashlib.sha256(
        impl.read_bytes()
    ).hexdigest()

    run_meta = {
        "paper": {
            "paper_id": "P1",
            "enabled": True,
            "documents": [],
        },
        "document_sources": [],
        "model": "m",
        "provider": "p",
        "prompt_version": "pv",
        "prompt_sha256": "ph",
        "schema_sha256": "sh",
        "vocabularies": [],
        "policy": {},
        "domain_profile_id": "d",
        "data_root": "data",
        "chunking_sha256": "ch",
        "implementation_files": [
            {
                "relative_path": "impl.py",
                "sha256": digest,
            }
        ],
    }

    current = core.run_contract(
        run_meta
    )

    ok, reason = (
        core.run_compatibility_reason(
            run_meta,
            current,
            freshness="full",
            project_root=tmp_path,
        )
    )

    assert ok
    assert reason == (
        "full contract matches"
    )

    impl.write_text(
        "x = 2\n",
        encoding="utf-8",
    )

    ok, reason = (
        core.run_compatibility_reason(
            run_meta,
            current,
            freshness="full",
            project_root=tmp_path,
        )
    )

    assert not ok
    assert reason == (
        "full contract changed: "
        "implementation_files"
    )
