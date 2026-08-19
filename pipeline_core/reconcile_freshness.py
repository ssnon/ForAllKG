from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline_core.reconcile_runtime import (
    FreshnessPolicy,
    sha256_file,
)


def semantic_paper_payload_from_run(
    value: Any,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    documents = value.get(
        "documents"
    )

    if not isinstance(
        documents,
        list,
    ):
        return None

    normalized: list[
        dict[str, Any]
    ] = []

    for raw in documents:
        if not isinstance(
            raw,
            dict,
        ):
            return None

        selection = (
            raw.get("selection")
            or {}
        )

        figure = (
            raw.get(
                "figure_processing"
            )
            or {}
        )

        normalized.append({
            "document_id": str(
                raw.get(
                    "document_id"
                )
                or ""
            ),
            "role": str(
                raw.get("role")
                or ""
            ),
            "selection": {
                "mode": str(
                    selection.get(
                        "mode"
                    )
                    or ""
                ),
                "headings": list(
                    selection.get(
                        "headings"
                    )
                    or []
                ),
                "fallback": str(
                    selection.get(
                        "fallback"
                    )
                    or "error"
                ),
                "reference_scope": str(
                    selection.get(
                        "reference_scope"
                    )
                    or "selected_main"
                ),
            },
            "figure_processing": {
                "mode": str(
                    figure.get(
                        "mode"
                    )
                    or "caption_first"
                ),
                "vision_assets": list(
                    figure.get(
                        "vision_assets"
                    )
                    or []
                ),
                "vision_model": (
                    figure.get(
                        "vision_model"
                    )
                ),
            },
        })

    return {
        "paper_id": str(
            value.get("paper_id")
            or ""
        ),
        "enabled": bool(
            value.get(
                "enabled",
                True,
            )
        ),
        "documents": normalized,
    }


def semantic_policy_payload(
    value: Any,
) -> dict[str, Any] | None:
    if not isinstance(
        value,
        dict,
    ):
        return None

    # Scheduling/runtime controls change execution
    # cost and timing, not the intended semantic
    # extraction contract.
    operational = {
        "logical_batch_size",
        "concurrency",
        "max_api_retries",
    }

    return {
        str(key): item
        for key, item
        in value.items()
        if str(key)
        not in operational
    }


def run_contract(
    run_meta: dict[str, Any],
) -> dict[str, Any]:
    source = {
        "paper": (
            semantic_paper_payload_from_run(
                run_meta.get("paper")
            )
        ),
        "document_sources": (
            run_meta.get(
                "document_sources"
            )
        ),
    }

    semantic = {
        **source,
        "model": run_meta.get(
            "model"
        ),
        "provider": run_meta.get(
            "provider"
        ),
        "prompt_version": (
            run_meta.get(
                "prompt_version"
            )
        ),
        "prompt_sha256": (
            run_meta.get(
                "prompt_sha256"
            )
        ),
        "schema_sha256": (
            run_meta.get(
                "schema_sha256"
            )
        ),
        "vocabularies": (
            run_meta.get(
                "vocabularies"
            )
        ),
        "policy": (
            semantic_policy_payload(
                run_meta.get(
                    "policy"
                )
            )
        ),
    }

    return {
        "source": source,
        "semantic": semantic,
        "full": {
            **semantic,
            "domain_profile_id": (
                run_meta.get(
                    "domain_profile_id"
                )
            ),
            "data_root": str(
                run_meta.get(
                    "data_root"
                )
                or ""
            ),
            "chunking_sha256": (
                run_meta.get(
                    "chunking_sha256"
                )
            ),
            "policy_full": (
                run_meta.get(
                    "policy"
                )
            ),
        },
    }


def contract_diff(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> list[str]:
    return [
        key
        for key in expected
        if (
            expected.get(key)
            != actual.get(key)
        )
    ]


def run_compatibility_reason(
    run_meta: dict[str, Any],
    current: dict[str, Any],
    *,
    freshness: FreshnessPolicy,
    project_root: str | Path,
) -> tuple[bool, str]:
    actual = (
        run_contract(run_meta)[
            freshness
        ]
    )

    expected = current[
        freshness
    ]

    differences = contract_diff(
        expected,
        actual,
    )

    if freshness == "full":
        # Full mode additionally verifies every
        # implementation file recorded by the run
        # against the current checkout.
        implementation_changes: list[
            str
        ] = []

        rows = run_meta.get(
            "implementation_files"
        )

        if isinstance(rows, list):
            for row in rows:
                if not isinstance(
                    row,
                    dict,
                ):
                    implementation_changes.append(
                        "implementation_files"
                    )
                    break

                relative = str(
                    row.get(
                        "relative_path"
                    )
                    or ""
                )

                path = (
                    Path(project_root)
                    / relative
                )

                if (
                    not relative
                    or not path.is_file()
                    or row.get(
                        "sha256"
                    )
                    != sha256_file(path)
                ):
                    implementation_changes.append(
                        "implementation_files"
                    )
                    break

        if implementation_changes:
            differences.extend(
                implementation_changes
            )

    if differences:
        unique = ", ".join(
            dict.fromkeys(
                differences
            )
        )

        return (
            False,
            f"{freshness} contract "
            f"changed: {unique}",
        )

    return (
        True,
        f"{freshness} contract matches",
    )
