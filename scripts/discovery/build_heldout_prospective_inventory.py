from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEVELOPMENT_CASES = {
    "sers_raman_orientation_direct_ho_reference",
    "sers_au_ag_structure_broad_control",
    "dac_her_multifactor_optimal_regime",
    "dac_her_charge_transfer_conditional",
    "dac_her_metal_distance_stability",
    "dac_her_low_overpotential_comparison",
}


def _load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _write(path: Path, value: Any) -> None:
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


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Inventory candidate held-out runs without selecting "
            "or evaluating them. The six architecture-development "
            "cases are excluded by name."
        )
    )
    p.add_argument(
        "--evaluation-root",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return p


def main() -> int:
    args = parser().parse_args()
    root = args.evaluation_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    rows = []
    for context_path in root.rglob(
        "hypothesis.context.json"
    ):
        run = context_path.parent
        if run.name in DEVELOPMENT_CASES:
            continue

        legacy = (
            run / "hypothesis_axis_a4.portfolio.json"
        )
        augmented = run / "frontier_augmented"
        population = (
            augmented
            / "frontier_idea_population.full.shadow.json"
        )
        frontier_audit = (
            augmented / "frontier_exploration.audit.json"
        )
        evolution = (
            augmented
            / "idea_evolution"
            / "idea_evolution.shadow.json"
        )
        scientific = (
            augmented / "scientific_portfolio"
        )
        selected = (
            scientific
            / "materialized.shadow.portfolio.json"
        )

        if not (
            legacy.is_file()
            and population.is_file()
            and frontier_audit.is_file()
            and evolution.is_file()
        ):
            continue

        try:
            context = _load(context_path)
            legacy_payload = _load(legacy)
            selected_payload = (
                _load(selected)
                if selected.is_file()
                else None
            )
        except Exception:
            continue

        rows.append(
            {
                "run_dir": str(run.resolve()),
                "run_name": run.name,
                "domain_profile_id":
                    context.get("domain_profile_id"),
                "question":
                    context.get("question"),
                "context_id":
                    context.get("context_id"),
                "legacy_hypothesis_count":
                    len(
                        legacy_payload.get(
                            "hypotheses",
                            [],
                        )
                    ),
                "preexisting_portfolio_selected_output":
                    selected.is_file(),
                "preexisting_selected_hypothesis_count":
                    (
                        len(
                            selected_payload.get(
                                "hypotheses",
                                [],
                            )
                        )
                        if isinstance(
                            selected_payload,
                            dict,
                        )
                        else None
                    ),
                "required_frozen_inputs_present": True,
            }
        )

    rows.sort(
        key=lambda row: (
            str(
                row.get("domain_profile_id")
                or ""
            ),
            str(row.get("run_name") or ""),
        )
    )

    payload = {
        "schema_version":
            "held-out-prospective-inventory-v1",
        "evaluation_root": str(root),
        "development_case_names_excluded":
            sorted(DEVELOPMENT_CASES),
        "candidate_count": len(rows),
        "candidates": rows,
        "selection_performed": False,
        "scientific_quality_ranking_performed": False,
        "production_selection_authority": False,
    }
    _write(
        args.output.expanduser().resolve(),
        payload,
    )

    print("Held-out prospective inventory complete")
    print("candidates:", len(rows))
    for row in rows:
        print(
            row["domain_profile_id"],
            "|",
            row["run_name"],
            "| legacy=",
            row["legacy_hypothesis_count"],
            "| preexisting selected=",
            row["preexisting_selected_hypothesis_count"],
        )
    print("SELECTION_PERFORMED=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print(
        "output:",
        args.output.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
