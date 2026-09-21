from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class ReplicationStage:
    stage_id: str
    module: str
    args: tuple[str, ...]
    outputs: tuple[Path, ...]

    def command(self, python_executable: str | None = None) -> list[str]:
        return [python_executable or sys.executable, "-m", self.module, *self.args]


@dataclass(frozen=True)
class SparseReplicationPlan:
    case_key: str
    canonical_dir: Path
    context: Path
    relational_portfolio: Path
    reframe_shadow: Path
    proxy_shadow: Path | None
    contradiction_shadow: Path | None
    stages: tuple[ReplicationStage, ...]


@dataclass
class StageExecutionRecord:
    stage_id: str
    status: str
    command: list[str]
    outputs: list[str]


@dataclass
class TaskExecutionReport:
    case_key: str
    canonical_dir: str
    records: list[StageExecutionRecord] = field(default_factory=list)


_REQUIRED = {
    "context": "hypothesis.context.json",
    "relational_portfolio": "hypothesis_axis_a4.portfolio.json",
}


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def discover_reframe_shadow(canonical_dir: Path) -> Path:
    preferred = _first_existing(
        [
            canonical_dir / "scientific_reframing_shadow.json",
            canonical_dir / "scientific_reframing_shadow_s38.json",
        ]
    )
    if preferred is not None:
        return preferred
    candidates = sorted(
        path
        for path in canonical_dir.glob("scientific_reframing_shadow*.json")
        if path.is_file()
        and "critic" not in path.name
        and "portfolio" not in path.name
    )
    if not candidates:
        raise FileNotFoundError(
            f"no scientific reframing shadow found in {canonical_dir}"
        )
    return candidates[-1]


def _require(canonical_dir: Path, filename: str, label: str) -> Path:
    path = canonical_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _optional(canonical_dir: Path, filename: str) -> Path | None:
    path = canonical_dir / filename
    return path if path.is_file() else None


def _path_arg(flag: str, path: Path) -> tuple[str, str]:
    return flag, str(path)


def build_sparse_replication_plan(
    *,
    canonical_dir: Path,
    case_key: str | None = None,
    max_comparisons: int = 12,
) -> SparseReplicationPlan:
    if max_comparisons < 1:
        raise ValueError("max_comparisons must be >= 1")
    canonical_dir = canonical_dir.expanduser().resolve()
    context = _require(
        canonical_dir,
        _REQUIRED["context"],
        "hypothesis context",
    )
    relational = _require(
        canonical_dir,
        _REQUIRED["relational_portfolio"],
        "relational hypothesis portfolio",
    )
    reframe = discover_reframe_shadow(canonical_dir)
    proxy = _optional(canonical_dir, "scientific_proxy_challenge_shadow.json")
    contradiction = _optional(
        canonical_dir,
        "scientific_contradiction_resolution_shadow.json",
    )

    mode_contrast = canonical_dir / "scientific_reframing_mode_contrast.json"
    reframing_portfolio = canonical_dir / "scientific_reframing_reasoning_portfolio.json"
    cross_lane = canonical_dir / "scientific_cross_lane_reasoning_portfolio.json"
    ablation = canonical_dir / "scientific_reasoning_ablation_packet.json"
    ablation_key = canonical_dir / "scientific_reasoning_ablation_key.json"
    normalized = canonical_dir / "scientific_reasoning_ablation_schema_normalized_packet.json"
    normalization_report = (
        canonical_dir / "scientific_reasoning_ablation_schema_normalization_report.json"
    )
    matched = canonical_dir / "scientific_reasoning_ablation_generalized_matched_packet.json"
    matched_key = canonical_dir / "scientific_reasoning_ablation_generalized_matched_key.json"
    matched_report = (
        canonical_dir / "scientific_reasoning_ablation_generalized_matched_build_report.json"
    )

    mode_args: list[str] = [
        "--reframe-shadow",
        str(reframe),
        "--output",
        str(mode_contrast),
    ]
    if proxy is not None:
        mode_args.extend(["--proxy-shadow", str(proxy)])
    if contradiction is not None:
        mode_args.extend(["--contradiction-shadow", str(contradiction)])

    cross_lane_args: list[str] = [
        "--context",
        str(context),
        "--relational-portfolio",
        str(relational),
        "--reframing-portfolio",
        str(reframing_portfolio),
        "--reframe-shadow",
        str(reframe),
        "--output",
        str(cross_lane),
    ]
    if proxy is not None:
        cross_lane_args.extend(["--proxy-shadow", str(proxy)])
    if contradiction is not None:
        cross_lane_args.extend(["--contradiction-shadow", str(contradiction)])

    ablation_args: list[str] = [
        "--context",
        str(context),
        "--cross-lane-portfolio",
        str(cross_lane),
        "--relational-portfolio",
        str(relational),
        "--reframe-shadow",
        str(reframe),
        "--output",
        str(ablation),
        "--key-output",
        str(ablation_key),
    ]
    if proxy is not None:
        ablation_args.extend(["--proxy-shadow", str(proxy)])
    if contradiction is not None:
        ablation_args.extend(["--contradiction-shadow", str(contradiction)])

    stages = (
        ReplicationStage(
            stage_id="mode_contrast",
            module="scripts.discovery.compare_scientific_reframing_modes",
            args=tuple(mode_args),
            outputs=(mode_contrast,),
        ),
        ReplicationStage(
            stage_id="reframing_portfolio",
            module="scripts.discovery.build_scientific_reasoning_portfolio",
            args=(
                "--mode-contrast",
                str(mode_contrast),
                "--output",
                str(reframing_portfolio),
            ),
            outputs=(reframing_portfolio,),
        ),
        ReplicationStage(
            stage_id="cross_lane_portfolio",
            module="scripts.discovery.build_cross_lane_scientific_reasoning_portfolio",
            args=tuple(cross_lane_args),
            outputs=(cross_lane,),
        ),
        ReplicationStage(
            stage_id="ablation_packet",
            module="scripts.discovery.build_scientific_reasoning_ablation_packet",
            args=tuple(ablation_args),
            outputs=(ablation, ablation_key),
        ),
        ReplicationStage(
            stage_id="schema_normalization",
            module=(
                "scripts.discovery."
                "build_schema_normalized_scientific_reasoning_ablation_packet"
            ),
            args=(
                "--packet",
                str(ablation),
                "--output",
                str(normalized),
                "--report-output",
                str(normalization_report),
            ),
            outputs=(normalized, normalization_report),
        ),
        ReplicationStage(
            stage_id="generalized_matching",
            module=(
                "scripts.discovery."
                "build_generalized_matched_count_scientific_reasoning_ablation_packet"
            ),
            args=(
                "--packet",
                str(normalized),
                "--key",
                str(ablation_key),
                "--max-comparisons",
                str(max_comparisons),
                "--output",
                str(matched),
                "--output-key",
                str(matched_key),
                "--report",
                str(matched_report),
            ),
            outputs=(matched, matched_key, matched_report),
        ),
    )
    return SparseReplicationPlan(
        case_key=case_key or canonical_dir.parent.parent.name,
        canonical_dir=canonical_dir,
        context=context,
        relational_portfolio=relational,
        reframe_shadow=reframe,
        proxy_shadow=proxy,
        contradiction_shadow=contradiction,
        stages=stages,
    )


def _outputs_complete(stage: ReplicationStage) -> bool:
    return bool(stage.outputs) and all(path.is_file() for path in stage.outputs)


def execute_sparse_replication_plan(
    *,
    plan: SparseReplicationPlan,
    repository_root: Path,
    force: bool = False,
    python_executable: str | None = None,
    runner: Callable[..., object] = subprocess.run,
) -> TaskExecutionReport:
    repository_root = repository_root.expanduser().resolve()
    report = TaskExecutionReport(
        case_key=plan.case_key,
        canonical_dir=str(plan.canonical_dir),
    )
    for stage in plan.stages:
        command = stage.command(python_executable)
        if not force and _outputs_complete(stage):
            report.records.append(
                StageExecutionRecord(
                    stage_id=stage.stage_id,
                    status="skipped_existing_outputs",
                    command=command,
                    outputs=[str(path) for path in stage.outputs],
                )
            )
            continue
        runner(command, cwd=repository_root, check=True)
        missing = [path for path in stage.outputs if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"stage {stage.stage_id} completed without expected outputs: "
                + ", ".join(str(path) for path in missing)
            )
        report.records.append(
            StageExecutionRecord(
                stage_id=stage.stage_id,
                status="executed",
                command=command,
                outputs=[str(path) for path in stage.outputs],
            )
        )
    return report


def plan_to_dict(plan: SparseReplicationPlan) -> dict[str, object]:
    return {
        "case_key": plan.case_key,
        "canonical_dir": str(plan.canonical_dir),
        "context": str(plan.context),
        "relational_portfolio": str(plan.relational_portfolio),
        "reframe_shadow": str(plan.reframe_shadow),
        "proxy_shadow": str(plan.proxy_shadow) if plan.proxy_shadow else None,
        "contradiction_shadow": (
            str(plan.contradiction_shadow) if plan.contradiction_shadow else None
        ),
        "stages": [
            {
                "stage_id": stage.stage_id,
                "module": stage.module,
                "args": list(stage.args),
                "outputs": [str(path) for path in stage.outputs],
            }
            for stage in plan.stages
        ],
    }


def execution_report_to_dict(report: TaskExecutionReport) -> dict[str, object]:
    return {
        "case_key": report.case_key,
        "canonical_dir": report.canonical_dir,
        "records": [
            {
                "stage_id": row.stage_id,
                "status": row.status,
                "command": row.command,
                "outputs": row.outputs,
            }
            for row in report.records
        ],
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
