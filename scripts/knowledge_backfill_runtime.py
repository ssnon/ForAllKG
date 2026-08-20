from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import yaml


KnowledgeTargetStatus = Literal[
    "STRICT_USABLE",
    "BRIDGE_USEFUL",
    "CORPUS_ELIGIBLE",
]


class KnowledgeBackfillError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise KnowledgeBackfillError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise KnowledgeBackfillError(f"Expected JSONL objects: {path}")
        rows.append(value)
    return rows


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_work_ids(path: str | Path) -> list[str]:
    rows = _read_jsonl(Path(path))
    ids = [str(row.get("work_id") or "").strip() for row in rows]
    if any(not value for value in ids):
        raise KnowledgeBackfillError(f"selected_works contains missing work_id: {path}")
    if len(ids) != len(set(ids)):
        raise KnowledgeBackfillError(f"selected_works contains duplicate work_id: {path}")
    return ids


def outcome_meets_target(
    outcome: dict[str, Any],
    target_status: KnowledgeTargetStatus,
) -> bool:
    strict_usable = str(outcome.get("strict_status") or "") == "STRICT_USABLE"
    projection_usable = (
        str(outcome.get("projection_status") or "") == "PROJECTION_USABLE"
    )
    corpus_eligible = bool(outcome.get("corpus_eligible"))
    bridge_useful = str(outcome.get("bridge_status") or "") == "BRIDGE_USEFUL"

    if target_status == "STRICT_USABLE":
        return strict_usable
    if target_status == "BRIDGE_USEFUL":
        # A Bridge result only counts toward the production target if it can
        # actually reach the corpus.  This prevents a projection failure from
        # being hidden behind a successful Bridge extraction.
        return strict_usable and bridge_useful and projection_usable and corpus_eligible
    if target_status == "CORPUS_ELIGIBLE":
        return strict_usable and projection_usable and corpus_eligible
    raise KnowledgeBackfillError(f"Unsupported target status: {target_status}")


def summarize_outcomes(
    rows: Sequence[dict[str, Any]],
    *,
    target_status: KnowledgeTargetStatus,
) -> dict[str, Any]:
    latest_by_paper: dict[str, dict[str, Any]] = {}
    for row in rows:
        paper_id = str(row.get("paper_id") or "").strip()
        if paper_id:
            latest_by_paper[paper_id] = dict(row)

    latest = list(latest_by_paper.values())
    target_ids = sorted(
        str(row["paper_id"])
        for row in latest
        if outcome_meets_target(row, target_status)
    )
    strict_usable = sum(
        str(row.get("strict_status") or "") == "STRICT_USABLE" for row in latest
    )
    bridge_useful = sum(
        str(row.get("bridge_status") or "") == "BRIDGE_USEFUL" for row in latest
    )
    bridge_empty = sum(
        str(row.get("bridge_status") or "") == "BRIDGE_EMPTY" for row in latest
    )
    corpus_eligible = sum(bool(row.get("corpus_eligible")) for row in latest)
    return {
        "paper_outcome_count": len(latest),
        "strict_usable_count": strict_usable,
        "bridge_useful_count_raw": bridge_useful,
        "bridge_empty_count": bridge_empty,
        "corpus_eligible_count": corpus_eligible,
        "target_status": target_status,
        "target_status_count": len(target_ids),
        "target_status_paper_ids": target_ids,
    }


def _apportion_axis_targets(
    *,
    axes: Sequence[dict[str, Any]],
    source_target_total: int,
    target_total: int,
) -> list[dict[str, Any]]:
    """Scale quota targets while preserving the source quota proportions.

    AcquisitionProfile requires sum(axis.target_selected) <= selection.target_total.
    Knowledge backfill creates smaller/larger round-local target totals, so copying
    the production quotas verbatim can make the generated profile invalid.

    Preserve the production profile's quota-vs-global-fill fraction, then use the
    Hamilton/largest-remainder method to assign the integer per-axis quotas in a
    deterministic axis-order tie break.
    """
    if source_target_total < 1:
        raise KnowledgeBackfillError(
            "source acquisition selection.target_total must be >= 1"
        )

    copied = [dict(axis) for axis in axes]
    original_targets: list[int] = []
    for axis in copied:
        raw = axis.get("target_selected", 0)
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise KnowledgeBackfillError(
                f"invalid axis target_selected: {raw!r}"
            ) from exc
        if value < 0:
            raise KnowledgeBackfillError("axis target_selected must be >= 0")
        original_targets.append(value)

    quota_sum = sum(original_targets)
    if quota_sum > source_target_total:
        raise KnowledgeBackfillError(
            "source axis quota sum exceeds source selection.target_total"
        )
    if quota_sum == 0:
        for axis in copied:
            axis["target_selected"] = 0
        return copied

    # Preserve the original fraction of the corpus reserved for primary-axis
    # quotas.  floor(x + 0.5) gives deterministic half-up rounding and avoids
    # Python's banker rounding at .5.
    quota_fraction = quota_sum / float(source_target_total)
    dynamic_quota_total = min(
        target_total,
        max(0, int(math.floor(target_total * quota_fraction + 0.5))),
    )
    if dynamic_quota_total == 0:
        for axis in copied:
            axis["target_selected"] = 0
        return copied

    raw_targets = [
        dynamic_quota_total * value / float(quota_sum)
        for value in original_targets
    ]
    apportioned = [int(math.floor(value)) for value in raw_targets]
    remaining = dynamic_quota_total - sum(apportioned)
    order = sorted(
        range(len(copied)),
        key=lambda index: (
            -(raw_targets[index] - apportioned[index]),
            index,
        ),
    )
    for index in order[:remaining]:
        apportioned[index] += 1

    for axis, value in zip(copied, apportioned):
        axis["target_selected"] = int(value)
    return copied


def write_dynamic_target_profile(
    *,
    source_profile: str | Path,
    output_path: str | Path,
    target_total: int,
) -> Path:
    if target_total < 1:
        raise KnowledgeBackfillError("dynamic acquisition target_total must be >= 1")
    source = Path(source_profile)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise KnowledgeBackfillError(f"Acquisition profile must be a mapping: {source}")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise KnowledgeBackfillError("Acquisition profile is missing selection mapping")
    try:
        source_target_total = int(selection.get("target_total"))
    except (TypeError, ValueError) as exc:
        raise KnowledgeBackfillError(
            "Acquisition profile selection.target_total must be an integer"
        ) from exc
    axes = payload.get("axes")
    if not isinstance(axes, list) or not all(isinstance(axis, dict) for axis in axes):
        raise KnowledgeBackfillError("Acquisition profile is missing axes list")

    selection = dict(selection)
    selection["target_total"] = int(target_total)
    payload = dict(payload)
    payload["selection"] = selection
    payload["axes"] = _apportion_axis_targets(
        axes=axes,
        source_target_total=source_target_total,
        target_total=target_total,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output


@dataclass(frozen=True)
class KnowledgeBackfillOptions:
    target_count: int
    target_status: KnowledgeTargetStatus = "BRIDGE_USEFUL"
    oversample_factor: float = 1.0
    max_rounds: int = 5
    max_extra_candidates: int = 100
    extract_concurrency: int = 4
    bridge_concurrency: int = 4
    heartbeat_seconds: float = 30.0
    retry_failed_acquisition: bool = False
    retry_access_misses: bool = False
    retry_failed_materialization: bool = False
    retry_failed_supplementary: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.target_count < 1:
            raise ValueError("target_count must be >= 1")
        if self.oversample_factor < 1.0:
            raise ValueError("oversample_factor must be >= 1.0")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if self.max_extra_candidates < 1:
            raise ValueError("max_extra_candidates must be >= 1")
        if self.extract_concurrency < 1 or self.bridge_concurrency < 1:
            raise ValueError("concurrency must be >= 1")


@dataclass(frozen=True)
class KnowledgeBackfillPaths:
    acquisition_profile: Path
    backfill_policy: Path
    source_policy: Path
    catalog: Path
    m2_assessments: Path
    quality_assessments: Path
    quality_gate_report: Path
    starting_m3_dir: Path
    materialization_policy: Path
    m4_dir: Path
    m4_config: Path
    gate_policy: Path
    m4_5_dir: Path
    strict_config: Path
    data_root: Path
    run_root: Path
    m3_1_dir: Path | None = None
    supplementary_policy: Path | None = None


CommandRunner = Callable[[list[str], str], bool]


class KnowledgeAwareBackfillCoordinator:
    """Feedback loop from Strict/Bridge outcomes into acquisition reserve fill.

    M3.2 remains the authority for scientific ranking, quality eligibility, OA
    resolution, and axis-aware reserve choice.  This coordinator changes only
    the stopping criterion: acquire more candidates until the requested
    knowledge-layer outcome count is reached or the configured safety budget is
    exhausted.
    """

    def __init__(
        self,
        *,
        project_root: str | Path,
        corpus_id: str,
        domain_profile: str,
        paper_id_prefix: str,
        paths: KnowledgeBackfillPaths,
        options: KnowledgeBackfillOptions,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.corpus_id = str(corpus_id)
        self.domain_profile = str(domain_profile)
        self.paper_id_prefix = str(paper_id_prefix)
        self.options = options
        self.paths = self._resolve_paths(paths)
        self.paths.run_root.mkdir(parents=True, exist_ok=True)
        self.command_records: list[dict[str, Any]] = []
        self.command_runner = command_runner or self._run_command

        profile = yaml.safe_load(
            self.paths.acquisition_profile.read_text(encoding="utf-8")
        )
        if not isinstance(profile, dict):
            raise KnowledgeBackfillError("Acquisition profile must be a mapping")
        self.profile_id = str(profile.get("profile_id") or "").strip()
        if not self.profile_id:
            raise KnowledgeBackfillError("Acquisition profile is missing profile_id")
        profile_domain = str(profile.get("domain_profile_id") or "").strip()
        if profile_domain and profile_domain != self.domain_profile:
            raise KnowledgeBackfillError(
                f"Acquisition/domain mismatch: {profile_domain!r} != "
                f"{self.domain_profile!r}"
            )

        self.outcomes_path = (
            self.paths.data_root
            / "pipeline_runs"
            / self.corpus_id
            / "strict_bridge"
            / "paper_outcomes.jsonl"
        )
        if not self.outcomes_path.is_file():
            raise KnowledgeBackfillError(
                "An initial Strict-Bridge run is required before knowledge-aware "
                f"backfill: {self.outcomes_path}"
            )

        for required in (
            self.paths.acquisition_profile,
            self.paths.backfill_policy,
            self.paths.source_policy,
            self.paths.catalog,
            self.paths.m2_assessments,
            self.paths.quality_assessments,
            self.paths.quality_gate_report,
            self.paths.materialization_policy,
            self.paths.gate_policy,
        ):
            if not required.is_file():
                raise KnowledgeBackfillError(f"Required input not found: {required}")
        if not self.paths.starting_m3_dir.is_dir():
            raise KnowledgeBackfillError(
                f"Starting M3/M3.2 directory not found: {self.paths.starting_m3_dir}"
            )
        if self.paths.supplementary_policy is not None and self.paths.m3_1_dir is None:
            raise KnowledgeBackfillError(
                "--supplementary-policy requires an M3.1 output directory"
            )

        self.materialization_id = self._materialization_id()

    def _resolve_paths(self, value: KnowledgeBackfillPaths) -> KnowledgeBackfillPaths:
        def resolve(path: Path | None) -> Path | None:
            if path is None:
                return None
            return (path if path.is_absolute() else self.root / path).resolve()

        return KnowledgeBackfillPaths(
            acquisition_profile=resolve(value.acquisition_profile),  # type: ignore[arg-type]
            backfill_policy=resolve(value.backfill_policy),  # type: ignore[arg-type]
            source_policy=resolve(value.source_policy),  # type: ignore[arg-type]
            catalog=resolve(value.catalog),  # type: ignore[arg-type]
            m2_assessments=resolve(value.m2_assessments),  # type: ignore[arg-type]
            quality_assessments=resolve(value.quality_assessments),  # type: ignore[arg-type]
            quality_gate_report=resolve(value.quality_gate_report),  # type: ignore[arg-type]
            starting_m3_dir=resolve(value.starting_m3_dir),  # type: ignore[arg-type]
            materialization_policy=resolve(value.materialization_policy),  # type: ignore[arg-type]
            m4_dir=resolve(value.m4_dir),  # type: ignore[arg-type]
            m4_config=resolve(value.m4_config),  # type: ignore[arg-type]
            gate_policy=resolve(value.gate_policy),  # type: ignore[arg-type]
            m4_5_dir=resolve(value.m4_5_dir),  # type: ignore[arg-type]
            strict_config=resolve(value.strict_config),  # type: ignore[arg-type]
            data_root=resolve(value.data_root),  # type: ignore[arg-type]
            run_root=resolve(value.run_root),  # type: ignore[arg-type]
            m3_1_dir=resolve(value.m3_1_dir),
            supplementary_policy=resolve(value.supplementary_policy),
        )

    def _materialization_id(self) -> str:
        report_path = self.paths.m4_dir / "materialization_report.json"
        if report_path.is_file():
            try:
                value = _read_json(report_path).get("materialization_id")
                if value:
                    return str(value)
            except Exception:
                pass
        return f"knowledge_backfill_{self.corpus_id}"

    def _run_command(self, command: list[str], label: str) -> bool:
        print(f"[knowledge-backfill] {label} | start", flush=True)
        print(f"[knowledge-backfill]   $ {shlex.join(command)}", flush=True)
        if self.options.dry_run:
            self.command_records.append(
                {"label": label, "status": "dry_run", "command": command}
            )
            return True
        started = time.monotonic()
        completed = subprocess.run(command, cwd=self.root, check=False)
        elapsed = time.monotonic() - started
        status = "passed" if completed.returncode == 0 else "failed"
        self.command_records.append(
            {
                "label": label,
                "status": status,
                "return_code": completed.returncode,
                "elapsed_seconds": elapsed,
                "command": command,
            }
        )
        print(
            f"[knowledge-backfill] {label} | {status} | {elapsed:.1f}s",
            flush=True,
        )
        return completed.returncode == 0

    def _current_summary(self) -> dict[str, Any]:
        return summarize_outcomes(
            _read_jsonl(self.outcomes_path),
            target_status=self.options.target_status,
        )

    def _selected_paths(self, m3_dir: Path) -> tuple[Path, Path]:
        selected = m3_dir / "selected_works.jsonl"
        report = m3_dir / "selection_report.json"
        if not selected.is_file() or not report.is_file():
            raise KnowledgeBackfillError(
                f"Backfill starting directory lacks selected/report artifacts: {m3_dir}"
            )
        return selected, report

    def _access_recovery_command(
        self,
        *,
        current_m3_dir: Path,
        output_m3_dir: Path,
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "scripts.prepare_access_recovery",
            "--source-policy",
            str(self.paths.source_policy),
            "--source-m3-dir",
            str(current_m3_dir),
            "--output-m3-dir",
            str(output_m3_dir),
        ]
        if self.options.retry_failed_acquisition:
            command.append("--retry-failed")
        if self.options.retry_access_misses:
            command.append("--retry-access-misses")
        return command

    def _round_backfill_command(
        self,
        *,
        dynamic_profile: Path,
        current_m3_dir: Path,
        output_m3_dir: Path,
        round_id: str,
    ) -> list[str]:
        selected, report = self._selected_paths(current_m3_dir)
        command = [
            sys.executable,
            "-m",
            "scripts.backfill_acquisition_ready_corpus",
            "--profile",
            str(dynamic_profile),
            "--backfill-policy",
            str(self.paths.backfill_policy),
            "--source-policy",
            str(self.paths.source_policy),
            "--catalog",
            str(self.paths.catalog),
            "--m2-assessments",
            str(self.paths.m2_assessments),
            "--quality-assessments",
            str(self.paths.quality_assessments),
            "--quality-gate-report",
            str(self.paths.quality_gate_report),
            "--m2-1-selected-works",
            str(selected),
            "--m2-1-selection-report",
            str(report),
            "--m3-dir",
            str(current_m3_dir),
            "--output-dir",
            str(output_m3_dir),
            "--backfill-id",
            round_id,
        ]
        if self.options.retry_failed_acquisition:
            command.append("--retry-failed")
        return command

    def _supplementary_command(self, *, m3_dir: Path, round_id: str) -> list[str] | None:
        if self.paths.m3_1_dir is None or self.paths.supplementary_policy is None:
            return None
        selected, report = self._selected_paths(m3_dir)
        command = [
            sys.executable,
            "-m",
            "scripts.discover_supplementary_artifacts",
            "--profile-id",
            self.profile_id,
            "--catalog",
            str(self.paths.catalog),
            "--selected-works",
            str(selected),
            "--selection-report",
            str(report),
            "--m3-dir",
            str(m3_dir),
            "--supplementary-policy",
            str(self.paths.supplementary_policy),
            "--output-dir",
            str(self.paths.m3_1_dir),
            "--acquisition-id",
            f"{round_id}_supplementary",
        ]
        if self.options.retry_failed_supplementary:
            command.append("--retry-failed")
        return command

    def _materialize_command(self, *, m3_dir: Path) -> list[str]:
        selected, report = self._selected_paths(m3_dir)
        command = [
            sys.executable,
            "-m",
            "scripts.materialize_corpus_documents",
            "--profile-id",
            self.profile_id,
            "--domain-profile-id",
            self.domain_profile,
            "--data-root",
            str(self.paths.data_root),
            "--catalog",
            str(self.paths.catalog),
            "--selected-works",
            str(selected),
            "--selection-report",
            str(report),
            "--m3-dir",
            str(m3_dir),
            "--materialization-policy",
            str(self.paths.materialization_policy),
            "--output-dir",
            str(self.paths.m4_dir),
            "--generated-config",
            str(self.paths.m4_config),
            "--materialization-id",
            self.materialization_id,
            "--paper-id-prefix",
            self.paper_id_prefix,
        ]
        if self.paths.m3_1_dir is not None:
            command.extend(["--m3-1-dir", str(self.paths.m3_1_dir)])
        if self.options.retry_failed_materialization:
            command.append("--retry-failed")
        return command

    def _gate_command(self, *, m3_dir: Path) -> list[str]:
        selected, _ = self._selected_paths(m3_dir)
        return [
            sys.executable,
            "-m",
            "scripts.apply_pre_extraction_gate",
            "--acquisition-profile",
            str(self.paths.acquisition_profile),
            "--gate-policy",
            str(self.paths.gate_policy),
            "--catalog",
            str(self.paths.catalog),
            "--selected-works",
            str(selected),
            "--m4-dir",
            str(self.paths.m4_dir),
            "--input-config",
            str(self.paths.m4_config),
            "--output-dir",
            str(self.paths.m4_5_dir),
            "--output-config",
            str(self.paths.strict_config),
            "--domain-profile-id",
            self.domain_profile,
            "--data-root",
            str(self.paths.data_root),
        ]

    def _strict_bridge_command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "scripts.run_strict_bridge_corpus",
            "--config",
            str(self.paths.strict_config),
            "--source-manifest",
            str(self.paths.m4_5_dir / "extraction_plan.jsonl"),
            "--domain-profile",
            self.domain_profile,
            "--data-root",
            str(self.paths.data_root),
            "--corpus-id",
            self.corpus_id,
            "--extract-concurrency",
            str(self.options.extract_concurrency),
            "--bridge-concurrency",
            str(self.options.bridge_concurrency),
            "--heartbeat-seconds",
            str(self.options.heartbeat_seconds),
        ]

    def _write_round_report(self, round_dir: Path, payload: dict[str, Any]) -> None:
        _write_json_atomic(round_dir / "round.json", payload)

    def run(self) -> dict[str, Any]:
        initial = self._current_summary()
        current_m3_dir = self.paths.starting_m3_dir
        current_selected_path, _ = self._selected_paths(current_m3_dir)
        initial_selected_count = len(load_work_ids(current_selected_path))
        extra_candidates_added = 0
        rounds: list[dict[str, Any]] = []

        if initial["target_status_count"] >= self.options.target_count:
            result = {
                "schema_version": "knowledge-aware-backfill-run-v1",
                "status": "target_already_satisfied",
                "corpus_id": self.corpus_id,
                "target_status": self.options.target_status,
                "target_count": self.options.target_count,
                "initial": initial,
                "final": initial,
                "initial_selected_count": initial_selected_count,
                "extra_candidates_added": 0,
                "rounds": [],
                "command_records": [],
                "updated_at": _utc_now(),
            }
            _write_json_atomic(self.paths.run_root / "run.json", result)
            print(
                "[knowledge-backfill] target already satisfied: "
                f"{initial['target_status_count']}/{self.options.target_count}",
                flush=True,
            )
            return result

        if self.options.dry_run:
            # Dry-run intentionally plans only the next round.  Later deficits
            # depend on real acquisition/gate/LLM outcomes and must not be
            # fabricated.
            round_limit = 1
        else:
            round_limit = self.options.max_rounds

        terminal_status = "max_rounds_exhausted"
        for round_number in range(1, round_limit + 1):
            before = self._current_summary()
            if before["target_status_count"] >= self.options.target_count:
                terminal_status = "target_reached"
                break

            deficit = self.options.target_count - int(before["target_status_count"])
            remaining_budget = self.options.max_extra_candidates - extra_candidates_added
            if remaining_budget <= 0:
                terminal_status = "candidate_budget_exhausted"
                break
            requested_slots = max(1, int(math.ceil(deficit * self.options.oversample_factor)))
            requested_slots = min(requested_slots, remaining_budget)

            current_selected, _ = self._selected_paths(current_m3_dir)
            selected_before_ids = load_work_ids(current_selected)
            acquisition_target_total = len(selected_before_ids) + requested_slots
            round_id = f"knowledge_backfill_{self.corpus_id}_r{round_number:03d}"
            round_dir = self.paths.run_root / f"round_{round_number:03d}"
            output_m3_dir = round_dir / "m3_2"
            dynamic_profile = write_dynamic_target_profile(
                source_profile=self.paths.acquisition_profile,
                output_path=round_dir / "acquisition_profile.yaml",
                target_total=acquisition_target_total,
            )

            round_payload: dict[str, Any] = {
                "round": round_number,
                "round_id": round_id,
                "before": before,
                "knowledge_deficit": deficit,
                "requested_acquisition_slots": requested_slots,
                "selected_count_before": len(selected_before_ids),
                "dynamic_acquisition_target_total": acquisition_target_total,
                "dynamic_profile": str(dynamic_profile),
                "m3_output_dir": str(output_m3_dir),
                "access_recovery_report": str(
                    output_m3_dir / "access_recovery_report.json"
                ),
                "status": "running",
                "started_at": _utc_now(),
            }
            self._write_round_report(round_dir, round_payload)

            stages: list[tuple[str, list[str] | None]] = [
                (
                    "access_recovery",
                    self._access_recovery_command(
                        current_m3_dir=current_m3_dir,
                        output_m3_dir=output_m3_dir,
                    ),
                ),
                (
                    "m3_2_backfill",
                    self._round_backfill_command(
                        dynamic_profile=dynamic_profile,
                        current_m3_dir=current_m3_dir,
                        output_m3_dir=output_m3_dir,
                        round_id=round_id,
                    ),
                ),
            ]

            failed_stage: str | None = None
            for label, command in stages:
                if command is not None and not self.command_runner(command, f"r{round_number:03d}:{label}"):
                    failed_stage = label
                    break
            if failed_stage is not None:
                round_payload.update({"status": "failed", "failed_stage": failed_stage})
                self._write_round_report(round_dir, round_payload)
                terminal_status = "stage_failure"
                rounds.append(round_payload)
                break

            if self.options.dry_run:
                round_payload.update(
                    {
                        "status": "dry_run_planned",
                        "planned_followup_stages": [
                            "supplementary" if self.paths.supplementary_policy else None,
                            "m4_materialize",
                            "m4_5_gate",
                            "strict_bridge",
                        ],
                    }
                )
                round_payload["planned_followup_stages"] = [
                    value for value in round_payload["planned_followup_stages"] if value
                ]
                self._write_round_report(round_dir, round_payload)
                rounds.append(round_payload)
                terminal_status = "dry_run"
                break

            selected_after_path, _ = self._selected_paths(output_m3_dir)
            selected_after_ids = load_work_ids(selected_after_path)
            new_work_ids = [
                work_id for work_id in selected_after_ids if work_id not in set(selected_before_ids)
            ]
            round_payload["selected_count_after"] = len(selected_after_ids)
            round_payload["new_work_ids"] = new_work_ids
            round_payload["new_candidate_count"] = len(new_work_ids)
            extra_candidates_added += len(new_work_ids)

            if not new_work_ids:
                round_payload.update(
                    {
                        "status": "reserve_exhausted",
                        "completed_at": _utc_now(),
                    }
                )
                self._write_round_report(round_dir, round_payload)
                rounds.append(round_payload)
                terminal_status = "reserve_exhausted"
                current_m3_dir = output_m3_dir
                break

            followups: list[tuple[str, list[str] | None]] = [
                (
                    "supplementary",
                    self._supplementary_command(m3_dir=output_m3_dir, round_id=round_id),
                ),
                ("m4_materialize", self._materialize_command(m3_dir=output_m3_dir)),
                ("m4_5_gate", self._gate_command(m3_dir=output_m3_dir)),
                ("strict_bridge", self._strict_bridge_command()),
            ]
            for label, command in followups:
                if command is None:
                    continue
                if not self.command_runner(command, f"r{round_number:03d}:{label}"):
                    failed_stage = label
                    break

            if failed_stage is not None:
                round_payload.update(
                    {
                        "status": "failed",
                        "failed_stage": failed_stage,
                        "completed_at": _utc_now(),
                    }
                )
                self._write_round_report(round_dir, round_payload)
                rounds.append(round_payload)
                terminal_status = "stage_failure"
                current_m3_dir = output_m3_dir
                break

            current_m3_dir = output_m3_dir
            after = self._current_summary()
            round_payload.update(
                {
                    "after": after,
                    "knowledge_gain": (
                        int(after["target_status_count"])
                        - int(before["target_status_count"])
                    ),
                    "status": (
                        "target_reached"
                        if after["target_status_count"] >= self.options.target_count
                        else "round_complete"
                    ),
                    "completed_at": _utc_now(),
                }
            )
            self._write_round_report(round_dir, round_payload)
            rounds.append(round_payload)
            if after["target_status_count"] >= self.options.target_count:
                terminal_status = "target_reached"
                break

        final = self._current_summary()
        if self.options.dry_run:
            final_status = "dry_run"
        elif final["target_status_count"] >= self.options.target_count:
            final_status = "target_reached"
        else:
            final_status = terminal_status

        result = {
            "schema_version": "knowledge-aware-backfill-run-v1",
            "status": final_status,
            "corpus_id": self.corpus_id,
            "domain_profile": self.domain_profile,
            "target_status": self.options.target_status,
            "target_count": self.options.target_count,
            "initial": initial,
            "final": final,
            "initial_selected_count": initial_selected_count,
            "final_selected_count": (
                len(load_work_ids(self._selected_paths(current_m3_dir)[0]))
                if not self.options.dry_run
                else initial_selected_count
            ),
            "extra_candidates_added": extra_candidates_added,
            "max_extra_candidates": self.options.max_extra_candidates,
            "round_count": len(rounds),
            "rounds": rounds,
            "latest_m3_dir": str(current_m3_dir),
            "m4_dir": str(self.paths.m4_dir),
            "m4_5_dir": str(self.paths.m4_5_dir),
            "strict_config": str(self.paths.strict_config),
            "outcomes_path": str(self.outcomes_path),
            "supplementary_mode": (
                "refresh_expanded_selection"
                if self.paths.supplementary_policy is not None
                else (
                    "reuse_existing_only" if self.paths.m3_1_dir is not None else "disabled"
                )
            ),
            "command_records": self.command_records,
            "updated_at": _utc_now(),
        }
        _write_json_atomic(self.paths.run_root / "run.json", result)
        return result
