from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Sequence


KnowledgeTargetStatus = Literal[
    "STRICT_USABLE",
    "BRIDGE_USEFUL",
    "CORPUS_ELIGIBLE",
]
PublicationMode = Literal["evidence", "mechanism", "exploratory"]


class CorpusPublicationError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | Path | None) -> str:
    if path is None:
        return ""
    resolved = Path(path)
    if not resolved.is_file():
        return ""
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temp, path)


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise CorpusPublicationError(f"Required JSON file not found: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusPublicationError(f"Invalid JSON file: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CorpusPublicationError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.is_file():
        if required:
            raise CorpusPublicationError(f"Required JSONL file not found: {path}")
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusPublicationError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise CorpusPublicationError(
                    f"Expected JSON object at {path}:{line_number}"
                )
            rows.append(payload)
    return rows


def _unique_index(
    rows: Sequence[dict[str, Any]],
    key: str,
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "").strip()
        if not value:
            continue
        if value in indexed:
            raise CorpusPublicationError(
                f"Duplicate {key}={value!r} in {source}"
            )
        indexed[value] = dict(row)
    return indexed


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def outcome_meets_target(
    outcome: dict[str, Any],
    target_status: KnowledgeTargetStatus,
) -> bool:
    if target_status == "STRICT_USABLE":
        return str(outcome.get("strict_status") or "") == "STRICT_USABLE"
    if target_status == "BRIDGE_USEFUL":
        return str(outcome.get("bridge_status") or "") == "BRIDGE_USEFUL"
    if target_status == "CORPUS_ELIGIBLE":
        return _as_bool(outcome.get("corpus_eligible"))
    raise ValueError(target_status)


def _axis_values(row: dict[str, Any]) -> list[str]:
    raw = row.get("matched_axes")
    if isinstance(raw, list):
        return sorted({str(value) for value in raw if str(value).strip()})
    return []


def _assessment_reason(assessment: dict[str, Any]) -> tuple[str, list[str]]:
    identity = assessment.get("identity")
    suitability = assessment.get("suitability")
    identity = identity if isinstance(identity, dict) else {}
    suitability = suitability if isinstance(suitability, dict) else {}
    codes = [
        f"identity:{str(identity.get('status') or 'unknown')}",
        f"suitability:{str(suitability.get('status') or 'unknown')}",
    ]
    reasons: list[str] = []
    for prefix, block in (("identity", identity), ("suitability", suitability)):
        raw = block.get("reasons")
        if isinstance(raw, list):
            reasons.extend(
                f"{prefix}:{str(value)}"
                for value in raw
                if str(value).strip()
            )
    return ";".join(codes), reasons


def _terminal_status(row: dict[str, Any]) -> tuple[str, str | None, list[str]]:
    if _as_bool(row.get("corpus_eligible")):
        return "corpus", None, []

    strict_status = str(row.get("strict_status") or "NOT_RUN")
    bridge_status = str(row.get("bridge_status") or "NOT_RUN")
    projection_status = str(row.get("projection_status") or "NOT_RUN")
    if strict_status != "NOT_RUN":
        if strict_status != "STRICT_USABLE":
            return "strict", strict_status, [strict_status]
        if bridge_status in {"BRIDGE_ERROR", "NOT_RUN"}:
            return "bridge", bridge_status, [bridge_status]
        if projection_status != "PROJECTION_USABLE":
            return "projection", projection_status, [projection_status]
        return "strict_bridge", "not_corpus_eligible", ["not_corpus_eligible"]

    gate_evaluated = _as_bool(row.get("m4_5_evaluated"))
    if gate_evaluated and not _as_bool(row.get("m4_5_auto_extraction_allowed")):
        reason = str(row.get("m4_5_block_reason") or "pre_extraction_gate_blocked")
        details = row.get("m4_5_reason_details")
        return (
            "m4_5",
            reason,
            [str(value) for value in details] if isinstance(details, list) else [],
        )

    if _as_bool(row.get("m4_extraction_ready")):
        return "m4_5", "missing_pre_extraction_gate_outcome", [
            "missing_pre_extraction_gate_outcome"
        ]

    m4_status = str(row.get("m4_main_document_status") or "")
    if m4_status:
        return "m4", f"main_document:{m4_status}", [f"main_document:{m4_status}"]

    m3_status = str(row.get("m3_main_artifact_status") or "missing")
    if m3_status == "downloaded":
        return "m4", "missing_materialization_record", [
            "missing_materialization_record"
        ]
    return "m3", f"main_artifact:{m3_status}", [f"main_artifact:{m3_status}"]


def build_paper_lifecycle(
    *,
    selected_works_path: str | Path,
    m3_dir: str | Path,
    m4_dir: str | Path,
    m4_5_dir: str | Path,
    outcomes_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join acquisition through corpus eligibility without losing selected works.

    The selected-work set is the accounting authority.  Downstream artifacts may
    legitimately contain fewer rows; those absences are represented explicitly
    rather than silently disappearing from the production funnel.
    """
    selected_path = Path(selected_works_path)
    m3_root = Path(m3_dir)
    m4_root = Path(m4_dir)
    gate_root = Path(m4_5_dir)
    outcomes_file = Path(outcomes_path)

    selected = _read_jsonl(selected_path)
    if not selected:
        raise CorpusPublicationError("Selected-work set is empty")
    selected_by_work = _unique_index(selected, "work_id", source=str(selected_path))
    if len(selected_by_work) != len(selected):
        raise CorpusPublicationError("Every selected work must have a unique work_id")

    m3_artifacts = _read_jsonl(m3_root / "artifacts.jsonl", required=False)
    main_artifact_by_work: dict[str, dict[str, Any]] = {}
    for artifact in m3_artifacts:
        if str(artifact.get("role") or "") != "main":
            continue
        work_id = str(artifact.get("work_id") or "").strip()
        if not work_id:
            continue
        if work_id in main_artifact_by_work:
            raise CorpusPublicationError(
                f"Duplicate main SourceArtifact for work_id={work_id!r}"
            )
        main_artifact_by_work[work_id] = dict(artifact)

    paper_map_rows = _read_jsonl(m4_root / "paper_map.jsonl", required=False)
    paper_map_by_work = _unique_index(
        paper_map_rows, "work_id", source=str(m4_root / "paper_map.jsonl")
    )
    materialization_rows = _read_jsonl(
        m4_root / "paper_materialization_records.jsonl", required=False
    )
    materialization_by_work = _unique_index(
        materialization_rows,
        "work_id",
        source=str(m4_root / "paper_materialization_records.jsonl"),
    )

    gate_rows = _read_jsonl(
        gate_root / "pre_extraction_gate_assessments.jsonl", required=False
    )
    gate_by_work = _unique_index(
        gate_rows,
        "work_id",
        source=str(gate_root / "pre_extraction_gate_assessments.jsonl"),
    )
    outcomes = _read_jsonl(outcomes_file, required=False)
    outcome_by_paper = _unique_index(
        outcomes, "paper_id", source=str(outcomes_file)
    )

    lifecycle: list[dict[str, Any]] = []
    for selected_row in selected:
        work_id = str(selected_row["work_id"])
        mapped = paper_map_by_work.get(work_id, {})
        materialization = materialization_by_work.get(work_id, {})
        gate = gate_by_work.get(work_id, {})
        paper_id = str(
            mapped.get("paper_id")
            or materialization.get("paper_id")
            or gate.get("paper_id")
            or ""
        ).strip()
        outcome = outcome_by_paper.get(paper_id, {}) if paper_id else {}
        artifact = main_artifact_by_work.get(work_id, {})

        identity = gate.get("identity") if isinstance(gate.get("identity"), dict) else {}
        suitability = (
            gate.get("suitability")
            if isinstance(gate.get("suitability"), dict)
            else {}
        )
        gate_reason, gate_details = (
            _assessment_reason(gate) if gate else ("", [])
        )

        row: dict[str, Any] = {
            "schema_version": "graphagentsdac-paper-lifecycle-v1",
            "work_id": work_id,
            "paper_id": paper_id or None,
            "title": selected_row.get("title"),
            "doi": selected_row.get("doi"),
            "matched_axes": _axis_values(selected_row),
            "primary_quota_axis": selected_row.get("primary_quota_axis"),
            "selection_total_score": selected_row.get("total_score"),
            "m3_main_artifact_status": str(artifact.get("status") or "missing"),
            "m3_main_artifact_sha256": artifact.get("sha256"),
            "m3_main_artifact_error": artifact.get("error"),
            "m4_seen": bool(materialization),
            "m4_main_document_status": materialization.get("main_document_status"),
            "m4_extraction_ready": _as_bool(materialization.get("extraction_ready")),
            "m4_5_evaluated": bool(gate),
            "m4_5_identity_status": identity.get("status"),
            "m4_5_identity_method": identity.get("method"),
            "m4_5_suitability_status": suitability.get("status"),
            "m4_5_suitable_axes": suitability.get("suitable_axes", []),
            "m4_5_auto_extraction_allowed": _as_bool(
                gate.get("auto_extraction_allowed")
            ),
            "m4_5_block_reason": gate_reason if gate and not _as_bool(
                gate.get("auto_extraction_allowed")
            ) else None,
            "m4_5_reason_details": gate_details,
            "strict_status": str(outcome.get("strict_status") or "NOT_RUN"),
            "bridge_status": str(outcome.get("bridge_status") or "NOT_RUN"),
            "projection_status": str(outcome.get("projection_status") or "NOT_RUN"),
            "corpus_eligible": _as_bool(outcome.get("corpus_eligible")),
            "strict_extraction_identity": outcome.get("strict_extraction_identity"),
            "canonical_graph_sha256": outcome.get("canonical_graph_sha256"),
            "bridge_graph_sha256": outcome.get("bridge_graph_sha256"),
            "projection_sha256": outcome.get("projection_sha256"),
        }
        stage, reason, details = _terminal_status(row)
        row["terminal_stage"] = stage
        row["terminal_reason"] = reason
        row["terminal_reason_details"] = details
        lifecycle.append(row)

    selected_ids = set(selected_by_work)
    lifecycle_paper_ids = {
        str(row.get("paper_id") or "")
        for row in lifecycle
        if str(row.get("paper_id") or "").strip()
    }
    non_lifecycle_counts = {
        "m3_main_artifact": len(set(main_artifact_by_work) - selected_ids),
        "m4_paper_map": len(set(paper_map_by_work) - selected_ids),
        "m4_materialization": len(set(materialization_by_work) - selected_ids),
        "m4_5_gate": len(set(gate_by_work) - selected_ids),
        "strict_bridge_outcome": len(set(outcome_by_paper) - lifecycle_paper_ids),
    }
    # M4/M4.5 are canonical, resumable acquisition artifacts.  They may
    # legitimately retain records from a broader historical selection than
    # the current latest-M3 lifecycle snapshot.  M3 and Strict/Bridge outputs,
    # however, are expected to describe the active snapshot/run exactly.
    allowed_superset_counts = {
        key: non_lifecycle_counts[key]
        for key in ("m4_paper_map", "m4_materialization", "m4_5_gate")
    }
    unexpected_record_counts = {
        key: non_lifecycle_counts[key]
        for key in ("m3_main_artifact", "strict_bridge_outcome")
    }

    selected_axis_counts: Counter[str] = Counter()
    corpus_axis_counts: Counter[str] = Counter()
    selected_primary_counts: Counter[str] = Counter()
    corpus_primary_counts: Counter[str] = Counter()
    for row in lifecycle:
        for axis in row["matched_axes"]:
            selected_axis_counts[axis] += 1
            if row["corpus_eligible"]:
                corpus_axis_counts[axis] += 1
        primary = str(row.get("primary_quota_axis") or "").strip()
        if primary:
            selected_primary_counts[primary] += 1
            if row["corpus_eligible"]:
                corpus_primary_counts[primary] += 1

    summary = {
        "schema_version": "graphagentsdac-corpus-funnel-v1",
        "selected_work_count": len(lifecycle),
        "m3_main_downloaded_count": sum(
            row["m3_main_artifact_status"] == "downloaded" for row in lifecycle
        ),
        "m4_seen_count": sum(row["m4_seen"] for row in lifecycle),
        "m4_main_materialized_count": sum(
            row["m4_main_document_status"] == "materialized" for row in lifecycle
        ),
        "m4_extraction_ready_count": sum(row["m4_extraction_ready"] for row in lifecycle),
        "m4_5_evaluated_count": sum(row["m4_5_evaluated"] for row in lifecycle),
        "m4_5_auto_extraction_ready_count": sum(
            row["m4_5_auto_extraction_allowed"] for row in lifecycle
        ),
        "strict_outcome_count": sum(row["strict_status"] != "NOT_RUN" for row in lifecycle),
        "strict_usable_count": sum(
            row["strict_status"] == "STRICT_USABLE" for row in lifecycle
        ),
        "bridge_useful_count": sum(
            row["bridge_status"] == "BRIDGE_USEFUL" for row in lifecycle
        ),
        "projection_usable_count": sum(
            row["projection_status"] == "PROJECTION_USABLE" for row in lifecycle
        ),
        "corpus_eligible_count": sum(row["corpus_eligible"] for row in lifecycle),
        "terminal_stage_counts": dict(
            sorted(Counter(str(row["terminal_stage"]) for row in lifecycle).items())
        ),
        "terminal_reason_counts": dict(
            sorted(
                Counter(
                    str(row["terminal_reason"])
                    for row in lifecycle
                    if row["terminal_reason"]
                ).items()
            )
        ),
        "selected_matched_axis_counts": dict(sorted(selected_axis_counts.items())),
        "corpus_eligible_matched_axis_counts": dict(sorted(corpus_axis_counts.items())),
        "selected_primary_quota_axis_counts": dict(sorted(selected_primary_counts.items())),
        "corpus_eligible_primary_quota_axis_counts": dict(sorted(corpus_primary_counts.items())),
        # Backward-compatible diagnostic name retained for existing readers.
        # Non-zero M4/M4.5 values are not necessarily errors: those stages may
        # retain a superset of the active lifecycle selection.
        "orphan_downstream_record_counts": non_lifecycle_counts,
        "allowed_superset_record_counts": allowed_superset_counts,
        "unexpected_non_lifecycle_record_counts": unexpected_record_counts,
        "selected_work_accounting_complete": len(lifecycle) == len(selected),
    }
    return lifecycle, summary


CommandRunner = Callable[[list[str], str], bool]


@dataclass(frozen=True)
class CorpusPublicationOptions:
    mode: PublicationMode = "mechanism"
    target_count: int = 0
    target_status: KnowledgeTargetStatus = "CORPUS_ELIGIBLE"
    build_node_index: bool = True
    embedding_model: str | None = None
    embedding_device: str | None = None
    embedding_batch_size: int = 32
    include_alignment_hubs_in_index: bool = False
    resume: bool = True
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.target_count < 0:
            raise ValueError("target_count must be >= 0")
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be >= 1")


class StrictBridgeCorpusPublisher:
    """Publish a validated corpus into the traversal-ready Explorer layout."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        corpus_id: str,
        domain_profile: str,
        data_root: str | Path,
        selected_works_path: str | Path,
        m3_dir: str | Path,
        m4_dir: str | Path,
        m4_5_dir: str | Path,
        outcomes_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        options: CorpusPublicationOptions | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.corpus_id = str(corpus_id)
        self.domain_profile = str(domain_profile)
        self.options = options or CorpusPublicationOptions()
        self.data_root = self._resolve(data_root)
        self.selected_works_path = self._resolve(selected_works_path)
        self.m3_dir = self._resolve(m3_dir)
        self.m4_dir = self._resolve(m4_dir)
        self.m4_5_dir = self._resolve(m4_5_dir)
        self.outcomes_path = self._resolve(
            outcomes_path
            or (
                self.data_root
                / "pipeline_runs"
                / self.corpus_id
                / "strict_bridge"
                / "paper_outcomes.jsonl"
            )
        )
        self.mode_root = (
            self.data_root / "corpus" / self.corpus_id / self.options.mode
        )
        self.navigation_root = self.mode_root / "navigation"
        self.output_dir = self._resolve(output_dir) if output_dir else (
            self.mode_root / "publication"
        )
        self.lifecycle_path = self.output_dir / "paper_lifecycle.jsonl"
        self.funnel_path = self.output_dir / "funnel_summary.json"
        self.manifest_path = self.output_dir / "corpus_publish_manifest.json"
        self.command_records: list[dict[str, Any]] = []
        self.command_runner = command_runner or self._run_command

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else self.root / path).resolve()

    def _run_command(self, command: list[str], label: str) -> bool:
        print(f"[corpus-publish] {label} | start", flush=True)
        print(f"[corpus-publish]   $ {shlex.join(command)}", flush=True)
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
                "command": command,
                "return_code": completed.returncode,
                "elapsed_seconds": round(elapsed, 6),
            }
        )
        return completed.returncode == 0

    @property
    def corpus_graph_path(self) -> Path:
        return self.mode_root / "graph.graphml"

    @property
    def node_text_path(self) -> Path:
        return self.mode_root / "node_text.jsonl"

    @property
    def corpus_manifest_path(self) -> Path:
        return self.mode_root / "manifest.json"

    @property
    def corpus_audit_path(self) -> Path:
        return self.mode_root / "audit.json"

    @property
    def navigation_graph_path(self) -> Path:
        return self.navigation_root / "graph.graphml"

    @property
    def navigation_summary_path(self) -> Path:
        return self.navigation_root / "summary.json"

    @property
    def node_index_manifest_path(self) -> Path:
        return self.navigation_root / "node_index" / "manifest.json"

    def _validate_inputs(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        for path in (
            self.selected_works_path,
            self.outcomes_path,
            self.corpus_graph_path,
            self.node_text_path,
            self.corpus_manifest_path,
            self.corpus_audit_path,
        ):
            if not path.is_file():
                raise CorpusPublicationError(f"Required publication input not found: {path}")

        corpus_manifest = _read_json(self.corpus_manifest_path)
        corpus_audit = _read_json(self.corpus_audit_path)
        structural_gate = corpus_audit.get(
            "passes_structural_gate",
            corpus_manifest.get("passes_structural_gate"),
        )
        if structural_gate is not True:
            raise CorpusPublicationError(
                "Corpus structural gate is not passing; refusing production publish"
            )

        lifecycle, funnel = build_paper_lifecycle(
            selected_works_path=self.selected_works_path,
            m3_dir=self.m3_dir,
            m4_dir=self.m4_dir,
            m4_5_dir=self.m4_5_dir,
            outcomes_path=self.outcomes_path,
        )
        if not funnel["selected_work_accounting_complete"]:
            raise CorpusPublicationError("Selected-work lifecycle accounting is incomplete")

        unexpected_counts = funnel.get(
            "unexpected_non_lifecycle_record_counts", {}
        )
        if any(int(value) for value in unexpected_counts.values()):
            raise CorpusPublicationError(
                "Active-run artifacts contain records outside the selected-work lifecycle: "
                f"{unexpected_counts}"
            )

        target_count = sum(
            outcome_meets_target(row, self.options.target_status)
            for row in lifecycle
        )
        corpus_eligible_count = int(funnel["corpus_eligible_count"])
        if self.options.target_count and target_count < self.options.target_count:
            raise CorpusPublicationError(
                f"Knowledge target not reached: {target_count}/{self.options.target_count} "
                f"{self.options.target_status}"
            )
        # A production publish promises a traversal-ready corpus.  Reaching an
        # earlier Strict/Bridge target is insufficient if projection/corpus
        # eligibility subsequently lost papers.
        if self.options.target_count and corpus_eligible_count < self.options.target_count:
            raise CorpusPublicationError(
                "Traversal-ready corpus is below the requested production target: "
                f"{corpus_eligible_count}/{self.options.target_count} CORPUS_ELIGIBLE"
            )

        lifecycle_corpus_ids = sorted(
            str(row["paper_id"])
            for row in lifecycle
            if row.get("paper_id") and row["corpus_eligible"]
        )
        raw_manifest_paper_ids = corpus_manifest.get("paper_ids")
        if not isinstance(raw_manifest_paper_ids, list):
            raise CorpusPublicationError(
                "Corpus manifest is missing authoritative paper_ids; rebuild the corpus "
                "before production publication"
            )
        manifest_paper_ids = sorted(
            str(value)
            for value in raw_manifest_paper_ids
            if str(value).strip()
        )
        if manifest_paper_ids != lifecycle_corpus_ids:
            raise CorpusPublicationError(
                "Corpus manifest paper_ids do not match lifecycle CORPUS_ELIGIBLE papers: "
                f"manifest={manifest_paper_ids!r} lifecycle={lifecycle_corpus_ids!r}"
            )
        return lifecycle, funnel

    def _input_hashes(self) -> dict[str, str]:
        paths = {
            "selected_works": self.selected_works_path,
            "m3_artifacts": self.m3_dir / "artifacts.jsonl",
            "m3_acquisition_report": self.m3_dir / "acquisition_report.json",
            "m4_paper_map": self.m4_dir / "paper_map.jsonl",
            "m4_materialization_records": self.m4_dir / "paper_materialization_records.jsonl",
            "m4_materialization_report": self.m4_dir / "materialization_report.json",
            "m4_5_assessments": self.m4_5_dir / "pre_extraction_gate_assessments.jsonl",
            "m4_5_report": self.m4_5_dir / "pre_extraction_gate_report.json",
            "strict_bridge_outcomes": self.outcomes_path,
            "corpus_graph": self.corpus_graph_path,
            "corpus_node_text": self.node_text_path,
            "corpus_manifest": self.corpus_manifest_path,
            "corpus_audit": self.corpus_audit_path,
        }
        return {
            name: _sha256_file(path)
            for name, path in paths.items()
            if path.is_file()
        }

    def _input_fingerprint(self, hashes: dict[str, str]) -> str:
        payload = {
            "corpus_id": self.corpus_id,
            "domain_profile": self.domain_profile,
            "mode": self.options.mode,
            "target_count": self.options.target_count,
            "target_status": self.options.target_status,
            "build_node_index": self.options.build_node_index,
            "embedding_model": self.options.embedding_model,
            "include_alignment_hubs_in_index": (
                self.options.include_alignment_hubs_in_index
            ),
            "input_hashes": hashes,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _navigation_current(self, corpus_sha: str) -> bool:
        if not self.navigation_graph_path.is_file() or not self.navigation_summary_path.is_file():
            return False
        summary = _read_json(self.navigation_summary_path, required=False)
        return (
            str(summary.get("source_graph_sha256") or "") == corpus_sha
            and str(summary.get("graphml_sha256") or "")
            == _sha256_file(self.navigation_graph_path)
        )

    def _node_index_current(self) -> bool:
        if not self.node_index_manifest_path.is_file() or not self.navigation_graph_path.is_file():
            return False
        manifest = _read_json(self.node_index_manifest_path, required=False)
        if str(manifest.get("navigation_graph_sha256") or "") != _sha256_file(
            self.navigation_graph_path
        ):
            return False
        if str(manifest.get("node_text_sha256") or "") != _sha256_file(self.node_text_path):
            return False
        if bool(manifest.get("include_alignment_hubs", False)) != bool(
            self.options.include_alignment_hubs_in_index
        ):
            return False
        if self.options.embedding_model and str(manifest.get("model_name") or "") != str(
            self.options.embedding_model
        ):
            return False
        return True

    def _navigation_command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "scripts.discovery.build_navigation_graph",
            "--corpus-id",
            self.corpus_id,
            "--mode",
            self.options.mode,
            "--data-root",
            str(self.data_root),
        ]

    def _node_index_command(self) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "scripts.discovery.build_node_index",
            "--corpus-id",
            self.corpus_id,
            "--mode",
            self.options.mode,
            "--data-root",
            str(self.data_root),
            "--batch-size",
            str(self.options.embedding_batch_size),
        ]
        if self.options.embedding_model:
            command.extend(["--model", self.options.embedding_model])
        if self.options.embedding_device:
            command.extend(["--device", self.options.embedding_device])
        if self.options.include_alignment_hubs_in_index:
            command.append("--include-alignment-hubs")
        return command

    def _integrity_chain(self) -> dict[str, Any]:
        corpus_sha = _sha256_file(self.corpus_graph_path)
        navigation_sha = _sha256_file(self.navigation_graph_path)
        navigation_summary = _read_json(self.navigation_summary_path, required=False)
        node_manifest = _read_json(self.node_index_manifest_path, required=False)
        node_manifest_sha = _sha256_file(self.node_index_manifest_path)

        navigation_bound = bool(navigation_sha) and (
            str(navigation_summary.get("source_graph_sha256") or "") == corpus_sha
            and str(navigation_summary.get("graphml_sha256") or "") == navigation_sha
        )
        if self.options.build_node_index:
            index_bound = bool(node_manifest_sha) and (
                str(node_manifest.get("navigation_graph_sha256") or "") == navigation_sha
                and str(node_manifest.get("node_text_sha256") or "")
                == _sha256_file(self.node_text_path)
            )
        else:
            index_bound = True

        return {
            "corpus_graph_sha256": corpus_sha,
            "navigation_source_graph_sha256": navigation_summary.get(
                "source_graph_sha256"
            ),
            "navigation_graph_sha256": navigation_sha,
            "node_text_sha256": _sha256_file(self.node_text_path),
            "node_index_manifest_sha256": node_manifest_sha or None,
            "node_index_navigation_graph_sha256": node_manifest.get(
                "navigation_graph_sha256"
            ),
            "node_index_node_text_sha256": node_manifest.get("node_text_sha256"),
            "navigation_binding_valid": navigation_bound,
            "node_index_binding_valid": index_bound,
            "chain_valid": navigation_bound and index_bound,
        }

    def _previous_publish_current(self, input_fingerprint: str) -> bool:
        if not self.options.resume or not self.manifest_path.is_file():
            return False
        manifest = _read_json(self.manifest_path, required=False)
        if str(manifest.get("input_fingerprint") or "") != input_fingerprint:
            return False
        integrity = self._integrity_chain()
        return bool(integrity["chain_valid"])

    def run(self) -> dict[str, Any]:
        lifecycle, funnel = self._validate_inputs()
        input_hashes = self._input_hashes()
        input_fingerprint = self._input_fingerprint(input_hashes)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lifecycle/funnel are cheap and deterministic, so rewrite them even on
        # an otherwise no-op publish.  This keeps the human-facing accounting
        # synchronized with the exact inputs referenced by the publish manifest.
        if not self.options.dry_run:
            _write_jsonl(self.lifecycle_path, lifecycle)
            funnel_payload = dict(funnel)
            funnel_payload.update(
                {
                    "corpus_id": self.corpus_id,
                    "domain_profile": self.domain_profile,
                    "mode": self.options.mode,
                    "selected_works": str(self.selected_works_path),
                    "outcomes": str(self.outcomes_path),
                    "updated_at": _utc_now(),
                }
            )
            _write_json_atomic(self.funnel_path, funnel_payload)

        if self._previous_publish_current(input_fingerprint):
            previous = _read_json(self.manifest_path)
            result = {
                "status": "already_current",
                "corpus_id": self.corpus_id,
                "publication_manifest": str(self.manifest_path),
                "paper_lifecycle": str(self.lifecycle_path),
                "funnel_summary": str(self.funnel_path),
                "integrity_chain": previous.get("integrity_chain", {}),
                "command_records": [],
            }
            print("[corpus-publish] publication already current", flush=True)
            return result

        corpus_sha = _sha256_file(self.corpus_graph_path)
        navigation_action = "resume_skip"
        if not self._navigation_current(corpus_sha):
            navigation_action = "dry_run" if self.options.dry_run else "run"
            if not self.command_runner(self._navigation_command(), "navigation"):
                raise CorpusPublicationError("Navigation graph build failed")
            if not self.options.dry_run and not self._navigation_current(corpus_sha):
                raise CorpusPublicationError(
                    "Navigation graph build completed but source fingerprint binding is invalid"
                )

        node_index_action: str | None = None
        if self.options.build_node_index:
            node_index_action = "resume_skip"
            if not self._node_index_current():
                node_index_action = "dry_run" if self.options.dry_run else "run"
                if not self.command_runner(self._node_index_command(), "node_index"):
                    raise CorpusPublicationError("Node embedding index build failed")
                if not self.options.dry_run and not self._node_index_current():
                    raise CorpusPublicationError(
                        "Node index build completed but graph/node-text fingerprint binding is invalid"
                    )

        if self.options.dry_run:
            return {
                "status": "dry_run",
                "corpus_id": self.corpus_id,
                "selected_work_count": funnel["selected_work_count"],
                "corpus_eligible_count": funnel["corpus_eligible_count"],
                "navigation_action": navigation_action,
                "node_index_action": node_index_action,
                "command_records": self.command_records,
            }

        integrity = self._integrity_chain()
        if not integrity["chain_valid"]:
            raise CorpusPublicationError(
                "Publication fingerprint chain is invalid after downstream builds"
            )

        manifest = {
            "schema_version": "graphagentsdac-corpus-publish-manifest-v1",
            "status": "published",
            "corpus_id": self.corpus_id,
            "domain_profile": self.domain_profile,
            "data_root": str(self.data_root),
            "mode": self.options.mode,
            "target_count": self.options.target_count,
            "target_status": self.options.target_status,
            "selected_work_count": funnel["selected_work_count"],
            "corpus_eligible_count": funnel["corpus_eligible_count"],
            "input_fingerprint": input_fingerprint,
            "input_hashes": input_hashes,
            "paper_lifecycle": str(self.lifecycle_path),
            "funnel_summary": str(self.funnel_path),
            "corpus_graph": str(self.corpus_graph_path),
            "corpus_manifest": str(self.corpus_manifest_path),
            "corpus_audit": str(self.corpus_audit_path),
            "navigation_graph": str(self.navigation_graph_path),
            "navigation_summary": str(self.navigation_summary_path),
            "node_index_manifest": (
                str(self.node_index_manifest_path)
                if self.options.build_node_index
                else None
            ),
            "navigation_action": navigation_action,
            "node_index_action": node_index_action,
            "integrity_chain": integrity,
            "command_records": self.command_records,
            "updated_at": _utc_now(),
        }
        _write_json_atomic(self.manifest_path, manifest)
        print(
            "[corpus-publish] published | "
            f"selected={funnel['selected_work_count']} "
            f"corpus_eligible={funnel['corpus_eligible_count']}",
            flush=True,
        )
        return manifest
