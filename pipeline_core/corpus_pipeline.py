from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import yaml


Mode = Literal["evidence", "mechanism", "exploratory"]


class CorpusPipelineError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_component(value: str) -> str:
    value = value.replace("/", "_").replace("\\", "_").strip()
    return value or "paper"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def load_paper_ids_from_yaml(path: str | Path) -> list[str]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("papers"), dict):
        raise CorpusPipelineError("Generated papers YAML must contain a papers mapping")
    paper_ids = [str(value) for value in payload["papers"].keys()]
    if not paper_ids:
        raise CorpusPipelineError("Generated papers YAML contains no papers")
    if len(paper_ids) != len(set(paper_ids)):
        raise CorpusPipelineError("Generated papers YAML contains duplicate paper IDs")
    return paper_ids


def select_paper_ids(
    available: Sequence[str],
    *,
    requested: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    selected = list(available)
    if requested:
        unknown = [paper_id for paper_id in requested if paper_id not in available]
        if unknown:
            raise CorpusPipelineError(
                "Requested paper IDs are not present in the generated config: "
                + ", ".join(unknown)
            )
        # Preserve explicit CLI order while removing accidental repeats.
        seen: set[str] = set()
        selected = []
        for paper_id in requested:
            if paper_id not in seen:
                selected.append(paper_id)
                seen.add(paper_id)
    if limit is not None:
        if limit < 1:
            raise CorpusPipelineError("--limit must be at least 1")
        selected = selected[:limit]
    if not selected:
        raise CorpusPipelineError("Paper selection is empty")
    return selected


@dataclass(frozen=True)
class PipelineOptions:
    mode: Mode = "evidence"
    extract_concurrency: int = 4
    bridge_concurrency: int = 4
    fail_fast: bool = False
    skip_node_index: bool = False
    include_alignment_hubs_in_index: bool = False
    force_extract: bool = False
    force_bridge: bool = False
    allow_partial: bool = False
    device: str | None = None
    index_batch_size: int = 32
    heartbeat_seconds: float = 30.0
    dry_run: bool = False
    resume: bool = True


class FrozenCorpusPipeline:
    def __init__(
        self,
        *,
        project_root: str | Path,
        papers_yaml: str | Path,
        frozen_manifest: str | Path,
        corpus_id: str,
        options: PipelineOptions | None = None,
        selected_paper_ids: Sequence[str] | None = None,
        paper_limit: int | None = None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.papers_yaml = Path(papers_yaml)
        if not self.papers_yaml.is_absolute():
            self.papers_yaml = (self.root / self.papers_yaml).resolve()
        self.frozen_manifest = Path(frozen_manifest)
        if not self.frozen_manifest.is_absolute():
            self.frozen_manifest = (self.root / self.frozen_manifest).resolve()
        self.corpus_id = corpus_id
        self.options = options or PipelineOptions()
        if self.options.extract_concurrency < 1 or self.options.bridge_concurrency < 1:
            raise CorpusPipelineError("Concurrency must be at least 1")
        if self.options.index_batch_size < 1:
            raise CorpusPipelineError("index_batch_size must be at least 1")
        if self.options.heartbeat_seconds < 0:
            raise CorpusPipelineError("heartbeat_seconds cannot be negative")
        if not self.papers_yaml.is_file():
            raise CorpusPipelineError(f"Generated papers YAML not found: {self.papers_yaml}")
        if not self.frozen_manifest.is_file():
            raise CorpusPipelineError(f"Frozen manifest not found: {self.frozen_manifest}")

        self.available_paper_ids = load_paper_ids_from_yaml(self.papers_yaml)
        self.paper_ids = select_paper_ids(
            self.available_paper_ids,
            requested=selected_paper_ids,
            limit=paper_limit,
        )
        self.pipeline_fingerprint = self._fingerprint()
        self.state_path = (
            self.root
            / "data_dac"
            / "pipeline_state"
            / self.corpus_id
            / f"{self.options.mode}.json"
        )
        self.logs_root = (
            self.root
            / "data_dac"
            / "pipeline_logs"
            / self.corpus_id
            / self.options.mode
        )
        self.state = self._load_state()

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(_sha256_file(self.frozen_manifest).encode("ascii"))
        digest.update(_sha256_file(self.papers_yaml).encode("ascii"))
        digest.update(self.corpus_id.encode("utf-8"))
        digest.update(self.options.mode.encode("ascii"))
        digest.update(json.dumps(self.paper_ids, ensure_ascii=False).encode("utf-8"))
        return digest.hexdigest()

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema_version": "graphagentsdac-corpus-pipeline-state-v02",
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "corpus_id": self.corpus_id,
            "mode": self.options.mode,
            "available_paper_count": len(self.available_paper_ids),
            "paper_count": len(self.paper_ids),
            "paper_ids": self.paper_ids,
            "papers": {},
            "global_stages": {},
            "updated_at": _utc_now(),
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.options.resume or not self.state_path.exists():
            return self._new_state()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return self._new_state()
        if payload.get("pipeline_fingerprint") != self.pipeline_fingerprint:
            return self._new_state()
        return payload

    def _save_state(self) -> None:
        self.state["updated_at"] = _utc_now()
        _write_json_atomic(self.state_path, self.state)

    def _paper_root(self, paper_id: str) -> Path:
        return self.root / "data_dac" / "extracted" / paper_id

    def _expected_output(self, paper_id: str, stage: str) -> Path:
        paper_root = self._paper_root(paper_id)
        if stage == "extract":
            return paper_root / "latest_run.json"
        if stage == "paper_graph":
            return paper_root / f"{paper_id}.graphml"
        if stage == "bridge":
            return paper_root / f"{paper_id}.bridge.graphml"
        if stage == "projection":
            return paper_root / "graphagents" / self.options.mode / "graph.graphml"
        raise KeyError(stage)

    def _stage_already_done(self, paper_id: str, stage: str) -> bool:
        if not self.options.resume:
            return False
        record = (
            self.state.get("papers", {})
            .get(paper_id, {})
            .get("stages", {})
            .get(stage, {})
        )
        if record.get("status") != "passed":
            return False
        expected = self._expected_output(paper_id, stage)
        if not expected.exists():
            return False
        if stage == "bridge" and self.options.mode == "exploratory":
            candidate = self._paper_root(paper_id) / f"{paper_id}.bridge.candidates.graphml"
            if not candidate.exists():
                return False
        return True

    def _paper_command(self, paper_id: str, stage: str) -> list[str]:
        py = sys.executable
        config = str(self.papers_yaml)
        if stage == "extract":
            command = [
                py,
                "-m",
                "scripts.extract_paper",
                "--paper-id",
                paper_id,
                "--config",
                config,
                "--concurrency",
                str(self.options.extract_concurrency),
            ]
            if self.options.force_extract:
                command.append("--force")
            if self.options.allow_partial:
                command.append("--allow-partial")
            return command
        if stage == "paper_graph":
            command = [
                py,
                "-m",
                "scripts.build_paper_graph",
                "--paper-id",
                paper_id,
                "--config",
                config,
            ]
            if self.options.allow_partial:
                command.append("--allow-incomplete")
            return command
        if stage == "bridge":
            command = [
                py,
                "-m",
                "scripts.extract_bridge_graph",
                "--paper-id",
                paper_id,
                "--config",
                config,
                "--concurrency",
                str(self.options.bridge_concurrency),
            ]
            if self.options.force_bridge:
                command.append("--force")
            if self.options.allow_partial:
                command.append("--allow-incomplete")
            return command
        if stage == "projection":
            return [
                py,
                "-m",
                "scripts.build_graphagents_projection",
                "--paper-id",
                paper_id,
                "--mode",
                self.options.mode,
            ]
        raise KeyError(stage)

    def _record_stage(
        self,
        paper_id: str,
        stage: str,
        *,
        status: str,
        return_code: int | None,
        elapsed_seconds: float,
        stdout_path: Path | None,
        stderr_path: Path | None,
        command: list[str],
    ) -> None:
        papers = self.state.setdefault("papers", {})
        paper = papers.setdefault(paper_id, {"stages": {}})
        stages = paper.setdefault("stages", {})
        stages[stage] = {
            "status": status,
            "return_code": return_code,
            "elapsed_seconds": elapsed_seconds,
            "command": command,
            "stdout_path": str(stdout_path) if stdout_path else None,
            "stderr_path": str(stderr_path) if stderr_path else None,
            "expected_output": str(self._expected_output(paper_id, stage)),
            "updated_at": _utc_now(),
        }
        self._save_state()

    def _wait_with_heartbeat(
        self,
        process: subprocess.Popen[str],
        *,
        label: str,
        started: float,
    ) -> int:
        interval = self.options.heartbeat_seconds
        if interval <= 0:
            return process.wait()
        while True:
            try:
                return process.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                print(
                    f"[corpus-pipeline] {label} | still running | elapsed={elapsed:.0f}s",
                    flush=True,
                )

    def _run_logged_command(
        self,
        command: list[str],
        *,
        stdout_path: Path,
        stderr_path: Path,
        heartbeat_label: str,
    ) -> tuple[int, float]:
        started = time.monotonic()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                command,
                cwd=self.root,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            return_code = self._wait_with_heartbeat(
                process,
                label=heartbeat_label,
                started=started,
            )
        return return_code, time.monotonic() - started

    def _run_paper_stage(self, paper_id: str, stage: str, ordinal: int) -> bool:
        total = len(self.paper_ids)
        prefix = f"[{ordinal:>3}/{total}] {paper_id} | {stage}"
        if self._stage_already_done(paper_id, stage):
            print(f"[corpus-pipeline] {prefix} | resume-skip", flush=True)
            return True

        command = self._paper_command(paper_id, stage)
        print(f"[corpus-pipeline] {prefix} | start", flush=True)
        print(f"[corpus-pipeline]   $ {shlex.join(command)}", flush=True)
        if self.options.dry_run:
            return True

        log_dir = self.logs_root / "papers" / _safe_component(paper_id)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{stage}.stdout.log"
        stderr_path = log_dir / f"{stage}.stderr.log"
        return_code, elapsed = self._run_logged_command(
            command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            heartbeat_label=prefix,
        )
        status = "passed" if return_code == 0 else "failed"
        self._record_stage(
            paper_id,
            stage,
            status=status,
            return_code=return_code,
            elapsed_seconds=elapsed,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            command=command,
        )
        print(
            f"[corpus-pipeline] {prefix} | {status} | {elapsed:.1f}s",
            flush=True,
        )
        if return_code != 0:
            print(f"[corpus-pipeline]   stderr: {stderr_path}", flush=True)
        return return_code == 0

    def _global_command(self, stage: str) -> list[str]:
        py = sys.executable
        mode = self.options.mode
        if stage == "corpus_graph":
            command = [
                py,
                "-m",
                "scripts.build_corpus_graph",
                "--corpus-id",
                self.corpus_id,
                "--mode",
                mode,
                "--paper-ids",
                *self.paper_ids,
            ]
            if self.options.allow_partial:
                command.append("--allow-critical-partial")
            return command
        if stage == "navigation":
            return [
                py,
                "-m",
                "scripts.build_navigation_graph",
                "--corpus-id",
                self.corpus_id,
                "--mode",
                mode,
            ]
        if stage == "node_index":
            command = [
                py,
                "-m",
                "scripts.build_node_index",
                "--corpus-id",
                self.corpus_id,
                "--mode",
                mode,
                "--batch-size",
                str(self.options.index_batch_size),
            ]
            if self.options.device:
                command.extend(["--device", self.options.device])
            if self.options.include_alignment_hubs_in_index:
                command.append("--include-alignment-hubs")
            return command
        raise KeyError(stage)

    def _global_expected_output(self, stage: str) -> Path:
        mode_root = self.root / "data_dac" / "corpus" / self.corpus_id / self.options.mode
        if stage == "corpus_graph":
            return mode_root / "graph.graphml"
        if stage == "navigation":
            return mode_root / "navigation" / "graph.graphml"
        if stage == "node_index":
            return mode_root / "navigation" / "node_index"
        raise KeyError(stage)

    def _global_already_done(self, stage: str) -> bool:
        if not self.options.resume:
            return False
        record = self.state.get("global_stages", {}).get(stage, {})
        return (
            record.get("status") == "passed"
            and self._global_expected_output(stage).exists()
        )

    def _run_global_stage(self, stage: str) -> bool:
        if self._global_already_done(stage):
            print(f"[corpus-pipeline] global {stage} | resume-skip", flush=True)
            return True
        command = self._global_command(stage)
        print(f"[corpus-pipeline] global {stage} | start", flush=True)
        print(f"[corpus-pipeline]   $ {shlex.join(command)}", flush=True)
        if self.options.dry_run:
            return True

        log_dir = self.logs_root / "global"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{stage}.stdout.log"
        stderr_path = log_dir / f"{stage}.stderr.log"
        return_code, elapsed = self._run_logged_command(
            command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            heartbeat_label=f"global {stage}",
        )
        status = "passed" if return_code == 0 else "failed"
        globals_ = self.state.setdefault("global_stages", {})
        globals_[stage] = {
            "status": status,
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "command": command,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "expected_output": str(self._global_expected_output(stage)),
            "updated_at": _utc_now(),
        }
        self._save_state()
        print(f"[corpus-pipeline] global {stage} | {status} | {elapsed:.1f}s", flush=True)
        if return_code != 0:
            print(f"[corpus-pipeline]   stderr: {stderr_path}", flush=True)
        return return_code == 0

    def run(self) -> dict[str, Any]:
        if self.options.mode not in {"evidence", "mechanism", "exploratory"}:
            raise CorpusPipelineError(f"Unsupported mode: {self.options.mode}")

        print(
            f"[corpus-pipeline] corpus={self.corpus_id} mode={self.options.mode} "
            f"papers={len(self.paper_ids)}/{len(self.available_paper_ids)}",
            flush=True,
        )
        paper_stages = ["extract", "paper_graph"]
        if self.options.mode in {"mechanism", "exploratory"}:
            paper_stages.append("bridge")
        paper_stages.append("projection")

        failures: list[dict[str, str]] = []
        for ordinal, paper_id in enumerate(self.paper_ids, start=1):
            for stage in paper_stages:
                ok = self._run_paper_stage(paper_id, stage, ordinal)
                if not ok:
                    failures.append({"paper_id": paper_id, "stage": stage})
                    break
            if failures and self.options.fail_fast:
                break

        if failures:
            summary = {
                "status": "paper_failures",
                "corpus_id": self.corpus_id,
                "mode": self.options.mode,
                "paper_count": len(self.paper_ids),
                "failure_count": len(failures),
                "failures": failures,
                "state_path": str(self.state_path),
            }
            print(
                "[corpus-pipeline] per-paper phase finished with failures; "
                "global corpus build was not started.",
                flush=True,
            )
            return summary

        global_stages = ["corpus_graph", "navigation"]
        if not self.options.skip_node_index:
            global_stages.append("node_index")

        global_failures: list[str] = []
        for stage in global_stages:
            if not self._run_global_stage(stage):
                global_failures.append(stage)
                break

        status = "passed" if not global_failures else "global_failure"
        return {
            "status": status,
            "corpus_id": self.corpus_id,
            "mode": self.options.mode,
            "paper_count": len(self.paper_ids),
            "paper_ids": self.paper_ids,
            "failure_count": len(global_failures),
            "global_failures": global_failures,
            "state_path": str(self.state_path),
            "corpus_root": str(
                self.root / "data_dac" / "corpus" / self.corpus_id / self.options.mode
            ),
        }
