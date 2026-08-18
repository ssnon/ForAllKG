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

import networkx as nx
import yaml


StrictBridgeMode = Literal["mechanism", "exploratory"]


class StrictBridgeCorpusPipelineError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_component(value: str) -> str:
    safe = value.replace("/", "_").replace("\\", "_").strip()
    return safe or "paper"


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


def _sha256_source_tree(root: Path) -> str:
    """Conservative implementation fingerprint for orchestration resume safety.

    Strict and Bridge extraction already own finer-grained fingerprints.  This
    tree hash only decides whether the orchestration state itself may be reused;
    a changed tree causes the CLI stages to be revisited, after which their own
    caches/fingerprints can still avoid unnecessary LLM work.

    Shared implementation under ``pipeline_core`` participates in the same
    resume-safety boundary as the historical ``dac_her`` compatibility/domain
    tree.  Candidate paths are de-duplicated and ordered by repository-relative
    POSIX path so the fingerprint is deterministic across equivalent checkouts.
    """
    digest = hashlib.sha256()

    script_candidates = (
        root / "scripts" / "extract_paper.py",
        root / "scripts" / "build_paper_graph.py",
        root / "scripts" / "extract_bridge_graph.py",
        root / "scripts" / "build_graphagents_projection.py",
        root / "scripts" / "build_corpus_graph.py",
    )

    candidates = {
        *root.glob("dac_her/**/*.py"),
        *root.glob("domains/**/*.py"),
        *root.glob("pipeline_core/**/*.py"),
        *script_candidates,
    }

    ordered_candidates = sorted(
        (
            path
            for path in candidates
            if path.is_file()
        ),
        key=lambda item: (
            item.relative_to(root).as_posix()
        ),
    )

    for path in ordered_candidates:
        relative = (
            path.relative_to(root).as_posix()
        )
        digest.update(
            relative.encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(
            _sha256_file(path).encode("ascii")
        )
        digest.update(b"\0")

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
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_strict_ready_paper_ids(path: str | Path) -> list[str]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("papers"), dict):
        raise StrictBridgeCorpusPipelineError(
            "Strict-ready config must contain a top-level papers mapping"
        )
    paper_ids = [str(value) for value in payload["papers"].keys()]
    if not paper_ids:
        raise StrictBridgeCorpusPipelineError("Strict-ready config contains no papers")
    if len(paper_ids) != len(set(paper_ids)):
        raise StrictBridgeCorpusPipelineError("Strict-ready config contains duplicate IDs")
    return paper_ids


def select_strict_ready_paper_ids(
    available: Sequence[str],
    *,
    requested: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    selected = list(available)
    if requested:
        unknown = [paper_id for paper_id in requested if paper_id not in available]
        if unknown:
            raise StrictBridgeCorpusPipelineError(
                "Requested paper IDs are not present in the strict-ready config: "
                + ", ".join(unknown)
            )
        seen: set[str] = set()
        selected = []
        for paper_id in requested:
            if paper_id in seen:
                continue
            selected.append(paper_id)
            seen.add(paper_id)
    if limit is not None:
        if limit < 1:
            raise StrictBridgeCorpusPipelineError("--limit must be at least 1")
        selected = selected[:limit]
    if not selected:
        raise StrictBridgeCorpusPipelineError("Paper selection is empty")
    return selected


@dataclass(frozen=True)
class StrictBridgePipelineOptions:
    mode: StrictBridgeMode = "mechanism"
    extract_concurrency: int = 4
    bridge_concurrency: int = 4
    heartbeat_seconds: float = 30.0
    allow_partial: bool = False
    force_extract: bool = False
    force_bridge: bool = False
    continue_on_error: bool = True
    resume: bool = True
    dry_run: bool = False
    skip_corpus: bool = False


class StrictBridgeCorpusPipeline:
    """Tolerant acquisition-ready -> Strict -> Bridge -> corpus orchestrator.

    The runner is intentionally thin: it invokes the existing extraction,
    paper-graph, Bridge, projection, and corpus CLIs.  It owns only orchestration
    state, failure isolation, and provenance/freshness checks.
    """

    def __init__(
        self,
        *,
        project_root: str | Path,
        config: str | Path,
        corpus_id: str,
        domain_profile: str,
        data_root: str | Path,
        source_manifest: str | Path | None = None,
        options: StrictBridgePipelineOptions | None = None,
        requested_paper_ids: Sequence[str] | None = None,
        paper_limit: int | None = None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.config = Path(config)
        if not self.config.is_absolute():
            self.config = (self.root / self.config).resolve()
        if not self.config.is_file():
            raise StrictBridgeCorpusPipelineError(
                f"Strict-ready config not found: {self.config}"
            )

        self.source_manifest = Path(source_manifest) if source_manifest else None
        if self.source_manifest is not None and not self.source_manifest.is_absolute():
            self.source_manifest = (self.root / self.source_manifest).resolve()
        elif self.source_manifest is not None:
            self.source_manifest = self.source_manifest.resolve()
        if self.source_manifest is not None and not self.source_manifest.is_file():
            raise StrictBridgeCorpusPipelineError(
                f"Source manifest not found: {self.source_manifest}"
            )

        self.data_root = Path(data_root)
        if not self.data_root.is_absolute():
            self.data_root = (self.root / self.data_root).resolve()
        self.corpus_id = str(corpus_id)
        self.domain_profile = str(domain_profile)
        self.options = options or StrictBridgePipelineOptions()
        if self.options.mode not in {"mechanism", "exploratory"}:
            raise StrictBridgeCorpusPipelineError(
                f"Unsupported strict-Bridge mode: {self.options.mode}"
            )
        if self.options.extract_concurrency < 1 or self.options.bridge_concurrency < 1:
            raise StrictBridgeCorpusPipelineError("Concurrency must be at least 1")
        if self.options.heartbeat_seconds < 0:
            raise StrictBridgeCorpusPipelineError("heartbeat_seconds cannot be negative")

        self.available_paper_ids = load_strict_ready_paper_ids(self.config)
        self.paper_ids = select_strict_ready_paper_ids(
            self.available_paper_ids,
            requested=requested_paper_ids,
            limit=paper_limit,
        )

        self.run_root = self.data_root / "pipeline_runs" / self.corpus_id / "strict_bridge"
        self.logs_root = self.run_root / "logs"
        self.state_path = self.run_root / "state.json"
        self.outcomes_path = self.run_root / "paper_outcomes.jsonl"
        self.manifest_path = self.run_root / "run.json"
        self.run_root.mkdir(parents=True, exist_ok=True)

        self.pipeline_fingerprint = self._pipeline_fingerprint()
        self.state = self._load_state()
        self.records: list[dict[str, Any]] = []

    def _pipeline_fingerprint(self) -> str:
        payload = {
            "config_sha256": _sha256_file(self.config),
            "source_manifest_sha256": _sha256_file(self.source_manifest),
            "domain_profile": self.domain_profile,
            "data_root": str(self.data_root),
            "mode": self.options.mode,
            "paper_ids": self.paper_ids,
            "implementation_tree_sha256": _sha256_source_tree(self.root),
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema_version": "graphagentsdac-strict-bridge-pipeline-state-v1",
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "corpus_id": self.corpus_id,
            "domain_profile": self.domain_profile,
            "data_root": str(self.data_root),
            "mode": self.options.mode,
            "paper_ids": self.paper_ids,
            "papers": {},
            "global": {},
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
        return self.data_root / "extracted" / paper_id

    def _canonical_path(self, paper_id: str) -> Path:
        return self._paper_root(paper_id) / f"{paper_id}.graphml"

    def _bridge_path(self, paper_id: str) -> Path:
        return self._paper_root(paper_id) / f"{paper_id}.bridge.graphml"

    def _candidate_bridge_path(self, paper_id: str) -> Path:
        return self._paper_root(paper_id) / f"{paper_id}.bridge.candidates.graphml"

    def _projection_root(self, paper_id: str) -> Path:
        return self._paper_root(paper_id) / "graphagents" / self.options.mode

    def _projection_path(self, paper_id: str) -> Path:
        return self._projection_root(paper_id) / "graph.graphml"

    def _latest_extraction_identity(self, paper_id: str) -> dict[str, str] | None:
        pointer_path = self._paper_root(paper_id) / "latest_run.json"
        if not pointer_path.exists():
            return None
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            family_dir = Path(str(pointer["run_directory"]))
            attempt_raw = pointer.get("attempt_directory")
            attempt_dir = Path(str(attempt_raw)) if attempt_raw else None
            if attempt_dir is not None and attempt_dir.exists():
                run_dir = attempt_dir
            else:
                latest_attempt_path = family_dir / "latest_attempt.json"
                latest_attempt = (
                    json.loads(latest_attempt_path.read_text(encoding="utf-8"))
                    if latest_attempt_path.exists()
                    else {}
                )
                latest_raw = latest_attempt.get("attempt_directory")
                latest_dir = Path(str(latest_raw)) if latest_raw else None
                run_dir = latest_dir if latest_dir is not None and latest_dir.exists() else family_dir

            active_path = run_dir / "active_chunks.json"
            if not active_path.exists():
                return None
            active = json.loads(active_path.read_text(encoding="utf-8"))
            run_meta_path = run_dir / "run.json"
            run_meta = (
                json.loads(run_meta_path.read_text(encoding="utf-8"))
                if run_meta_path.exists()
                else {}
            )
            return {
                "status": str(active.get("graph_materialization_status") or "").strip(),
                "run_id": str(
                    active.get("run_id") or pointer.get("run_id") or run_meta.get("run_id") or ""
                ).strip(),
                "run_fingerprint": str(
                    active.get("run_fingerprint")
                    or pointer.get("run_fingerprint")
                    or run_meta.get("run_fingerprint")
                    or ""
                ).strip(),
                "attempt_id": str(
                    active.get("attempt_id")
                    or pointer.get("attempt_id")
                    or run_meta.get("attempt_id")
                    or ""
                ).strip(),
                "run_directory": str(run_dir.resolve()),
                "run_family_directory": str(family_dir.resolve()),
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _strict_identity_usable(self, identity: dict[str, str] | None) -> bool:
        if identity is None:
            return False
        status = identity.get("status")
        if status in {"complete", "partial_acceptable"}:
            return True
        return status == "partial_critical" and self.options.allow_partial

    @staticmethod
    def _identity_tuple(identity: dict[str, str] | None) -> dict[str, str]:
        if identity is None:
            return {}
        return {
            key: str(identity.get(key) or "")
            for key in ("run_id", "run_fingerprint", "attempt_id")
        }

    def _strict_run_matches_current_inputs(
        self,
        paper_id: str,
        identity: dict[str, str] | None,
    ) -> bool:
        """Verify a cached Strict run against current source/config/code inputs.

        The Strict extractor already persists the authoritative run fingerprint
        payload in the attempt ``run.json``.  Resume therefore must not depend
        on this orchestrator's state.json surviving unchanged: state is an
        execution journal, not the source of truth for expensive LLM cache
        validity.
        """
        if not self._strict_identity_usable(identity):
            return False
        assert identity is not None
        run_dir = Path(str(identity.get("run_directory") or ""))
        run_meta_path = run_dir / "run.json"
        if not run_meta_path.is_file():
            return False
        try:
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
            from pipeline_core.document_config import (
                get_paper_config,
                paper_config_fingerprint_payload,
            )
            from pipeline_core.document_provenance import document_source_fingerprints

            paper = get_paper_config(
                self.config,
                project_root=self.root,
                paper_id=paper_id,
            )
        except Exception:
            return False

        if str(run_meta.get("run_id") or "") != str(identity.get("run_id") or ""):
            return False
        if str(run_meta.get("run_fingerprint") or "") != str(
            identity.get("run_fingerprint") or ""
        ):
            return False
        if str(run_meta.get("domain_profile_id") or "") != self.domain_profile:
            return False

        if run_meta.get("paper") != paper_config_fingerprint_payload(paper):
            return False
        try:
            current_sources = document_source_fingerprints(paper)
        except Exception:
            return False
        if run_meta.get("document_sources") != current_sources:
            return False

        # A cached run is invalid if any implementation or vocabulary file
        # that participated in its own fingerprint has changed.
        for collection_name in ("implementation_files", "vocabularies"):
            records = run_meta.get(collection_name, [])
            if not isinstance(records, list):
                return False
            for record in records:
                if not isinstance(record, dict):
                    return False
                relative = str(record.get("relative_path") or "").strip()
                expected_sha = str(record.get("sha256") or "").strip()
                if not relative or not expected_sha:
                    return False
                if _sha256_file(self.root / relative) != expected_sha:
                    return False

        # Respect explicit runtime environment overrides when present.  If no
        # override is set, the persisted run metadata remains authoritative.
        env_model = str(os.getenv("OPENROUTER_EXTRACT_MODEL") or "").strip()
        env_provider = str(os.getenv("OPENROUTER_PROVIDER") or "").strip()
        if env_model and str(run_meta.get("model") or "").strip() != env_model:
            return False
        if env_provider and str(run_meta.get("provider") or "").strip() != env_provider:
            return False

        return True

    def _extract_resume_safe(self, paper_id: str) -> bool:
        if not self.options.resume or self.options.force_extract:
            return False
        identity = self._latest_extraction_identity(paper_id)
        return self._strict_run_matches_current_inputs(paper_id, identity)

    def _paper_graph_matches_latest_extraction(self, paper_id: str) -> bool:
        identity = self._latest_extraction_identity(paper_id)
        path = self._canonical_path(paper_id)
        if identity is None or not path.exists():
            return False
        try:
            graph = nx.read_graphml(path, force_multigraph=True)
        except Exception:
            return False
        actual = {
            "run_id": str(graph.graph.get("run_id") or "").strip(),
            "run_fingerprint": str(graph.graph.get("run_fingerprint") or "").strip(),
            "attempt_id": str(
                graph.graph.get("source_extraction_attempt_id") or ""
            ).strip(),
        }
        expected = self._identity_tuple(identity)
        if not expected.get("run_id") or actual["run_id"] != expected["run_id"]:
            return False
        if expected.get("run_fingerprint") and actual["run_fingerprint"] != expected["run_fingerprint"]:
            return False
        if expected.get("attempt_id") and actual["attempt_id"] != expected["attempt_id"]:
            return False
        return True

    def _current_bridge_binding(self, paper_id: str) -> dict[str, str] | None:
        identity = self._latest_extraction_identity(paper_id)
        canonical_path = self._canonical_path(paper_id)
        bridge_path = self._bridge_path(paper_id)
        if identity is None or not canonical_path.exists() or not bridge_path.exists():
            return None
        strict_run_dir = Path(identity["run_directory"])
        pointer_path = strict_run_dir / "latest_bridge_policy_run.json"
        if not pointer_path.exists():
            return None
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            policy_dir = Path(str(pointer["bridge_policy_run_directory"])).resolve()
            policy_run_path = policy_dir / "run.json"
            if not policy_run_path.exists():
                return None
            policy = json.loads(policy_run_path.read_text(encoding="utf-8"))
            graph = nx.read_graphml(bridge_path, force_multigraph=True)
        except Exception:
            return None

        canonical_sha = _sha256_file(canonical_path)
        policy_run_id = str(pointer.get("bridge_policy_run_id") or "").strip()
        policy_fingerprint = str(pointer.get("bridge_policy_run_fingerprint") or "").strip()
        extraction_id = str(pointer.get("bridge_extraction_id") or "").strip()
        if not policy_run_id or not policy_fingerprint or not extraction_id:
            return None
        if str(policy.get("strict_run_directory") or "").strip() != str(strict_run_dir.resolve()):
            return None
        if str(policy.get("canonical_graph_sha256") or "").strip() != canonical_sha:
            return None
        if str(policy.get("bridge_policy_run_id") or "").strip() != policy_run_id:
            return None
        if str(policy.get("bridge_policy_run_fingerprint") or "").strip() != policy_fingerprint:
            return None
        if str(policy.get("bridge_extraction_id") or "").strip() != extraction_id:
            return None
        if str(policy.get("domain_profile_id") or "").strip() != self.domain_profile:
            return None
        if str(graph.graph.get("bridge_policy_run_id") or "").strip() != policy_run_id:
            return None
        if str(graph.graph.get("bridge_policy_run_fingerprint") or "").strip() != policy_fingerprint:
            return None
        if str(graph.graph.get("bridge_extraction_id") or "").strip() != extraction_id:
            return None
        if str(graph.graph.get("domain_profile_id") or "").strip() != self.domain_profile:
            return None
        if self.options.mode == "exploratory" and not self._candidate_bridge_path(paper_id).exists():
            return None

        return {
            **self._identity_tuple(identity),
            "canonical_sha256": canonical_sha,
            "bridge_extraction_id": extraction_id,
            "bridge_policy_run_id": policy_run_id,
            "bridge_policy_run_fingerprint": policy_fingerprint,
            "bridge_sha256": _sha256_file(bridge_path),
        }

    def _projection_matches_current_sources(self, paper_id: str) -> bool:
        binding = self._current_bridge_binding(paper_id)
        summary_path = self._projection_root(paper_id) / "summary.json"
        graph_path = self._projection_path(paper_id)
        if binding is None or not summary_path.exists() or not graph_path.exists():
            return False
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if str(summary.get("mode") or "") != self.options.mode:
            return False
        if str(summary.get("domain_profile_id") or "") != self.domain_profile:
            return False
        if str(summary.get("bridge_extraction_id") or "") != binding["bridge_extraction_id"]:
            return False
        if str(summary.get("bridge_policy_run_id") or "") != binding["bridge_policy_run_id"]:
            return False
        if self.options.mode == "exploratory":
            if str(summary.get("candidate_bridge_policy_run_id") or "") != binding["bridge_policy_run_id"]:
                return False
        return True

    def _bridge_status(self, paper_id: str) -> tuple[str, dict[str, int]]:
        path = self._bridge_path(paper_id)
        if not path.exists():
            return "BRIDGE_ERROR", {"bridge_concepts": 0, "bridge_edges": 0}
        try:
            graph = nx.read_graphml(path, force_multigraph=True)
        except Exception:
            return "BRIDGE_ERROR", {"bridge_concepts": 0, "bridge_edges": 0}
        concept_count = sum(
            1
            for _, attrs in graph.nodes(data=True)
            if str(attrs.get("type") or "") == "BridgeConcept"
        )
        counts = {
            "bridge_concepts": concept_count,
            "bridge_edges": int(graph.number_of_edges()),
        }
        return ("BRIDGE_USEFUL" if concept_count > 0 else "BRIDGE_EMPTY"), counts

    def _paper_command(self, paper_id: str, stage: str) -> list[str]:
        py = sys.executable
        common = [
            "--paper-id",
            paper_id,
            "--config",
            str(self.config),
            "--domain-profile",
            self.domain_profile,
            "--data-root",
            str(self.data_root),
        ]
        if stage == "extract":
            command = [
                py,
                "-m",
                "scripts.extract_paper",
                *common,
                "--concurrency",
                str(self.options.extract_concurrency),
            ]
            if self.options.force_extract:
                command.append("--force")
            if self.options.allow_partial:
                command.append("--allow-partial")
            return command
        if stage == "paper_graph":
            command = [py, "-m", "scripts.build_paper_graph", *common]
            if self.options.allow_partial:
                command.append("--allow-incomplete")
            return command
        if stage == "bridge":
            command = [
                py,
                "-m",
                "scripts.extract_bridge_graph",
                *common,
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
                "--domain-profile",
                self.domain_profile,
                "--data-root",
                str(self.data_root),
                "--mode",
                self.options.mode,
            ]
        raise KeyError(stage)

    def _corpus_command(self, usable_paper_ids: Sequence[str]) -> list[str]:
        if not usable_paper_ids:
            raise StrictBridgeCorpusPipelineError("Cannot build a corpus with zero usable papers")
        command = [
            sys.executable,
            "-m",
            "scripts.build_corpus_graph",
            "--corpus-id",
            self.corpus_id,
            "--domain-profile",
            self.domain_profile,
            "--data-root",
            str(self.data_root),
            "--mode",
            self.options.mode,
            "--paper-ids",
            *usable_paper_ids,
        ]
        if self.options.allow_partial:
            command.append("--allow-critical-partial")
        return command

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
                    f"[strict-bridge] {label} | still running | elapsed={elapsed:.0f}s",
                    flush=True,
                )

    def _run_logged_command(self, command: list[str], *, label: str) -> tuple[bool, float]:
        print(f"[strict-bridge] {label} | start", flush=True)
        print(f"[strict-bridge]   $ {shlex.join(command)}", flush=True)
        if self.options.dry_run:
            self.records.append(
                {"label": label, "status": "dry_run", "command": command, "elapsed_seconds": 0.0}
            )
            return True, 0.0

        safe_label = _safe_component(label.replace(":", "__"))
        stdout_path = self.logs_root / f"{safe_label}.stdout.log"
        stderr_path = self.logs_root / f"{safe_label}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
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
            return_code = self._wait_with_heartbeat(process, label=label, started=started)
        elapsed = time.monotonic() - started
        status = "passed" if return_code == 0 else "failed"
        self.records.append(
            {
                "label": label,
                "status": status,
                "return_code": return_code,
                "command": command,
                "elapsed_seconds": elapsed,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        )
        print(f"[strict-bridge] {label} | {status} | {elapsed:.1f}s", flush=True)
        if return_code != 0:
            print(f"[strict-bridge]   stderr: {stderr_path}", flush=True)
        return return_code == 0, elapsed

    def _record_stage(
        self,
        paper_id: str,
        stage: str,
        *,
        status: str,
        action: str,
        binding: dict[str, str] | None = None,
    ) -> None:
        paper_state = self.state.setdefault("papers", {}).setdefault(paper_id, {"stages": {}})
        paper_state.setdefault("stages", {})[stage] = {
            "status": status,
            "action": action,
            "binding": dict(binding or {}),
            "updated_at": _utc_now(),
        }
        self._save_state()

    def _run_extract(self, paper_id: str) -> tuple[bool, str, dict[str, str] | None]:
        if self._extract_resume_safe(paper_id):
            identity = self._latest_extraction_identity(paper_id)
            print(f"[strict-bridge] {paper_id}:extract | resume-skip", flush=True)
            return True, "resume_skip", identity

        ok, _ = self._run_logged_command(
            self._paper_command(paper_id, "extract"), label=f"{paper_id}:extract"
        )
        if self.options.dry_run:
            return True, "dry_run", None
        identity = self._latest_extraction_identity(paper_id)
        if not ok:
            self._record_stage(paper_id, "extract", status="failed", action="run")
            return False, "run", identity
        if not self._strict_identity_usable(identity):
            self._record_stage(
                paper_id,
                "extract",
                status="rejected",
                action="run",
                binding=self._identity_tuple(identity),
            )
            return False, "run", identity
        self._record_stage(
            paper_id,
            "extract",
            status="passed",
            action="run",
            binding=self._identity_tuple(identity),
        )
        return True, "run", identity

    def _run_paper_graph(self, paper_id: str) -> tuple[bool, str]:
        if self.options.resume and self._paper_graph_matches_latest_extraction(paper_id):
            binding = {
                **self._identity_tuple(self._latest_extraction_identity(paper_id)),
                "canonical_sha256": _sha256_file(self._canonical_path(paper_id)),
            }
            self._record_stage(
                paper_id, "paper_graph", status="passed", action="resume_skip", binding=binding
            )
            print(f"[strict-bridge] {paper_id}:paper_graph | resume-skip", flush=True)
            return True, "resume_skip"
        ok, _ = self._run_logged_command(
            self._paper_command(paper_id, "paper_graph"), label=f"{paper_id}:paper_graph"
        )
        if self.options.dry_run:
            return True, "dry_run"
        verified = ok and self._paper_graph_matches_latest_extraction(paper_id)
        binding = {
            **self._identity_tuple(self._latest_extraction_identity(paper_id)),
            "canonical_sha256": _sha256_file(self._canonical_path(paper_id)),
        }
        self._record_stage(
            paper_id,
            "paper_graph",
            status="passed" if verified else "failed",
            action="run",
            binding=binding if verified else None,
        )
        return verified, "run"

    def _run_bridge(self, paper_id: str) -> tuple[bool, str]:
        current = self._current_bridge_binding(paper_id)
        if self.options.resume and not self.options.force_bridge and current is not None:
            self._record_stage(
                paper_id, "bridge", status="passed", action="resume_skip", binding=current
            )
            print(f"[strict-bridge] {paper_id}:bridge | resume-skip", flush=True)
            return True, "resume_skip"
        ok, _ = self._run_logged_command(
            self._paper_command(paper_id, "bridge"), label=f"{paper_id}:bridge"
        )
        if self.options.dry_run:
            return True, "dry_run"
        current = self._current_bridge_binding(paper_id) if ok else None
        verified = ok and current is not None
        self._record_stage(
            paper_id,
            "bridge",
            status="passed" if verified else "failed",
            action="run",
            binding=current,
        )
        return verified, "run"

    def _run_projection(self, paper_id: str) -> tuple[bool, str]:
        if self.options.resume and self._projection_matches_current_sources(paper_id):
            binding = self._current_bridge_binding(paper_id) or {}
            binding = {
                **binding,
                "projection_sha256": _sha256_file(self._projection_path(paper_id)),
            }
            self._record_stage(
                paper_id, "projection", status="passed", action="resume_skip", binding=binding
            )
            print(f"[strict-bridge] {paper_id}:projection | resume-skip", flush=True)
            return True, "resume_skip"
        ok, _ = self._run_logged_command(
            self._paper_command(paper_id, "projection"), label=f"{paper_id}:projection"
        )
        if self.options.dry_run:
            return True, "dry_run"
        verified = ok and self._projection_matches_current_sources(paper_id)
        binding = self._current_bridge_binding(paper_id) or {}
        if verified:
            binding = {
                **binding,
                "projection_sha256": _sha256_file(self._projection_path(paper_id)),
            }
        self._record_stage(
            paper_id,
            "projection",
            status="passed" if verified else "failed",
            action="run",
            binding=binding if verified else None,
        )
        return verified, "run"

    def _corpus_input_binding(self, usable_paper_ids: Sequence[str]) -> dict[str, str]:
        return {
            paper_id: _sha256_file(self._projection_path(paper_id))
            for paper_id in usable_paper_ids
        }

    def _corpus_resume_safe(self, usable_paper_ids: Sequence[str]) -> bool:
        if not self.options.resume:
            return False
        record = self.state.get("global", {}).get("corpus", {})
        if record.get("status") != "passed":
            return False
        expected_binding = self._corpus_input_binding(usable_paper_ids)
        if record.get("input_binding") != expected_binding:
            return False
        corpus_root = self.data_root / "corpus" / self.corpus_id / self.options.mode
        return (corpus_root / "graph.graphml").exists() and (corpus_root / "audit.json").exists()

    def _run_corpus(self, usable_paper_ids: Sequence[str]) -> tuple[bool, str]:
        if self._corpus_resume_safe(usable_paper_ids):
            print("[strict-bridge] global:corpus | resume-skip", flush=True)
            return True, "resume_skip"
        ok, _ = self._run_logged_command(
            self._corpus_command(usable_paper_ids), label="global:corpus"
        )
        if not self.options.dry_run:
            corpus_root = self.data_root / "corpus" / self.corpus_id / self.options.mode
            ok = ok and (corpus_root / "graph.graphml").exists() and (corpus_root / "audit.json").exists()
        self.state.setdefault("global", {})["corpus"] = {
            "status": "passed" if ok else "failed",
            "action": "run",
            "input_binding": self._corpus_input_binding(usable_paper_ids),
            "updated_at": _utc_now(),
        }
        self._save_state()
        return ok, "run"

    def _write_manifest(
        self,
        *,
        status: str,
        outcomes: Sequence[dict[str, Any]],
        usable_paper_ids: Sequence[str],
        corpus_action: str | None,
    ) -> Path:
        strict_counts: dict[str, int] = {}
        bridge_counts: dict[str, int] = {}
        for outcome in outcomes:
            strict = str(outcome.get("strict_status") or "UNKNOWN")
            bridge = str(outcome.get("bridge_status") or "NOT_RUN")
            strict_counts[strict] = strict_counts.get(strict, 0) + 1
            bridge_counts[bridge] = bridge_counts.get(bridge, 0) + 1
        payload = {
            "schema_version": "graphagentsdac-strict-bridge-corpus-run-v1",
            "status": status,
            "corpus_id": self.corpus_id,
            "domain_profile": self.domain_profile,
            "data_root": str(self.data_root),
            "mode": self.options.mode,
            "strict_ready_config": str(self.config),
            "strict_ready_config_sha256": _sha256_file(self.config),
            "source_manifest": str(self.source_manifest) if self.source_manifest else None,
            "source_manifest_sha256": _sha256_file(self.source_manifest),
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "requested_paper_count": len(self.paper_ids),
            "requested_paper_ids": self.paper_ids,
            "usable_paper_count": len(usable_paper_ids),
            "usable_paper_ids": list(usable_paper_ids),
            "paper_success_fraction": (
                len(usable_paper_ids) / len(self.paper_ids) if self.paper_ids else 0.0
            ),
            "strict_status_counts": strict_counts,
            "bridge_status_counts": bridge_counts,
            "corpus_action": corpus_action,
            "paper_outcomes": str(self.outcomes_path),
            "state_path": str(self.state_path),
            "records": self.records,
            "updated_at": _utc_now(),
        }
        _write_json_atomic(self.manifest_path, payload)
        return self.manifest_path

    def run(self) -> dict[str, Any]:
        outcomes: list[dict[str, Any]] = []
        usable_paper_ids: list[str] = []
        aborted = False

        print(
            f"[strict-bridge] corpus={self.corpus_id} mode={self.options.mode} "
            f"papers={len(self.paper_ids)}/{len(self.available_paper_ids)}",
            flush=True,
        )

        for ordinal, paper_id in enumerate(self.paper_ids, start=1):
            print(
                f"[strict-bridge] paper {ordinal}/{len(self.paper_ids)}: {paper_id}",
                flush=True,
            )
            outcome: dict[str, Any] = {
                "paper_id": paper_id,
                "strict_status": "NOT_RUN",
                "bridge_status": "NOT_RUN",
                "projection_status": "NOT_RUN",
                "corpus_eligible": False,
                "stage_actions": {},
            }

            strict_ok, action, identity = self._run_extract(paper_id)
            outcome["stage_actions"]["extract"] = action
            outcome["strict_extraction_identity"] = self._identity_tuple(identity)
            if not strict_ok:
                outcome["strict_status"] = (
                    "STRICT_REJECTED"
                    if identity is not None and not self._strict_identity_usable(identity)
                    else "STRICT_ERROR"
                )
                outcomes.append(outcome)
                if not self.options.continue_on_error:
                    aborted = True
                    break
                continue
            outcome["strict_status"] = "STRICT_USABLE"

            graph_ok, action = self._run_paper_graph(paper_id)
            outcome["stage_actions"]["paper_graph"] = action
            if not graph_ok:
                outcome["projection_status"] = "PAPER_GRAPH_ERROR"
                outcomes.append(outcome)
                if not self.options.continue_on_error:
                    aborted = True
                    break
                continue
            outcome["canonical_graph_sha256"] = _sha256_file(self._canonical_path(paper_id))

            bridge_ok, action = self._run_bridge(paper_id)
            outcome["stage_actions"]["bridge"] = action
            if not bridge_ok:
                outcome["bridge_status"] = "BRIDGE_ERROR"
                outcomes.append(outcome)
                if not self.options.continue_on_error:
                    aborted = True
                    break
                continue

            if self.options.dry_run:
                bridge_status, bridge_counts = "BRIDGE_DRY_RUN", {
                    "bridge_concepts": 0,
                    "bridge_edges": 0,
                }
            else:
                bridge_status, bridge_counts = self._bridge_status(paper_id)
            outcome["bridge_status"] = bridge_status
            outcome.update(bridge_counts)
            bridge_binding = self._current_bridge_binding(paper_id) or {}
            outcome["bridge_extraction_id"] = bridge_binding.get("bridge_extraction_id", "")
            outcome["bridge_policy_run_id"] = bridge_binding.get("bridge_policy_run_id", "")
            outcome["bridge_graph_sha256"] = bridge_binding.get("bridge_sha256", "")

            projection_ok, action = self._run_projection(paper_id)
            outcome["stage_actions"]["projection"] = action
            if not projection_ok:
                outcome["projection_status"] = "PROJECTION_ERROR"
                outcomes.append(outcome)
                if not self.options.continue_on_error:
                    aborted = True
                    break
                continue

            outcome["projection_status"] = "PROJECTION_USABLE"
            outcome["projection_sha256"] = _sha256_file(self._projection_path(paper_id))
            outcome["corpus_eligible"] = True
            usable_paper_ids.append(paper_id)
            outcomes.append(outcome)

        _write_jsonl(self.outcomes_path, outcomes)

        corpus_action: str | None = None
        corpus_ok = False
        if aborted:
            corpus_ok = False
        elif self.options.skip_corpus:
            corpus_ok = True
            corpus_action = "skipped_by_option"
        elif usable_paper_ids:
            corpus_ok, corpus_action = self._run_corpus(usable_paper_ids)
        else:
            corpus_action = "not_started_zero_usable_papers"

        if self.options.dry_run:
            status = "dry_run"
        elif aborted:
            status = "aborted"
        elif not usable_paper_ids:
            status = "no_usable_papers"
        elif not corpus_ok:
            status = "corpus_failure"
        elif len(usable_paper_ids) < len(self.paper_ids):
            status = "passed_with_paper_skips"
        else:
            status = "passed"

        self._write_manifest(
            status=status,
            outcomes=outcomes,
            usable_paper_ids=usable_paper_ids,
            corpus_action=corpus_action,
        )
        summary = {
            "status": status,
            "corpus_id": self.corpus_id,
            "requested_paper_count": len(self.paper_ids),
            "planned_paper_count": len(self.paper_ids) if self.options.dry_run else None,
            "usable_paper_count": (None if self.options.dry_run else len(usable_paper_ids)),
            "usable_paper_ids": ([] if self.options.dry_run else usable_paper_ids),
            "paper_outcomes": str(self.outcomes_path),
            "run_manifest": str(self.manifest_path),
            "state_path": str(self.state_path),
            "corpus_root": str(
                self.data_root / "corpus" / self.corpus_id / self.options.mode
            ),
        }
        return summary
