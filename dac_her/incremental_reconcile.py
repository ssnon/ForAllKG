from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import networkx as nx

from dac_her.bridge_policy import BRIDGE_POLICY_VERSION
from dac_her.bridge_prompts import BRIDGE_PROMPT_VERSION
from dac_her.config import get_paper_config
from dac_her.domains.extraction_registry import get_extraction_adapter
from dac_her.domains.registry import get_domain_profile
from dac_her.extraction_policy import ExtractionPolicy
from dac_her.run_state import document_source_fingerprints
import pipeline_core.chunking as chunking_module
import dac_her.schemas as schemas_module


Mode = Literal["evidence", "mechanism", "exploratory"]
FreshnessPolicy = Literal["source", "semantic", "full"]
StageName = Literal[
    "strict",
    "strict_graph",
    "bridge",
    "projection",
    "corpus",
    "navigation",
    "index",
]


class ReconcileError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").strip() or "paper"


def _same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except Exception:
        return str(left) == str(right)


@dataclass(frozen=True)
class StageState:
    valid: bool
    reason: str
    path: Path | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def ready(
        cls,
        reason: str,
        path: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "StageState":
        return cls(True, reason, path, metadata)

    @classmethod
    def pending(
        cls,
        reason: str,
        path: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "StageState":
        return cls(False, reason, path, metadata)


@dataclass(frozen=True)
class ReconcileOptions:
    mode: Mode = "exploratory"
    freshness: FreshnessPolicy = "semantic"
    extract_model: str | None = None
    extract_provider: str | None = None
    domain_profile: str = "dac_her"
    kg_data_root: str = "data_dac"
    extract_concurrency: int = 4
    bridge_concurrency: int = 4
    heartbeat_seconds: float = 30.0
    fail_fast: bool = False
    allow_partial: bool = False
    skip_node_index: bool = False
    include_alignment_hubs_in_index: bool = False
    index_batch_size: int = 32
    device: str | None = None
    force_stages: frozenset[str] = frozenset()
    dry_run: bool = False


@dataclass(frozen=True)
class StrictRunState:
    stage: StageState
    run_id: str | None = None
    run_dir: Path | None = None


class IncrementalCorpusReconciler:
    """Reconcile local artifacts to the desired frozen-corpus state.

    The reconciler is deliberately artifact-driven rather than state-file-driven.
    Existing outputs produced before this orchestrator was installed can still be
    recognized and reused.
    """

    def __init__(
        self,
        *,
        project_root: str | Path,
        papers_yaml: str | Path,
        frozen_manifest: str | Path,
        corpus_id: str,
        options: ReconcileOptions | None = None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.papers_yaml = Path(papers_yaml)
        if not self.papers_yaml.is_absolute():
            self.papers_yaml = (self.root / self.papers_yaml).resolve()
        self.frozen_manifest = Path(frozen_manifest)
        if not self.frozen_manifest.is_absolute():
            self.frozen_manifest = (self.root / self.frozen_manifest).resolve()
        self.corpus_id = corpus_id
        self.options = options or ReconcileOptions()
        self.data_root = Path(self.options.kg_data_root)
        if not self.data_root.is_absolute():
            self.data_root = (self.root / self.data_root).resolve()

        if not self.papers_yaml.is_file():
            raise ReconcileError(f"papers.yaml not found: {self.papers_yaml}")
        if not self.frozen_manifest.is_file():
            raise ReconcileError(f"Frozen manifest not found: {self.frozen_manifest}")
        if self.options.extract_concurrency < 1:
            raise ReconcileError("extract_concurrency must be >= 1")
        if self.options.bridge_concurrency < 1:
            raise ReconcileError("bridge_concurrency must be >= 1")
        if self.options.heartbeat_seconds < 0:
            raise ReconcileError("heartbeat_seconds cannot be negative")
        if self.options.freshness not in {"source", "semantic", "full"}:
            raise ReconcileError(
                f"Unknown freshness policy: {self.options.freshness!r}"
            )

        frozen = _read_json(self.frozen_manifest)
        if not frozen:
            raise ReconcileError(f"Invalid frozen manifest: {self.frozen_manifest}")
        self.paper_ids = [
            str(row.get("paper_id") or "")
            for row in (frozen.get("documents") or [])
            if isinstance(row, dict) and str(row.get("paper_id") or "").strip()
        ]
        if not self.paper_ids:
            raise ReconcileError("Frozen manifest contains no paper IDs")
        if len(self.paper_ids) != len(set(self.paper_ids)):
            raise ReconcileError("Frozen manifest contains duplicate paper IDs")

        self.logs_root = (
            self.data_root
            / "reconcile_logs"
            / self.corpus_id
            / self.options.mode
        )
        self.latest_report = (
            self.data_root
            / "reconcile_runs"
            / self.corpus_id
            / self.options.mode
            / "latest.json"
        )
        self.changed_any_projection = False

    def paper_root(self, paper_id: str) -> Path:
        return self.data_root / "extracted" / paper_id

    @staticmethod
    def _semantic_paper_payload_from_config(paper: Any) -> dict[str, Any]:
        return {
            "paper_id": str(paper.paper_id),
            "enabled": bool(paper.enabled),
            "documents": [
                {
                    "document_id": str(document.document_id),
                    "role": str(document.role),
                    "selection": {
                        "mode": str(document.selection.mode),
                        "headings": list(document.selection.headings),
                        "fallback": str(document.selection.fallback),
                        "reference_scope": str(document.selection.reference_scope),
                    },
                    "figure_processing": {
                        "mode": str(document.figure_processing.mode),
                        "vision_assets": list(document.figure_processing.vision_assets),
                        "vision_model": document.figure_processing.vision_model,
                    },
                }
                for document in paper.documents
            ],
        }

    @staticmethod
    def _semantic_paper_payload_from_run(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        documents = value.get("documents")
        if not isinstance(documents, list):
            return None
        normalized: list[dict[str, Any]] = []
        for raw in documents:
            if not isinstance(raw, dict):
                return None
            selection = raw.get("selection") or {}
            figure = raw.get("figure_processing") or {}
            normalized.append({
                "document_id": str(raw.get("document_id") or ""),
                "role": str(raw.get("role") or ""),
                "selection": {
                    "mode": str(selection.get("mode") or ""),
                    "headings": list(selection.get("headings") or []),
                    "fallback": str(selection.get("fallback") or "error"),
                    "reference_scope": str(
                        selection.get("reference_scope") or "selected_main"
                    ),
                },
                "figure_processing": {
                    "mode": str(figure.get("mode") or "caption_first"),
                    "vision_assets": list(figure.get("vision_assets") or []),
                    "vision_model": figure.get("vision_model"),
                },
            })
        return {
            "paper_id": str(value.get("paper_id") or ""),
            "enabled": bool(value.get("enabled", True)),
            "documents": normalized,
        }

    @staticmethod
    def _semantic_policy_payload(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        # Batch/concurrency/retry scheduling changes execution cost and timing,
        # not the intended scientific extraction contract.
        operational = {"logical_batch_size", "concurrency", "max_api_retries"}
        return {
            str(key): item
            for key, item in value.items()
            if str(key) not in operational
        }

    def _current_contract(self, paper_id: str) -> dict[str, Any]:
        paper = get_paper_config(
            self.papers_yaml,
            project_root=self.root,
            paper_id=paper_id,
        )
        profile = get_domain_profile(self.options.domain_profile)
        adapter = get_extraction_adapter(profile.profile_id)
        model = self.options.extract_model or os.getenv("OPENROUTER_EXTRACT_MODEL")
        provider = (
            self.options.extract_provider
            if self.options.extract_provider is not None
            else (os.getenv("OPENROUTER_PROVIDER") or None)
        )
        policy = asdict(ExtractionPolicy())
        vocabularies = [
            {
                "relative_path": str(path.relative_to(self.root)),
                "sha256": _sha256_file(path),
            }
            for path in sorted((self.root / "configs" / "vocabularies").glob("*.yaml"))
        ]
        source = {
            "paper": self._semantic_paper_payload_from_config(paper),
            "document_sources": document_source_fingerprints(paper),
        }
        semantic = {
            **source,
            "model": model,
            "provider": provider,
            "prompt_version": str(adapter.prompt_version),
            "prompt_sha256": hashlib.sha256(
                str(adapter.system_prompt).encode("utf-8")
            ).hexdigest(),
            "schema_sha256": _sha256_file(Path(schemas_module.__file__)),
            "vocabularies": vocabularies,
            "policy": self._semantic_policy_payload(policy),
        }
        return {
            "paper": paper,
            "source": source,
            "semantic": semantic,
            "full": {
                **semantic,
                "domain_profile_id": profile.profile_id,
                "data_root": str(self.data_root),
                "chunking_sha256": _sha256_file(Path(chunking_module.__file__)),
                "policy_full": policy,
            },
        }

    def _run_contract(self, run_meta: dict[str, Any]) -> dict[str, Any]:
        source = {
            "paper": self._semantic_paper_payload_from_run(run_meta.get("paper")),
            "document_sources": run_meta.get("document_sources"),
        }
        semantic = {
            **source,
            "model": run_meta.get("model"),
            "provider": run_meta.get("provider"),
            "prompt_version": run_meta.get("prompt_version"),
            "prompt_sha256": run_meta.get("prompt_sha256"),
            "schema_sha256": run_meta.get("schema_sha256"),
            "vocabularies": run_meta.get("vocabularies"),
            "policy": self._semantic_policy_payload(run_meta.get("policy")),
        }
        return {
            "source": source,
            "semantic": semantic,
            "full": {
                **semantic,
                "domain_profile_id": run_meta.get("domain_profile_id"),
                "data_root": str(run_meta.get("data_root") or ""),
                "chunking_sha256": run_meta.get("chunking_sha256"),
                "policy_full": run_meta.get("policy"),
            },
        }

    @staticmethod
    def _contract_diff(
        expected: dict[str, Any], actual: dict[str, Any]
    ) -> list[str]:
        return [
            key
            for key in expected
            if expected.get(key) != actual.get(key)
        ]

    def _run_compatibility_reason(
        self, run_meta: dict[str, Any], current: dict[str, Any]
    ) -> tuple[bool, str]:
        policy = self.options.freshness
        actual = self._run_contract(run_meta)[policy]
        expected = current[policy]
        differences = self._contract_diff(expected, actual)

        if policy == "full":
            # Full mode additionally verifies every implementation file recorded
            # by the run against the current checkout. This intentionally stays
            # stricter than the default semantic policy.
            implementation_changes: list[str] = []
            rows = run_meta.get("implementation_files")
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        implementation_changes.append("implementation_files")
                        break
                    relative = str(row.get("relative_path") or "")
                    path = self.root / relative
                    if (
                        not relative
                        or not path.is_file()
                        or row.get("sha256") != _sha256_file(path)
                    ):
                        implementation_changes.append("implementation_files")
                        break
            if implementation_changes:
                differences.extend(implementation_changes)

        if differences:
            unique = ", ".join(dict.fromkeys(differences))
            return False, f"{policy} contract changed: {unique}"
        return True, f"{policy} contract matches"

    def _validate_strict_run(
        self, paper_id: str, run_dir: Path, current: dict[str, Any]
    ) -> StrictRunState:
        run_path = run_dir / "run.json"
        active_path = run_dir / "active_chunks.json"
        run_meta = _read_json(run_path)
        active = _read_json(active_path)
        run_id = run_dir.name
        if not run_meta:
            return StrictRunState(
                StageState.pending("strict run.json missing/invalid", run_path),
                run_id,
                run_dir,
            )
        if not active:
            return StrictRunState(
                StageState.pending("active_chunks.json missing/invalid", active_path),
                run_id,
                run_dir,
            )
        if str(run_meta.get("run_id") or "") != run_id:
            return StrictRunState(
                StageState.pending("strict run directory/metadata mismatch", run_path),
                run_id,
                run_dir,
            )
        if str(active.get("run_id") or "") != run_id:
            return StrictRunState(
                StageState.pending("strict active/run metadata mismatch", active_path),
                run_id,
                run_dir,
            )
        chunks = active.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            return StrictRunState(
                StageState.pending("strict run has no active chunks", active_path),
                run_id,
                run_dir,
            )
        quality = str(active.get("graph_materialization_status") or "")
        if not quality and isinstance(active.get("quality"), dict):
            quality = str(active["quality"].get("graph_materialization_status") or "")
        if quality == "rejected":
            return StrictRunState(
                StageState.pending("strict run is rejected", active_path),
                run_id,
                run_dir,
            )
        if quality == "partial_critical" and not self.options.allow_partial:
            return StrictRunState(
                StageState.pending(
                    "strict run is partial_critical and override is disabled",
                    active_path,
                ),
                run_id,
                run_dir,
            )
        compatible, reason = self._run_compatibility_reason(run_meta, current)
        if not compatible:
            return StrictRunState(
                StageState.pending(reason, run_path, run_meta),
                run_id,
                run_dir,
            )
        return StrictRunState(
            StageState.ready(reason, run_dir, run_meta),
            run_id,
            run_dir,
        )

    def _strict_run_candidates(self, paper_id: str) -> tuple[list[Path], str]:
        paper_root = self.paper_root(paper_id)
        pointer_path = paper_root / "latest_run.json"
        pointer = _read_json(pointer_path)
        candidates: list[Path] = []
        pointer_reason = "no latest strict run"
        if pointer:
            run_id = str(pointer.get("run_id") or "").strip()
            if run_id:
                candidates.append(paper_root / "runs" / run_id)
                pointer_reason = f"latest pointer={run_id}"
            else:
                pointer_reason = "latest_run.json has no run_id"

        runs_root = paper_root / "runs"
        if runs_root.is_dir():
            others = sorted(
                (path for path in runs_root.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            seen = {path.resolve() for path in candidates}
            candidates.extend(
                path for path in others if path.resolve() not in seen
            )
        return candidates, pointer_reason

    def strict_state(self, paper_id: str) -> StrictRunState:
        try:
            current = self._current_contract(paper_id)
        except Exception as error:
            return StrictRunState(
                StageState.pending(
                    f"cannot compute current {self.options.freshness} contract: {error}"
                )
            )
        if self.options.freshness in {"semantic", "full"}:
            if not current["semantic"].get("model"):
                return StrictRunState(
                    StageState.pending(
                        "current extraction model is unknown; set "
                        "OPENROUTER_EXTRACT_MODEL or --extract-model"
                    )
                )

        candidates, pointer_reason = self._strict_run_candidates(paper_id)
        if not candidates:
            return StrictRunState(StageState.pending(pointer_reason))

        failures: list[str] = []
        for index, run_dir in enumerate(candidates):
            state = self._validate_strict_run(paper_id, run_dir, current)
            if state.stage.valid:
                if index == 0:
                    return state
                return StrictRunState(
                    StageState.ready(
                        "recovered compatible usable strict run from runs/*; "
                        f"{pointer_reason}; selected={state.run_id}",
                        state.run_dir,
                        state.stage.metadata,
                    ),
                    state.run_id,
                    state.run_dir,
                )
            failures.append(f"{run_dir.name}: {state.stage.reason}")

        detail = "; ".join(failures[:3])
        if len(failures) > 3:
            detail += f"; +{len(failures) - 3} older run(s)"
        return StrictRunState(
            StageState.pending(
                "no compatible usable strict run exists"
                + (f" ({detail})" if detail else "")
            )
        )

    def _repair_latest_run_pointer(self, paper_id: str, strict: StrictRunState) -> None:
        if not strict.stage.valid or not strict.run_id or not strict.run_dir:
            return
        if not strict.stage.metadata:
            return
        pointer_path = self.paper_root(paper_id) / "latest_run.json"
        pointer = _read_json(pointer_path)
        if pointer and str(pointer.get("run_id") or "") == strict.run_id:
            return
        metadata = strict.stage.metadata
        payload = {
            "paper_id": paper_id,
            "run_id": strict.run_id,
            "run_fingerprint": str(metadata.get("run_fingerprint") or ""),
            "run_directory": str(strict.run_dir.resolve()),
            "updated_at_utc": _utc_now(),
            "recovered_by": "incremental_reconciler",
        }
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[reconcile] {paper_id} | latest_run | repaired -> {strict.run_id}",
            flush=True,
        )

    def _latest_usable_run_without_freshness(
        self, paper_id: str
    ) -> StrictRunState | None:
        runs_root = self.paper_root(paper_id) / "runs"
        if not runs_root.is_dir():
            return None
        candidates = sorted(
            (path for path in runs_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for run_dir in candidates:
            run_id = run_dir.name
            run_meta = _read_json(run_dir / "run.json")
            active = _read_json(run_dir / "active_chunks.json")
            if not run_meta or not active:
                continue
            if str(run_meta.get("run_id") or "") != run_id:
                continue
            if str(active.get("run_id") or "") != run_id:
                continue
            chunks = active.get("chunks")
            if not isinstance(chunks, list) or not chunks:
                continue
            quality = str(active.get("graph_materialization_status") or "")
            if not quality and isinstance(active.get("quality"), dict):
                quality = str(
                    active["quality"].get("graph_materialization_status") or ""
                )
            if quality == "rejected":
                continue
            if quality == "partial_critical" and not self.options.allow_partial:
                continue
            return StrictRunState(
                StageState.ready(
                    "latest structurally usable strict run", run_dir, run_meta
                ),
                run_id,
                run_dir,
            )
        return None

    def _repair_broken_latest_pointer_before_attempt(self, paper_id: str) -> None:
        pointer_path = self.paper_root(paper_id) / "latest_run.json"
        pointer = _read_json(pointer_path)
        if pointer:
            run_id = str(pointer.get("run_id") or "").strip()
            if run_id:
                run_dir = self.paper_root(paper_id) / "runs" / run_id
                run_meta = _read_json(run_dir / "run.json")
                active = _read_json(run_dir / "active_chunks.json")
                chunks = active.get("chunks") if active else None
                quality = (
                    str(active.get("graph_materialization_status") or "")
                    if active
                    else ""
                )
                structurally_usable = (
                    bool(run_meta)
                    and bool(active)
                    and str(run_meta.get("run_id") or "") == run_id
                    and str(active.get("run_id") or "") == run_id
                    and isinstance(chunks, list)
                    and bool(chunks)
                    and quality != "rejected"
                    and not (
                        quality == "partial_critical"
                        and not self.options.allow_partial
                    )
                )
                if structurally_usable:
                    return

        fallback = self._latest_usable_run_without_freshness(paper_id)
        if fallback is not None:
            self._repair_latest_run_pointer(paper_id, fallback)

    def strict_graph_state(
        self,
        paper_id: str,
        strict: StrictRunState | None = None,
    ) -> StageState:
        strict = strict or self.strict_state(paper_id)
        path = self.paper_root(paper_id) / f"{paper_id}.graphml"
        if not strict.stage.valid or not strict.run_id:
            return StageState.pending("strict extraction is not current", path)
        if not path.is_file():
            return StageState.pending("canonical strict KG missing", path)
        try:
            graph = nx.read_graphml(path, force_multigraph=True)
        except Exception as error:
            return StageState.pending(f"canonical GraphML unreadable: {error}", path)
        if str(graph.graph.get("run_id") or "") != strict.run_id:
            return StageState.pending("canonical KG belongs to another strict run", path)
        profile = str(graph.graph.get("domain_profile_id") or "")
        if profile and profile != self.options.domain_profile:
            return StageState.pending(
                f"canonical KG domain mismatch: {profile!r}",
                path,
            )
        return StageState.ready("current canonical strict KG exists", path)

    def bridge_state(
        self,
        paper_id: str,
        strict: StrictRunState | None = None,
        strict_graph: StageState | None = None,
    ) -> StageState:
        if self.options.mode == "evidence":
            return StageState.ready("Bridge not required in evidence mode")
        strict = strict or self.strict_state(paper_id)
        strict_graph = strict_graph or self.strict_graph_state(paper_id, strict)
        path = self.paper_root(paper_id) / f"{paper_id}.bridge.graphml"
        if not strict.stage.valid or not strict.run_dir or not strict.run_id:
            return StageState.pending("strict extraction is not current", path)
        if not strict_graph.valid:
            return StageState.pending("canonical KG is not current", path)

        pointer_path = strict.run_dir / "latest_bridge_policy_run.json"
        pointer = _read_json(pointer_path)
        if not pointer:
            return StageState.pending("no Bridge policy run for current strict run", path)
        if str(pointer.get("strict_run_id") or "") != strict.run_id:
            return StageState.pending("Bridge points to another strict run", path)
        if not path.is_file():
            return StageState.pending("Bridge KG missing", path)
        try:
            graph = nx.read_graphml(path, force_multigraph=True)
        except Exception as error:
            return StageState.pending(f"Bridge GraphML unreadable: {error}", path)

        if self.options.freshness != "source":
            graph_prompt_version = str(graph.graph.get("bridge_prompt_version") or "")
            graph_policy_version = str(graph.graph.get("bridge_policy_version") or "")
            if graph_prompt_version and graph_prompt_version != str(BRIDGE_PROMPT_VERSION):
                return StageState.pending(
                    "Bridge prompt version changed since extraction", path
                )
            if graph_policy_version and graph_policy_version != str(BRIDGE_POLICY_VERSION):
                return StageState.pending(
                    "Bridge policy version changed since materialization", path
                )

        expected_policy = str(pointer.get("bridge_policy_run_id") or "")
        expected_extract = str(pointer.get("bridge_extraction_id") or "")
        if not expected_policy or str(graph.graph.get("bridge_policy_run_id") or "") != expected_policy:
            return StageState.pending("Bridge policy run is not current", path)
        if not expected_extract or str(graph.graph.get("bridge_extraction_id") or "") != expected_extract:
            return StageState.pending("Bridge extraction is not current", path)
        if strict_graph.path and strict_graph.path.stat().st_mtime_ns > path.stat().st_mtime_ns:
            return StageState.pending("canonical KG is newer than Bridge KG", path)

        if self.options.mode == "exploratory":
            candidate = self.paper_root(paper_id) / f"{paper_id}.bridge.candidates.graphml"
            if not candidate.is_file():
                return StageState.pending("candidate Bridge KG missing", candidate)
            try:
                candidate_graph = nx.read_graphml(candidate, force_multigraph=True)
            except Exception as error:
                return StageState.pending(f"candidate Bridge unreadable: {error}", candidate)
            if str(candidate_graph.graph.get("bridge_policy_run_id") or "") != expected_policy:
                return StageState.pending("candidate Bridge policy run is not current", candidate)
        return StageState.ready("current Bridge KG exists", path, pointer)

    def projection_state(
        self,
        paper_id: str,
        strict_graph: StageState | None = None,
        bridge: StageState | None = None,
    ) -> StageState:
        strict_graph = strict_graph or self.strict_graph_state(paper_id)
        bridge = bridge or self.bridge_state(paper_id)
        root = self.paper_root(paper_id) / "graphagents" / self.options.mode
        graph_path = root / "graph.graphml"
        summary_path = root / "summary.json"
        node_text = root / "node_text.jsonl"
        evidence = root / "edge_evidence.jsonl"
        required = (graph_path, summary_path, node_text, evidence)
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            return StageState.pending(
                "projection bundle incomplete: " + ", ".join(missing),
                graph_path,
            )
        if not strict_graph.valid or not strict_graph.path:
            return StageState.pending("canonical KG is not current", graph_path)
        if self.options.mode != "evidence" and not bridge.valid:
            return StageState.pending("Bridge KG is not current", graph_path)

        summary = _read_json(summary_path)
        if not summary:
            return StageState.pending("projection summary invalid", summary_path)
        if str(summary.get("paper_id") or "") != paper_id:
            return StageState.pending("projection paper_id mismatch", summary_path)
        if str(summary.get("mode") or "") != self.options.mode:
            return StageState.pending("projection mode mismatch", summary_path)
        profile = str(summary.get("domain_profile_id") or "")
        if profile and profile != self.options.domain_profile:
            return StageState.pending("projection domain mismatch", summary_path)

        output_time = min(path.stat().st_mtime_ns for path in required)
        if strict_graph.path.stat().st_mtime_ns > output_time:
            return StageState.pending("canonical KG is newer than projection", graph_path)

        if self.options.mode != "evidence" and bridge.path:
            if bridge.path.stat().st_mtime_ns > output_time:
                return StageState.pending("Bridge KG is newer than projection", graph_path)
            try:
                bridge_graph = nx.read_graphml(bridge.path, force_multigraph=True)
            except Exception as error:
                return StageState.pending(f"Bridge GraphML unreadable: {error}", bridge.path)
            expected_policy = str(bridge_graph.graph.get("bridge_policy_run_id") or "")
            if str(summary.get("bridge_policy_run_id") or "") != expected_policy:
                return StageState.pending("projection references another Bridge policy run", graph_path)
            if self.options.mode == "exploratory":
                if str(summary.get("candidate_bridge_policy_run_id") or "") != expected_policy:
                    return StageState.pending(
                        "projection candidate Bridge policy run mismatch",
                        graph_path,
                    )
        return StageState.ready("current GraphAgents projection exists", graph_path, summary)

    def _paper_command(
        self, paper_id: str, stage: StageName, *, run_id: str | None = None
    ) -> list[str]:
        py = sys.executable
        common = [
            "--paper-id",
            paper_id,
            "--domain-profile",
            self.options.domain_profile,
        ]
        if self.options.kg_data_root:
            common += ["--data-root", str(self.data_root)]

        if stage == "strict":
            command = [
                py,
                "-m",
                "scripts.extract_paper",
                *common,
                "--config",
                str(self.papers_yaml),
                "--concurrency",
                str(self.options.extract_concurrency),
            ]
            if self.options.extract_model:
                command += ["--model", self.options.extract_model]
            if self.options.extract_provider:
                command += ["--provider", self.options.extract_provider]
            if self.options.allow_partial:
                command.append("--allow-partial")
            return command
        if stage == "strict_graph":
            command = [
                py,
                "-m",
                "scripts.build_paper_graph",
                *common,
                "--config",
                str(self.papers_yaml),
            ]
            if run_id:
                command += ["--run-id", run_id]
            if self.options.allow_partial:
                command.append("--allow-incomplete")
            return command
        if stage == "bridge":
            command = [
                py,
                "-m",
                "scripts.extract_bridge_graph",
                *common,
                "--config",
                str(self.papers_yaml),
                "--concurrency",
                str(self.options.bridge_concurrency),
            ]
            if run_id:
                command += ["--run-id", run_id]
            if self.options.allow_partial:
                command.append("--allow-incomplete")
            return command
        if stage == "projection":
            return [
                py,
                "-m",
                "scripts.build_graphagents_projection",
                *common,
                "--mode",
                self.options.mode,
            ]
        raise KeyError(stage)

    def _global_command(self, stage: StageName) -> list[str]:
        py = sys.executable
        mode_root = self.data_root / "corpus" / self.corpus_id / self.options.mode
        if stage == "corpus":
            command = [
                py,
                "-m",
                "scripts.build_corpus_graph",
                "--corpus-id",
                self.corpus_id,
                "--domain-profile",
                self.options.domain_profile,
                "--data-root",
                str(self.data_root),
                "--mode",
                self.options.mode,
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
                self.options.mode,
                "--corpus-graphml",
                str(mode_root / "graph.graphml"),
                "--output-dir",
                str(mode_root / "navigation"),
            ]
        if stage == "index":
            command = [
                py,
                "-m",
                "scripts.build_node_index",
                "--corpus-id",
                self.corpus_id,
                "--mode",
                self.options.mode,
                "--navigation-graphml",
                str(mode_root / "navigation" / "graph.graphml"),
                "--node-text",
                str(mode_root / "node_text.jsonl"),
                "--output-dir",
                str(mode_root / "navigation" / "node_index"),
                "--batch-size",
                str(self.options.index_batch_size),
            ]
            if self.options.device:
                command += ["--device", self.options.device]
            if self.options.include_alignment_hubs_in_index:
                command.append("--include-alignment-hubs")
            return command
        raise KeyError(stage)

    def _run_logged(self, command: Sequence[str], *, label: str, log_dir: Path) -> bool:
        print(f"[reconcile] {label} | start", flush=True)
        print(f"[reconcile]   $ {shlex.join(list(command))}", flush=True)
        if self.options.dry_run:
            return True

        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"
        started = time.monotonic()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                list(command),
                cwd=self.root,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            while True:
                try:
                    code = process.wait(
                        timeout=(self.options.heartbeat_seconds or None)
                    )
                    break
                except subprocess.TimeoutExpired:
                    print(
                        f"[reconcile] {label} | still running | "
                        f"elapsed={time.monotonic() - started:.0f}s",
                        flush=True,
                    )
        elapsed = time.monotonic() - started
        if code == 0:
            print(f"[reconcile] {label} | passed | {elapsed:.1f}s", flush=True)
            return True
        print(
            f"[reconcile] {label} | failed({code}) | {elapsed:.1f}s | "
            f"stderr={stderr_path}",
            flush=True,
        )
        return False

    def _force(self, stage: StageName) -> bool:
        return stage in self.options.force_stages

    def reconcile_paper(self, paper_id: str, ordinal: int) -> dict[str, Any]:
        total = len(self.paper_ids)
        prefix = f"[{ordinal:>3}/{total}] {paper_id}"
        result: dict[str, Any] = {"paper_id": paper_id, "stages": {}}

        strict = self.strict_state(paper_id)
        if self._force("strict") or not strict.stage.valid:
            reason = (
                "forced by --force-stage"
                if self._force("strict")
                else strict.stage.reason
            )
            result["stages"]["strict"] = reason
            print(f"[reconcile] {prefix} | strict | RUN ({reason})", flush=True)
            if not self.options.dry_run:
                # Migration/self-heal: an older interrupted extractor may have
                # overwritten latest_run.json before completion. Restore the
                # newest usable run first so another interruption cannot hide it.
                self._repair_broken_latest_pointer_before_attempt(paper_id)
            if not self._run_logged(
                self._paper_command(paper_id, "strict"),
                label=f"{prefix} | strict",
                log_dir=self.logs_root / "papers" / _safe_component(paper_id) / "strict",
            ):
                result["failed_stage"] = "strict"
                return result
            if not self.options.dry_run:
                strict = self.strict_state(paper_id)
                if not strict.stage.valid:
                    result["failed_stage"] = "strict_validation"
                    result["error"] = strict.stage.reason
                    return result
        else:
            print(f"[reconcile] {prefix} | strict | SKIP ({strict.stage.reason})", flush=True)
            if not self.options.dry_run:
                self._repair_latest_run_pointer(paper_id, strict)

        graph = self.strict_graph_state(paper_id, strict)
        if self._force("strict_graph") or not graph.valid:
            reason = (
                "forced by --force-stage"
                if self._force("strict_graph")
                else graph.reason
            )
            result["stages"]["strict_graph"] = reason
            print(f"[reconcile] {prefix} | strict_graph | RUN ({reason})", flush=True)
            if not self._run_logged(
                self._paper_command(paper_id, "strict_graph", run_id=strict.run_id),
                label=f"{prefix} | strict_graph",
                log_dir=self.logs_root / "papers" / _safe_component(paper_id) / "strict_graph",
            ):
                result["failed_stage"] = "strict_graph"
                return result
            if not self.options.dry_run:
                graph = self.strict_graph_state(paper_id, strict)
                if not graph.valid:
                    result["failed_stage"] = "strict_graph_validation"
                    result["error"] = graph.reason
                    return result
        else:
            print(f"[reconcile] {prefix} | strict_graph | SKIP ({graph.reason})", flush=True)

        bridge = self.bridge_state(paper_id, strict, graph)
        if self.options.mode != "evidence":
            if self._force("bridge") or not bridge.valid:
                reason = (
                    "forced by --force-stage"
                    if self._force("bridge")
                    else bridge.reason
                )
                result["stages"]["bridge"] = reason
                print(f"[reconcile] {prefix} | bridge | RUN ({reason})", flush=True)
                if not self._run_logged(
                    self._paper_command(paper_id, "bridge", run_id=strict.run_id),
                    label=f"{prefix} | bridge",
                    log_dir=self.logs_root / "papers" / _safe_component(paper_id) / "bridge",
                ):
                    result["failed_stage"] = "bridge"
                    return result
                if not self.options.dry_run:
                    bridge = self.bridge_state(paper_id, strict, graph)
                    if not bridge.valid:
                        result["failed_stage"] = "bridge_validation"
                        result["error"] = bridge.reason
                        return result
            else:
                print(f"[reconcile] {prefix} | bridge | SKIP ({bridge.reason})", flush=True)

        projection = self.projection_state(paper_id, graph, bridge)
        if self._force("projection") or not projection.valid:
            reason = (
                "forced by --force-stage"
                if self._force("projection")
                else projection.reason
            )
            result["stages"]["projection"] = reason
            print(f"[reconcile] {prefix} | projection | RUN ({reason})", flush=True)
            if not self._run_logged(
                self._paper_command(paper_id, "projection"),
                label=f"{prefix} | projection",
                log_dir=self.logs_root / "papers" / _safe_component(paper_id) / "projection",
            ):
                result["failed_stage"] = "projection"
                return result
            self.changed_any_projection = True
            if not self.options.dry_run:
                projection = self.projection_state(paper_id, graph, bridge)
                if not projection.valid:
                    result["failed_stage"] = "projection_validation"
                    result["error"] = projection.reason
                    return result
        else:
            print(f"[reconcile] {prefix} | projection | SKIP ({projection.reason})", flush=True)

        result["status"] = "ready"
        return result

    def _projection_hashes(self, paper_id: str) -> dict[str, str] | None:
        root = self.paper_root(paper_id) / "graphagents" / self.options.mode
        paths = {
            "graphml": root / "graph.graphml",
            "node_text": root / "node_text.jsonl",
            "edge_evidence": root / "edge_evidence.jsonl",
            "summary": root / "summary.json",
        }
        if any(not path.is_file() for path in paths.values()):
            return None
        return {key: _sha256_file(path) for key, path in paths.items()}

    def corpus_state(self) -> StageState:
        root = self.data_root / "corpus" / self.corpus_id / self.options.mode
        graph = root / "graph.graphml"
        manifest_path = root / "manifest.json"
        audit_path = root / "audit.json"
        manifest = _read_json(manifest_path)
        audit = _read_json(audit_path)
        if not graph.is_file() or not manifest or not audit:
            return StageState.pending("corpus bundle missing/incomplete", graph)
        if manifest.get("paper_ids") != self.paper_ids:
            return StageState.pending("corpus paper set differs from frozen corpus", graph)
        if not bool(manifest.get("passes_structural_gate")):
            return StageState.pending("corpus structural gate is not passing", graph)

        old_rows = {
            str(row.get("paper_id") or ""): row
            for row in (manifest.get("papers") or [])
            if isinstance(row, dict)
        }
        for paper_id in self.paper_ids:
            current = self._projection_hashes(paper_id)
            old = old_rows.get(paper_id)
            if current is None or old is None:
                return StageState.pending(f"corpus lacks current projection for {paper_id}", graph)
            if old.get("sha256") != current:
                return StageState.pending(f"projection changed for {paper_id}", graph)
        return StageState.ready("corpus includes the current projection set", graph, manifest)

    def navigation_state(self) -> StageState:
        mode_root = self.data_root / "corpus" / self.corpus_id / self.options.mode
        corpus_graph = mode_root / "graph.graphml"
        root = mode_root / "navigation"
        graph = root / "graph.graphml"
        summary_path = root / "summary.json"
        summary = _read_json(summary_path)
        if not corpus_graph.is_file() or not graph.is_file() or not summary:
            return StageState.pending("navigation bundle missing/incomplete", graph)
        if corpus_graph.stat().st_mtime_ns > graph.stat().st_mtime_ns:
            return StageState.pending("corpus graph is newer than navigation graph", graph)
        source = summary.get("source_graphml")
        if source and not _same_path(str(source), corpus_graph):
            return StageState.pending("navigation summary points to another corpus graph", graph)
        return StageState.ready("navigation graph is current", graph, summary)

    def index_state(self) -> StageState:
        mode_root = self.data_root / "corpus" / self.corpus_id / self.options.mode
        navigation = mode_root / "navigation" / "graph.graphml"
        node_text = mode_root / "node_text.jsonl"
        root = mode_root / "navigation" / "node_index"
        manifest_path = root / "manifest.json"
        records = root / "records.jsonl"
        embeddings = root / "embeddings.npy"
        manifest = _read_json(manifest_path)
        if not manifest or not records.is_file() or not embeddings.is_file():
            return StageState.pending("node index missing/incomplete", root)
        if not navigation.is_file() or not node_text.is_file():
            return StageState.pending("node-index source files are missing", root)
        if manifest.get("navigation_graph_sha256") != _sha256_file(navigation):
            return StageState.pending("navigation graph changed since indexing", root)
        if manifest.get("node_text_sha256") != _sha256_file(node_text):
            return StageState.pending("corpus node_text changed since indexing", root)
        return StageState.ready("node index is current", root, manifest)

    def reconcile_globals(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        corpus = self.corpus_state()
        if self._force("corpus") or self.changed_any_projection or not corpus.valid:
            if not self._run_logged(
                self._global_command("corpus"),
                label="global | corpus",
                log_dir=self.logs_root / "global" / "corpus",
            ):
                return {"failed_stage": "corpus"}
            if not self.options.dry_run:
                corpus = self.corpus_state()
                if not corpus.valid:
                    return {"failed_stage": "corpus_validation", "error": corpus.reason}
        else:
            print(f"[reconcile] global | corpus | skip ({corpus.reason})", flush=True)
        result["corpus"] = corpus.reason

        navigation = self.navigation_state()
        if self._force("navigation") or not navigation.valid:
            if not self._run_logged(
                self._global_command("navigation"),
                label="global | navigation",
                log_dir=self.logs_root / "global" / "navigation",
            ):
                return {"failed_stage": "navigation"}
            if not self.options.dry_run:
                navigation = self.navigation_state()
                if not navigation.valid:
                    return {
                        "failed_stage": "navigation_validation",
                        "error": navigation.reason,
                    }
        else:
            print(
                f"[reconcile] global | navigation | skip ({navigation.reason})",
                flush=True,
            )
        result["navigation"] = navigation.reason

        if self.options.skip_node_index:
            result["index"] = "skipped by option"
            return result

        index = self.index_state()
        if self._force("index") or not index.valid:
            if not self._run_logged(
                self._global_command("index"),
                label="global | index",
                log_dir=self.logs_root / "global" / "index",
            ):
                return {"failed_stage": "index"}
            if not self.options.dry_run:
                index = self.index_state()
                if not index.valid:
                    return {"failed_stage": "index_validation", "error": index.reason}
        else:
            print(f"[reconcile] global | index | skip ({index.reason})", flush=True)
        result["index"] = index.reason
        return result

    def status_table(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for paper_id in self.paper_ids:
            strict = self.strict_state(paper_id)
            graph = self.strict_graph_state(paper_id, strict)
            bridge = self.bridge_state(paper_id, strict, graph)
            projection = self.projection_state(paper_id, graph, bridge)
            rows.append({
                "paper_id": paper_id,
                "strict": "ready" if strict.stage.valid else strict.stage.reason,
                "strict_graph": "ready" if graph.valid else graph.reason,
                "bridge": "n/a" if self.options.mode == "evidence" else (
                    "ready" if bridge.valid else bridge.reason
                ),
                "projection": "ready" if projection.valid else projection.reason,
            })
        return rows

    def run(self) -> dict[str, Any]:
        print(
            f"[reconcile] corpus={self.corpus_id} mode={self.options.mode} "
            f"papers={len(self.paper_ids)}",
            flush=True,
        )
        paper_results: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for ordinal, paper_id in enumerate(self.paper_ids, start=1):
            result = self.reconcile_paper(paper_id, ordinal)
            paper_results.append(result)
            if result.get("failed_stage"):
                failures.append(result)
                if self.options.fail_fast:
                    break

        global_result: dict[str, Any] = {}
        if not failures:
            global_result = self.reconcile_globals()
            if global_result.get("failed_stage"):
                failures.append({"paper_id": None, **global_result})
        else:
            print(
                "[reconcile] paper failures remain; global corpus rebuild is blocked.",
                flush=True,
            )

        report = {
            "schema_version": "graphagentsdac-incremental-reconcile-v01",
            "created_at": _utc_now(),
            "corpus_id": self.corpus_id,
            "mode": self.options.mode,
            "domain_profile": self.options.domain_profile,
            "paper_count": len(self.paper_ids),
            "changed_any_projection": self.changed_any_projection,
            "status": "passed" if not failures else "failed",
            "failures": failures,
            "papers": paper_results,
            "global": global_result,
        }
        if not self.options.dry_run:
            self.latest_report.parent.mkdir(parents=True, exist_ok=True)
            temp = self.latest_report.with_suffix(".tmp")
            temp.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temp, self.latest_report)
        return report
