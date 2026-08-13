from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml


class BroadCorpusPipelineError(RuntimeError):
    pass


def _count_source_tokens(text: str) -> int:
    """Use the project tokenizer when available, with a dependency-light fallback."""
    try:
        from dac_her.chunking import count_tokens

        return int(count_tokens(text))
    except Exception:
        # The fallback is only for preflight safety/tests. Runtime extraction
        # uses dac_her.chunking.count_tokens, so this intentionally errs toward
        # treating long prose as expensive rather than under-counting it.
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def load_broad_paper_configs(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("papers"), dict):
        raise BroadCorpusPipelineError("papers.yaml must contain a papers mapping")
    configs: dict[str, dict[str, Any]] = {}
    for paper_id, raw in payload["papers"].items():
        if not isinstance(raw, dict):
            raise BroadCorpusPipelineError(
                f"paper config must be a mapping: {paper_id!r}"
            )
        configs[str(paper_id)] = dict(raw)
    if not configs:
        raise BroadCorpusPipelineError("papers.yaml contains no papers")
    return configs


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_broad_paper_ids(path: str | Path) -> list[str]:
    return list(load_broad_paper_configs(path).keys())


def select_broad_paper_ids(
    available: Sequence[str],
    *,
    requested: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    if requested:
        unknown = [paper_id for paper_id in requested if paper_id not in available]
        if unknown:
            raise BroadCorpusPipelineError(
                "Requested paper IDs are not in papers.yaml: " + ", ".join(unknown)
            )
        seen: set[str] = set()
        selected = []
        for paper_id in requested:
            if paper_id not in seen:
                selected.append(paper_id)
                seen.add(paper_id)
    else:
        selected = list(available)
    if limit is not None:
        if limit < 1:
            raise BroadCorpusPipelineError("limit must be at least 1")
        selected = selected[:limit]
    if not selected:
        raise BroadCorpusPipelineError("Paper selection is empty")
    return selected


@dataclass(frozen=True)
class BroadPilotOptions:
    data_root: str = "data_broad"
    domain_profile: str = "catalysis_mechanism"
    extract_concurrency: int = 1
    force_extract: bool = False
    broad_compact_schema: bool = False
    broad_prune_metric_vocabulary: bool = False
    allow_partial: bool = False
    skip_extraction: bool = False
    dry_run: bool = False
    # Broad-corpus collection is tolerant at the paper level by default.
    # Strict validation inside each paper remains unchanged; rejected papers
    # are excluded from the corpus rather than aborting the whole run.
    continue_on_error: bool = True
    retry_rejected: bool = False
    resume: bool = True
    # Runtime guard for malformed discovery records that contain full-text-like
    # content in the abstract field. Set <=0 to disable explicitly.
    max_abstract_source_tokens: int = 1200


class BroadCorpusPilotPipeline:
    """Sequential, resumable pilot runner for the abstract Broad KG."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        papers_yaml: str | Path,
        corpus_id: str,
        options: BroadPilotOptions | None = None,
        requested_paper_ids: Sequence[str] | None = None,
        paper_limit: int | None = None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.papers_yaml = Path(papers_yaml)
        if not self.papers_yaml.is_absolute():
            self.papers_yaml = (self.root / self.papers_yaml).resolve()
        if not self.papers_yaml.exists():
            raise BroadCorpusPipelineError(
                f"papers.yaml not found: {self.papers_yaml}"
            )
        self.corpus_id = corpus_id
        self.options = options or BroadPilotOptions()
        if self.options.extract_concurrency < 1:
            raise BroadCorpusPipelineError("extract_concurrency must be at least 1")
        self.data_root = Path(self.options.data_root)
        if not self.data_root.is_absolute():
            self.data_root = self.root / self.data_root
        self.paper_configs = load_broad_paper_configs(self.papers_yaml)
        available = list(self.paper_configs.keys())
        self.paper_ids = select_broad_paper_ids(
            available,
            requested=requested_paper_ids,
            limit=paper_limit,
        )
        self.run_root = (
            self.data_root / "pipeline_runs" / self.corpus_id
        )
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self._freshly_extracted_papers: set[str] = set()
        self._rebuilt_paper_graphs: set[str] = set()
        self._preflight_outlier_ids: set[str] = set()

    def _paper_root(self, paper_id: str) -> Path:
        return self.data_root / "extracted" / paper_id

    def _expected_output(self, paper_id: str, stage: str) -> Path:
        root = self._paper_root(paper_id)
        if stage == "extract":
            return root / "latest_run.json"
        if stage == "paper_graph":
            return root / f"{paper_id}.graphml"
        if stage == "projection":
            return root / "graphagents" / "mechanism" / "graph.graphml"
        raise KeyError(stage)

    def paper_command(self, paper_id: str, stage: str) -> list[str]:
        py = sys.executable
        common = [
            "--paper-id", paper_id,
            "--domain-profile", self.options.domain_profile,
            "--data-root", str(self.data_root),
        ]
        if stage == "extract":
            command = [
                py, "-m", "scripts.extract_paper",
                *common,
                "--config", str(self.papers_yaml),
                "--concurrency", str(self.options.extract_concurrency),
            ]
            if self.options.force_extract:
                command.append("--force")
            if self.options.broad_compact_schema:
                command.append("--broad-compact-schema")
            if self.options.broad_prune_metric_vocabulary:
                command.append("--broad-prune-metric-vocabulary")
            if self.options.allow_partial:
                command.append("--allow-partial")
            return command
        if stage == "paper_graph":
            command = [
                py, "-m", "scripts.build_paper_graph",
                *common,
                "--config", str(self.papers_yaml),
            ]
            if self.options.allow_partial:
                command.append("--allow-incomplete")
            return command
        if stage == "projection":
            return [
                py, "-m", "scripts.build_broad_projection",
                *common,
            ]
        raise KeyError(stage)

    def corpus_command(self, paper_ids: Sequence[str] | None = None) -> list[str]:
        paper_ids = list(paper_ids or self.paper_ids)
        if not paper_ids:
            raise BroadCorpusPipelineError("Cannot build a corpus with zero usable papers")
        command = [
            sys.executable,
            "-m",
            "scripts.build_corpus_graph",
            "--corpus-id",
            self.corpus_id,
            "--domain-profile",
            self.options.domain_profile,
            "--data-root",
            str(self.data_root),
            "--mode",
            "mechanism",
            "--no-pattern-alignment",
            "--paper-ids",
            *paper_ids,
        ]
        if self.options.allow_partial:
            command.append("--allow-critical-partial")
        return command

    def audit_command(self, paper_ids: Sequence[str] | None = None) -> list[str]:
        paper_ids = list(paper_ids or self.paper_ids)
        if not paper_ids:
            raise BroadCorpusPipelineError("Cannot audit a corpus with zero usable papers")
        return [
            sys.executable,
            "-m",
            "scripts.audit_broad_corpus",
            "--corpus-id",
            self.corpus_id,
            "--domain-profile",
            self.options.domain_profile,
            "--data-root",
            str(self.data_root),
            "--paper-ids",
            *paper_ids,
        ]

    def diagnostics_command(
        self,
        paper_ids: Sequence[str] | None = None,
    ) -> list[str]:
        paper_ids = list(paper_ids or self.paper_ids)
        if not paper_ids:
            raise BroadCorpusPipelineError(
                "Cannot diagnose a Broad extraction run with zero papers"
            )
        command = [
            sys.executable,
            "-m",
            "scripts.audit_broad_extraction",
            "--corpus-id",
            self.corpus_id,
            "--domain-profile",
            self.options.domain_profile,
            "--data-root",
            str(self.data_root),
            "--paper-ids",
            *paper_ids,
        ]
        for paper_id in sorted(self._preflight_outlier_ids):
            command.extend(["--preflight-outlier", paper_id])
        return command

    def _run(self, command: list[str], *, label: str) -> bool:
        print(f"[broad-pilot] {label}")
        print("[broad-pilot]   $", shlex.join(command))
        if self.options.dry_run:
            self.records.append({
                "label": label,
                "status": "dry_run",
                "command": command,
                "elapsed_seconds": 0.0,
            })
            return True
        started = time.monotonic()
        completed = subprocess.run(command, cwd=self.root, check=False)
        elapsed = time.monotonic() - started
        status = "passed" if completed.returncode == 0 else "failed"
        self.records.append({
            "label": label,
            "status": status,
            "return_code": completed.returncode,
            "command": command,
            "elapsed_seconds": elapsed,
        })
        print(f"[broad-pilot] {label} | {status} | {elapsed:.1f}s")
        return completed.returncode == 0

    def _latest_extraction_identity(self, paper_id: str) -> dict[str, str] | None:
        pointer_path = self._paper_root(paper_id) / "latest_run.json"
        if not pointer_path.exists():
            return None
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            family_dir = Path(str(pointer["run_directory"]))
            attempt_dir_raw = pointer.get("attempt_directory")
            attempt_dir = (
                Path(str(attempt_dir_raw))
                if attempt_dir_raw
                else None
            )
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
                run_dir = (
                    latest_dir
                    if latest_dir is not None and latest_dir.exists()
                    else family_dir
                )
            active_path = run_dir / "active_chunks.json"
            if not active_path.exists():
                return None
            payload = json.loads(active_path.read_text(encoding="utf-8"))
            run_meta_path = run_dir / "run.json"
            run_meta = (
                json.loads(run_meta_path.read_text(encoding="utf-8"))
                if run_meta_path.exists()
                else {}
            )
            status = str(payload.get("graph_materialization_status", "")).strip()
            run_id = str(
                payload.get("run_id")
                or pointer.get("run_id")
                or run_meta.get("run_id")
                or ""
            ).strip()
            run_fingerprint = str(
                payload.get("run_fingerprint")
                or pointer.get("run_fingerprint")
                or run_meta.get("run_fingerprint")
                or ""
            ).strip()
            attempt_id = str(
                payload.get("attempt_id")
                or pointer.get("attempt_id")
                or run_meta.get("attempt_id")
                or ""
            ).strip()
            return {
                "status": status,
                "run_id": run_id,
                "run_fingerprint": run_fingerprint,
                "attempt_id": attempt_id,
                "run_directory": str(run_dir),
                "run_family_directory": str(family_dir),
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _cached_extraction_status(self, paper_id: str) -> str | None:
        identity = self._latest_extraction_identity(paper_id)
        if identity is None:
            return None
        return identity.get("status") or None

    @staticmethod
    def _identity_matches(
        expected: dict[str, str],
        actual_run_id: str,
        actual_fingerprint: str,
        actual_attempt_id: str = "",
    ) -> bool:
        expected_run_id = str(expected.get("run_id") or "").strip()
        if not expected_run_id or actual_run_id != expected_run_id:
            return False
        expected_fingerprint = str(expected.get("run_fingerprint") or "").strip()
        if expected_fingerprint and actual_fingerprint != expected_fingerprint:
            return False
        expected_attempt_id = str(expected.get("attempt_id") or "").strip()
        if expected_attempt_id and actual_attempt_id != expected_attempt_id:
            return False
        return True

    def _paper_graph_matches_latest_extraction(self, paper_id: str) -> bool:
        identity = self._latest_extraction_identity(paper_id)
        graph_path = self._expected_output(paper_id, "paper_graph")
        if identity is None or not graph_path.exists():
            return False
        try:
            import networkx as nx

            graph = nx.read_graphml(graph_path, force_multigraph=True)
            return self._identity_matches(
                identity,
                str(graph.graph.get("run_id") or "").strip(),
                str(graph.graph.get("run_fingerprint") or "").strip(),
                str(
                    graph.graph.get("source_extraction_attempt_id") or ""
                ).strip(),
            )
        except Exception:
            return False

    def _projection_matches_latest_extraction(self, paper_id: str) -> bool:
        identity = self._latest_extraction_identity(paper_id)
        summary_path = (
            self._paper_root(paper_id)
            / "graphagents"
            / "mechanism"
            / "summary.json"
        )
        if identity is None or not summary_path.exists():
            return False
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            return self._identity_matches(
                identity,
                str(summary.get("source_extraction_run_id") or "").strip(),
                str(
                    summary.get("source_extraction_run_fingerprint") or ""
                ).strip(),
                str(summary.get("source_extraction_attempt_id") or "").strip(),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def _abstract_markdown_path(self, paper_id: str) -> Path | None:
        config = self.paper_configs.get(paper_id) or {}
        documents = config.get("documents")
        if not isinstance(documents, list) or not documents:
            return None
        raw = next(
            (
                item
                for item in documents
                if isinstance(item, dict) and item.get("document_id") == "abstract"
            ),
            documents[0],
        )
        if not isinstance(raw, dict):
            return None
        package_dir = raw.get("package_dir")
        markdown_file = raw.get("markdown_file") or "main.md"
        if not package_dir:
            return None
        package_path = Path(str(package_dir))
        if not package_path.is_absolute():
            package_path = self.root / package_path
        return package_path / str(markdown_file)

    def _abstract_source_preflight(self, paper_id: str) -> dict[str, Any]:
        threshold = int(self.options.max_abstract_source_tokens)
        path = self._abstract_markdown_path(paper_id)
        result: dict[str, Any] = {
            "paper_id": paper_id,
            "source_path": str(path) if path is not None else None,
            "max_abstract_source_tokens": threshold,
            "checked": False,
            "source_tokens_estimated": None,
            "outlier": False,
        }
        if threshold <= 0 or path is None or not path.exists():
            return result
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return result
        tokens = _count_source_tokens(text)
        result.update({
            "checked": True,
            "source_tokens_estimated": tokens,
            "outlier": tokens > threshold,
        })
        return result

    def _cached_extraction_is_usable(self, paper_id: str) -> bool:
        status = self._cached_extraction_status(paper_id)
        if status in {"complete", "partial_acceptable"}:
            return True
        return status == "partial_critical" and self.options.allow_partial

    def _record_cached_rejection(self, paper_id: str, status: str) -> None:
        self.records.append({
            "label": f"{paper_id}:extract",
            "status": "cached_rejected",
            "graph_materialization_status": status,
            "reason": (
                "The latest extraction is not graph-usable. Broad-corpus policy "
                "skips the paper instead of retrying it automatically."
            ),
        })
        print(
            f"[broad-pilot] {paper_id} | extract | cached-{status} | paper-skip",
            flush=True,
        )

    def _run_paper_stage(self, paper_id: str, stage: str) -> bool:
        expected = self._expected_output(paper_id, stage)
        if self.options.resume and stage == "extract" and not self.options.force_extract:
            cached_status = self._cached_extraction_status(paper_id)
            if self._cached_extraction_is_usable(paper_id):
                print(f"[broad-pilot] {paper_id} | extract | resume-skip")
                self.records.append({
                    "label": f"{paper_id}:extract",
                    "status": "resume_skip",
                    "expected_output": str(expected),
                    "graph_materialization_status": cached_status,
                })
                return True
            if cached_status in {"rejected", "partial_critical"} and not self.options.retry_rejected:
                self._record_cached_rejection(paper_id, cached_status)
                return False

        if self.options.resume and stage != "extract" and expected.exists():
            forced_by_upstream = (
                paper_id in self._freshly_extracted_papers
                or (
                    stage == "projection"
                    and paper_id in self._rebuilt_paper_graphs
                )
            )
            identity_matches = (
                self._paper_graph_matches_latest_extraction(paper_id)
                if stage == "paper_graph"
                else self._projection_matches_latest_extraction(paper_id)
            )
            if not forced_by_upstream and identity_matches:
                print(f"[broad-pilot] {paper_id} | {stage} | resume-skip")
                self.records.append({
                    "label": f"{paper_id}:{stage}",
                    "status": "resume_skip",
                    "expected_output": str(expected),
                    "source_extraction_identity_verified": True,
                })
                return True
            reason = (
                "upstream_rebuilt" if forced_by_upstream else "stale_extraction_identity"
            )
            print(
                f"[broad-pilot] {paper_id} | {stage} | rebuild ({reason})",
                flush=True,
            )
            self.records.append({
                "label": f"{paper_id}:{stage}",
                "status": "cache_invalidated",
                "reason": reason,
                "expected_output": str(expected),
            })

        passed = self._run(
            self.paper_command(paper_id, stage),
            label=f"{paper_id}:{stage}",
        )
        if passed and not self.options.dry_run:
            if stage == "extract":
                self._freshly_extracted_papers.add(paper_id)
            elif stage == "paper_graph":
                self._rebuilt_paper_graphs.add(paper_id)
        return passed

    def _write_manifest(
        self,
        *,
        status: str,
        usable_paper_ids: Sequence[str] = (),
        skipped_papers: Sequence[dict[str, Any]] = (),
    ) -> Path:
        path = self.run_root / "run.json"
        payload = {
            "schema_version": "graphagentsdac-broad-pilot-run-v3",
            "corpus_id": self.corpus_id,
            "domain_profile": self.options.domain_profile,
            "data_root": str(self.data_root),
            "papers_yaml": str(self.papers_yaml),
            "max_abstract_source_tokens": self.options.max_abstract_source_tokens,
            "requested_paper_count": len(self.paper_ids),
            "requested_paper_ids": self.paper_ids,
            "usable_paper_count": len(usable_paper_ids),
            "usable_paper_ids": list(usable_paper_ids),
            "skipped_paper_count": len(skipped_papers),
            "skipped_papers": list(skipped_papers),
            "paper_success_fraction": (
                len(usable_paper_ids) / len(self.paper_ids)
                if self.paper_ids else 0.0
            ),
            "status": status,
            "updated_at": _utc_now(),
            "records": self.records,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def run(self) -> Path:
        usable_paper_ids: list[str] = []
        skipped_papers: list[dict[str, Any]] = []

        for ordinal, paper_id in enumerate(self.paper_ids, start=1):
            print(
                f"[broad-pilot] paper {ordinal}/{len(self.paper_ids)}: {paper_id}"
            )
            preflight = self._abstract_source_preflight(paper_id)
            if preflight.get("outlier"):
                self._preflight_outlier_ids.add(paper_id)
                self.records.append({
                    "label": f"{paper_id}:abstract_preflight",
                    "status": "abstract_length_outlier",
                    **preflight,
                })
                skipped_record = {
                    "paper_id": paper_id,
                    "failed_stage": "abstract_preflight",
                    "action": "excluded_from_broad_corpus",
                    "reason": "ABSTRACT_LENGTH_OUTLIER",
                    "source_tokens_estimated": preflight.get(
                        "source_tokens_estimated"
                    ),
                    "max_abstract_source_tokens": preflight.get(
                        "max_abstract_source_tokens"
                    ),
                }
                skipped_papers.append(skipped_record)
                print(
                    f"[broad-pilot] {paper_id} | abstract_preflight | "
                    f"length-outlier tokens={preflight.get('source_tokens_estimated')} "
                    f"> {preflight.get('max_abstract_source_tokens')} | paper-skip",
                    flush=True,
                )
                if not self.options.continue_on_error:
                    manifest = self._write_manifest(
                        status="failed",
                        usable_paper_ids=usable_paper_ids,
                        skipped_papers=skipped_papers,
                    )
                    raise BroadCorpusPipelineError(
                        "Abstract source preflight failed for "
                        f"{paper_id}. Manifest: {manifest}"
                    )
                continue

            stages = [] if self.options.skip_extraction else ["extract"]
            stages.extend(["paper_graph", "projection"])
            failed_stage: str | None = None

            for stage in stages:
                if not self._run_paper_stage(paper_id, stage):
                    failed_stage = stage
                    break

            if failed_stage is None:
                usable_paper_ids.append(paper_id)
                continue

            skipped_record = {
                "paper_id": paper_id,
                "failed_stage": failed_stage,
                "action": "excluded_from_broad_corpus",
            }
            skipped_papers.append(skipped_record)
            print(
                f"[broad-pilot] {paper_id} | excluded | failed_stage={failed_stage}",
                flush=True,
            )

            if not self.options.continue_on_error:
                manifest = self._write_manifest(
                    status="failed",
                    usable_paper_ids=usable_paper_ids,
                    skipped_papers=skipped_papers,
                )
                raise BroadCorpusPipelineError(
                    f"Stage failed: {paper_id}:{failed_stage}. Manifest: {manifest}"
                )

        # Diagnostics are intentionally based on every requested paper, not
        # only the usable subset. This makes rejected/quarantined papers part
        # of the evidence used to tune Broad-specific recovery policy. A
        # diagnostics failure must not invalidate an otherwise usable corpus.
        diagnostics_ok = self._run(
            self.diagnostics_command(self.paper_ids),
            label="extraction_diagnostics",
        )
        if not diagnostics_ok:
            print(
                "[broad-pilot] extraction diagnostics failed; continuing with "
                "corpus construction",
                flush=True,
            )

        if not usable_paper_ids:
            manifest = self._write_manifest(
                status="failed_no_usable_papers",
                usable_paper_ids=usable_paper_ids,
                skipped_papers=skipped_papers,
            )
            raise BroadCorpusPipelineError(
                "No paper completed extraction -> paper graph -> projection; "
                f"corpus build was not attempted. Manifest: {manifest}"
            )

        if not self._run(
            self.corpus_command(usable_paper_ids),
            label="corpus_graph",
        ):
            manifest = self._write_manifest(
                status="failed",
                usable_paper_ids=usable_paper_ids,
                skipped_papers=skipped_papers,
            )
            raise BroadCorpusPipelineError(
                f"Corpus graph build failed. Manifest: {manifest}"
            )
        if not self._run(
            self.audit_command(usable_paper_ids),
            label="broad_audit",
        ):
            manifest = self._write_manifest(
                status="failed",
                usable_paper_ids=usable_paper_ids,
                skipped_papers=skipped_papers,
            )
            raise BroadCorpusPipelineError(
                f"Broad corpus audit failed. Manifest: {manifest}"
            )

        status = (
            "dry_run"
            if self.options.dry_run
            else ("complete_with_skips" if skipped_papers else "complete")
        )
        return self._write_manifest(
            status=status,
            usable_paper_ids=usable_paper_ids,
            skipped_papers=skipped_papers,
        )
