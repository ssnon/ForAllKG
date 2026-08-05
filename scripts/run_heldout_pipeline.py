from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FATAL_PATTERNS = (
    r"AuthenticationError",
    r"PermissionDenied",
    r"Unauthorized",
    r"invalid_api_key",
    r"invalid_json_schema",
    r"BadRequestError",
    r"invalid_request_error",
    r"model.+not found",
    r"Markdown not found",
    r"Unknown paper_id",
    r"Unsupported selection",
    r"FileNotFoundError",
    r"\b401\b",
    r"\b403\b",
    r"\b404\b",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def changed_paths(root: Path) -> list[str]:
    output = git_output(root, "status", "--porcelain")
    paths: list[str] = []
    for line in output.splitlines():
        raw = line[3:].strip()
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        if raw:
            paths.append(raw)
    return paths


def path_is_under(path: str, parent: str) -> bool:
    normalized_path = Path(path).as_posix().rstrip("/")
    normalized_parent = Path(parent).as_posix().rstrip("/")
    return (
        normalized_path == normalized_parent
        or normalized_path.startswith(normalized_parent + "/")
    )


def baseline_code_drift(
    root: Path,
    *,
    baseline_tag: str,
    protected_paths: Iterable[str],
) -> list[str]:
    output = git_output(
        root,
        "diff",
        "--name-only",
        f"{baseline_tag}..HEAD",
        "--",
        *protected_paths,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def run_streaming_command(
    *,
    command: list[str],
    cwd: Path,
    log_path: Path,
    environment: dict[str, str],
) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    collected: list[str] = []

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command_text(command)}\n\n")
        log.flush()

        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            collected.append(line)

        return_code = process.wait()

    return return_code, "".join(collected)


def classify_failure(output: str) -> str:
    for pattern in FATAL_PATTERNS:
        if re.search(pattern, output, flags=re.IGNORECASE):
            return "non_retryable"
    return "retryable_or_incomplete"


def ensure_stage_record(
    paper_record: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    stages = paper_record.setdefault("stages", {})
    record = stages.setdefault(
        stage,
        {
            "status": "pending",
            "attempts": [],
        },
    )
    return record


def run_bounded_stage(
    *,
    paper_record: dict[str, Any],
    stage: str,
    command: list[str],
    max_passes: int,
    backoff_seconds: list[int],
    root: Path,
    logs_root: Path,
    environment: dict[str, str],
    save_manifest,
    dry_run: bool,
) -> bool:
    stage_record = ensure_stage_record(paper_record, stage)

    if stage_record.get("status") == "complete":
        print(f"[SKIP COMPLETE] {paper_record['paper_id']} / {stage}")
        return True

    completed_attempts = len(stage_record.get("attempts", []))

    for pass_number in range(completed_attempts + 1, max_passes + 1):
        delay = (
            backoff_seconds[pass_number - 1]
            if pass_number - 1 < len(backoff_seconds)
            else backoff_seconds[-1]
        )

        if delay > 0:
            print(
                f"[BACKOFF] {paper_record['paper_id']} / {stage}: "
                f"{delay}s before pass {pass_number}"
            )
            if not dry_run:
                time.sleep(delay)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = (
            logs_root
            / paper_record["paper_id"]
            / f"{stage}_pass_{pass_number}_{timestamp}.log"
        )

        attempt = {
            "pass": pass_number,
            "started_at_utc": now_utc(),
            "delay_seconds": delay,
            "command": command,
            "log_path": str(log_path.relative_to(root)),
            "status": "running",
        }
        stage_record["status"] = "running"
        stage_record.setdefault("attempts", []).append(attempt)
        save_manifest()

        print()
        print("=" * 72)
        print(
            f"{paper_record['paper_id']} / {stage} "
            f"pass {pass_number}/{max_passes}"
        )
        print(command_text(command))
        print("=" * 72)

        if dry_run:
            return_code = 0
            output = "[DRY RUN]\n"
        else:
            return_code, output = run_streaming_command(
                command=command,
                cwd=root,
                log_path=log_path,
                environment=environment,
            )

        attempt["finished_at_utc"] = now_utc()
        attempt["exit_code"] = return_code

        if return_code == 0:
            attempt["status"] = "complete"
            stage_record["status"] = "complete"
            stage_record["completed_at_utc"] = now_utc()
            save_manifest()
            return True

        failure_class = classify_failure(output)
        attempt["status"] = "failed"
        attempt["failure_class"] = failure_class
        stage_record["status"] = "failed"
        stage_record["last_failure_class"] = failure_class
        save_manifest()

        if failure_class == "non_retryable":
            print(
                f"[STOP] Non-retryable error detected in "
                f"{paper_record['paper_id']} / {stage}."
            )
            return False

    stage_record["status"] = "failed"
    stage_record["failure_reason"] = "outer_retry_limit_reached"
    save_manifest()
    return False


def run_single_stage(
    *,
    paper_record: dict[str, Any],
    stage: str,
    command: list[str],
    root: Path,
    logs_root: Path,
    environment: dict[str, str],
    save_manifest,
    dry_run: bool,
) -> bool:
    return run_bounded_stage(
        paper_record=paper_record,
        stage=stage,
        command=command,
        max_passes=1,
        backoff_seconds=[0],
        root=root,
        logs_root=logs_root,
        environment=environment,
        save_manifest=save_manifest,
        dry_run=dry_run,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def assert_no_manual_resolution(paper_root: Path) -> None:
    decisions_path = paper_root / "resolution" / "decisions.jsonl"
    if not decisions_path.exists():
        return

    manual = []
    for record in read_jsonl(decisions_path):
        reviewer = str(record.get("reviewer") or "").strip()
        decision = str(record.get("decision") or "unreviewed").strip()
        approved = bool(record.get("approved", False))

        automatic = reviewer == "automatic_registry_rule"
        untouched = not reviewer and decision == "unreviewed" and not approved

        if not automatic and not untouched:
            manual.append(
                {
                    "candidate_id": record.get("candidate_id"),
                    "decision": decision,
                    "approved": approved,
                    "reviewer": reviewer,
                }
            )

    if manual:
        raise RuntimeError(
            "Pre-existing manual entity-resolution decisions were found for a "
            "held-out paper. Move them aside or start from a clean held-out "
            f"paper directory. Examples: {manual[:5]!r}"
        )


def resolve_latest_run_dir(paper_root: Path) -> Path:
    pointer = read_json(paper_root / "latest_run.json")
    run_directory = pointer.get("run_directory")
    if not run_directory:
        raise RuntimeError(
            f"latest_run.json has no run_directory: {paper_root}"
        )
    return Path(str(run_directory)).expanduser().resolve()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
        }
    return {
        "path": str(path.relative_to(root)),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def preview_selection(
    *,
    paper_id: str,
    config_path: Path,
    root: Path,
    environment: dict[str, str],
    paper_record: dict[str, Any],
    save_manifest,
    dry_run: bool,
) -> bool:
    stage_record = ensure_stage_record(paper_record, "selection_preview")
    if stage_record.get("status") == "complete":
        return True

    command = [
        sys.executable,
        "-m",
        "scripts.preview_document_selection",
        "--paper-id",
        paper_id,
        "--config",
        str(config_path),
        "--json",
    ]

    # Selection preview is local and does not call an LLM, so it is executed
    # even during --dry-run. This catches bad paths and SI-selection surprises.
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
    )
    return_code = completed.returncode
    stderr = completed.stderr

    if return_code == 0:
        payload = json.loads(completed.stdout)
    else:
        payload = {}

    stage_record["command"] = command
    stage_record["finished_at_utc"] = now_utc()
    stage_record["exit_code"] = return_code

    if return_code != 0:
        stage_record["status"] = "failed"
        stage_record["error"] = stderr[-5000:]
        save_manifest()
        return False

    warnings: list[str] = []
    for document in payload.get("documents", []):
        if (
            document.get("role") == "supporting_information"
            and document.get("selection_mode") == "referenced_blocks"
            and int(document.get("source_count", 0)) == 0
        ):
            warnings.append(
                f"{document.get('document_id')}: configured SI produced "
                "zero referenced source blocks"
            )

    stage_record["status"] = "complete"
    stage_record["preview"] = payload
    stage_record["warnings"] = warnings
    save_manifest()

    for warning in warnings:
        print(f"[SELECTION WARNING] {paper_id}: {warning}")

    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen GraphAgentsDAC held-out pipeline with bounded "
            "resume retries and a persistent campaign manifest."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/heldout/bridge_v2_3_3_protocol.json"),
    )
    parser.add_argument(
        "--campaign-id",
        required=True,
    )
    parser.add_argument(
        "--paper",
        action="append",
        default=[],
        help="Run one paper ID. Repeat for multiple papers.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all papers listed in the protocol.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--stop-on-paper-failure",
        action="store_true",
    )
    parser.add_argument(
        "--adopt-existing-cache",
        action="store_true",
        help=(
            "Allow the campaign to reuse a pre-existing held-out paper run. "
            "The manifest records that the cache predates this campaign."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.project_root.expanduser().resolve()
    protocol_path = (
        args.protocol
        if args.protocol.is_absolute()
        else root / args.protocol
    ).resolve()
    protocol = read_json(protocol_path)

    if not args.all and not args.paper:
        raise ValueError("Pass --all or at least one --paper.")

    protocol_papers = [
        str(value)
        for value in protocol.get("papers", [])
    ]
    papers = protocol_papers if args.all else args.paper

    unknown = sorted(set(papers) - set(protocol_papers))
    if unknown:
        raise ValueError(
            f"Papers are not listed in the held-out protocol: {unknown!r}"
        )

    baseline_tag = str(protocol["baseline_tag"])
    baseline_commit = git_output(
        root,
        "rev-list",
        "-n",
        "1",
        baseline_tag,
    )
    head_commit = git_output(root, "rev-parse", "HEAD")

    ancestor_check = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            baseline_tag,
            "HEAD",
        ],
        cwd=root,
    )
    if ancestor_check.returncode != 0:
        raise RuntimeError(
            f"Baseline tag {baseline_tag!r} is not an ancestor of HEAD."
        )

    drift = baseline_code_drift(
        root,
        baseline_tag=baseline_tag,
        protected_paths=[
            str(value)
            for value in protocol.get(
                "protected_paths",
                [],
            )
        ],
    )
    if drift:
        raise RuntimeError(
            "Frozen extraction/graph/Bridge implementation differs from the "
            f"baseline tag. Held-out execution aborted. Changed: {drift!r}"
        )

    config_path = (
        Path(str(protocol["config_path"]))
        if Path(str(protocol["config_path"])).is_absolute()
        else root / str(protocol["config_path"])
    ).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    strict_model_env = str(
        protocol["model_environment"]["strict_model"]
    )
    bridge_model_env = str(
        protocol["model_environment"]["bridge_model"]
    )
    provider_env = str(
        protocol["model_environment"]["provider"]
    )

    strict_model = os.environ.get(strict_model_env, "").strip()
    bridge_model = (
        os.environ.get(bridge_model_env, "").strip()
        or strict_model
    )
    provider = os.environ.get(provider_env, "").strip()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not defined.")
    if not strict_model:
        raise RuntimeError(f"{strict_model_env} is not defined.")
    if not bridge_model:
        raise RuntimeError(
            f"Neither {bridge_model_env} nor {strict_model_env} is defined."
        )

    campaign_root = (
        root
        / "evaluation"
        / "bridge_semantic"
        / "held_out"
        / args.campaign_id
    )
    live_root = (
        root
        / "data_dac"
        / "held_out_runs"
        / args.campaign_id
    )
    manifest_path = campaign_root / "manifest.json"
    logs_root = live_root / "logs"

    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if bool(manifest.get("dry_run", False)) != bool(args.dry_run):
            raise RuntimeError(
                "A dry-run campaign cannot be resumed as a real campaign, "
                "or vice versa. Use a different --campaign-id."
            )
        if manifest.get("protocol_sha256") != sha256_file(protocol_path):
            raise RuntimeError(
                "Protocol file changed after campaign creation."
            )
        if manifest.get("git", {}).get("head_commit") != head_commit:
            raise RuntimeError(
                "Git HEAD changed after campaign creation."
            )
        if manifest.get("config_sha256") != sha256_file(config_path):
            raise RuntimeError(
                "papers.yaml changed after campaign creation."
            )
        resolved = manifest.get("resolved_runtime", {})
        if (
            resolved.get("strict_model") != strict_model
            or resolved.get("bridge_model") != bridge_model
            or resolved.get("provider", "") != provider
        ):
            raise RuntimeError(
                "Resolved model/provider changed after campaign creation."
            )

        allowed_dirty_root = str(
            campaign_root.relative_to(root)
        )
        disallowed_dirty = [
            path
            for path in changed_paths(root)
            if not path_is_under(
                path,
                allowed_dirty_root,
            )
        ]
        if disallowed_dirty:
            raise RuntimeError(
                "Unexpected working-tree changes during campaign resume: "
                f"{disallowed_dirty!r}"
            )
    else:
        dirty = changed_paths(root)
        if dirty:
            raise RuntimeError(
                "Start a held-out campaign from a clean Git working tree. "
                f"Current changes: {dirty!r}"
            )

        manifest = {
            "campaign_id": args.campaign_id,
            "campaign_name": protocol.get(
                "campaign_name",
                args.campaign_id,
            ),
            "status": "running",
            "dry_run": bool(args.dry_run),
            "created_at_utc": now_utc(),
            "updated_at_utc": now_utc(),
            "baseline": {
                "tag": baseline_tag,
                "commit": baseline_commit,
            },
            "git": {
                "head_commit": head_commit,
                "branch": git_output(
                    root,
                    "rev-parse",
                    "--abbrev-ref",
                    "HEAD",
                ),
            },
            "protocol_path": str(
                protocol_path.relative_to(root)
            ),
            "protocol_sha256": sha256_file(
                protocol_path
            ),
            "protocol": protocol,
            "config_path": str(
                config_path.relative_to(root)
            ),
            "config_sha256": sha256_file(
                config_path
            ),
            "resolved_runtime": {
                "strict_model": strict_model,
                "bridge_model": bridge_model,
                "provider": provider,
                "strict_concurrency": protocol[
                    "operational_retry_protocol"
                ]["strict_concurrency"],
                "bridge_concurrency": protocol[
                    "operational_retry_protocol"
                ]["bridge_concurrency"],
            },
            "papers": {},
        }

    def save_manifest() -> None:
        manifest["updated_at_utc"] = now_utc()
        atomic_write_json(manifest_path, manifest)

    save_manifest()

    environment = dict(os.environ)
    retry = protocol["operational_retry_protocol"]

    for paper_id in papers:
        paper_record = manifest["papers"].setdefault(
            paper_id,
            {
                "paper_id": paper_id,
                "status": "pending",
                "started_at_utc": now_utc(),
                "stages": {},
            },
        )

        if paper_record.get("status") == "complete":
            print(f"[SKIP PAPER COMPLETE] {paper_id}")
            continue

        paper_root = (
            root
            / "data_dac"
            / "extracted"
            / paper_id
        )

        if (
            not args.adopt_existing_cache
            and not paper_record.get(
                "campaign_cache_checked",
                False,
            )
            and (
                paper_root / "latest_run.json"
            ).exists()
        ):
            raise RuntimeError(
                f"{paper_id} already has latest_run.json before its first "
                "campaign attempt. Use --adopt-existing-cache only when this "
                "pre-existing cache was produced under the same frozen setup."
            )

        if not paper_record.get(
            "campaign_cache_checked",
            False,
        ):
            paper_record[
                "preexisting_cache_adopted"
            ] = bool(
                (paper_root / "latest_run.json").exists()
            )
            paper_record[
                "campaign_cache_checked"
            ] = True
            save_manifest()

        paper_record["status"] = "running"
        save_manifest()

        try:
            if protocol["selection_preflight"].get(
                "run_preview",
                True,
            ):
                if not preview_selection(
                    paper_id=paper_id,
                    config_path=config_path,
                    root=root,
                    environment=environment,
                    paper_record=paper_record,
                    save_manifest=save_manifest,
                    dry_run=args.dry_run,
                ):
                    raise RuntimeError(
                        "selection preview failed"
                    )

            strict_command = [
                sys.executable,
                "-m",
                "scripts.extract_paper",
                "--paper-id",
                paper_id,
                "--config",
                str(config_path),
                "--model",
                strict_model,
                "--concurrency",
                str(
                    retry[
                        "strict_concurrency"
                    ]
                ),
            ]
            if provider:
                strict_command.extend(
                    ["--provider", provider]
                )

            if not run_bounded_stage(
                paper_record=paper_record,
                stage="strict_extraction",
                command=strict_command,
                max_passes=int(
                    retry[
                        "strict_max_outer_passes"
                    ]
                ),
                backoff_seconds=[
                    int(value)
                    for value in retry[
                        "strict_backoff_seconds"
                    ]
                ],
                root=root,
                logs_root=logs_root,
                environment=environment,
                save_manifest=save_manifest,
                dry_run=args.dry_run,
            ):
                raise RuntimeError(
                    "strict extraction did not complete"
                )

            if args.dry_run:
                strict_run_id = "DRY_RUN"
                strict_run_dir = paper_root / "runs" / strict_run_id
            else:
                strict_run_dir = resolve_latest_run_dir(
                    paper_root
                )
                run_json = read_json(
                    strict_run_dir / "run.json"
                )
                active_chunks = read_json(
                    strict_run_dir
                    / "active_chunks.json"
                )
                strict_summary = read_json(
                    strict_run_dir / "summary.json"
                )
                strict_run_id = str(
                    run_json["run_id"]
                )
                if not active_chunks.get(
                    "complete",
                    False,
                ):
                    raise RuntimeError(
                        "strict command returned success "
                        "but active_chunks.complete is false"
                    )

                paper_record["strict_run"] = {
                    "run_id": strict_run_id,
                    "run_fingerprint": run_json.get(
                        "run_fingerprint",
                        "",
                    ),
                    "run_directory": str(
                        strict_run_dir.relative_to(root)
                    ),
                    "summary": strict_summary,
                    "active_chunk_count": (
                        active_chunks.get(
                            "active_chunk_count",
                            len(
                                active_chunks.get(
                                    "chunks",
                                    [],
                                )
                            ),
                        )
                    ),
                    "failed_chunk_count": len(
                        active_chunks.get(
                            "failed_chunks",
                            [],
                        )
                    ),
                }
                save_manifest()

            assert_no_manual_resolution(
                paper_root
            )

            build_command = [
                sys.executable,
                "-m",
                "scripts.build_paper_graph",
                "--paper-id",
                paper_id,
                "--config",
                str(config_path),
                "--run-id",
                strict_run_id,
            ]

            if not run_single_stage(
                paper_record=paper_record,
                stage="canonical_graph_build",
                command=build_command,
                root=root,
                logs_root=logs_root,
                environment=environment,
                save_manifest=save_manifest,
                dry_run=args.dry_run,
            ):
                raise RuntimeError(
                    "canonical graph build failed"
                )

            canonical_path = (
                paper_root / f"{paper_id}.graphml"
            )

            if not args.dry_run:
                build_summary = read_json(
                    strict_run_dir
                    / "build_summary.json"
                )
                paper_record["canonical_graph"] = {
                    "build_summary": build_summary,
                    "file": file_record(
                        canonical_path,
                        root,
                    ),
                }
                save_manifest()

            audit_output_dir = (
                live_root / "audit" / paper_id
            )
            audit_command = [
                sys.executable,
                "-m",
                "scripts.inspect_graphml",
                "--graphml",
                str(canonical_path),
                "--output-dir",
                str(audit_output_dir),
            ]

            if not run_single_stage(
                paper_record=paper_record,
                stage="canonical_graph_audit",
                command=audit_command,
                root=root,
                logs_root=logs_root,
                environment=environment,
                save_manifest=save_manifest,
                dry_run=args.dry_run,
            ):
                raise RuntimeError(
                    "canonical graph audit failed"
                )

            if not args.dry_run:
                readiness_path = (
                    audit_output_dir
                    / "pilot_readiness.json"
                )
                semantic_path = (
                    audit_output_dir
                    / "semantic_readiness.json"
                )
                paper_record["audit"] = {
                    "pilot_readiness": (
                        read_json(readiness_path)
                        if readiness_path.exists()
                        else {}
                    ),
                    "semantic_readiness": (
                        read_json(semantic_path)
                        if semantic_path.exists()
                        else {}
                    ),
                    "audit_report": file_record(
                        audit_output_dir
                        / "audit_report.txt",
                        root,
                    ),
                    "note": (
                        "Audit gate values are recorded as held-out outcomes. "
                        "They do not trigger paper-specific policy tuning."
                    ),
                }
                save_manifest()

            bridge_command = [
                sys.executable,
                "-m",
                "scripts.extract_bridge_graph",
                "--paper-id",
                paper_id,
                "--config",
                str(config_path),
                "--run-id",
                strict_run_id,
                "--model",
                bridge_model,
                "--concurrency",
                str(
                    retry[
                        "bridge_concurrency"
                    ]
                ),
            ]
            if provider:
                bridge_command.extend(
                    ["--provider", provider]
                )

            if not run_bounded_stage(
                paper_record=paper_record,
                stage="bridge_extraction",
                command=bridge_command,
                max_passes=int(
                    retry[
                        "bridge_max_outer_passes"
                    ]
                ),
                backoff_seconds=[
                    int(value)
                    for value in retry[
                        "bridge_backoff_seconds"
                    ]
                ],
                root=root,
                logs_root=logs_root,
                environment=environment,
                save_manifest=save_manifest,
                dry_run=args.dry_run,
            ):
                raise RuntimeError(
                    "Bridge extraction did not complete"
                )

            bridge_path = (
                paper_root
                / f"{paper_id}.bridge.graphml"
            )
            candidate_bridge_path = (
                paper_root
                / (
                    f"{paper_id}"
                    ".bridge.candidates.graphml"
                )
            )

            if not args.dry_run:
                extraction_pointer = read_json(
                    strict_run_dir
                    / "latest_bridge_extraction.json"
                )
                policy_pointer = read_json(
                    strict_run_dir
                    / "latest_bridge_policy_run.json"
                )
                extraction_dir = Path(
                    str(
                        extraction_pointer[
                            "bridge_extraction_directory"
                        ]
                    )
                )
                policy_dir = Path(
                    str(
                        policy_pointer[
                            "bridge_policy_run_directory"
                        ]
                    )
                )
                paper_record["bridge"] = {
                    "bridge_extraction_id": (
                        extraction_pointer.get(
                            "bridge_extraction_id",
                            "",
                        )
                    ),
                    "bridge_policy_run_id": (
                        policy_pointer.get(
                            "bridge_policy_run_id",
                            "",
                        )
                    ),
                    "extraction_summary": read_json(
                        extraction_dir
                        / "summary.json"
                    ),
                    "policy_summary": read_json(
                        policy_dir
                        / "summary.json"
                    ),
                    "confirmed_graph": file_record(
                        bridge_path,
                        root,
                    ),
                    "candidate_graph": file_record(
                        candidate_bridge_path,
                        root,
                    ),
                }
                save_manifest()

            projection_records: dict[
                str,
                Any,
            ] = {}

            for mode in protocol.get(
                "projection_modes",
                [],
            ):
                projection_command = [
                    sys.executable,
                    "-m",
                    "scripts.build_graphagents_projection",
                    "--paper-id",
                    paper_id,
                    "--mode",
                    str(mode),
                    "--canonical-graphml",
                    str(canonical_path),
                ]

                if mode in {
                    "mechanism",
                    "exploratory",
                }:
                    projection_command.extend(
                        [
                            "--bridge-graphml",
                            str(bridge_path),
                        ]
                    )

                if mode == "exploratory":
                    projection_command.extend(
                        [
                            "--candidate-bridge-graphml",
                            str(
                                candidate_bridge_path
                            ),
                        ]
                    )

                stage_name = (
                    f"projection_{mode}"
                )
                if not run_single_stage(
                    paper_record=paper_record,
                    stage=stage_name,
                    command=projection_command,
                    root=root,
                    logs_root=logs_root,
                    environment=environment,
                    save_manifest=save_manifest,
                    dry_run=args.dry_run,
                ):
                    raise RuntimeError(
                        f"{mode} projection failed"
                    )

                if not args.dry_run:
                    output_dir = (
                        paper_root
                        / "graphagents"
                        / str(mode)
                    )
                    summary = read_json(
                        output_dir / "summary.json"
                    )
                    projection_records[
                        str(mode)
                    ] = {
                        "summary": summary,
                        "graph": file_record(
                            output_dir
                            / "graph.graphml",
                            root,
                        ),
                        "node_text": file_record(
                            output_dir
                            / "node_text.jsonl",
                            root,
                        ),
                        "edge_evidence": file_record(
                            output_dir
                            / "edge_evidence.jsonl",
                            root,
                        ),
                    }

            paper_record[
                "projections"
            ] = projection_records
            paper_record["status"] = "complete"
            paper_record[
                "completed_at_utc"
            ] = now_utc()
            save_manifest()
            print(
                f"[PAPER COMPLETE] {paper_id}"
            )

        except Exception as error:
            paper_record["status"] = "failed"
            paper_record["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
                "recorded_at_utc": now_utc(),
            }
            save_manifest()
            print(
                f"[PAPER FAILED] {paper_id}: {error}",
                file=sys.stderr,
            )
            if args.stop_on_paper_failure:
                break

    selected_statuses = {
        paper_id: manifest["papers"].get(
            paper_id,
            {},
        ).get("status", "missing")
        for paper_id in papers
    }
    all_statuses = {
        paper_id: manifest["papers"].get(
            paper_id,
            {},
        ).get("status", "not_started")
        for paper_id in protocol_papers
    }

    selected_complete = bool(
        selected_statuses
        and all(
            value == "complete"
            for value in selected_statuses.values()
        )
    )
    campaign_complete = bool(
        all_statuses
        and all(
            value == "complete"
            for value in all_statuses.values()
        )
    )

    manifest["status"] = (
        "complete"
        if campaign_complete
        else "incomplete"
    )
    manifest[
        "selected_paper_statuses"
    ] = selected_statuses
    manifest[
        "all_paper_statuses"
    ] = all_statuses
    manifest["finished_at_utc"] = now_utc()
    save_manifest()

    print()
    print("Selected papers complete:", selected_complete)
    print("Held-out campaign status:", manifest["status"])
    print("Manifest:", manifest_path)
    print("Operational logs:", logs_root)

    if not selected_complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
