from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LiveReframingPaths:
    run_dir: Path
    packet: Path
    explorer_report: Path
    context: Path
    readiness: Path
    evidence: Path
    tensions: Path
    triggers: Path
    reframe_shadow: Path
    mode_contrast: Path
    reasoning_portfolio: Path
    live_manifest: Path


def resolve_live_reframing_paths(run_dir: str | Path) -> LiveReframingPaths:
    run = Path(run_dir).expanduser().resolve()
    return LiveReframingPaths(
        run_dir=run,
        packet=run / "explorer.packet.json",
        explorer_report=run / "explorer.report.json",
        context=run / "hypothesis.context.json",
        readiness=run / "reframing_operator_readiness.live.json",
        evidence=run / "scientific_reframing_evidence.live.json",
        tensions=run / "scientific_reframing_evidence_tensions.live.json",
        triggers=run / "scientific_reframing_triggers.live.json",
        reframe_shadow=run / "scientific_reframing_shadow.live.json",
        mode_contrast=run / "scientific_reframing_mode_contrast.live.json",
        reasoning_portfolio=run / "scientific_reframing_reasoning_portfolio.live.json",
        live_manifest=run / "scientific_live_reframing_companion_manifest.json",
    )


def root_args(roots: list[str | Path]) -> list[str]:
    if not roots:
        raise ValueError("at least one semantic extraction root is required")
    argv: list[str] = []
    for root in roots:
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"semantic extraction root does not exist: {path}")
        argv += ["--root", str(path)]
    return argv


def common_scope_args(
    *,
    roots: list[str | Path],
    duplicate_paper_policy: str,
    cross_root_duplicate_policy: str,
) -> list[str]:
    return [
        *root_args(roots),
        "--duplicate-paper-policy",
        duplicate_paper_policy,
        "--cross-root-duplicate-policy",
        cross_root_duplicate_policy,
    ]


def extract_run_dir_from_e2e_args(argv: list[str]) -> Path:
    for index, value in enumerate(argv):
        if value == "--run-dir":
            if index + 1 >= len(argv):
                raise ValueError("--run-dir requires a value")
            return Path(argv[index + 1]).expanduser().resolve()
        if value.startswith("--run-dir="):
            return Path(value.split("=", 1)[1]).expanduser().resolve()
    raise ValueError("legacy E2E arguments must include --run-dir")


def strip_remainder_separator(argv: list[str]) -> list[str]:
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


__all__ = [
    "LiveReframingPaths",
    "common_scope_args",
    "extract_run_dir_from_e2e_args",
    "resolve_live_reframing_paths",
    "root_args",
    "strip_remainder_separator",
]
