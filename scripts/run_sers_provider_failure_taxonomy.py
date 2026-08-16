from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from dac_her.provider_failure_taxonomy import (
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_OUTPUT_ROOT,
    EXPECTED_CLEAN_BRANCH,
    EXPECTED_CLEAN_HEAD,
    EXPECTED_DIAGNOSTIC_BRANCH,
    EXPECTED_DIAGNOSTIC_COMMIT,
    atomic_json,
    atomic_text,
    build_audit,
    render_markdown,
    verify_audit,
)


ROOT = Path.cwd()


def _git(
    repo: Path,
    *args: str,
) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only taxonomy of historical literature-provider failures. "
            "Reads diagnostic artifacts from the preserved sibling worktree."
        )
    )
    mode = parser.add_mutually_exclusive_group(
        required=True
    )
    mode.add_argument(
        "--preflight",
        action="store_true",
    )
    mode.add_argument(
        "--run",
        action="store_true",
    )
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    branch = _git(
        ROOT,
        "branch",
        "--show-current",
    )
    head = _git(
        ROOT,
        "rev-parse",
        "HEAD",
    )
    if branch != EXPECTED_CLEAN_BRANCH:
        raise SystemExit(
            f"Expected clean branch {EXPECTED_CLEAN_BRANCH!r}, "
            f"observed {branch!r}"
        )
    if head != EXPECTED_CLEAN_HEAD:
        raise SystemExit(
            f"Expected clean HEAD {EXPECTED_CLEAN_HEAD}, observed {head}"
        )

    diagnostic_root = (
        args.diagnostic_root.expanduser()
        .resolve()
    )
    if not (
        diagnostic_root / ".git"
    ).exists():
        # In a linked worktree .git is a file.
        if not (
            diagnostic_root / ".git"
        ).is_file():
            raise SystemExit(
                f"Diagnostic worktree not found: {diagnostic_root}"
            )

    diag_branch = _git(
        diagnostic_root,
        "branch",
        "--show-current",
    )
    diag_head = _git(
        diagnostic_root,
        "rev-parse",
        "--short=7",
        "HEAD",
    )
    if diag_branch != EXPECTED_DIAGNOSTIC_BRANCH:
        raise SystemExit(
            "Unexpected diagnostic branch: "
            f"{diag_branch!r}"
        )
    if diag_head != EXPECTED_DIAGNOSTIC_COMMIT:
        raise SystemExit(
            "Unexpected diagnostic commit: "
            f"{diag_head!r}"
        )

    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else ROOT / args.output_root
    )
    if output_root.exists():
        print(
            "provider-failure taxonomy preflight: FAIL"
        )
        print(
            " - output root already exists:",
            output_root,
        )
        print("Write performed: False")
        return 2

    try:
        prospective = build_audit(
            diagnostic_root=
                diagnostic_root
        )
    except Exception as exc:
        print(
            "provider-failure taxonomy preflight: FAIL"
        )
        print(
            " -",
            f"{type(exc).__name__}: {exc}",
        )
        print("Write performed: False")
        return 2

    baseline = prospective[
        "baseline_taxonomy"
    ]
    conclusion = prospective[
        "semantic_scholar_conclusion"
    ]

    print(
        "SERS Provider Failure Taxonomy"
    )
    print(
        "Prospective ID:",
        prospective["audit_id"],
    )
    print(
        "Prospective SHA256:",
        prospective["audit_sha256"],
    )
    print(
        "Baseline providers:",
        baseline["provider_summary"],
    )
    print(
        "Semantic Scholar failure category counts:",
        baseline[
            "provider_summary"
        ].get(
            "semantic_scholar",
            {},
        ).get(
            "failure_categories",
            {},
        ),
    )
    print(
        "Semantic Scholar exception types:",
        baseline[
            "provider_summary"
        ].get(
            "semantic_scholar",
            {},
        ).get(
            "exception_types",
            {},
        ),
    )
    print(
        "Affected hypotheses:",
        baseline[
            "provider_failure_recurrence"
        ].get(
            "semantic_scholar",
            {},
        ).get(
            "affected_hypothesis_count",
            0,
        ),
    )
    print(
        "Dominant failure category:",
        conclusion[
            "dominant_failure_category"
        ],
    )
    print(
        "Dominant failure fraction:",
        conclusion[
            "dominant_failure_fraction"
        ],
    )
    print(
        "Generic provider hardening justified:",
        conclusion[
            "generic_provider_hardening_justified"
        ],
    )
    print(
        "Specific policy change authorized:",
        False,
    )
    print("Network searches:", 0)
    print("LLM calls:", 0)

    if args.preflight:
        print("Preflight: PASS")
        print("Write performed: False")
        return 0

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )
    audit_path = (
        output_root / "audit.json"
    )
    report_path = (
        output_root / "audit.md"
    )
    pass_path = (
        output_root / "AUDIT_PASS.json"
    )
    try:
        audit = build_audit(
            diagnostic_root=
                diagnostic_root
        )
        atomic_json(
            audit_path,
            audit,
        )
        atomic_text(
            report_path,
            render_markdown(
                audit
            ),
        )
        issues, verified = (
            verify_audit(
                audit_path=audit_path,
                diagnostic_root=
                    diagnostic_root,
            )
        )
        if issues:
            raise RuntimeError(
                "post-write taxonomy verification failed:\n- "
                + "\n- ".join(issues)
            )

        atomic_json(
            pass_path,
            {
                "status":
                    "audit_pass",
                "audit_id":
                    verified["audit_id"],
                "audit_sha256":
                    verified[
                        "audit_sha256"
                    ],
                "semantic_scholar_dominant_failure_category":
                    verified[
                        "semantic_scholar_conclusion"
                    ][
                        "dominant_failure_category"
                    ],
                "generic_provider_hardening_justified":
                    verified[
                        "semantic_scholar_conclusion"
                    ][
                        "generic_provider_hardening_justified"
                    ],
                "specific_policy_change_authorized":
                    False,
                "network_searches": 0,
                "llm_calls": 0,
            },
        )
    except Exception:
        for path in (
            pass_path,
            report_path,
            audit_path,
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            output_root.rmdir()
        except OSError:
            pass
        raise

    print()
    print(
        "provider-failure taxonomy audit: PASS"
    )
    print(
        "Audit ID:",
        audit["audit_id"],
    )
    print(
        "Dominant failure category:",
        audit[
            "semantic_scholar_conclusion"
        ][
            "dominant_failure_category"
        ],
    )
    print(
        "Dominant failure fraction:",
        audit[
            "semantic_scholar_conclusion"
        ][
            "dominant_failure_fraction"
        ],
    )
    print(
        "Conditional probe failure categories:",
        audit[
            "conditional_probe_context"
        ][
            "failure_category_counts"
        ],
    )
    print(
        "Retrieval behavior modified:",
        False,
    )
    print("Network searches:", 0)
    print("LLM calls:", 0)
    print("Report:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
