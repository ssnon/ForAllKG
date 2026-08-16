from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.provider_failure_taxonomy import (
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_OUTPUT_ROOT,
    verify_audit,
)


ROOT = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_ROOT,
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT
        / "audit.json",
    )
    args = parser.parse_args()

    diagnostic_root = (
        args.diagnostic_root.expanduser()
        .resolve()
    )
    audit_path = (
        args.audit
        if args.audit.is_absolute()
        else ROOT / args.audit
    )

    issues, value = verify_audit(
        audit_path=audit_path,
        diagnostic_root=diagnostic_root,
    )
    if issues:
        print(
            "provider-failure taxonomy verification: FAIL"
        )
        for issue in issues:
            print(" -", issue)
        print("Network searches:", 0)
        print("LLM calls:", 0)
        return 2

    baseline = value[
        "baseline_taxonomy"
    ]
    conclusion = value[
        "semantic_scholar_conclusion"
    ]
    print(
        "provider-failure taxonomy verification: PASS"
    )
    print(
        "Audit ID:",
        value["audit_id"],
    )
    print(
        "Audit SHA256:",
        value["audit_sha256"],
    )
    print(
        "Provider summary:",
        baseline["provider_summary"],
    )
    print(
        "Semantic Scholar affected hypotheses:",
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
        "Failure shape:",
        conclusion[
            "failure_shape"
        ],
    )
    print(
        "Generic provider hardening justified:",
        conclusion[
            "generic_provider_hardening_justified"
        ],
    )
    print(
        "Specific retry/auth/rate-limit policy change authorized:",
        False,
    )
    print("Network searches:", 0)
    print("LLM calls:", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
