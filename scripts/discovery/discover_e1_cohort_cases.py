from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def discover(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in root.rglob("adaptive_yield.case_synthesis.json"):
        try:
            payload = load(path)
        except Exception:
            continue
        rows.append(
            {
                "case_id": str(payload.get("case_id") or ""),
                "path": str(path),
                "candidate_count": len(payload.get("candidates", []) or []),
                "mtime_ns": path.stat().st_mtime_ns,
            }
        )
    return sorted(rows, key=lambda x: (x["case_id"], x["path"]))


def resolve_roles(
    rows: list[dict[str, Any]],
    role_tokens: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    resolved = {}
    for role, tokens in role_tokens.items():
        candidates = []
        for row in rows:
            hay = (row["case_id"] + " " + row["path"]).lower()
            if any(token.lower() in hay for token in tokens):
                candidates.append(row)
        if not candidates:
            raise RuntimeError(
                f"no E1 synthesis matched role={role!r}; "
                f"tokens={tokens!r}"
            )
        resolved[role] = max(candidates, key=lambda x: x["mtime_ns"])
    return resolved


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--role-token",
        action="append",
        default=[],
        help="ROLE=TOKEN1,TOKEN2",
    )
    return p


def main() -> int:
    args = parser().parse_args()
    root = args.eval_root.expanduser().resolve()
    rows = discover(root)

    role_tokens = {
        "q1": ["q1", "hotspot_geometry"],
        "q2": ["q2", "tradeoff", "reproducibility"],
        "q3": ["q3", "excitation_reporter_matching"],
    }
    for spec in args.role_token:
        if "=" not in spec:
            raise SystemExit(f"invalid --role-token: {spec!r}")
        role, raw = spec.split("=", 1)
        tokens = [x.strip() for x in raw.split(",") if x.strip()]
        if not role.strip() or not tokens:
            raise SystemExit(f"invalid --role-token: {spec!r}")
        role_tokens[role.strip()] = tokens

    try:
        resolved = resolve_roles(rows, role_tokens)
        status = "COMPLETE"
        error = None
    except RuntimeError as exc:
        resolved = {}
        status = "INCOMPLETE_ROLE_RESOLUTION"
        error = str(exc)

    body = {
        "schema_version": "e1-cohort-case-discovery-v1",
        "status": status,
        "eval_root": str(root),
        "available_case_count": len(rows),
        "available_cases": rows,
        "role_tokens": role_tokens,
        "resolved_roles": resolved,
        "error": error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("available E1 syntheses:")
    for row in rows:
        print(
            " -",
            row["case_id"] or "<missing-case-id>",
            "| candidates=", row["candidate_count"],
            "|", row["path"],
        )

    print()
    if status != "COMPLETE":
        print("ROLE RESOLUTION FAILED:", error)
        print("discovery artifact:", args.output)
        return 2

    print("resolved cohort:")
    for role in sorted(resolved):
        row = resolved[role]
        print(role, "=>", row["case_id"], "|", row["path"])
    print("discovery artifact:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
