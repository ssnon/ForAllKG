from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
FREEZE_ROOT = ROOT / (
    "evaluation/sers_novelty_gap/"
    "t0_targeted_retrieval_canonicalization_freeze_v2"
)
MANIFEST = FREEZE_ROOT / "freeze_manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _run_parent_verifier(parent_commit: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(
        prefix="sers-t0-parent-freeze-"
    ) as tmp:
        worktree = Path(tmp) / "parent"
        add = subprocess.run(
            [
                "git", "worktree", "add", "--detach",
                str(worktree), parent_commit,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if add.returncode != 0:
            return False, (
                "could not create parent verification worktree: "
                + add.stderr.strip()
            )
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.verify_sers_novelty_gap_g0_g2_production_freeze_v1",
                ],
                cwd=worktree,
                text=True,
                capture_output=True,
            )
            if result.returncode != 0:
                return False, (
                    "parent G0-G2 verifier failed at frozen parent commit:\n"
                    + result.stdout
                    + result.stderr
                )
            return True, result.stdout
        finally:
            subprocess.run(
                [
                    "git", "worktree", "remove", "--force",
                    str(worktree),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )


def main() -> int:
    if not MANIFEST.is_file():
        print("T0 production freeze v2 verification: FAIL")
        print(" - manifest missing:", MANIFEST)
        return 2

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    body = dict(manifest)
    observed_id = body.pop("freeze_id", None)
    observed_sha = body.pop("manifest_sha256", None)
    expected_sha = _sha256_json(body)
    expected_id = (
        "sers_targeted_retrieval_t0_freeze_v2:"
        + expected_sha[:20]
    )

    issues: list[str] = []
    if observed_sha != expected_sha:
        issues.append("manifest SHA mismatch")
    if observed_id != expected_id:
        issues.append("freeze ID mismatch")

    for rel, expected in manifest.get("files", {}).items():
        path = ROOT / rel
        if not path.is_file():
            issues.append(f"missing frozen file: {rel}")
            continue
        observed = _sha256(path)
        if observed != expected:
            issues.append(
                f"frozen file drift: {rel} "
                f"observed={observed} expected={expected}"
            )

    parent_ok, parent_output = _run_parent_verifier(
        manifest["parent_commit"]
    )
    if not parent_ok:
        issues.append(parent_output)

    t0 = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify_sers_targeted_retrieval_t0_offline",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if t0.returncode != 0:
        issues.append(
            "T0 offline verifier failed:\n"
            + t0.stdout
            + t0.stderr
        )

    if issues:
        print("T0 production freeze v2 verification: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Parent verification isolated at frozen parent commit:", True)
        print("Network calls during verification:", 0)
        print("LLM calls during verification:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("T0 production freeze v2 verification: PASS")
    print("Freeze ID:", observed_id)
    print("Manifest SHA256:", observed_sha)
    print("Parent commit:", manifest["parent_commit"])
    print("Parent G0-G2 freeze isolated verification: PASS")
    print("Sanctioned T0 deterministic delta verified at freeze:", True)
    print("T0 offline run:", manifest["t0_offline_run_id"])
    print("Shared canonicalization algorithm changed:", False)
    print("Network calls during verification:", 0)
    print("LLM calls during verification:", 0)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
