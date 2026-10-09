"""M4-B1: bounded, read-only local evidence discovery against M4-A2 requirements.

File/column matches are NOT independent measurements, evidence, or certification.
No models, network requests, source writes, or downstream promotions occur.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_SCHEMA = "m4a2-confrontation-readiness-audit-v1"
SCHEMA = "m4b1-evidence-readiness-inventory-v1"
SOURCE_STATUS = "REVIEW_REQUIRED_NO_SCIENCE_CERTIFICATION"
DATA_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl"}
METADATA_SUFFIXES = {".parquet", ".xlsx", ".xls", ".h5", ".hdf5", ".npy", ".npz", ".mat", ".pdf", ".md", ".txt"}
IGNORE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".cache"}
TAG_PATTERNS = {
    "SERS": r"sers|raman|spectr|wavenumber|band.?ratio|peak.?intens|mode.?intens",
    "ORIENTATION": r"orient|tilt|angular|polariz|alignment|azimuth|sfg|sum.?frequency",
    "FIELD": r"near.?field|electric.?field|plasmon|enhance|hotspot|hot.?spot|electromag",
    "SURFACE": r"surface|morpholog|roughness|afm|microscop|oxide|topograph",
    "ELECTRONIC": r"electronic|charge.?transfer|charge.?state|binding.?energy|xps|xanes|oxidation",
    "TIME": r"time|exposure|timestamp|kinetic|dwell|residence|transient|intermitten",
    "THICKNESS": r"thickness|capping|cap.?layer|spacer|distance|depth",
    "POPULATION": r"subpop|occupan|molecule.?count|adsorp|density|coverage|fraction",
    "SAMPLE_ID": r"sample.?id|specimen|substrate.?id|experiment.?id|replicate",
}
COMPILED = {k: re.compile(v, re.I) for k, v in TAG_PATTERNS.items()}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("M4B1_INPUT_REJECTED: " + reason)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "M4A2 JSON root")
    require(data.get("schema_version") == SOURCE_SCHEMA, "M4A2 schema mismatch")
    require(data.get("status") == SOURCE_STATUS, "M4A2 status mismatch")
    for field in ("scientific_truth_or_novelty_certified", "physical_feasibility_certified",
                  "reviewer_template_completed", "deletion_or_production_authority", "source_files_mutated"):
        require(data.get(field) is False, f"{field} must be false")
    require(data.get("fresh_llm_or_network_calls") == 0, "source call count")
    cases = data.get("cases")
    require(isinstance(cases, list) and cases and len(cases) == data.get("case_count"), "case count")
    ids = []
    for case in cases:
        require(isinstance(case, dict) and isinstance(case.get("case_id"), str) and case["case_id"], "case identity")
        ids.append(case["case_id"])
        require(case.get("readiness") == "NOT_READY_EMPIRICAL_ADJUDICATION", "unsupported ready/validated case")
        require(case.get("scientific_truth_or_falsification_certified") is False, "source science authority")
        require(case.get("independent_measurement_certified") is False, "source measurement authority")
        exp = case.get("proposed_experiment_UNREVIEWED")
        require(isinstance(exp, dict), "missing experiment")
        obs = exp.get("observables")
        require(isinstance(obs, list) and len(obs) >= 2, "missing observables")
        names = set()
        for o in obs:
            require(isinstance(o, dict), "malformed observable")
            name = o.get("name")
            require(isinstance(name, str) and name.strip() and name.casefold() not in names, "observable name")
            names.add(name.casefold())
            require(o.get("role") in {"TARGET_OUTCOME", "INDEPENDENT_DISCRIMINATOR", "CONTROL"}, "observable role")
    require(len(ids) == len(set(ids)), "duplicate case")
    return data


def columns_for(path: Path, *, max_bytes: int) -> tuple[list[str], str]:
    """Read only a small bounded data header; no numerical measurements are ingested."""
    if path.stat().st_size > max_bytes:
        return [], "TOO_LARGE_FOR_HEADER_INSPECTION"
    ext = path.suffix.casefold()
    try:
        if ext in {".csv", ".tsv"}:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
                header = next(csv.reader(fh, delimiter="\t" if ext == ".tsv" else ","), [])
            return [str(v)[:160] for v in header[:200] if str(v).strip()], "HEADER_SAMPLED"
        if ext == ".jsonl":
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = []
                for _ in range(8):
                    line = fh.readline()
                    if not line: break
                    if line.strip(): lines.append(line)
            obj = json.loads(lines[0]) if lines else {}
        elif ext == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
        else:
            return [], "METADATA_ONLY_UNSUPPORTED_FORMAT"
        if isinstance(obj, list):
            obj = next((x for x in obj[:10] if isinstance(x, dict)), {})
        if isinstance(obj, dict):
            cols = list(obj.keys())
            for val in obj.values():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    cols.extend(val[0].keys())
                    break
            return list(dict.fromkeys(str(v)[:160] for v in cols[:200])), "JSON_KEYS_SAMPLED"
        return [], "NO_TABULAR_KEYS_OBSERVED"
    except (OSError, UnicodeError, ValueError, csv.Error, json.JSONDecodeError) as exc:
        return [], f"HEADER_READ_UNAVAILABLE_{type(exc).__name__}"


def tags(value: str) -> list[str]:
    return [k for k, p in COMPILED.items() if p.search(value)]


def walk_files(root: Path, *, max_depth: int, max_files: int) -> tuple[list[Path], bool, int]:
    files: list[Path] = []
    seen = 0
    truncated = False
    for current, dirs, names in os.walk(root, followlinks=False):
        cur = Path(current)
        depth = len(cur.relative_to(root).parts)
        dirs[:] = sorted(d for d in dirs if d not in IGNORE_DIRS and not (cur / d).is_symlink()) if depth < max_depth else []
        for name in sorted(names):
            path = cur / name
            if path.is_symlink() or path.suffix.lower() not in DATA_SUFFIXES | METADATA_SUFFIXES:
                continue
            seen += 1
            if seen > max_files:
                truncated = True
                break
            if path.is_file():
                files.append(path)
        if truncated: break
    return files, truncated, seen


def required_tags(name: str, *, role: str = "TARGET_OUTCOME") -> list[str]:
    # Heuristic discovery hints, NOT operationalization or data equivalence.
    t = set(tags(name)) - {"SAMPLE_ID"}
    # "independent of SERS" must not turn the SERS outcome into a supposed
    # source for the independent physical measurement.
    if role == "INDEPENDENT_DISCRIMINATOR" and len(t) > 1:
        t.discard("SERS")
    return sorted(t)


def collect_files(roots: list[Path], *, max_depth: int, max_files: int, max_bytes: int) -> tuple[list[dict[str, Any]], bool, list[str]]:
    seen_paths: set[str] = set()
    all_rows: list[dict[str, Any]] = []
    truncated = False
    warnings: list[str] = []
    budget = max_files
    for root in roots:
        if budget < 1:
            truncated = True
            warnings.append("FILE_BUDGET_EXHAUSTED_BEFORE_ALL_ROOTS")
            break
        paths, limit, _ = walk_files(root, max_depth=max_depth, max_files=budget)
        truncated = truncated or limit
        for path in paths:
            key = str(path.resolve())
            if key in seen_paths: continue
            seen_paths.add(key)
            try:
                stat = path.stat()
                fields, read_status = columns_for(path, max_bytes=max_bytes)
                row = {
                    "path_PRIVATE": key, "root_PRIVATE": str(root),
                    "relative_path": str(path.relative_to(root)),
                    "suffix": path.suffix.lower(), "size_bytes": stat.st_size,
                    "columns_or_keys": fields, "inspection_status": read_status,
                    "path_tags_HEURISTIC": tags(str(path.relative_to(root))),
                    "field_tags_HEURISTIC": tags(" ".join(fields)),
                    "small_file_sha256": digest(path) if stat.st_size <= max_bytes else None,
                    "actual_measurements_read": False,
                    "measurement_independence_verified": False,
                }
            except OSError as exc:
                warnings.append(f"UNREADABLE_ENTRY: {path.name}: {type(exc).__name__}")
                continue
            all_rows.append(row)
        budget = max_files - len(all_rows)
        if limit: warnings.append(f"SCAN_LIMIT_REACHED_IN_ROOT: {root}")
    return all_rows, truncated, warnings


def build_audit(source: dict[str, Any], *, source_sha: str, rows: list[dict[str, Any]], roots: list[Path], truncated: bool, warnings: list[str]) -> dict[str, Any]:
    cases = []
    for case in source["cases"]:
        observables = []
        for obs in case["proposed_experiment_UNREVIEWED"]["observables"]:
            hints = required_tags(obs["name"], role=obs["role"])
            matches: list[dict[str, Any]] = []
            for row in rows:
                found = set(row["path_tags_HEURISTIC"]) | set(row["field_tags_HEURISTIC"])
                overlap = sorted(found.intersection(hints))
                if not overlap: continue
                matches.append({
                    "file_path_PRIVATE": row["path_PRIVATE"],
                    "matched_heuristic_tags": overlap,
                    "matched_field_tags": sorted(set(row["field_tags_HEURISTIC"]).intersection(hints)),
                    "inspection_status": row["inspection_status"],
                    "measurement_status": "UNREVIEWED_POSSIBLE_MATCH_ONLY",
                })
            matches.sort(key=lambda v: (-len(v["matched_field_tags"]), -len(v["matched_heuristic_tags"]), v["file_path_PRIVATE"]))
            observables.append({
                "name": obs["name"], "role": obs["role"],
                "source_independence_assertion": obs["independent_of_sers_outcome"],
                "search_tags_HEURISTIC": hints,
                "candidate_file_count": len(matches),
                "candidate_files_HEURISTIC": matches[:12],
                "availability_status": "CANDIDATE_FILES_REQUIRE_MANUAL_LINKING" if matches else "NOT_LOCATED_IN_SCANNED_SCOPE",
                "independent_measurement_witness_verified": False,
                "observations_validated": False,
            })
        cases.append({
            "case_id": case["case_id"], "hypothesis_refs": case.get("hypothesis_refs", []),
            "readiness": "UNKNOWN_PENDING_DATA_LINKAGE",
            "observables": observables,
            "empirical_discrimination_established": False,
        })
    return {
        "schema_version": SCHEMA,
        "status": "PARTIAL_INVENTORY_REVIEW_REQUIRED" if truncated or warnings else "INVENTORY_COMPLETE_WITHIN_DECLARED_SCOPE_REVIEW_REQUIRED",
        "m4a2_input_sha256": source_sha,
        "data_roots_PRIVATE": [str(x) for x in roots],
        "file_count": len(rows), "file_type_counts": dict(Counter(x["suffix"] for x in rows)),
        "scan_truncated": truncated, "scan_warnings": warnings,
        "files": rows, "cases": cases,
        "claims_about_unscanned_data": False, "science_truth_or_novelty_certified": False,
        "physical_feasibility_certified": False, "measurement_independence_certified": False,
        "source_files_modified": False, "llm_or_network_calls": 0,
        "automatic_research_idea_or_hypothesis_promotion": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    in_path = args.m4a2_report.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    script_repo = Path(__file__).resolve().parents[2]
    require(output != script_repo and script_repo not in output.parents, "private output must be outside repo")
    require(not output.exists(), "output directory must not already exist")
    require(args.max_depth >= 0 and args.max_files >= 1 and args.max_header_bytes >= 1, "positive scan budgets")
    roots = [p.expanduser().resolve(strict=True) for p in args.data_root]
    require(roots and len(roots) == len(set(roots)), "missing/duplicate scan roots")
    for root in roots:
        require(root.is_dir(), f"not a directory: {root}")
        require(root != script_repo and script_repo not in root.parents, "do not scan code repo as data")
        require(output != root and root not in output.parents, "output must be outside scan roots")
    require(in_path.is_file(), "M4A2 input not a file")
    source_sha = digest(in_path)
    if args.expected_m4a2_sha256:
        require(source_sha == args.expected_m4a2_sha256, "M4A2 digest mismatch")
    source = load_source(in_path)
    rows, truncated, warnings = collect_files(roots, max_depth=args.max_depth, max_files=args.max_files, max_bytes=args.max_header_bytes)
    result = build_audit(source, source_sha=source_sha, rows=rows, roots=roots, truncated=truncated, warnings=warnings)
    output.mkdir(parents=True, exist_ok=False)
    (output / "M4B1_INVENTORY_PRIVATE.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "M4B1_EVIDENCE_LINKAGE_TEMPLATE_PRIVATE.csv").open("w", encoding="utf-8", newline="") as fh:
        cols = ["case_id", "observable", "role", "linked_source_file_PRIVATE", "linked_field_or_locator", "measurement_method", "same_sample_or_condition", "independent_of_target_witness", "measurement_uncertainty", "source_validation_status", "reviewer_notes"]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for case in result["cases"]:
            for obs in case["observables"]:
                writer.writerow({"case_id": case["case_id"], "observable": obs["name"], "role": obs["role"], "source_validation_status": "UNREVIEWED"})
    md = ["# M4-B1 — Scoped Evidence Inventory (PRIVATE)", "", f"Status: `{result['status']}`", f"M4-A2 SHA256: `{source_sha}`", f"Files indexed: {len(rows)}; scan truncated: {str(truncated).lower()}", "", "File names and column-key overlaps are NOT evidence or independent measurement certification.", "Data outside scanned roots/depth/budget has not been assessed.", "", "| Case | Observable | Candidate files (unreviewed) |", "|---|---|---:|"]
    for case in result["cases"]:
        for obs in case["observables"]:
            safe = obs["name"].replace("|", "/").replace("\n", " ")
            md.append(f"| {case['case_id']} | {safe} | {obs['candidate_file_count']} |")
    md.extend(["", "Next: manually establish exact field/sample/time linkage, measurement independence, units, provenance, and uncertainty. Do not use this inventory to accept/reject mechanisms."])
    (output / "M4B1_REPORT_PRIVATE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m4a2-report", required=True, type=Path)
    ap.add_argument("--expected-m4a2-sha256", default=None)
    ap.add_argument("--data-root", action="append", type=Path, required=True, help="Repeat for each explicit local data root")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--max-files", type=int, default=3000)
    ap.add_argument("--max-header-bytes", type=int, default=2_000_000)
    args = ap.parse_args()
    try:
        result = run(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        ap.exit(2, f"M4B1 STOPPED (source unmodified): {exc}\n")
    print("M4B1:", result["status"])
    print("files:", result["file_count"], "cases:", len(result["cases"]))
    print("measurement independently verified: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
