"""M4-B2: offline, metadata-only triage of an existing M4-B1 inventory.

This ranks where a *human* should inspect first. It never parses measurements,
confirms sample linkage, verifies measurement independence or grants claim authority.
No rescan of data roots and no model or network calls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_SCHEMA = "m4b1-evidence-readiness-inventory-v1"
SCHEMA = "m4b2-evidence-metadata-triage-v1"
ACCEPTED_SOURCE_STATUS = {
    "PARTIAL_INVENTORY_REVIEW_REQUIRED",
    "INVENTORY_COMPLETE_WITHIN_DECLARED_SCOPE_REVIEW_REQUIRED",
}
DATA_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl"}
UNREADABLE_STATUSES = {"TOO_LARGE_FOR_HEADER_INSPECTION", "METADATA_ONLY_UNSUPPORTED_FORMAT"}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
METHOD_RE = re.compile(
    r"(?:^|[^a-z])(?:sfg|sum.frequency|xps|xanes|xafs|afm|tem|sem|"
    r"ellipsometr|reflectometr|stm|nexafs|xrd|second.harmonic|"
    r"spectroellipsometr|angle.resolved|pump.probe|microscop)(?:[^a-z]|$)", re.I
)
DERIVED_RE = re.compile(
    r"(?:^|[/_.-])(?:report|audit|manifest|hypothesis|metadata|summary|"
    r"bibliograph|references|prompt|cache|knowledge.graph|graph|shadow|"
    r"evaluation|assessment|selection)(?:[/_.-]|$)", re.I
)
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
    "ANCHOR": r"anchor|binding|functionaliz|chemisorp",
    "SPATIAL": r"spatial|site.?resolved|pixel|position|mapping",
}
COMPILED = {name: re.compile(pattern, re.I) for name, pattern in TAG_PATTERNS.items()}

# Case-specific query refinements, not assertions of available measurements.
# A group is an AND of tags. Groups are alternatives (OR). An independent
# measurement remains UNVERIFIED even if all tokens match.
QA_HINT_GROUPS = {
    ("QA_B12_AIR_EXPOSURE", "time-resolved relative SERS band ratios"): [("SERS", "TIME")],
    ("QA_B12_AIR_EXPOSURE", "molecular orientation independent of relative SERS band ratios"): [("ORIENTATION",)],
    ("QA_B12_AIR_EXPOSURE", "surface and local-field evolution"): [("FIELD", "TIME"), ("SURFACE", "TIME")],
    ("QA_B01_ELECTRONIC_VS_EM", "anchoring-sensitive relative SERS band ratios"): [("SERS", "ANCHOR")],
    ("QA_B01_ELECTRONIC_VS_EM", "interfacial electronic state"): [("ELECTRONIC",)],
    ("QA_B01_ELECTRONIC_VS_EM", "electromagnetic field response and accessibility"): [("FIELD",)],
    ("QA_B03_B06_PROXY_COVARIANCE", "spatial or polarization-resolved relative SERS ratios"): [("SERS", "SPATIAL"), ("SERS", "ORIENTATION")],
    ("QA_B03_B06_PROXY_COVARIANCE", "active-subpopulation-sensitive orientation"): [("ORIENTATION", "POPULATION")],
    ("QA_B03_B06_PROXY_COVARIANCE", "registered orientation-by-field covariance"): [("ORIENTATION", "FIELD")],
    ("QA_B17_DYNAMIC_VS_STATIC", "time-resolved SERS band ratios"): [("SERS", "TIME")],
    ("QA_B17_DYNAMIC_VS_STATIC", "molecular residence or orientation states"): [("TIME", "POPULATION"), ("TIME", "ORIENTATION")],
    ("QA_B17_DYNAMIC_VS_STATIC", "hotspot intermittency or near-field activity"): [("FIELD", "TIME")],
}


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError("M4B2_INTEGRITY_FAILURE: " + message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tags(text: str) -> set[str]:
    return {k for k, p in COMPILED.items() if p.search(text)}


def validate_inventory(src: dict[str, Any]) -> None:
    require(src.get("schema_version") == SOURCE_SCHEMA, "input schema")
    require(src.get("status") in ACCEPTED_SOURCE_STATUS, "input status")
    for k in ("science_truth_or_novelty_certified", "physical_feasibility_certified",
              "measurement_independence_certified", "source_files_modified",
              "automatic_research_idea_or_hypothesis_promotion", "claims_about_unscanned_data"):
        require(src.get(k) is False, f"unsafe input flag {k}")
    require(src.get("llm_or_network_calls") == 0, "source model/network call status")
    require(isinstance(src.get("m4a2_input_sha256"), str) and SHA_RE.fullmatch(src["m4a2_input_sha256"]) is not None, "M4A2 pin")
    rows = src.get("files")
    cases = src.get("cases")
    require(isinstance(rows, list) and len(rows) == src.get("file_count"), "file count mismatch")
    require(isinstance(cases, list) and bool(cases), "cases missing")
    require(isinstance(src.get("scan_truncated"), bool), "scan coverage unknown")
    seen_paths: set[str] = set()
    for r in rows:
        require(isinstance(r, dict), "malformed file metadata")
        p = r.get("path_PRIVATE")
        require(isinstance(p, str) and p and Path(p).is_absolute() and p not in seen_paths, "file identity duplicate/relative")
        seen_paths.add(p)
        require(isinstance(r.get("relative_path"), str), "missing relative path")
        require(isinstance(r.get("columns_or_keys"), list) and all(isinstance(c, str) for c in r["columns_or_keys"]), "invalid columns")
        require(isinstance(r.get("inspection_status"), str), "missing inspection status")
        require(r.get("actual_measurements_read") is False and r.get("measurement_independence_verified") is False, "input promoted measurement")
        digest = r.get("small_file_sha256")
        require(digest is None or (isinstance(digest, str) and SHA_RE.fullmatch(digest) is not None), "file digest malformed")
    seen_cases: set[str] = set()
    for case in cases:
        require(isinstance(case, dict), "malformed case")
        cid = case.get("case_id")
        require(isinstance(cid, str) and cid and cid not in seen_cases, "duplicate case")
        seen_cases.add(cid)
        require(case.get("readiness") == "UNKNOWN_PENDING_DATA_LINKAGE" and case.get("empirical_discrimination_established") is False, "case wrongly promoted")
        obs_seen: set[str] = set()
        for obs in case.get("observables", []):
            require(isinstance(obs, dict), "malformed observable")
            name = obs.get("name")
            role = obs.get("role")
            require(isinstance(name, str) and name and name not in obs_seen, "duplicate/missing observable")
            obs_seen.add(name)
            require(role in {"TARGET_OUTCOME", "INDEPENDENT_DISCRIMINATOR", "CONTROL"}, "role invalid")
            require(obs.get("independent_measurement_witness_verified") is False and obs.get("observations_validated") is False, "observable promoted")
            require(isinstance(obs.get("candidate_file_count"), int) and obs["candidate_file_count"] >= 0, "candidate count invalid")
            require(isinstance(obs.get("candidate_files_HEURISTIC"), list), "candidate list invalid")
            for m in obs["candidate_files_HEURISTIC"]:
                require(m.get("file_path_PRIVATE") in seen_paths, "candidate references unknown file")


def required_groups(case_id: str, observable: dict[str, Any]) -> list[tuple[str, ...]]:
    role = observable["role"]
    hints = QA_HINT_GROUPS.get((case_id, observable["name"]))
    if hints:
        # Refine only: ensure every result still overlaps B1's observable hints.
        return hints
    t = sorted(tags(observable["name"]) - {"SAMPLE_ID"})
    if role == "INDEPENDENT_DISCRIMINATOR":
        t = [x for x in t if x != "SERS"]
    return [tuple(t)] if len(t) >= 2 else []


def evidence_cues(file_row: dict[str, Any], *, groups: list[tuple[str, ...]], role: str) -> dict[str, Any] | None:
    cols = file_row["columns_or_keys"]
    leaf = Path(file_row["relative_path"]).name
    ft = tags(" ".join(cols))
    lt = tags(leaf)
    # The parent directory is deliberately ignored: generic corpus naming
    # otherwise matches nearly every entry in an SERS repository.
    hdr_groups = [g for g in groups if set(g).issubset(ft)]
    leaf_groups = [g for g in groups if set(g).issubset(lt)]
    if not hdr_groups and not leaf_groups:
        return None
    derived = bool(DERIVED_RE.search(file_row["relative_path"]))
    method_cue = bool(METHOD_RE.search(" ".join(cols) + " " + leaf))
    has_header = file_row["suffix"].lower() in DATA_SUFFIXES and bool(cols) and file_row["inspection_status"] in {"HEADER_SAMPLED", "JSON_KEYS_SAMPLED"}
    if has_header and hdr_groups:
        tier = "HEADER_FIELDS_REQUIRE_HUMAN_INSPECTION"
        priority = 1
    elif leaf_groups:
        tier = "FILENAME_ONLY_NOT_MEASUREMENT"
        priority = 2
    else:
        return None
    if derived:
        tier = "DERIVED_OR_REPORT_ARTIFACT_VERIFY_ORIGIN"
        priority = 3
    if role == "INDEPENDENT_DISCRIMINATOR" and not method_cue:
        method_status = "NO_INDEPENDENT_METHOD_CUE"
    elif role == "INDEPENDENT_DISCRIMINATOR":
        method_status = "METHOD_STRING_CUE_UNVERIFIED"
    else:
        method_status = "NOT_REQUIRED_FOR_TARGET_OUTCOME"
    return {
        "path_PRIVATE": file_row["path_PRIVATE"],
        "relative_path_PRIVATE": file_row["relative_path"],
        "inspection_status": file_row["inspection_status"],
        "signal_tier": tier,
        "matched_header_tags": sorted(ft.intersection(set().union(*(set(g) for g in groups)))),
        "matched_leaf_tags": sorted(lt.intersection(set().union(*(set(g) for g in groups)))),
        "header_or_keys": cols[:40],
        "method_cue": method_status,
        "small_file_sha256": file_row.get("small_file_sha256"),
        "derived_or_report_cue": derived,
        "measurement_values_inspected": False,
        "sample_time_linkage_verified": False,
        "measurement_independence_verified": False,
        "scientific_evidence_confirmed": False,
        "_priority": priority,
        "_tie": (-len(hdr_groups), -len(ft), leaf),
    }


def triage(src: dict[str, Any], *, input_sha: str, top_k: int) -> dict[str, Any]:
    require(1 <= top_k <= 20, "top_k outside 1..20")
    validate_inventory(src)
    cases = []
    summary = Counter()
    for case in src["cases"]:
        requirements = []
        for obs in case["observables"]:
            groups = required_groups(case["case_id"], obs)
            if not groups:
                shortlist: list[dict[str, Any]] = []
                status = "NO_SPECIFIC_METADATA_QUERY_SAFE_TO_USE"
                candidate_total = 0
            else:
                all_candidates = []
                for file_row in src["files"]:
                    e = evidence_cues(file_row, groups=groups, role=obs["role"])
                    if e is not None:
                        all_candidates.append(e)
                all_candidates.sort(key=lambda x: (x["_priority"], x["_tie"], x["path_PRIVATE"]))
                candidate_total = len(all_candidates)
                shortlist = [{k:v for k,v in item.items() if not k.startswith("_")} for item in all_candidates[:top_k]]
                status = "METADATA_CUES_NEED_SOURCE_REVIEW" if candidate_total else "NO_STRICT_METADATA_CUE_IN_SCANNED_FILES"
            summary[status] += 1
            requirements.append({
                "observable": obs["name"], "role": obs["role"],
                "m4b1_weak_overlap_count": obs["candidate_file_count"],
                "strict_metadata_cue_count_NOT_VERIFIED": candidate_total,
                "query_tag_groups_HEURISTIC": [list(g) for g in groups],
                "status": status,
                "top_unreviewed_files_PRIVATE": shortlist,
                "measurements_available_verified": False,
                "independence_verified": False,
            })
        cases.append({"case_id": case["case_id"], "requirements": requirements,
                      "same_sample_time_link_verified": False,
                      "case_empirically_testable_certified": False})
    return {
        "schema_version": SCHEMA,
        "status": "METADATA_TRIAGE_ONLY_REVIEW_REQUIRED",
        "source_m4b1_sha256": input_sha,
        "source_m4a2_sha256": src["m4a2_input_sha256"],
        "scanned_file_count": src["file_count"],
        "source_scan_truncated": src["scan_truncated"],
        "source_scan_warnings": src.get("scan_warnings", []),
        "cases": cases,
        "requirement_status_counts": dict(sorted(summary.items())),
        "model_or_network_calls": 0,
        "measured_values_read": False,
        "measurement_independence_certified": False,
        "empirical_truth_or_falsification_certified": False,
        "no_claims_about_unscanned_data": True,
        "source_files_modified": False,
        "no_automatic_linkage_or_promotion": True,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    source = args.m4b1_inventory.expanduser().resolve(strict=True)
    out = args.output_dir.expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    require(source.is_file(), "inventory source not file")
    require(out != repo and repo not in out.parents, "private output inside repo")
    require(not out.exists(), "output exists")
    require(source != out and out not in source.parents, "source inside output")
    sha = sha256(source)
    if args.expected_m4b1_sha256:
        require(sha == args.expected_m4b1_sha256, "pinned inventory SHA mismatch")
    src = json.loads(source.read_text(encoding="utf-8"))
    require(isinstance(src, dict), "inventory JSON root")
    result = triage(src, input_sha=sha, top_k=args.top_k)
    out.mkdir(parents=True, exist_ok=False)
    (out / "M4B2_METADATA_TRIAGE_PRIVATE.json").write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["case_id", "observable", "role", "candidate_path_PRIVATE", "candidate_field_or_key", "observed_quantity_and_units", "measurement_method_and_protocol", "sample_id_and_time_alignment", "independence_from_target_witness", "uncertainty_and_detection_limit", "source_paper_or_instrument_record", "review_outcome", "reviewer_notes"]
    with (out / "M4B2_REVIEW_TEMPLATE_PRIVATE.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in result["cases"]:
            for req in case["requirements"]:
                picks = req["top_unreviewed_files_PRIVATE"]
                # Keep an explicit blank row when no plausible metadata cue exists.
                for p in picks if picks else [{}]:
                    writer.writerow({"case_id":case["case_id"], "observable":req["observable"], "role":req["role"], "candidate_path_PRIVATE":p.get("path_PRIVATE", ""), "review_outcome":"UNREVIEWED"})
    md = ["# M4-B2 — Evidence Metadata Triage (PRIVATE)", "", "Status: `METADATA_TRIAGE_ONLY_REVIEW_REQUIRED`", f"Pinned M4-B1 SHA256: `{sha}`", f"Source files indexed: {src['file_count']}; truncated: {str(src['scan_truncated']).lower()}", "", "This report ranks **unverified metadata cues**, not measured values or independent experimental evidence. File-name-only matches are weaker than structured header matches. A negative search cannot establish absence of data outside the scanned formats/roots.", "", "| Case | Observable | M4-B1 weak matches | M4-B2 metadata cues | Review shortlist |", "|---|---|---:|---:|---:|"]
    for c in result["cases"]:
        for req in c["requirements"]:
            s = req["observable"].replace("|", "/").replace("\n", " ")
            md.append(f"| {c['case_id']} | {s} | {req['m4b1_weak_overlap_count']} | {req['strict_metadata_cue_count_NOT_VERIFIED']} | {len(req['top_unreviewed_files_PRIVATE'])} |")
    md.extend(["", "**Interpretation:** None of the shortlisted entries certifies sample/time pairing, instrument provenance, physical measurement independence, or scientific truth. Empty shortlists require a targeted manual data-format/source review, not an absence claim.", "", "**Next:** Use the CSV to record independently inspected primary measurements, methods, sample/time/condition linkage, units and uncertainty. Leave `UNREVIEWED` if these are unknown. Do not promote or reject ResearchIdeas from this tool."])
    (out / "M4B2_REPORT_PRIVATE.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m4b1-inventory", type=Path, required=True)
    ap.add_argument("--expected-m4b1-sha256", default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    try:
        data = execute(args)
    except (OSError, ValueError, json.JSONDecodeError) as ex:
        ap.exit(2, f"M4B2 STOPPED (no source modified): {ex}\n")
    print("M4B2:", data["status"])
    print("source files:", data["scanned_file_count"], "cases:", len(data["cases"]))
    print("independent measurement certified: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
