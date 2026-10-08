"""M2-A: offline Stage7 -> P0 lineage attribution, observational only.

Consumes the M1 frozen evidence without importing production code or invoking APIs.
A selected source is not evidence of a counterfactual scientific-quality gain.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

NEEDED = ("frontier", "stage7_idea_evolution", "stage7_candidate_pool", "stage7_selection", "frozen_p0_execution")
MODES = ("KG_AXIS", "OPEN_WORLD_AXIS", "HIGHER_ORDER", "DIRECT_HIGHER_ORDER", "TENSION_DERIVED")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def rows(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    values = data.get(key)
    if not isinstance(values, list) or not all(isinstance(x, dict) for x in values):
        raise ValueError(f"expected list of objects in field {key}")
    return values


def unique_index(values: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for value in values:
        k = str(value.get(key) or "")
        if not k or k in output:
            raise ValueError(f"missing/duplicate {key}: {k}")
        output[k] = value
    return output


def list_strings(value: Any) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


def lineage_kinds(frontier: dict[str, Any]) -> list[str]:
    kinds = {str(x.get("source_kind")) for x in frontier.get("source_lineage", []) if isinstance(x, dict)}
    if not kinds:
        kinds = {str(frontier.get("source_kind") or "UNKNOWN")}
    return sorted(kinds)


def resolve_files(freeze_dir: Path, manifest: dict[str, Any], case_dir: Path | None) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    entries = {x["key"]: x for x in manifest.get("artifacts", [])}
    found = {}
    proofs = []
    for key in NEEDED:
        if key not in entries:
            raise ValueError(f"M1 manifest missing lineage artifact key: {key}")
        entry = entries[key]
        candidates = [freeze_dir / str(entry["frozen_relpath"])]
        if case_dir is not None:
            candidates.append(case_dir / str(entry["source_relpath"]))
        src = next((p for p in candidates if p.is_file()), None)
        if src is None:
            raise FileNotFoundError(f"Missing {key}. Looked in: {[str(p) for p in candidates]}")
        actual = sha256(src)
        expected = entry.get("source_sha256")
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {key}: {actual} != {expected} ({src})")
        found[key] = src
        proofs.append({"key": key, "path": str(src), "sha256": actual, "manifest_match": True})
    return found, proofs


def analyze(frozen: dict[str, dict[str, Any]], *, graph: dict[str, Any] | None = None) -> dict[str, Any]:
    frontier, evo, pool, selection, p0 = (frozen[key] for key in NEEDED)
    frows, erows, crows, srows = (
        rows(frontier, "ideas"), rows(evo, "ideas"), rows(pool, "candidates"), rows(selection, "entries")
    )
    p0_nodes = rows(p0, "g4_population_nodes")
    f_by = unique_index(frows, "idea_id")
    e_by = unique_index(erows, "evolution_id")
    c_by = unique_index(crows, "candidate_id")
    s_by = unique_index(srows, "candidate_id")
    p0_by_source = unique_index(p0_nodes, "source_object_id")
    if len(p0_nodes) != len(srows):
        raise ValueError("P0 population size != stage7 selected candidate count")
    if set(selection.get("retained_candidate_ids", [])) != set(s_by):
        raise ValueError("selection entries do not match retained candidate IDs")
    if len(set(selection.get("retained_candidate_ids", []))) != len(srows):
        raise ValueError("retained candidate IDs are not unique")
    if pool.get("source_population_id") != frontier.get("population_id"):
        raise ValueError("candidate pool / frontier ID mismatch")
    if pool.get("source_evolution_report_id") != evo.get("report_id"):
        raise ValueError("candidate pool / evolution report ID mismatch")
    if evo.get("source_population_id") != frontier.get("population_id"):
        raise ValueError("evolution report / frontier ID mismatch")
    if selection.get("source_pool_id") != pool.get("pool_id"):
        raise ValueError("selection / candidate pool ID mismatch")
    if str(p0.get("source_plan_id")) != str(selection.get("selection_id")):
        raise ValueError("P0 execution is not linked to frozen Stage7 selection")

    selected = []
    covered = set()
    origin_counts = Counter()
    ancestry_counts = Counter()
    source_kind_counts = Counter()
    evolution_ops = Counter()
    for choice in srows:
        cid = str(choice["candidate_id"])
        cand = c_by.get(cid)
        if cand is None:
            raise ValueError(f"selection references missing pool candidate: {cid}")
        if str(choice.get("source_object_id")) != str(cand.get("source_object_id")):
            raise ValueError(f"selection/pool source-object mismatch: {cid}")
        source_id = str(cand["source_object_id"])
        node = p0_by_source.get(source_id)
        if node is None:
            raise ValueError(f"selected source not represented in P0: {source_id}")
        origin = str(cand.get("origin"))
        if choice.get("origin") != origin:
            raise ValueError(f"selection/pool origin mismatch for {cid}")
        if (origin == "FRONTIER" and node.get("origin_kind") != "FRONTIER") or (
            origin == "EVOLUTION" and node.get("origin_kind") not in {"EVOLUTION", "IMPORTED_REFRAME"}
        ):
            raise ValueError(f"P0 node origin mismatch: {source_id}")
        if source_id in covered:
            raise ValueError(f"selected source duplicated: {source_id}")
        covered.add(source_id)
        kinds = set()
        parent_frontier = []
        upstream_lineage = []
        op = None
        if origin == "FRONTIER":
            f = f_by.get(source_id)
            if f is None:
                raise ValueError(f"frontier source absent: {source_id}")
            kinds.update(lineage_kinds(f))
            upstream_lineage = list(f.get("source_lineage") or [])
        elif origin == "EVOLUTION":
            e = e_by.get(source_id)
            if e is None:
                raise ValueError(f"evolution source absent: {source_id}")
            op = str(e.get("operator_id") or "UNKNOWN")
            evolution_ops[op] += 1
            parent_frontier = list_strings(e.get("parent_idea_ids"))
            for fid in parent_frontier:
                parent = f_by.get(fid)
                if parent is None:
                    raise ValueError(f"evolution parent missing from frozen frontier: {fid}")
                kinds.update(lineage_kinds(parent))
            for ref in e.get("lineage_refs") or []:
                if not isinstance(ref, dict):
                    raise ValueError(f"invalid evolution lineage ref for {source_id}")
                upstream_lineage.append(ref)
                if ref.get("lineage_kind") == "FRONTIER_IDEA":
                    if ref.get("source_object_id") not in f_by:
                        raise ValueError(f"evolution lineage ref missing frontier source: {source_id}")
            if not parent_frontier:
                kinds.update(list_strings(e.get("parent_source_kinds")))
            if not kinds:
                kinds.add("IMPORTED_REFRAME_OR_UNRESOLVED")
        else:
            raise ValueError(f"unexpected candidate origin: {origin}")
        kinds = sorted(kinds)
        if origin == "FRONTIER":
            role = "FRONTIER_DIRECT"
        elif len(kinds) > 1:
            role = "EVOLUTION_CROSS_SOURCE"
        elif parent_frontier:
            role = "EVOLUTION_FROM_FRONTIER"
        else:
            role = "EVOLUTION_IMPORTED_OR_NO_FRONTIER_PARENT"
        origin_counts[origin] += 1
        ancestry_counts[role] += 1
        source_kind_counts.update(kinds)
        selected.append({
            "idea_id": node["idea_id"], "candidate_id": cid, "source_object_id": source_id,
            "origin": origin, "idea_origin_kind": node.get("origin_kind"), "operator_id": op,
            "source_kinds": kinds, "attribution_role": role, "source_frontier_parent_ids": parent_frontier,
            "external_literature_lineage": bool(cand.get("external_literature_lineage")),
            "candidate_or_unverified_lineage": bool(cand.get("candidate_or_unverified_lineage")),
            "cross_source_composition": bool(cand.get("cross_source_composition")),
            "scientific_intent": (node.get("kernel") or {}).get("canonical_intent", ""),
            "conceptual_family_signature": cand.get("conceptual_family_signature"),
            "selection_profile": choice.get("assigned_profile"),
            "upstream_lineage": upstream_lineage,
        })
    if set(p0_by_source) != covered:
        raise ValueError("P0 contains a source object absent from selection")

    pool_origin = Counter(str(x.get("origin")) for x in crows)
    raw_kind = Counter(str(x.get("source_kind")) for x in frows)
    pool_sources = {(str(x.get("origin")), str(x.get("source_object_id"))) for x in crows}
    omitted_raw = {
        "FRONTIER": sorted(set(f_by)-{source for origin,source in pool_sources if origin == "FRONTIER"}),
        "EVOLUTION": sorted(set(e_by)-{source for origin,source in pool_sources if origin == "EVOLUTION"}),
    }
    not_selected = {
        "FRONTIER": sorted({str(x["source_object_id"]) for x in crows if x.get("origin") == "FRONTIER" and x["candidate_id"] not in s_by}),
        "EVOLUTION": sorted({str(x["source_object_id"]) for x in crows if x.get("origin") == "EVOLUTION" and x["candidate_id"] not in s_by}),
    }
    direct_p0 = sum(x["origin"] == "FRONTIER" for x in selected)
    evo_p0 = len(selected)-direct_p0
    findings = [
        "Observed Stage7 selection attribution only; no ablation, no novelty or quality causality inferred.",
        "Without Stage7 Evolution, exactly the selected evolution-origin P0 IDs lose their direct source; this is NOT a rerun of selection.",
        "Frontier origin may itself contain external/HO inspiration; origin=FRONTIER is not equivalent to KG-only.",
        "No safe-to-delete conclusion follows from a static dependency graph or provenance count.",
    ]
    graph_info = {}
    if graph:
        graph_info = {
            "classification_counts": graph.get("classification_counts", {}),
            "known_dynamic_site_count": len(graph.get("dynamic_sites") or {}),
            "stage7_dynamic_runner_site_count": len((graph.get("dynamic_sites") or {}).get("scripts.discovery.run_dac_discovery_e2e", [])),
            "static_reachability_is_not_actual_execution": graph.get("static_reachability_is_not_actual_execution"),
        }
    return {
        "schema_version": "ai-scientist-m2a-p0-scientific-lineage-audit-v1",
        "status": "ATTRIBUTION_COMPLETE",
        "counts": {
            "raw_frontier_count": len(frows), "raw_frontier_by_source_kind": dict(sorted(raw_kind.items())),
            "raw_evolution_count": len(erows), "raw_evolution_by_operator": dict(sorted(Counter(x.get("operator_id") for x in erows).items())),
            "projected_pool_count": len(crows), "pool_by_origin": dict(sorted(pool_origin.items())),
            "selected_p0_count": len(selected), "selected_p0_by_origin": dict(sorted(origin_counts.items())),
            "selected_p0_by_attribution_role": dict(sorted(ancestry_counts.items())),
            "selected_p0_upstream_source_kind_incidence": dict(sorted(source_kind_counts.items())),
            "selected_p0_by_evolution_operator": dict(sorted(evolution_ops.items())),
            "removed_pre_pool_by_origin": {k:len(v) for k,v in omitted_raw.items()},
            "not_selected_from_pool_by_origin": {k:len(v) for k,v in not_selected.items()},
        },
        "counterfactual_input_coverage_only": {
            "if_evolution_source_removed_selected_p0_lost": evo_p0,
            "if_evolution_source_removed_selected_p0_directly_retained": direct_p0,
            "not_a_counterfactual_selection_or_quality_outcome": True,
        },
        "selected_p0": selected,
        "omitted_before_pool": omitted_raw,
        "not_selected_from_pool": not_selected,
        "dependency_graph_context": graph_info,
        "interpretation_limits": findings,
    }


def emit(out: Path, report: dict[str, Any]) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out}")
    out.mkdir(parents=True)
    (out / "lineage.attribution.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    selected = report.get("selected_p0", [])
    with (out / "p0.lineage.csv").open("w", newline="", encoding="utf-8") as fp:
        names = ["idea_id", "candidate_id", "source_object_id", "origin", "idea_origin_kind", "operator_id",
                 "source_kinds", "attribution_role", "source_frontier_parent_ids", "external_literature_lineage",
                 "candidate_or_unverified_lineage", "cross_source_composition", "conceptual_family_signature",
                 "selection_profile", "scientific_intent"]
        w = csv.DictWriter(fp, names)
        w.writeheader()
        for item in selected:
            row = {k:item.get(k) for k in names}
            row["source_kinds"] = ";".join(item["source_kinds"])
            row["source_frontier_parent_ids"] = ";".join(item["source_frontier_parent_ids"])
            w.writerow(row)
    counts=report.get("counts", {})
    lines=["# M2-A — Stage7 → P0 Scientific Lineage Attribution", "",f"Status: **{report['status']}**", "",
           "Observational provenance; **NOT** scientific quality, novelty, or counterfactual effect.", ""]
    if report["status"] == "ATTRIBUTION_COMPLETE":
        lines += ["## Funnel", "", "| Layer | Count |", "|---|---:|",
                  f"| Frontier ideas (raw) | {counts['raw_frontier_count']} |",
                  f"| Stage7 Evolution ideas (raw) | {counts['raw_evolution_count']} |",
                  f"| Candidate Pool | {counts['projected_pool_count']} |",
                  f"| Selected P0 | {counts['selected_p0_count']} |", "",
                  "## Selected sources", "",
                  f"- P0 by direct origin: `{counts['selected_p0_by_origin']}`",
                  f"- P0 by attribution role: `{counts['selected_p0_by_attribution_role']}`",
                  f"- Source-kind incidence (one P0 may count more than once): `{counts['selected_p0_upstream_source_kind_incidence']}`",
                  f"- Evolution operators retained: `{counts['selected_p0_by_evolution_operator']}`",
                  "", "## Selected P0", "",
                  "| ResearchIdea | Source | Operator | Source kinds | Intent |", "|---|---|---|---|---|"]
        for r in selected:
            clean = str(r["scientific_intent"]).replace("|", "/").replace("\n"," ")
            lines.append(f"| `{r['idea_id']}` | {r['origin']} | {r['operator_id'] or '—'} | {', '.join(r['source_kinds'])} | {clean[:190]} |")
    else:
        lines += ["## Missing data", "", "Full frozen JSON snapshots are required for per-P0 attribution.", "", *[f"- {x}" for x in report.get("missing_inputs", [])]]
    lines += ["", "## Reduction decision", "", "**No deletion authorized by M2-A alone.**", "",
             "Retain Stage7 Evolution as an experimental source until a frozen-input ablation demonstrates nonessential scientific contribution.",
             "A selected P0 count by source is not a quality score. Do not infer quality or counterfactual effectiveness from counts.",
             "The next milestone is a cost-bounded minimal-path design; guard identity, grounding and provenance.", ""]
    (out / "M2A_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze-dir", type=Path, help="M1 folder containing frozen_artifacts/ and freeze.manifest.json")
    p.add_argument("--manifest", type=Path, help="Alternative explicit M1 freeze.manifest.json")
    p.add_argument("--case-dir", type=Path, help="Optional original Q-A case as artifact fallback")
    p.add_argument("--dependency-graph", type=Path, help="M1 dependency.graph.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--preflight-only", action="store_true", help="Manifest/graph-only summary; no lineage attribution")
    args=p.parse_args(argv)
    freeze = (args.freeze_dir or (args.manifest.parent if args.manifest else None))
    if freeze is None:
        p.error("--freeze-dir or --manifest required")
    manifest_path = args.manifest or freeze / "freeze.manifest.json"
    manifest = read_json(manifest_path)
    graph=read_json(args.dependency_graph) if args.dependency_graph else None
    if args.preflight_only:
        by_key = {x['key']: x for x in manifest['artifacts']}
        report={"schema_version":"ai-scientist-m2a-p0-scientific-lineage-audit-v1", "status":"MANIFEST_ONLY_PREFLIGHT",
                "freeze_git_head": manifest.get("git_head"),
                "snapshot_keys_available_in_manifest": {k:by_key.get(k, {}).get("present", False) for k in NEEDED},
                "missing_inputs": ["No frozen Stage7/P0 JSON content was read: lineage attribution is not yet known"],
                "dependency_graph_context": {"classification_counts":(graph or {}).get("classification_counts", {}),
                  "stage7_dynamic_runner_site_count":len(((graph or {}).get("dynamic_sites") or {}).get("scripts.discovery.run_dac_discovery_e2e", []))},
                "selected_p0": [],"counts":{}, "interpretation_limits":["Manifest presence cannot establish origin attribution"]}
    else:
        files,proof = resolve_files(freeze, manifest, args.case_dir)
        report=analyze({key: read_json(path) for key,path in files.items()}, graph=graph)
        report["snapshot_integrity_proofs"] = proof
        report["freeze_git_head"] = manifest.get("git_head")
    emit(args.output_dir,report)
    print("M2-A:",report["status"])
    if report["status"] == "ATTRIBUTION_COMPLETE":
        print("Selected P0 by origin:", report["counts"]["selected_p0_by_origin"])
        print("Source kind incidence:", report["counts"]["selected_p0_upstream_source_kind_incidence"])
    print("Output:", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
