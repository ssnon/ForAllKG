"""M1: non-destructive, offline Q-A freeze and static dependency inventory.

No repo production module is imported and no LLM/network API is contacted.
Static reachability is an upper bound, not proof of runtime execution or unused code.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOTS: dict[str, tuple[str, ...]] = {
    "full_current_v3_1": (
        "scripts.discovery.run_dac_discovery_e2e",
        "scripts.discovery.run_research_idea_e2e_search_v3_2",
        "scripts.discovery.run_standard_portfolio_verification_shadow",
    ),
    "sis_v3_4_continuation": (
        "scripts.discovery.run_research_idea_e2e_search_v3_4",
    ),
}
# V3.4 continuation consumes a frozen Stage7 P0 artifact. It does not re-run Stage7.
# The Stage7 runtime is the shared *provenance upstream*, not a V3.4 subprocess.

FREEZE_FILES: dict[str, str] = {
    "task_context": "hypothesis.context.json",
    "frontier": "frontier_idea_population.shadow.json",
    "stage7_idea_evolution": "idea_evolution.shadow.json",
    "stage7_candidate_pool": "scientific_portfolio_shadow/candidate_pool.json",
    "stage7_selection": "scientific_portfolio_shadow/selection.json",
    "stage7_materialized": "scientific_portfolio_shadow/materialized.shadow.portfolio.json",
    "frozen_p0_execution": "scientific_portfolio_shadow/sis_v3_2_e2e/p0.execution.json",
    "frozen_p0_population": "scientific_portfolio_shadow/sis_v3_2_e2e/p0.population.json",
    "v3_1_arm_summary": "scientific_portfolio_shadow/sis_v3_2_e2e/arms/v3_1/arm.summary.json",
    "v3_1_final": "scientific_portfolio_shadow/sis_v3_2_e2e/arms/v3_1/final.materialized.portfolio.json",
    "v3_1_verification": "full_current_verification/verification.summary.json",
    "v3_4_arm_summary": "scientific_portfolio_shadow/sis_v3_4_e2e/arm.summary.json",
    "v3_4_cycle_01": "scientific_portfolio_shadow/sis_v3_4_e2e/cycle_01.summary.json",
    "v3_4_cycle_02": "scientific_portfolio_shadow/sis_v3_4_e2e/cycle_02.summary.json",
    "v3_4_terminal_summary": "scientific_portfolio_shadow/sis_v3_4_e2e/terminal_realization.summary.json",
    "v3_4_final": "scientific_portfolio_shadow/sis_v3_4_e2e/final.materialized.portfolio.json",
    "v3_4_g3_raw": "scientific_portfolio_shadow/sis_v3_4_e2e/case_root/scientific_portfolio_shadow/sis_v3_4.g3_raw_generation_execution.json",
    "v3_4_g4_raw": "scientific_portfolio_shadow/sis_v3_4_e2e/case_root/scientific_portfolio_shadow/sis_v3_4.g4_raw_generation_execution.json",
    "v3_4_g3_population": "scientific_portfolio_shadow/sis_v3_4_e2e/case_root/scientific_portfolio_shadow/sis_v3_4.g3_adaptive_population_execution.json",
    "v3_4_g4_population": "scientific_portfolio_shadow/sis_v3_4_e2e/case_root/scientific_portfolio_shadow/sis_v3_4.g4_adaptive_population_execution.json",
}
REQUIRED = {
    "task_context", "frozen_p0_execution", "v3_4_arm_summary", "v3_4_cycle_01",
    "v3_4_cycle_02", "v3_4_final",
}
PINNED_CODE = (
    "run_full_current_ai_scientist_v1.sh",
    "scripts/discovery/run_dac_discovery_e2e.py",
    "scripts/discovery/run_research_idea_e2e_search_v3_2.py",
    "scripts/discovery/run_standard_portfolio_verification_shadow.py",
    "scripts/discovery/run_research_idea_e2e_search_v3_4.py",
    "scripts/discovery/run_research_idea_novelty_aware_cycle_v3_4.py",
    "pipeline_core/discovery/research_idea_novelty_aware_reproduction_v3_4.py",
    "pipeline_core/discovery/research_idea_search_pressure_v3_3_2.py",
    "pipeline_core/discovery/research_idea_program_family_v3_3_1.py",
    "pipeline_core/discovery/research_idea_adaptive_fertility.py",
    "pipeline_core/discovery/research_idea_epistemic_generational_evolution.py",
    "pipeline_core/discovery/scientific_portfolio_selection.py",
    "pipeline_core/discovery/idea_evolution_runtime.py",
)
MODULE_PATTERN = re.compile(r"\b(?:scripts|pipeline_core|domains)\.[A-Za-z_][A-Za-z_0-9.]*\b")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"not a JSON object: {path}")
    return obj


def _git(repo: Path, *args: str) -> str | None:
    cp = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return cp.stdout.strip() if cp.returncode == 0 else None


def _module_catalog(repo: Path) -> dict[str, Path]:
    catalog: dict[str, Path] = {}
    for top in ("scripts", "pipeline_core", "domains"):
        parent = repo / top
        if not parent.is_dir():
            continue
        for p in parent.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(repo)
            parts = list(rel.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            if parts:
                catalog[".".join(parts)] = p
    return catalog


def _local_references(module: str, path: Path, catalog: dict[str, Path]) -> tuple[set[tuple[str, str]], list[str], str | None]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeError) as exc:
        return set(), [], f"{type(exc).__name__}: {exc}"
    edges: set[tuple[str, str]] = set()
    dynamic: list[str] = []
    package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in catalog:
                    edges.add((alias.name, "STATIC_IMPORT"))
        elif isinstance(node, ast.ImportFrom):
            base = package[:max(0, len(package) - (node.level - 1))] if node.level else []
            suffix = (node.module or "").split(".") if node.module else []
            target = ".".join(base + suffix) if node.level else (node.module or "")
            if target in catalog:
                edges.add((target, "STATIC_IMPORT"))
            for alias in node.names:
                candidate = f"{target}.{alias.name}" if target else alias.name
                if candidate in catalog:
                    edges.add((candidate, "STATIC_IMPORT"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for m in MODULE_PATTERN.findall(node.value):
                if m in catalog and m != module:
                    edges.add((m, "LITERAL_MODULE_REFERENCE"))
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
            if name in {"run_stage", "import_module", "__import__", "Popen", "run", "check_call", "check_output"}:
                # These call sites may use a computed command/module string.
                dynamic.append(f"{path.name}:{node.lineno}:{name}")
    return edges, dynamic, None


def build_dependency_inventory(repo: Path) -> dict[str, Any]:
    catalog = _module_catalog(repo)
    edges: dict[str, list[dict[str, str]]] = {}
    dynamic_sites: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    # Parse each local module once; static analysis does not import production code.
    for module, path in catalog.items():
        refs, dyn, error = _local_references(module, path, catalog)
        edges[module] = [{"target": target, "kind": kind} for target, kind in sorted(refs)]
        if dyn:
            dynamic_sites[module] = dyn
        if error:
            errors[module] = error
    reached: dict[str, set[str]] = {}
    for group, roots in ROOTS.items():
        seen: set[str] = set()
        q = deque(r for r in roots if r in catalog)
        while q:
            cur = q.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            q.extend(row["target"] for row in edges.get(cur, []) if row["target"] not in seen)
        reached[group] = seen
    inventory = []
    for module, path in sorted(catalog.items()):
        in_old = module in reached["full_current_v3_1"]
        in_new = module in reached["sis_v3_4_continuation"]
        role = (
            "SHARED_STATIC_POSSIBLE" if in_old and in_new else
            "FULL_CURRENT_STATIC_POSSIBLE" if in_old else
            "V3_4_STATIC_POSSIBLE" if in_new else
            "NOT_REACHED_BY_STATIC_DISCOVERY"
        )
        inventory.append({
            "module": module, "path": str(path.relative_to(repo)), "classification": role,
            "old_possible": in_old, "v3_4_possible": in_new,
            "dynamic_callsite_count": len(dynamic_sites.get(module, [])),
            "code_sha256": _sha(path),
        })
    return {
        "method": "AST_TRANSITIVE_IMPORTS_AND_LITERAL_MODULE_REFERENCES_OVER_APPROXIMATION",
        "static_reachability_is_not_actual_execution": True,
        "unreached_is_not_safe_to_delete": True,
        "not_traversed_bash_commands_are_pinned_in_separate_entrypoint_manifest": True,
        "v3_4_consumes_frozen_stage7_p0_not_stage7_runtime": True,
        "roots": {k: list(v) for k, v in ROOTS.items()},
        "inventory": inventory,
        "edges": edges,
        "dynamic_sites": dynamic_sites,
        "parse_errors": errors,
        "missing_roots": {k: [r for r in v if r not in catalog] for k, v in ROOTS.items()},
        "classification_counts": dict(sorted(Counter(x["classification"] for x in inventory).items())),
    }


def _comparison(sources: dict[str, Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prefix, name in (("v3_1", "v3_1_arm_summary"), ("v3_4", "v3_4_arm_summary")):
        p = sources[name]
        if not p.is_file():
            result[prefix] = {"available": False}
            continue
        x = _json(p)
        fields = (
            "final_hypothesis_count", "continuation_generation_llm_calls",
            "continuation_realization_llm_calls", "genuine_child_count_across_cycles",
            "selected_parent_set_changed_across_cycles", "selected_parent_sets",
            "source_p0_seed_sha256", "source_p0_report_id",
        )
        result[prefix] = {"available": True, **{f: x.get(f) for f in fields}}
    cycles = []
    for name in ("v3_4_cycle_01", "v3_4_cycle_02"):
        p = sources[name]
        if p.is_file():
            obj = _json(p)
            cycles.append({
                "cycle": name, "source": FREEZE_FILES[name],
                **{f: obj.get(f) for f in (
                    "current_generation_index", "next_generation_index",
                    "current_population_count", "materialized_hypothesis_count",
                    "next_generation_llm_call_count", "next_generation_genuine_child_count",
                    "carried_forward_idea_count", "replaced_parent_idea_count",
                    "next_generation_population_count", "selected_parent_idea_ids",
                )},
            })
    result["v3_4_cycles"] = cycles
    result["comparison_is_not_equal_compute_or_scientific_quality_judgment"] = True
    result["generation_llm_calls_exclude_prior_art_family_and_realization_cost"] = True
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_inventory_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = ["module", "path", "classification", "old_possible", "v3_4_possible", "dynamic_callsite_count", "code_sha256"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def freeze_and_audit(repo: Path, case: Path, output: Path) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    case = case.resolve(strict=True)
    output = output.resolve()
    if not (repo / ".git").exists():
        raise ValueError(f"not a git checkout: {repo}")
    if output.exists():
        raise FileExistsError(f"freeze destination already exists: {output}")
    sources = {k: case / value for k, value in FREEZE_FILES.items()}
    missing_required = sorted(k for k in REQUIRED if not sources[k].is_file())
    if missing_required:
        raise FileNotFoundError(f"Required frozen QA artifacts missing: {missing_required}")
    if not (repo / "run_full_current_ai_scientist_v1.sh").is_file():
        raise FileNotFoundError("run_full_current_ai_scientist_v1.sh not found")
    dep = build_dependency_inventory(repo)
    required_root_absent = {k: v for k, v in dep["missing_roots"].items() if v}
    if required_root_absent:
        raise FileNotFoundError(f"Required root modules absent: {required_root_absent}")
    meta = {
        "schema_version": "ai-scientist-m1-freeze-audit-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(repo), "source_case": str(case),
        "git_branch": _git(repo, "branch", "--show-current"),
        "git_head": _git(repo, "rev-parse", "HEAD"),
        "git_status_porcelain": (_git(repo, "status", "--porcelain=v1", "--untracked-files=normal") or "").splitlines(),
        "repo_local_changes_may_prevent_exact_remote_sha_reproduction": True,
        "no_network_or_llm_called": True,
        "no_original_artifacts_modified": True,
        "v3_4_not_invoked_by_full_current_v1_shell": True,
        "claim_grounding_and_child_identity_boundaries_not_modified": True,
        "known_population_policy": "V3.4_WITH_ZERO_GROWTH_BUDGET_REPLACES_SELECTED_PARENT_WITH_CHILD",
    }
    code = {}
    for rel in PINNED_CODE:
        p = repo / rel
        code[rel] = {"present": p.is_file(), "sha256": _sha(p) if p.is_file() else None}
    file_rows = []
    comparison = _comparison(sources)
    # Must never overwrite an existing freeze. Stage everything then publish via rename.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".m1_freeze_staging_", dir=output.parent) as tmp:
        stage = Path(tmp)
        frozen = stage / "frozen_artifacts"
        frozen.mkdir()
        for key, src in sources.items():
            exists = src.is_file()
            row = {"key": key, "source_relpath": FREEZE_FILES[key], "present": exists,
                   "required": key in REQUIRED, "size_bytes": src.stat().st_size if exists else None,
                   "source_sha256": _sha(src) if exists else None}
            if exists:
                dest = frozen / f"{key}.json"
                shutil.copyfile(src, dest)
                if _sha(dest) != row["source_sha256"]:
                    raise RuntimeError(f"Snapshot copy hash mismatch: {key}")
                row["frozen_relpath"] = str(dest.relative_to(stage))
            file_rows.append(row)
        manifest = {**meta, "pinned_code": code, "artifacts": file_rows, "comparison": comparison,
                    "dependency_classification_counts": dep["classification_counts"],
                    "stage7_upstream_boundary_is_not_v3_4_subprocess": True}
        _write_json(stage / "freeze.manifest.json", manifest)
        _write_json(stage / "dependency.graph.json", {k: v for k, v in dep.items() if k != "inventory"})
        _write_inventory_csv(stage / "dependency.inventory.csv", dep["inventory"])
        _write_json(stage / "dependency.inventory.json", dep["inventory"])
        optional_count = dep["classification_counts"].get("NOT_REACHED_BY_STATIC_DISCOVERY", 0)
        md = [
            "# M1 — AI Scientist Freeze & Dependency Audit", "",
            "**Read-only snapshot and static inventory. NO deletion or code mutation was performed.**", "",
            f"- Git branch: `{meta['git_branch']}`; HEAD: `{meta['git_head']}`",
            f"- Dirty worktree entries: `{len(meta['git_status_porcelain'])}` (review before reproducibility claims)",
            f"- QA snapshots present: `{sum(r['present'] for r in file_rows)}/{len(file_rows)}`",
            f"- Static inventory size: `{len(dep['inventory'])}` local Python modules",
            f"- Not reached via known static edges: `{optional_count}` (NOT deletion-safe proof)",
            "", "## Two distinct runtime paths", "",
            "- **Full Current v1**: Stage7 `run_dac_discovery_e2e` → SIS `run_research_idea_e2e_search_v3_2 --mode v3_1` → `run_standard_portfolio_verification_shadow`.",
            "- **SIS v3.4**: frozen Stage7 P0 execution → `run_research_idea_e2e_search_v3_4` → per-cycle strict realization, external novelty, semantic program-family, bounded scheduler, mutation, terminal realization. It does **not** call the Full Current verifier by itself.",
            "- A v3.4 result must not be described as the same fully verified E2E as Full Current until the common verifier is applied.",
            "", "## Recorded policy deviation", "",
            "- In existing v3.4 persistence with `population_growth_budget=0`, a successfully reproduced selected parent is **replaced** by its first retained child, instead of preserving both in the active population.",
            "- Do not change this within M1; track as M2 scientific-search design decision.",
            "", "## Cost and outcome (not a quality verdict)", "",
            "```json", json.dumps(comparison, ensure_ascii=False, indent=2), "```", "",
            "## Reduction safety gate", "",
            "1. Preserve strict claim grounding, provenance, ResearchIdea identity, and semantic child validation.",
            "2. Unreached-in-static-graph means **needs dynamic reachability review**, not unused.",
            "3. Experimental and baseline runners must remain available for paired evaluation until a smaller path passes the same scientific-quality gate.",
            "4. No 3+/5 novelty target should be used as an unblinded generation stopping condition.",
            "", "## Next action", "",
            "M2: build a **parallel minimal AI Scientist execution path**, keep the existing paths unchanged, remove redundant early portfolio/evolution authority and repeated costly assessment **one at a time**, and compare against the frozen QA evidence.",
        ]
        (stage / "AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        stage.rename(output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--case-dir", type=Path, required=True, help="Existing Q-A case under Full Current evaluation root")
    p.add_argument("--output-dir", type=Path, required=True, help="New, nonexisting directory; never overwritten")
    args = p.parse_args(argv)
    result = freeze_and_audit(args.repo, args.case_dir, args.output_dir)
    print("M1 FREEZE COMPLETE — no LLM, no web, no production mutation")
    print("git HEAD:", result["git_head"])
    print("snapshot artifacts:", sum(row["present"] for row in result["artifacts"]))
    print("dependency counts:", result["dependency_classification_counts"])
    print("report:", args.output_dir.resolve() / "AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
