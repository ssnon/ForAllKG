"""M2-F: non-destructive scientific-idea archive and reviewed terminal view.

No network, LLM, model judgement, filesystem mutation outside a fresh output dir.
Original SIS runs, realization portfolios, and verifier authority are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def unique_nodes(execution: dict[str, Any], generation: int) -> dict[str, dict[str, Any]]:
    if execution.get("generation_index") != generation:
        raise ValueError(f"execution generation index mismatch at g{generation}")
    nodes = execution.get("g4_population_nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError(f"missing retained ResearchIdea population at g{generation}")
    if execution.get("g4_population_count") != len(nodes):
        raise ValueError(f"population count mismatch at g{generation}")
    found: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not node.get("idea_id") or not isinstance(node.get("kernel"), dict):
            raise ValueError(f"invalid ResearchIdea at g{generation}")
        ident = node["idea_id"]
        if ident in found:
            raise ValueError(f"duplicate ResearchIdea {ident} at g{generation}")
        found[ident] = node
    return found


def load_cohort(*, p0_path: Path, run_dir: Path) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, str]]:
    """Load every reachable population and every *checked* transition."""
    current = load(p0_path)
    start = int(current.get("generation_index", -1))
    if start != 2:
        raise ValueError("M2-F expects P0 execution generation 2")
    generations = {start: unique_nodes(current, start)}
    executions = {start: current}
    checked = {"p0": sha_file(p0_path)}
    cycle_index = 1
    while True:
        summary_path = run_dir / f"cycle_{cycle_index:02d}.summary.json"
        if not summary_path.is_file():
            if cycle_index == 1:
                raise FileNotFoundError(f"missing cycle summary: {summary_path}")
            break
        summary = load(summary_path)
        now = int(summary.get("current_generation_index", -1))
        following = int(summary.get("next_generation_index", -1))
        if now != max(generations) or following != now + 1:
            raise ValueError(f"broken cycle chain: {summary_path}")
        next_path = run_dir / "case_root" / "scientific_portfolio_shadow" / f"sis_v3_4.g{following}_adaptive_population_execution.json"
        next_exec = load(next_path)
        next_nodes = unique_nodes(next_exec, following)
        if summary.get("next_generation_population_count") != len(next_nodes):
            raise ValueError("cycle summary population mismatch")
        source_ids = set(generations[now])
        target_ids = set(next_nodes)
        for carried_id in source_ids & target_ids:
            if canonical(generations[now][carried_id]) != canonical(next_nodes[carried_id]):
                raise ValueError(f"carried idea mutated: {carried_id}")
        p_report = run_dir / "case_root" / "scientific_portfolio_shadow" / f"sis_v3_4.g{now}_population_persistence.json"
        persist = load(p_report)
        if int(persist.get("cycle_generation_index", -1)) != now or int(persist.get("next_generation_index", -1)) != following:
            raise ValueError("persistence generation mismatch")
        if set(persist.get("final_population_idea_ids", [])) != target_ids:
            raise ValueError("persistence final population differs from execution")
        replaced = set(persist.get("replaced_parent_idea_ids", []))
        carried = set(persist.get("carried_forward_idea_ids", []))
        children = set(persist.get("generated_child_idea_ids", []))
        if replaced & carried or not replaced <= source_ids or carried != source_ids & target_ids:
            raise ValueError("invalid replaced/carried parent ledger")
        if children != target_ids - source_ids:
            raise ValueError("generated child ledger inconsistent with population")
        if replaced != set(summary.get("replaced_parent_idea_ids", [])) and "replaced_parent_idea_ids" in summary:
            raise ValueError("summary/persistence replaced-parent mismatch")
        for parent_id in replaced:
            if not any(parent_id in next_nodes[cid].get("parent_idea_ids", []) for cid in children):
                raise ValueError(f"replaced parent has no retained child: {parent_id}")
        checked[f"cycle_{cycle_index:02d}"] = sha_file(summary_path)
        checked[f"g{following}_execution"] = sha_file(next_path)
        checked[f"g{now}_persistence"] = sha_file(p_report)
        generations[following] = next_nodes
        executions[following] = next_exec
        cycle_index += 1
    return generations, executions, {g: load(run_dir / "case_root" / "scientific_portfolio_shadow" / f"sis_v3_4.g{g}_population_persistence.json") for g in sorted(generations)[:-1]}, checked


def make_archive(generations: dict[int, dict[str, dict[str, Any]]], persistence: dict[int, dict[str, Any]]) -> dict[str, Any]:
    historic: dict[str, dict[str, Any]] = {}
    for generation, nodes in sorted(generations.items()):
        for ident, node in nodes.items():
            if ident not in historic:
                historic[ident] = {"idea_id": ident, "node": node, "node_sha256": digest(node), "present_in_epochs": []}
            elif historic[ident]["node_sha256"] != digest(node):
                raise ValueError(f"historical idea identity mutated: {ident}")
            historic[ident]["present_in_epochs"].append(generation)
    final_set = set(generations[max(generations)])
    replaced = {i for p in persistence.values() for i in p.get("replaced_parent_idea_ids", [])}
    generated = {i for p in persistence.values() for i in p.get("generated_child_idea_ids", [])}
    for item in historic.values():
        ident = item["idea_id"]
        item["terminal_active"] = ident in final_set
        item["was_replaced_as_parent"] = ident in replaced
        item["is_retained_generated_child"] = ident in generated
        item["direct_child_ids"] = sorted(j for j, child in historic.items() if ident in child["node"].get("parent_idea_ids", []))
        item["historical_only"] = ident not in final_set
        item["original_generation_index"] = item["node"].get("generation_index")
    return {
        "schema_version": "m2f-scientific-program-preservation-archive-v1",
        "epistemic_status": "INSPIRATION_ONLY",
        "positive_premise_authority": False,
        "novelty_certified": False,
        "parent_reintroduced_into_active_population": False,
        "epochs": sorted(generations),
        "total_unique_ideas": len(historic),
        "terminal_active_count": len(final_set),
        "historical_only_count": sum(x["historical_only"] for x in historic.values()),
        "replaced_parent_count": len(replaced),
        "ideas": [historic[i] for i in sorted(historic)],
    }


def terminal_links(*, portfolio: dict[str, Any], lifecycle: dict[str, Any], final_nodes: dict[str, dict[str, Any]]) -> dict[str, str]:
    if lifecycle.get("output_portfolio_id") != portfolio.get("portfolio_id"):
        raise ValueError("terminal lifecycle/portfolio ID mismatch")
    mapping: dict[str, str] = {}
    for row in lifecycle.get("links", []):
        if row.get("materialization_status") != "MATERIALIZED" or not row.get("hypothesis_id"):
            continue
        h, ident = str(row["hypothesis_id"]), str(row["idea_id"])
        if ident not in final_nodes:
            raise ValueError(f"realization references missing final idea: {ident}")
        if h in mapping and mapping[h] != ident:
            raise ValueError(f"hypothesis has multiple ResearchIdea parents: {h}")
        mapping[h] = ident
    ids = [r.get("hypothesis_id") for r in portfolio.get("hypotheses", [])]
    if not all(isinstance(x, str) and x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("invalid or duplicate final hypothesis IDs")
    if set(mapping) != set(ids):
        raise ValueError(f"terminal realization linkage mismatch, missing={sorted(set(ids)-set(mapping))}, extra={sorted(set(mapping)-set(ids))}")
    return mapping


def review_scope(ids: list[str], nodes: dict[str, dict[str, Any]]) -> str:
    return digest([{"idea_id": i, "node_sha256": digest(nodes[i])} for i in sorted(ids)])


def review_template(nodes: dict[str, dict[str, Any]], relevant_ids: list[str]) -> dict[str, Any]:
    ids = sorted(set(relevant_ids))
    return {
        "schema_version": "m2f-terminal-pair-expert-review-v1",
        "scope_sha256": review_scope(ids, nodes),
        "reviewer": "", "status": "UNREVIEWED",
        "idea_summaries": {i: {
            "canonical_intent": nodes[i]["kernel"].get("canonical_intent", ""),
            "core_scientific_commitments": nodes[i]["kernel"].get("core_scientific_commitments", []),
            "contrastive_commitments": nodes[i]["kernel"].get("contrastive_commitments", []),
            "differential_prediction": nodes[i].get("differential_prediction", ""),
            "falsification_condition": nodes[i].get("falsification_condition", ""),
            "discriminating_observation": nodes[i].get("discriminating_observation", ""),
        } for i in ids},
        "pairs": [{"idea_id_a": a, "idea_id_b": b, "relation": "UNREVIEWED", "rationale": ""} for a, b in itertools.combinations(ids, 2)],
        "policy": "SAME_PROGRAM requires same central causal mechanism and decisive experiment class; adjacent != duplicate; no scientific truth authority",
    }


def reviewed_groups(*, nodes: dict[str, dict[str, Any]], relevant_ids: list[str], reviews: dict[str, Any] | None) -> tuple[list[list[str]], bool]:
    ids = sorted(set(relevant_ids))
    if reviews is None:
        return [[x] for x in ids], False
    if reviews.get("schema_version") != "m2f-terminal-pair-expert-review-v1" or reviews.get("scope_sha256") != review_scope(ids, nodes):
        raise ValueError("terminal pair review scope mismatch (stale or wrong input)")
    if reviews.get("status") != "COMPLETE" or not str(reviews.get("reviewer", "")).strip():
        raise ValueError("only explicit completed expert review may compress scientific programs")
    expected = {tuple(row) for row in itertools.combinations(ids, 2)}
    observed: dict[tuple[str, str], str] = {}
    for row in reviews.get("pairs", []):
        a, b = row.get("idea_id_a"), row.get("idea_id_b")
        pair = tuple(sorted((a, b))) if isinstance(a, str) and isinstance(b, str) else ("", "")
        if pair not in expected or pair in observed:
            raise ValueError("unknown or duplicate expert pair")
        rel = row.get("relation")
        if rel not in {"SAME_PROGRAM", "ADJACENT_PROGRAM", "DISTINCT_PROGRAM"} or not str(row.get("rationale", "")).strip():
            raise ValueError("pair relation must be assessed with rationale")
        observed[pair] = rel
    if set(observed) != expected:
        raise ValueError("expert pair review must cover every terminal idea pair")
    parent = {i: i for i in ids}
    def root(x: str) -> str:
        while parent[x] != x:
            x = parent[x]
        return x
    for (a, b), rel in observed.items():
        if rel == "SAME_PROGRAM":
            parent[root(b)] = root(a)
    groups: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        groups[root(i)].append(i)
    out = sorted((sorted(v) for v in groups.values()), key=lambda v: v[0])
    for group in out:
        for pair in itertools.combinations(group, 2):
            if observed[tuple(sorted(pair))] != "SAME_PROGRAM":
                raise ValueError("non-clique SAME_PROGRAM chain; conflicting expert assessments")
    return out, True


def compressed_view(*, portfolio: dict[str, Any], mapping: dict[str, str], final_nodes: dict[str, dict[str, Any]], reviews: dict[str, Any] | None) -> dict[str, Any]:
    groups, reviewed = reviewed_groups(nodes=final_nodes, relevant_ids=list(mapping.values()), reviews=reviews)
    hypothesis_order = [x["hypothesis_id"] for x in portfolio["hypotheses"]]
    if not reviewed:
        # Even two distinct hypotheses from the same ResearchIdea are NOT
        # silently compressed without explicit review.
        groups = [[mapping[h]] for h in hypothesis_order]
    programs = []
    for ix, members in enumerate(groups, 1):
        hypotheses = ([h for h in hypothesis_order if mapping[h] in members]
                      if reviewed else [hypothesis_order[ix - 1]])
        programs.append({
            "view_program_id": f"M2F_PROGRAM_{ix:03d}", "idea_ids": members,
            "hypothesis_ids": hypotheses,
            "display_representative_hypothesis_id": hypotheses[0],
            "suppressed_from_display_only_hypothesis_ids": hypotheses[1:],
            "grounding_or_novelty_certified": False,
        })
    return {
        "schema_version": "m2f-terminal-scientific-program-view-v1",
        "source_portfolio_id": portfolio.get("portfolio_id"),
        "source_hypothesis_count": len(hypothesis_order),
        "program_groups": programs,
        "display_count": len(programs),
        "all_hypothesis_ids_preserved": sorted(hypothesis_order),
        "scientific_pair_review_completed": reviewed,
        "semantic_compression_applied": reviewed and len(programs) < len(set(mapping.values())),
        "unreviewed_singletons_not_independent_program_claim": not reviewed,
        "display_only_original_portfolio_unchanged": True,
        "review_does_not_certify_novelty_or_truth": True,
    }


def logic_review_template(portfolio: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Human review template; no speculative automated falsifier verdicts."""
    return {
        "schema_version": "m2f-terminal-falsifier-logic-review-v1",
        "review_status": "UNREVIEWED",
        "reviews": [
            {
                "hypothesis_id": h["hypothesis_id"],
                "research_idea_id": mapping[h["hypothesis_id"]],
                "title": h.get("title", ""),
                "hypothesis_statement": h.get("hypothesis_statement", ""),
                "predicted_observations": h.get("predicted_observations", []),
                "falsification_criteria": h.get("falsification_criteria", []),
                "causal_necessity_of_falsifier": "UNREVIEWED",
                "alternative_mechanisms_controlled": "UNREVIEWED",
                "measurement_independence": "UNREVIEWED",
                "confound_controls": "UNREVIEWED",
                "reviewer_notes": "",
            }
            for h in portfolio.get("hypotheses", [])
        ],
        "warning": "Neither this template nor program compression certifies valid falsification, evidence grounding, or scientific novelty.",
    }


def rescue_execution(*, execution: dict[str, Any], final_nodes: dict[str, dict[str, Any]], archive: dict[str, Any], requested: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    all_nodes = {row["idea_id"]: row["node"] for row in archive["ideas"]}
    selected = sorted(set(requested))
    if not selected or len(selected) != len(requested):
        raise ValueError("rescue IDs must be nonempty and unique")
    if any(i not in all_nodes or i in final_nodes for i in selected):
        raise ValueError("rescue idea IDs must be historic, absent from final population")
    combined = dict(final_nodes)
    combined.update({i: all_nodes[i] for i in selected})
    new = dict(execution)
    new["g4_population_nodes"] = [combined[k] for k in sorted(combined)]
    new["g4_population_count"] = len(combined)
    new["population_growth_budget"] = max(int(new.get("population_growth_budget", 0)), len(selected))
    new.pop("report_id", None)
    new.pop("report_sha256", None)
    sha = digest(new)
    new["report_id"] = f"m2f_terminal_rescue:{sha[:20]}"
    new["report_sha256"] = sha
    return new, {
        "schema_version": "m2f-terminal-rescue-handoff-v1",
        "selected_archive_idea_ids": selected,
        "source_execution_report_id": execution.get("report_id"),
        "rescue_execution_report_id": new["report_id"],
        "original_final_population_count": len(final_nodes),
        "rescued_terminal_population_count": len(combined),
        "new_llm_calls": 0,
        "requires_fresh_strict_realization_and_common_verification": True,
        "original_run_not_mutated": True,
        "not_a_quality_or_novelty_promotion": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.v34_dir.resolve()
    p0 = args.p0_execution.resolve()
    final_portfolio_path = run_dir / "final.materialized.portfolio.json"
    final_portfolio = load(final_portfolio_path)
    generations, executions, persistence, sources = load_cohort(p0_path=p0, run_dir=run_dir)
    last = max(generations)
    final_nodes = generations[last]
    arm = load(run_dir / "arm.summary.json")
    if arm.get("source_p0_report_id") and arm["source_p0_report_id"] != executions[min(generations)]["report_id"]:
        raise ValueError("arm summary does not use the supplied frozen P0 execution")
    if arm.get("child_generation_steps") != len(generations) - 1:
        raise ValueError("arm summary cycle count does not match archive")
    if arm.get("final_execution_report_id") != executions[last].get("report_id"):
        raise ValueError("arm summary final execution ID mismatch")
    if arm.get("final_portfolio") and Path(arm["final_portfolio"]).name != final_portfolio_path.name:
        raise ValueError("arm summary final portfolio path mismatch")
    lifecycle_path = run_dir / "case_root" / "scientific_portfolio_shadow" / f"sis_v3_4.g{last}_realization_lifecycle.json"
    lifecycle = load(lifecycle_path)
    mapping = terminal_links(portfolio=final_portfolio, lifecycle=lifecycle, final_nodes=final_nodes)
    archive = make_archive(generations, persistence)
    # Reuse, but do not promote, existing per-epoch scientific-program
    # classifications. Cross-generation program equivalence is NOT inferred.
    per_epoch_observations = []
    for generation, nodes in sorted(generations.items()):
        family_path = run_dir / "case_root" / "scientific_portfolio_shadow" / f"sis_v3_4.g{generation}_scientific_program_family.json"
        if not family_path.is_file():
            continue
        family = load(family_path)
        assignments = family.get("assignments", [])
        if set(r.get("idea_id") for r in assignments) != set(nodes):
            raise ValueError(f"program-family report and g{generation} idea IDs differ")
        if family.get("pair_count") != len(family.get("pair_assessments", [])):
            raise ValueError(f"program-family report has mismatched pair count at g{generation}")
        per_epoch_observations.append({
            "epoch": generation,
            "report_id": family.get("report_id"),
            "report_sha256_file": sha_file(family_path),
            "idea_count": len(nodes),
            "diagnostic_program_count": family.get("program_count"),
            "transitivity_diagnostic_count": family.get("transitivity_diagnostic_count"),
            "authority": "HISTORICAL_SEMANTIC_DIAGNOSTIC_ONLY",
        })
        sources[f"g{generation}_program_family"] = sha_file(family_path)
    archive["per_epoch_program_family_observations"] = per_epoch_observations
    template = review_template(final_nodes, list(mapping.values()))
    review = load(args.terminal_pair_review) if args.terminal_pair_review else None
    view = compressed_view(portfolio=final_portfolio, mapping=mapping, final_nodes=final_nodes, reviews=review)
    sources["final_portfolio"] = sha_file(final_portfolio_path)
    sources["terminal_lifecycle"] = sha_file(lifecycle_path)
    sources["arm_summary"] = sha_file(run_dir / "arm.summary.json")
    if args.expected_final_sha256 and sources["final_portfolio"] != args.expected_final_sha256:
        raise ValueError("final portfolio SHA256 differs from frozen reference")
    if args.terminal_pair_review:
        sources["expert_review"] = sha_file(args.terminal_pair_review)
    rescue = None
    if args.rescue_idea_id:
        rescue = rescue_execution(execution=executions[last], final_nodes=final_nodes, archive=archive, requested=args.rescue_idea_id)
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite prior audit: {out}")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "scientific_program_archive.json", archive)
    write(out / "terminal_program_view.json", view)
    write(out / "terminal_pairs_review_TEMPLATE.json", template)
    write(out / "terminal_falsifier_logic_review_TEMPLATE.json", logic_review_template(final_portfolio, mapping))
    write(out / "input_integrity.json", {"source_sha256": sources, "read_only_source": True})
    if rescue is not None:
        write(out / "terminal_rescue.execution.json", rescue[0])
        write(out / "terminal_rescue.handoff.json", rescue[1])
    status = {
        "status": "M2F_ARCHIVE_AND_TERMINAL_VIEW_READY",
        "epochs": sorted(generations),
        "archived_research_ideas": archive["total_unique_ideas"],
        "replaced_parent_ideas_preserved_in_archive": archive["replaced_parent_count"],
        "terminal_active_research_ideas": archive["terminal_active_count"],
        "terminal_hypotheses": len(mapping),
        "terminal_display_program_buckets": view["display_count"],
        "semantic_pair_review_completed": view["scientific_pair_review_completed"],
        "terminal_rescue_execution_prepared": rescue is not None,
        "scientific_novelty_certified": False,
        "original_pipeline_mutated": False,
        "new_llm_or_network_calls": 0,
    }
    write(out / "M2F_SUMMARY.json", status)
    (out / "M2F_REPORT.md").write_text(
        "# M2-F — Scientific Program Preservation & Terminal Compression\n\n"
        f"- Status: {status['status']}\n"
        f"- Epochs: {status['epochs']}\n"
        f"- ResearchIdea nodes ever retained: {archive['total_unique_ideas']}\n"
        f"- Replaced parent nodes preserved: {archive['replaced_parent_count']}\n"
        f"- Historical-only ideas: {archive['historical_only_count']}\n"
        f"- Final ResearchIdea active: {archive['terminal_active_count']}\n"
        f"- Final realized hypotheses: {len(mapping)}\n"
        f"- Terminal display buckets: {view['display_count']}\n"
        f"- Expert-scoped semantic grouping: {view['scientific_pair_review_completed']}\n"
        f"- Optional rescue execution built: {rescue is not None}\n\n"
        "**Important:** This is not proof of scientific novelty or quality. Archive preservation "
        "does not automatically keep parents active. Terminal compression is a display view "
        "only, and without complete independent pair review the singleton buckets are not "
        "claims of scientific program independence. A rescue execution must undergo fresh strict "
        "realization and downstream verification. Original outputs are unchanged.\n",
        encoding="utf-8",
    )
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v34-dir", required=True, type=Path, help="Frozen SIS v3.4 E2E output directory")
    ap.add_argument("--p0-execution", required=True, type=Path, help="M2-C P0 execution or original P0")
    ap.add_argument("--output-dir", required=True, type=Path, help="Must not exist")
    ap.add_argument("--expected-final-sha256", help="Optional SHA from M1 freeze manifest")
    ap.add_argument("--terminal-pair-review", type=Path, help="Explicit complete, fingerprint-scoped expert classification")
    ap.add_argument("--rescue-idea-id", action="append", default=[], help="Explicit archived ResearchIdea ID to add to separate terminal rescue execution")
    args = ap.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
