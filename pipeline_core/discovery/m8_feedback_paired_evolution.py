"""M8 opt-in, four-call matched G6 experiment. Never changes official population.

Known scientific critiques are research context, never positive evidence or proof.
The control and treatment share parents, source M7 program, model, task, and budget.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

TARGETS = ("P2_DESIGN_DECISIVE_EXPERIMENT", "P2_OPEN_ALTERNATE_BRANCH")
ARMS = ("CONTROL", "SCIENCE_FEEDBACK")
ALLOWED_OPERATORS = ("BACKBONE_MUTATION", "LATENT_VARIABLE", "PROXY_CHALLENGE", "REGIME_BOUNDARY", "AXIS_MUTATION")
DIMENSIONS = ("conceptual_change", "formal_prediction", "competing_null", "independent_identifiability", "falsification_logic", "research_value")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON object required: {path}")
    return obj


def write_new(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _unique_map(items: list[dict], key: str) -> dict:
    result = {}
    for item in items:
        ident = item[key]
        if ident in result:
            raise ValueError(f"duplicate {key}: {ident}")
        result[ident] = item
    return result


def make_experiment(*, raw: dict, m7: dict, m7_plan: dict, reviewed: dict, source_files: dict[str, Path]) -> dict:
    """Validate authentic source lineage before freezing the two-arm experiment."""
    if raw.get("generation_index") != 5 or raw.get("genuine_child_count") != 4:
        raise ValueError("Expected 4-child RAW native G5 execution")
    if raw.get("production_selection_authority") is not False or raw.get("canonical_graph_mutated") is not False:
        raise ValueError("Authority boundary violated")
    if m7.get("programs_developed") != 6 or m7.get("programs_planned") != 6:
        raise ValueError("Require actual complete M7 six-program report")
    if reviewed.get("official_g6_generation_executed") is not False or reviewed.get("use_as_search_context_only") is not True:
        raise ValueError("Review overlay must be nonofficial search context only")
    nodes = _unique_map(raw["offspring_nodes"], "idea_id")
    semantic = _unique_map([s if isinstance(s, dict) else s.model_dump(mode="json") for s in raw["semantic_records"]], "idea_id")
    programs = _unique_map(m7["rows"], "label")
    task_kernels = _unique_map(m7_plan["tasks"], "label")
    critiques = _unique_map(reviewed["primary_research_children"], "program_label")
    parents = []
    for label in TARGETS:
        cr = critiques[label]
        pid = cr["idea_id"]
        if pid not in nodes or pid not in semantic:
            raise ValueError(f"Reviewed parent not present in RAW G5: {label}")
        node = nodes[pid]
        source_task = task_kernels[label]
        if source_task.get("kernel_sha") != node.get("kernel_sha256") or digest(source_task["original_kernel"]) != node.get("kernel_sha256"):
            raise ValueError("Native G5 kernel does not match corresponding frozen M7 original")
        if semantic[pid].get("disposition") != "GENUINE_CHILD":
            raise ValueError(f"Source parent not genuine G5 child: {label}")
        if node.get("generation_index") != 5:
            raise ValueError("G5 parent generation mismatch")
        if node.get("source_context_id") != reviewed["source_context_id"]:
            raise ValueError("Reviewed source context mismatch")
        p = programs[label]
        if p.get("program", {}).get("idea_label") != label or p["program"].get("source_role") != label[3:]:
            raise ValueError("M7 scientific program label/role mismatch")
        if p["program"].get("novelty_certified") is not False or p["program"].get("scientific_truth_authority") is not False:
            raise ValueError("M7 has unapproved claim authority")
        if cr.get("scientific_truth_authority") is not False or cr.get("positive_premise_authority") is not False:
            raise ValueError("Review incorrectly claims authority")
        # One response per parent and arm, both arms use the same native G6 task ID.
        task_id = "m8_g6_matched_task:" + digest([raw["report_id"], pid, "G6_ONE_CHILD"] )[:20]
        parents.append({
            "label": label, "idea_id": pid, "node": node, "task_id": task_id,
            "source_program": p["program"], "review": cr,
        })
    if len({p["idea_id"] for p in parents}) != 2 or len({p["node"]["source_context_sha256"] for p in parents}) != 1:
        raise ValueError("Two distinct parents and single source context required")
    fixed = {
        "schema_version": "m8-scientific-feedback-paired-g6-plan-v1",
        "source_generation_index": 5, "target_generation_index": 6,
        "source_native_g5_report_id": raw["report_id"],
        "source_native_g5_sha256": raw["report_sha256"],
        "source_context_id": reviewed["source_context_id"],
        "source_context_sha256": parents[0]["node"]["source_context_sha256"],
        "parents": parents,
        "arms": list(ARMS), "max_candidates_per_parent_arm": 1,
        "planned_logical_provider_calls": 4, "model_temperature": 0,
        "operator_set": list(ALLOWED_OPERATORS),
        "selection_is_explicit_not_official": True,
        "official_scheduler_executed": False, "production_selection_authority": False,
        "positive_premise_authority": False, "scientific_truth_authority": False,
        "novelty_certified": False, "canonical_graph_mutated": False,
        "source_sha256": {key: {"path":str(value.resolve()), "sha256":sha_file(value)} for key,value in source_files.items()},
    }
    fixed["experiment_id"] = "m8_paired_g6:" + digest(fixed)[:20]
    return fixed


def checked_plan(path: Path) -> dict:
    cfg = read(path)
    if cfg.get("schema_version") != "m8-scientific-feedback-paired-g6-plan-v1":
        raise ValueError("Bad experiment plan schema")
    if len(cfg.get("parents", [])) != 2 or cfg.get("planned_logical_provider_calls") != 4:
        raise ValueError("Matched design modified")
    for v in cfg["source_sha256"].values():
        src=Path(v["path"])
        if not src.is_file() or sha_file(src) != v["sha256"]:
            raise ValueError(f"Frozen source changed/missing: {src}")
    return cfg


def _task(parent: dict, cfg: dict) -> Any:
    from pipeline_core.discovery.research_idea_epistemic_generational_evolution import EpistemicG4GenerationTask
    return EpistemicG4GenerationTask(
        task_id=parent["task_id"], handoff_id="m8_experimental_handoff:"+digest(parent["idea_id"])[:20],
        generation_index=6, primary_parent_idea_id=parent["idea_id"], channel="TRANSFORM",
        allowed_operator_ids=cfg["operator_set"], max_output_count=1,
        source_reason_codes=["M8_EXPERIMENTAL_MATCHED_PARENT_NOT_OFFICIAL_SCHEDULER"],
    )


def native_components(cfg: dict):
    from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode
    from pipeline_core.discovery.research_idea_epistemic_generational_evolution import EpistemicG4GenerationPlan
    nodes = {p["idea_id"]: ResearchIdeaNode.model_validate(p["node"]) for p in cfg["parents"]}
    tasks = [_task(p,cfg) for p in cfg["parents"]]
    # Transparent experimental shim: no invented real lifecycle/fertility evidence.
    par = SimpleNamespace(report_id="m8_experimental_parallel:"+cfg["experiment_id"].split(":")[-1],members=[],epistemic_debts=[])
    plan = EpistemicG4GenerationPlan(
        plan_id="m8_experimental_native_g6:"+cfg["experiment_id"].split(":")[-1],
        source_parallel_report_id=par.report_id, generation_index=6,
        available_handoff_count=0, candidate_parent_count=0, max_parent_count=2,
        selected_parent_idea_ids=[p["idea_id"] for p in cfg["parents"]],
        parent_candidates=[], tasks=tasks, task_count=2, planned_max_offspring=2,
        selected_parent_source_mode_counts={},task_count_by_channel={"TRANSFORM":2},
    )
    return plan, par, nodes


def make_prompt(*, cfg: dict, parent: dict, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("unknown arm")
    from pipeline_core.discovery.research_idea_epistemic_generational_evolution import build_epistemic_g4_prompt
    plan,par,nodes = native_components(cfg)
    task = next(t for t in plan.tasks if t.primary_parent_idea_id == parent["idea_id"])
    baseline = build_epistemic_g4_prompt(task=task, parent_by_id=nodes, parallel_report=par,
        research_question=parent["source_program"]["research_question"])
    # Same source development context for both arms. Neither is evidence.
    common = {
        "origin": "M7_LLM_SCIENTIFIC_DEVELOPMENT_SEARCH_CONTEXT_ONLY",
        "model": parent["source_program"]["mechanism_model"],
        "proposed_prediction": parent["source_program"]["formal_prediction"],
        "proposed_null": parent["source_program"]["competing_null_model"],
        "original_discriminator": parent["source_program"]["predicted_contrast"],
        "original_falsification_scope": parent["source_program"]["falsification_scope"],
    }
    critique = None
    if arm == "SCIENCE_FEEDBACK":
        r=parent["review"]
        critique={
            "observed_retention_failure":r["scientific_retention_verdict"],
            "scientific_weaknesses":r["lost_or_weakened_science"],
            "corrective_science":r["corrective_science"],
            "next_generation_mandate":r["next_generation_mandate"],
            "promotion_blockers":r["promotion_blockers"],
            "instruction":"Use this fallible scientific critique to develop a more discriminating research idea; do not merely restate it, and do not claim the critique is proof. You may reject it if physically unsound.",
        }
    suffix = ("\n\nM8 EXPERIMENTAL SCIENTIFIC DEVELOPMENT CONTEXT — NOT POSITIVE EVIDENCE:\n"
              +json.dumps(common,ensure_ascii=False,sort_keys=True)
              +"\n\nM8 SCIENTIFIC FEEDBACK (NULL FOR CONTROL):\n"
              +json.dumps(critique,ensure_ascii=False,sort_keys=True)
              +"\n\nReturn JSON with task_id, primary_parent_idea_id, channel=TRANSFORM, candidates (exactly 1 preferred, abstain if needed), abstention_reason. Candidate: local_id, chosen_operator_id, conceptual_change_summary, kernel (canonical_intent, core_scientific_commitments, scope_commitments, contrastive_commitments, question_commitment), differential_prediction, falsification_condition, discriminating_observation, task_relation_mode. Avoid unsupported numbers or invented citations; preserve machine identifiers exactly."
              )
    return {"task_id":task.task_id,"arm":arm,"label":parent["label"],"system":baseline.system_prompt,
            "user":baseline.user_prompt+suffix}


def parse_draft(raw: str, parent: dict, cfg: dict):
    from pipeline_core.discovery.research_idea_offspring_execution import GenerationalOffspringBatchDraft
    stripped=raw.strip()
    if stripped.startswith("```"):
        stripped=stripped.split("\n",1)[1].rsplit("```",1)[0].strip()
    payload=json.loads(stripped)
    draft=GenerationalOffspringBatchDraft.model_validate(payload)
    if draft.task_id != parent["task_id"] or draft.primary_parent_idea_id!=parent["idea_id"] or draft.channel!="TRANSFORM":
        raise ValueError("Provider returned identity/task mismatch")
    if len(draft.candidates)>1:
        raise ValueError("Paid output budget exceeded")
    for child in draft.candidates:
        if child.chosen_operator_id not in cfg["operator_set"] or child.secondary_parent_idea_id is not None:
            raise ValueError("Unexpected operator / non-matched secondary parent")
    return draft


def receipt_paths(root: Path,cfg: dict):
    return [(arm,p,root/"receipts"/arm/(p["label"]+".json"),root/"drafts"/arm/(p["label"]+".json"))
            for arm in ARMS for p in cfg["parents"]]


def create_blind_packet(cfg: dict, results: dict, out_dir: Path) -> None:
    rows=[];key=[]
    for parent in cfg["parents"]:
        label=parent["label"]
        conditions=[]
        for arm in ARMS:
            native=results[arm]
            task_id=parent["task_id"]
            nodes={s["idea_id"]:s for s in native.get("offspring_nodes",[])}
            semantic=[s for s in native.get("semantic_records",[]) if s["task_id"]==task_id]
            science=[nodes[s["idea_id"]] for s in semantic if s["idea_id"] in nodes]
            if len(science)>1:
                raise ValueError("More than one native generated offspring for a matched cell")
            run=next((r for r in native.get("run_records",[]) if r["task_id"]==task_id),None)
            conditions.append((arm,science[0] if science else None,run))
        if int(digest([cfg["experiment_id"],label])[-1],16)%2:
            conditions.reverse()
        for position,(arm,node,run) in zip(("A","B"),conditions):
            # Do not disclose arm, native identity score, feedback contents or condition metadata.
            rows.append({"parent_label":label,"blind_position":position,
                "parent_scientific_question":parent["source_program"]["research_question"],
                "offspring_science":({k:node.get(k) for k in (
                    "kernel","differential_prediction","falsification_condition",
                    "discriminating_observation","task_relation_mode")}
                    if node else None),
                "model_output_status":("CANDIDATE_PRESENT" if node else "NO_COMPILED_CANDIDATE"),
                "review_notice":"Blind content is speculative and not evidence. Native semantic classification is withheld to avoid anchoring."})
            key.append({"parent_label":label,"blind_position":position,"arm":arm,
                        "native_decision":run.get("decision") if run else "NO_RUN"})
    write_new(out_dir/"M8_BLINDED_SCIENCE_PACKET.json",{"rows":rows,
             "review_instruction":"Assess conceptual change, formal prediction, strong known-physics competing null, independent identifiability, valid falsification and research value without inferring novelty certification. Empty proposals must be scored accordingly."})
    write_new(out_dir/"M8_UNBLIND_KEY_KEEP_SEPARATE.json",{"rows":key})
    fields=["parent_label","blind_position",*DIMENSIONS,"scientific_reasoning","dominant_null_or_failure"]
    with (out_dir/"M8_BLIND_SCORES_TO_FILL.csv").open("x",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({"parent_label":row["parent_label"],"blind_position":row["blind_position"]})


def score_blind(reviewed_csv: Path, key_path: Path, out_path: Path) -> dict:
    with reviewed_csv.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    key=read(key_path)["rows"]
    match={(r["parent_label"],r["blind_position"]):r["arm"] for r in key}
    if len(rows)!=4 or len(match)!=4:raise ValueError("Expected exactly 4 blind reviews")
    normalized={}
    for r in rows:
        k=(r["parent_label"],r["blind_position"])
        if k not in match or k in normalized:raise ValueError("Unrecognized or duplicate blind candidate")
        if not r.get("scientific_reasoning","").strip(): raise ValueError("Science rationale mandatory")
        try: scores={d:int(r[d]) for d in DIMENSIONS}
        except (TypeError,ValueError,KeyError) as exc: raise ValueError("All 6 rubric scores required") from exc
        if any(not 0<=v<=3 for v in scores.values()): raise ValueError("Score must be 0..3")
        normalized[k]={"arm":match[k],"scores":scores,"rationale":r["scientific_reasoning"]}
    if len(normalized)!=4:raise ValueError("Incomplete blinded review")
    comparisons=[]
    for label in sorted({k[0] for k in match}):
        c=next(v for k,v in normalized.items() if k[0]==label and v["arm"]=="CONTROL")
        t=next(v for k,v in normalized.items() if k[0]==label and v["arm"]=="SCIENCE_FEEDBACK")
        comparisons.append({"parent_label":label,"dimension_deltas_feedback_minus_control":{d:t["scores"][d]-c["scores"][d] for d in DIMENSIONS},
                            "control":c,"science_feedback":t})
    result={"status":"M8_SMALL_N_BLIND_PAIRED_HUMAN_REVIEW_NOT_SCIENCE_PROOF",
         "paired_parent_count":2,"comparisons":comparisons,"claim_authority":False,"novelty_certified":False,
         "caveat":"Two matched parents, no statistical generalization; blind scoring remains subjective."}
    write_new(out_path,result)
    return result
