#!/usr/bin/env python3
"""M7 integrated opt-in loop: prepare -> develop-paid -> report -> realize-paid.

No commands change canonical KG or production SIS; preparatory actions are offline.
M7 pre-realization science is speculative, not proof/novelty/premise authority.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from pipeline_core.discovery.scientific_development_m7 import (
    ScientificProgramDraft, build_development_tasks, create_model_prompt,
    challenge_program, retention_brief, canonical, digest,
)
from pipeline_core.discovery.m7_sers_null_models import frozen_sers_challenges


def read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON object required: {path}")
    return data


def write_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args: argparse.Namespace) -> None:
    m64, m67 = read(args.m64), read(args.m67)
    expected = m67.get("source_file_sha256", {}).get("m64", {}).get("sha256")
    if expected != sha_file(args.m64):
        raise ValueError("M6.7 report references different M6.4 original bytes")
    plan = build_development_tasks(m64, m67, limit=args.limit)
    if args.out_dir.exists():
        if list(args.out_dir.iterdir()):
            raise FileExistsError("M7 output must be new/empty, never overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_new(args.out_dir / "M7_SCIENTIFIC_DEVELOPMENT_PLAN.json", plan)
    write_new(args.out_dir / "M7_SERS_REFERENCE_NULLS.json", frozen_sers_challenges())
    original_briefs = [
        {
            "label": t["label"], "source": "M6_4_GENERATED_RESEARCH_IDEA_NOT_M7_MODEL",
            "parent_idea_id": t["parent_idea_id"], "lane": t["population_lane"],
            "initial_prediction": t["original_differential_prediction"],
            "initial_discriminator": t["original_discriminating_observation"],
            "initial_falsification": t["original_falsification_condition"],
            "development_status": "NEEDS_FORMAL_MODELS_INDEPENDENT_MEASUREMENTS_AND_NULL_CHALLENGE",
            "truth_authority": False,
        } for t in plan["tasks"]
    ]
    write_new(args.out_dir / "M7_FROZEN_STARTING_PROGRAMS.json", {"rows": original_briefs})
    print(json.dumps({"status":"M7_FROZEN_SCIENCE_DEVELOPMENT_PREPARED",
                      "task_count":len(plan["tasks"]), "reserve_included":sum(t["population_lane"].startswith("RESERVE") for t in plan["tasks"]),
                      "new_api_calls":0, "out_dir":str(args.out_dir)},ensure_ascii=False,indent=2))


def _read_plan(path: Path) -> dict:
    data=read(path)
    if data.get("schema_version")!="m7-scientific-development-plan-v1":
        raise ValueError("wrong M7 plan schema")
    return data


def _receipts(root: Path, plan: dict) -> list[dict]:
    rows=[]
    for task in plan["tasks"]:
        name=task["label"]
        receipt=root/"receipts"/(name+".json")
        entry={"label":name,"role":task["role"],"population_lane":task["population_lane"],"status":"PENDING"}
        if receipt.is_file():
            r=read(receipt)
            if r.get("task_sha")!=digest(task) or r.get("label")!=name:
                raise ValueError("saved receipt task mismatch: "+name)
            if r.get("raw_response") is None:
                raise ValueError("saved receipt missing original response")
            try:
                revision_file=root/"revisions"/(name+".json")
                parsed_file=root/"parsed"/(name+".json")
                chosen=revision_file if revision_file.is_file() else parsed_file
                if not chosen.is_file():
                    raise ValueError("model response saved, but parsed scientific program missing")
                p=ScientificProgramDraft.parse(read(chosen))
                entry["version"]="FEEDBACK_REVISION" if chosen==revision_file else "INITIAL_DEVELOPMENT"
                if p.idea_label!=name or p.parent_idea_id!=task["parent_idea_id"] or p.source_role!=task["role"]:
                    raise ValueError("saved scientific program identity mismatch")
                entry.update({"status":"DEVELOPED_SPECULATIVE", "program":p.to_dict(),
                              "challenge":challenge_program(p),"retention_brief":retention_brief(p,task)})
            except Exception as e:
                entry.update({"status":"RECEIPT_SAVED_PROGRAM_INVALID",
                              "validation_error":str(e)})
        rows.append(entry)
    return rows


def make_report(plan_path: Path) -> dict:
    plan=_read_plan(plan_path); root=plan_path.parent
    rows=_receipts(root,plan)
    report={"schema_version":"m7-scientific-development-results-v1",
            "status":"M7_SPECULATIVE_PROGRAMS_UNREVIEWED",
            "original_plan_sha":digest(plan),
            "programs_developed":sum(r["status"]=="DEVELOPED_SPECULATIVE" for r in rows),
            "programs_planned":len(rows),"rows":rows,
            "actual_experiment_executed":False,"novelty_certified":False,
            "scientific_truth_authority":False,
            "strict_realization_completed":False}
    return report


def report_cmd(args: argparse.Namespace) -> None:
    report=make_report(args.plan)
    # State-versioned immutable report; safe to call repeatedly.
    target=args.plan.parent / ("M7_PROGRAM_REPORT_"+digest(report)[:14]+".json")
    if not target.exists():
        write_new(target,report)
    feedback={"schema_version":"m7-feedback-to-next-generation-v1",
              "source_report_sha":digest(report),
              "search_only_not_positive_premise":True,
              "ideas":[row["challenge"] for row in report["rows"] if row["status"]=="DEVELOPED_SPECULATIVE"],
              "novelty_or_truth_authority":False}
    ft=args.plan.parent / ("M7_EVOLUTION_FEEDBACK_"+digest(feedback)[:14]+".json")
    if not ft.exists():write_new(ft,feedback)
    print(json.dumps({"status":report["status"],"programs_developed":report["programs_developed"],
                      "programs_planned":report["programs_planned"],"program_report":str(target),
                      "evolution_feedback":str(ft)},ensure_ascii=False,indent=2))


def _extract_json(text: str) -> dict:
    parsed=json.loads(text)
    if isinstance(parsed,dict) and "program" in parsed and isinstance(parsed["program"],dict):
        # Allow explicit envelope, never accept other envelope fields blindly.
        parsed=parsed["program"]
    if not isinstance(parsed,dict): raise ValueError("expected JSON object")
    return parsed


def develop_paid(args: argparse.Namespace) -> None:
    if not (args.allow_paid and args.i_authorize_new_llm_calls):
        raise ValueError("Paid calls require both explicit flags")
    if not 1<=args.max_new_calls<=2 or not 600<=args.max_output_tokens<=5000:
        raise ValueError("New call budget must be 1..2 and token limit 600..5000")
    plan=_read_plan(args.plan); root=args.plan.parent
    # Critical paid safety: do not move to later tasks while an earlier billed
    # receipt remains malformed. Existing provider response cannot be regenerated
    # or silently discarded; inspect/recover it first.
    invalid=[r["label"] for r in _receipts(root,plan)
             if r["status"]=="RECEIPT_SAVED_PROGRAM_INVALID"]
    if invalid:
        raise RuntimeError("Saved paid receipt(s) lack valid program(s): " + repr(invalid)
                           + ". Run m7_receipt_recovery inspect/recover-offline first."
                           + " NO NEW PROVIDER CALL MADE.")
    token=os.getenv(args.api_key_env)
    if not token:raise RuntimeError("missing configured provider key: "+args.api_key_env)
    from openai import OpenAI
    kwargs={"api_key":token,"max_retries":0,"timeout":180.0}
    if args.base_url:kwargs["base_url"]=args.base_url
    else:kwargs["base_url"]="https://openrouter.ai/api/v1"
    client=OpenAI(**kwargs)
    revision_mode=getattr(args,"revision_mode",False)
    made=0
    for task in plan["tasks"]:
        label=task["label"]
        target=root/("revision_receipts" if revision_mode else "receipts")/(label+".json")
        attempt=root/("revision_attempts" if revision_mode else "attempts")/(label+".json")
        if target.exists(): continue
        if attempt.exists():
            raise RuntimeError("Prior payment attempt exists without validated receipt; inspect before retry: "+str(attempt))
        if made>=args.max_new_calls:break
        messages=create_model_prompt(task)
        # The full scientific-program schema is relatively wide. Prior 3k-token
        # runs produced unclosed JSON strings at provider length limits. Keep
        # complete, substantive but compact outputs rather than losing science.
        messages[0]["content"] += ("\nOUTPUT BUDGET IS A STRICT ENGINEERING CONSTRAINT: "
            "return the FULL JSON object with EVERY required key, without fences or commentary, "
            "preferably within 2000 output tokens even when the API cap is higher. "
            "Use 1-3 informative sentences per long string; at most 2-3 concise entries per list; "
            "provide up to 2 symbolic equations; explicitly state conditions, null contrasts, "
            "observable independence and failure modes. Completeness and physics matter more "
            "than verbosity. Do not omit fields to shorten output.")
        if revision_mode:
            previous=root/"parsed"/(label+".json")
            if not previous.is_file():
                continue  # Revision requires an actually model-generated first program.
            old=ScientificProgramDraft.parse(read(previous))
            challenge=challenge_program(old)
            messages[1]["content"] += "\n\nFEEDBACK FOR REVISION (SEARCH CONTEXT ONLY): "+canonical({
                "earlier_program":old.to_dict(), "critical_feedback":challenge,
                "instruction":"Repair specific scientific non-identifiability and formal discriminators, not just add prose. Retain original mechanism where possible, or explain replacement. Answer full schema again."})
        write_new(attempt,{"label":label,"task_sha":digest(task),"model":args.model,
                           "attempted":True,"revision_mode":revision_mode,"request_hash":digest(messages),"automatic_retry":False})
        made+=1
        try:
            response=client.chat.completions.create(model=args.model,messages=messages,
                          temperature=0.2,max_tokens=args.max_output_tokens,
                          response_format={"type":"json_object"})
        except Exception as e:
            print(f"M7_PAID_PROVIDER_ERROR_AFTER_ATTEMPT {label}: {type(e).__name__}: {e}",file=sys.stderr)
            raise
        choice=response.choices[0]
        raw_text=choice.message.content or ""
        receipt={"label":label,"task_sha":digest(task),"model":args.model,
                 "provider_response_id":response.id,
                 "finish_reason":choice.finish_reason,
                 "raw_response":raw_text,"parsed_program":None,
                 "usage":response.usage.model_dump(mode="json") if response.usage else None,
                 "new_api_calls_for_this_receipt":1}
        # Persist original text BEFORE validating. No automatic paid retries.
        target.parent.mkdir(parents=True,exist_ok=True)
        write_new(target,receipt)
        try:
            scientific=ScientificProgramDraft.parse(_extract_json(raw_text))
            if scientific.idea_label!=label or scientific.parent_idea_id!=task["parent_idea_id"] or scientific.source_role!=task["role"]:
                raise ValueError("scientific program identity mismatch")
            receipt["parsed_program"]=scientific.to_dict()
            # Keep original raw in fixed receipt; parsed result is saved separately.
            write_new(root/("revisions" if revision_mode else "parsed")/(label+".json"),scientific.to_dict())
            print(json.dumps({"saved_paid_receipt":label,"finish_reason":choice.finish_reason,
                              "scientific_program_structurally_valid":True}))
        except Exception as e:
            print("M7_SAVED_RAW_RECEIPT_BUT_PROGRAM_INVALID: "+label+": "+str(e),file=sys.stderr)
            print("M7_RECEIPT_STATUS finish_reason="+str(choice.finish_reason)
                  +" output_chars="+str(len(raw_text))
                  +"; raw receipt preserved. Inspect with m7_receipt_recovery."
                  +" Do not repeat paid generation.",file=sys.stderr)
            raise
    print(json.dumps({"status":"M7_BOUNDED_MODEL_DEVELOPMENT_COMPLETED_OR_PARTIAL",
                      "new_provider_calls":made,"receipt_dir":str(root/"receipts"),
                      "parsed_dir":str(root/"parsed")},indent=2))


def load_developed(plan_path: Path) -> tuple[dict,dict[str,ScientificProgramDraft]]:
    plan=_read_plan(plan_path);root=plan_path.parent
    programs={}
    for task in plan["tasks"]:
        label=task["label"]
        new_path=root/"revisions"/(label+".json")
        path=new_path if new_path.is_file() else root/"parsed"/(label+".json")
        if not path.exists():continue
        p=ScientificProgramDraft.parse(read(path))
        if p.idea_label!=label or p.parent_idea_id!=task["parent_idea_id"] or p.source_role!=task["role"]:
            raise ValueError("invalid parsed program identity")
        programs[label]=p
    return plan,programs



def _native_row_field(row: Any, field: str) -> Any:
    """Access native SIS list[Any] rows, which deserialize as dictionaries."""
    if isinstance(row, dict):
        return row[field]
    return getattr(row, field)


def _native_target_view(*, plan_path: Path, raw_g5: Path, context_path: Path,
                        requested_labels: list[str] | None, max_ideas: int):
    """Opt-in derived realization VIEW; native source G5 and evidence context stay immutable.

    Target selection is scientific-experiment selection, NOT official SIS fertility.
    Any selected child must be native accepted, mapped by its exact kernel, and developed.
    """
    from pipeline_core.discovery.research_idea_epistemic_generational_evolution import EpistemicG4ExecutionReport
    from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
    from pipeline_core.discovery.research_idea_closed_generation_cycle import adapt_epistemic_execution_for_realization
    from pipeline_core.discovery.research_idea_offspring_execution import (
        OffspringExecutionReport, select_offspring_for_realization,
    )

    if not 1 <= max_ideas <= 2:
        raise ValueError("realization budget must be 1..2")
    current = EpistemicG4ExecutionReport.model_validate_json(raw_g5.read_text(encoding="utf-8"))
    if current.generation_index != 5 or current.genuine_child_count != 4:
        raise ValueError("Expected M6.7 native G5 four-child RAW generation")
    context = HypothesisContext.model_validate_json(context_path.read_text(encoding="utf-8"))
    if not current.offspring_nodes or any(
        _native_row_field(n, "source_context_id") != context.context_id or
        _native_row_field(n, "source_context_sha256") != context.context_sha256
        for n in current.offspring_nodes
    ):
        raise ValueError("Native G5 child and HypothesisContext identity mismatch")
    plan, programs = load_developed(plan_path)
    if not programs:
        raise ValueError("No developed M7 research programs")
    by_kernel = {}
    for t in plan["tasks"]:
        if t["label"] not in programs:
            continue
        kernel_digest = digest(t["original_kernel"])
        if t.get("kernel_sha") != kernel_digest:
            raise ValueError("M7 source kernel hash inconsistent: " + t["label"])
        if kernel_digest in by_kernel:
            raise ValueError("Ambiguous original kernel maps to multiple M7 tasks")
        by_kernel[kernel_digest] = programs[t["label"]]
    per_idea = {
        _native_row_field(n, "idea_id"): by_kernel[_native_row_field(n, "kernel_sha256")]
        for n in current.offspring_nodes if _native_row_field(n, "kernel_sha256") in by_kernel
    }
    if not per_idea:
        raise ValueError("No M7 developed program matches the RAW G5 child kernels")

    legacy = adapt_epistemic_execution_for_realization(current)
    all_native_eligible = {
        row.idea_id for row in legacy.semantic_records if row.accepted_for_realization
    }
    if requested_labels:
        if len(requested_labels) > max_ideas or len(requested_labels) != len(set(requested_labels)):
            raise ValueError("Target labels must be unique and fit within paid budget")
        matched = {p.idea_label: idea_id for idea_id, p in per_idea.items()}
        chosen = []
        for label in requested_labels:
            if label not in matched:
                raise ValueError("Target is not a developed native G5 child: " + label)
            idea_id = matched[label]
            if idea_id not in all_native_eligible:
                raise ValueError("Native child was not accepted for realization: " + label)
            chosen.append(idea_id)
        # Validate a derived view against the actual SIS legacy contract.
        # Nothing in original RAW execution, canonical graph, or official population changes.
        keep = set(chosen)
        payload = legacy.model_dump(mode="json")
        payload["offspring_nodes"] = [n for n in payload["offspring_nodes"] if n["idea_id"] in keep]
        payload["semantic_records"] = [s for s in payload["semantic_records"] if s["idea_id"] in keep]
        payload["raw_offspring_count"] = len(payload["offspring_nodes"])
        payload["accepted_for_realization_count"] = sum(
            r["accepted_for_realization"] for r in payload["semantic_records"]
        )
        view_digest = digest({"source_report_id": legacy.report_id, "target_ids": sorted(chosen)})
        payload["report_id"] = "m7_explicit_target_view:" + view_digest[:20]
        payload["report_sha256"] = view_digest
        target_view = OffspringExecutionReport.model_validate(payload)
        actual = select_offspring_for_realization(
            target_view, max_realizations=len(chosen), max_per_parent=2,
        )
        if set(actual) != keep or len(actual) != len(chosen):
            raise ValueError("Native selection from explicit target view differs from requested ideas")
        selected_ids = actual
        mode = "EXPLICIT_SCIENTIFIC_TARGETS_DERIVED_NATIVE_VIEW_NOT_OFFICIAL"
    else:
        target_view = legacy
        selected_ids = select_offspring_for_realization(
            legacy, max_realizations=max_ideas, max_per_parent=2,
        )
        mode = "NATIVE_DEFAULT_SELECTION"
    if not selected_ids or set(selected_ids) - set(per_idea):
        raise ValueError("Native-selected child lacks a valid developed M7 program")
    info = {
        "status": "M7_NATIVE_REALIZATION_PREFLIGHT_PASS_NO_PROVIDER_CALLS",
        "source_g5_report_id": current.report_id,
        "source_context_id": context.context_id,
        "source_context_sha256": context.context_sha256,
        "selection_mode": mode,
        "source_g5_sha256": sha_file(raw_g5),
        "source_context_file_sha256": sha_file(context_path),
        "selected": [
            {"idea_id": idea_id, "label": per_idea[idea_id].idea_label}
            for idea_id in selected_ids
        ],
        "max_new_paid_calls": len(selected_ids),
        "official_scheduler_executed": False,
        "positive_premise_from_m7_program": False,
    }
    return current, context, per_idea, target_view, selected_ids, info


def preview_native_realization(args: argparse.Namespace) -> None:
    *_, info = _native_target_view(
        plan_path=args.plan, raw_g5=args.raw_g5, context_path=args.context,
        requested_labels=args.target_labels, max_ideas=args.max_new_calls,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


def native_realize_paid(args: argparse.Namespace) -> None:
    """Optional native strict-grounding realization; no paid call before explicit approval.

    Uses the four RAW G5 children, NOT a fabricated production-generation file.
    """
    if not (args.allow_paid and args.i_authorize_new_llm_calls):
        raise ValueError("Native realization requires explicit paid consent")
    if not 1<=args.max_new_calls<=2:
        raise ValueError("Native realization can call 1..2 logical generations per invocation")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("native realization output must be new and empty")
    if not os.getenv(args.api_key_env):
        raise RuntimeError("missing paid provider key")
    # This preflight does all native/context/selection checks before any paid API call.
    current, context, per_idea, legacy, selected_ids, preflight = _native_target_view(
        plan_path=args.plan, raw_g5=args.raw_g5, context_path=args.context,
        requested_labels=args.target_labels, max_ideas=args.max_new_calls,
    )
    # Imports only after all no-cost input checks.
    import instructor
    from openai import OpenAI
    from pipeline_core.discovery.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
    from pipeline_core.discovery.hypothesis_prompt import HypothesisPrompt
    from pipeline_core.discovery.research_idea_closed_loop import run_realization_lifecycle

    class M7StrictBackend:
        backend_name="m7_native_bounded_strict_hypothesis_maker"
        model_name=args.model
        def __init__(self):
            self.calls=0
            self.base=InstructorOpenAICompatibleHypothesisBackend(
                model=args.model,api_key_env=args.api_key_env,
                base_url=args.base_url or "https://openrouter.ai/api/v1",
                temperature=0.0,parse_retries=0,timeout=180.0,
                telemetry_path=args.out_dir/"native_realization.telemetry.jsonl",
                telemetry_context={"pipeline":"M7_OPT_IN_STRICT_REALIZATION"})
            # Avoid SDK retry (one logical call = one transport attempt).
            client=OpenAI(api_key=os.getenv(args.api_key_env),
                    base_url=args.base_url or "https://openrouter.ai/api/v1",
                    timeout=180,max_retries=0)
            self.base._client=instructor.from_openai(client,mode=instructor.Mode.JSON)
        def generate(self,prompt):
            if self.calls>=args.max_new_calls:
                raise RuntimeError("M7_PAID_REALIZATION_BUDGET_EXHAUSTED")
            self.calls+=1
            # The prompt is a valid native hypothesis prompt. Optional M7 development
            # is NOT eligible positive evidence and must not be injected as fact.
            attached = []
            for idea_id in selected_ids:
                if idea_id in prompt.user_prompt:
                    p=per_idea[idea_id]
                    attached.append({"idea_id":idea_id,
                                     "research_question_search_only":p.research_question,
                                     "differential_contrast_search_only":p.predicted_contrast,
                                     "possible_observables_search_only":p.independent_observables})
            if attached:
                user=prompt.user_prompt + "\n\nM7 SPECULATIVE SEARCH CONTEXT (NOT POSITIVE PREMISE, NO FACTUAL AUTHORITY):\n" + json.dumps(attached,ensure_ascii=False)
                prompt=HypothesisPrompt.create(system_prompt=prompt.system_prompt,user_prompt=user)
            return self.base.generate(prompt)
        def repair(self,prompt,previous_draft,feedback):
            raise RuntimeError("M7_REPAIRS_DISABLED_TO_BOUND_PAID_CALLS")

    args.out_dir.mkdir(parents=True,exist_ok=True)
    # Precall immutable intent: provider failure will NOT silently retry.
    write_new(args.out_dir/"M7_REALIZATION_AUTHORIZATION_RECEIPT.json",{
        "official_G5_scheduler_executed":False,"source_raw_g5_file_sha":sha_file(args.raw_g5),
        "source_context_file_sha":sha_file(args.context),"model":args.model,
        "max_logical_generation_calls":args.max_new_calls,
        "new_api_calls_claimed_before_execution":False,
        "speculative_context_not_positive_premise":True,
        "realization_preflight":preflight,
        "explicit_targeting_not_official_scheduler":bool(args.target_labels)})
    backend=M7StrictBackend()
    # Raw four-child execution ensures only newly generated G5 children are eligible.
    lifecycle, portfolio=run_realization_lifecycle(
        execution=legacy,context=context,backend=backend,
        output_dir=args.out_dir/"realization",
        max_ideas=len(selected_ids),max_realizations_per_idea=1,
        max_per_parent=2,max_repair_attempts=0,
        max_prospective_audits=0)
    write_new(args.out_dir/"M7_NATIVE_REALIZATION_LIFECYCLE.json",lifecycle.model_dump(mode="json"))
    write_new(args.out_dir/"M7_NATIVE_STRICT_HYPOTHESES.json",portfolio.model_dump(mode="json"))
    # Scientific differentiator retention stays a REVIEW QUEUE, never auto-certified.
    # Unmaterialized research ideas are preserved in the search context.
    reported_cards={c.hypothesis_id:c for c in portfolio.hypotheses}
    retention_rows=[]
    evolution_rows=[]
    for node in current.offspring_nodes:
        program=per_idea.get(_native_row_field(node, "idea_id"))
        if program is None: continue
        links=[link for link in lifecycle.links if link.idea_id==_native_row_field(node, "idea_id")]
        observations=[obs for obs in lifecycle.observations if obs.idea_id==_native_row_field(node, "idea_id")]
        materialized=[reported_cards[link.hypothesis_id] for link in links
                       if link.hypothesis_id in reported_cards]
        record={
            "idea_id":_native_row_field(node, "idea_id"),"source_program_label":program.idea_label,
            "program_status":"MODEL_SPECULATIVE_NOT_GROUNDED_EVIDENCE",
            "original_scientific_differentiator":program.predicted_contrast,
            "original_independent_observables":program.independent_observables,
            "materialized_hypothesis_cards":[x.model_dump(mode="json") for x in materialized],
            "realization_statuses":[x.materialization_status for x in links],
            "retention_verdict":"REVIEW_REQUIRED_NOT_AUTOMATIC",
            "non_materialization_is_not_idea_falsification":True,
            "scientific_truth_authority":False,
        }
        retention_rows.append(record)
        evolution_rows.append({
            "source_idea_id":_native_row_field(node, "idea_id"),
            "source_program_label":program.idea_label,
            "realization_observation_search_context_only":[x.model_dump(mode="json") for x in observations],
            "science_development_failure_modes":program.predicted_failure_modes,
            "next_search_prompt":"Develop an alternative research program from this unverified idea. Preserve "
                "any independently identifiable mechanism and explicitly improve or replace any "
                "unrealized contrast. Do not treat unavailable evidence, failed realization, or prior-art "
                "search gaps as scientific refutation or verification. Original differentiator: "
                +program.predicted_contrast,
            "not_positive_evidence":True,"production_authority":False
        })
    write_new(args.out_dir/"M7_RETENTION_REVIEW_QUEUE.json",{
        "rows":retention_rows,"automatic_scientific_retention_verdict":False,
        "source_g5_generation_index":5,"novelty_certified":False})
    write_new(args.out_dir/"M7_REALIZATION_TO_EVOLUTION_FEEDBACK.json",{
        "schema_version":"m7-native-realization-to-evolution-feedback-v1",
        "rows":evolution_rows,"evidence_positive_premise_authority":False,
        "official_G6_generation_executed":False,"new_model_calls_for_feedback":0})
    write_new(args.out_dir/"M7_NATIVE_REALIZATION_SUMMARY.json",{
        "status":"M7_NATIVE_STRICT_REALIZATION_COMPLETED_UNREVIEWED",
        "logical_model_calls":backend.calls,"materialized_hypotheses":len(portfolio.hypotheses),
        "realization_report_id":lifecycle.report_id,
        "grounded_differentiator_retention_scientifically_reviewed":False,
        "realized_target_labels":[per_idea[i].idea_label for i in selected_ids],
        "selection_mode":preflight["selection_mode"],
        "official_SIS_scheduler_executed":False,"novelty_certified":False,
        "production_population_modified":False})
    print(json.dumps({"status":"M7_NATIVE_STRICT_REALIZATION_COMPLETED_UNREVIEWED",
         "logical_calls":backend.calls,"materialized_hypotheses":len(portfolio.hypotheses),
         "out_dir":str(args.out_dir)},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("prepare")
    a.add_argument("--m64",type=Path,required=True)
    a.add_argument("--m67",type=Path,required=True)
    a.add_argument("--out-dir",type=Path,required=True)
    a.add_argument("--limit",type=int,default=6)
    b=sub.add_parser("develop-paid")
    b.add_argument("--plan",type=Path,required=True)
    b.add_argument("--model",default="openai/gpt-5.6-luna")
    b.add_argument("--api-key-env",default="OPENROUTER_API_KEY")
    b.add_argument("--base-url",default=None)
    b.add_argument("--max-new-calls",type=int,default=2)
    b.add_argument("--max-output-tokens",type=int,default=3000)
    b.add_argument("--allow-paid",action="store_true")
    b.add_argument("--i-authorize-new-llm-calls",action="store_true")
    r=sub.add_parser("revise-paid")
    r.add_argument("--plan",type=Path,required=True)
    r.add_argument("--model",default="openai/gpt-5.6-luna")
    r.add_argument("--api-key-env",default="OPENROUTER_API_KEY")
    r.add_argument("--base-url",default=None)
    r.add_argument("--max-new-calls",type=int,default=2)
    r.add_argument("--max-output-tokens",type=int,default=3000)
    r.add_argument("--allow-paid",action="store_true")
    r.add_argument("--i-authorize-new-llm-calls",action="store_true")
    r.set_defaults(revision_mode=True)
    c=sub.add_parser("report")
    c.add_argument("--plan",type=Path,required=True)
    e=sub.add_parser("preview-realization")
    e.add_argument("--plan",type=Path,required=True)
    e.add_argument("--raw-g5",type=Path,required=True)
    e.add_argument("--context",type=Path,required=True)
    e.add_argument("--max-new-calls",type=int,default=2)
    e.add_argument("--target-labels",nargs="+",default=None)
    d=sub.add_parser("realize-paid")
    d.add_argument("--plan",type=Path,required=True)
    d.add_argument("--raw-g5",type=Path,required=True)
    d.add_argument("--context",type=Path,required=True)
    d.add_argument("--out-dir",type=Path,required=True)
    d.add_argument("--model",default="openai/gpt-5.6-luna")
    d.add_argument("--api-key-env",default="OPENROUTER_API_KEY")
    d.add_argument("--base-url",default=None)
    d.add_argument("--max-new-calls",type=int,default=2)
    d.add_argument("--target-labels",nargs="+",default=None)
    d.add_argument("--allow-paid",action="store_true")
    d.add_argument("--i-authorize-new-llm-calls",action="store_true")
    args=p.parse_args()
    if args.command=="prepare":prepare(args)
    elif args.command=="develop-paid":develop_paid(args)
    elif args.command=="revise-paid":develop_paid(args)
    elif args.command=="report":report_cmd(args)
    elif args.command=="preview-realization":preview_native_realization(args)
    elif args.command=="realize-paid":native_realize_paid(args)

if __name__=="__main__":main()
