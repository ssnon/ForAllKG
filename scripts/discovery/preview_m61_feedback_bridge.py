"""Offline preview of M6.1 prompt injection; zero API calls and no SIS generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import EpistemicG4Prompt
from pipeline_core.discovery.research_idea_m61_feedback_bridge import (
    M61FeedbackPromptAdapter, load_m61_feedback_bundle,
)

class PreviewBackend:
    def __init__(self):
        self.prompts = []
    def generate(self, prompt):
        self.prompts.append(prompt)
        return {"preview_only": True}
    def repair(self, prompt, previous_draft, feedback):
        self.prompts.append(prompt)
        return {"preview_only": True}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--feedback-hints", type=Path, required=True)
    a.add_argument("--feedback-population", type=Path, required=True)
    a.add_argument("--selected-parent-id", action="append", default=[])
    a.add_argument("--generation-plan", type=Path, help="Optional existing native generation plan JSON")
    a.add_argument("--output", type=Path, required=True)
    args=a.parse_args()
    bundle=load_m61_feedback_bundle(args.feedback_hints,args.feedback_population)
    if args.generation_plan:
        plan=json.loads(args.generation_plan.read_text(encoding="utf-8"))
        selected=plan["selected_parent_idea_ids"]
    elif args.selected_parent_id:
        selected=args.selected_parent_id
    else:
        selected=sorted(bundle.by_parent)
    backend=PreviewBackend()
    mapping={f"offline-preview-task-{i}": parent for i,parent in enumerate(selected)}
    adapter=M61FeedbackPromptAdapter(backend=backend,bundle=bundle,task_to_idea=mapping)
    previews=[]
    for tid,parent in mapping.items():
        prompt=EpistemicG4Prompt(task_id=tid,system_prompt="SIS generation prompt placeholder, preview only",user_prompt=json.dumps({"primary_parent_idea_id":parent},ensure_ascii=False),prompt_sha256="preview-not-authentic")
        adapter.generate(prompt)
        used=backend.prompts[-1]
        parsed=json.loads(used.user_prompt)
        ctx=parsed.get("m61_adversarial_development_search_context_only")
        previews.append({"parent_idea_id":parent,"feedback_injected":bool(ctx),"prompt_sha256":used.prompt_sha256,"feedback_context":ctx})
    out={"status":"M62_OFFLINE_PROMPT_BRIDGE_PREVIEW_NO_GENERATION", "preview_is_not_native_g5":True,
         "source_report":bundle.selection_report(selected),"previews":previews,"new_api_calls":0}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({"status":out["status"],"matched_parent_count":out["source_report"]["hint_matched_parent_count"],"hint_matched_count":out["source_report"]["hint_matched_count"],"output":str(args.output),"new_api_calls":0},indent=2))

if __name__=="__main__": main()
