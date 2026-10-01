
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any

def _by_h(rows):
    out={}
    for r in rows:
        h=str(r.get("hypothesis_id") or "")
        if h: out.setdefault(h,[]).append(r)
    return out

def _find_hypotheses(obj):
    out={}
    def rec(x):
        if isinstance(x,dict):
            h=x.get("hypothesis_id") or x.get("id")
            if isinstance(h,str) and h.startswith("hypothesis:"):
                out.setdefault(h,x)
            for v in x.values(): rec(v)
        elif isinstance(x,list):
            for v in x: rec(v)
    rec(obj); return out

def _display(x):
    if not x: return None
    for k in ("title","hypothesis","hypothesis_text","statement","text","summary","name"):
        v=x.get(k)
        if isinstance(v,str) and v.strip(): return v.strip()
    return None

def _state(rows):
    ds={str(r.get("aggregation_disposition") or "") for r in rows}
    if "HOLD_FOR_TOPOLOGY" in ds: return "UNRESOLVED_TOPOLOGY_GAP"
    if "HOLD_FOR_BASE_EVIDENCE" in ds: return "UNRESOLVED_EVIDENCE_GAP"
    if "RESIDUAL_CANDIDATE_SHADOW" in ds: return "RESIDUAL_AUTHORITY_CANDIDATE_SHADOW"
    if "NO_RESIDUAL" in ds: return "PRIOR_ART_BACKED_OR_NO_RESIDUAL"
    return "UNCLASSIFIED"

def build_closeout(portfolio, agg_a, agg_b, readiness, selection_audit, stability, source_binding, sentinel):
    a=_by_h(agg_a.get("composites",[])); b=_by_h(agg_b.get("composites",[]))
    objs=_find_hypotheses(portfolio)
    src={}
    for r in source_binding.get("records",[]):
        h=str(r.get("hypothesis_id") or "")
        if h: src[h]=src.get(h,0)+1
    ready={str(r.get("hypothesis_id")):r for r in readiness.get("rows",[]) if r.get("hypothesis_id")}
    ids=sorted(set(a)|set(b)|set(src)|set(objs)|set(ready))
    rows=[]
    for h in ids:
        ra=a.get(h,[]); rb=b.get(h,[])
        total=sum(len(r.get("aggregated_component_claim_ids",[])) for r in ra)
        backed=sum(len(r.get("aggregated_relation_backed_component_claim_ids",[])) for r in ra)
        sig=lambda rs:[(r.get("claim_id"),r.get("aggregation_disposition"),r.get("aggregated_residual_state")) for r in rs]
        rr=ready.get(h,{})
        rows.append({
            "hypothesis_id":h,
            "display_text":_display(objs.get(h)),
            "source_bound_claim_count":src.get(h,0),
            "composite_count":len(ra),
            "composite_claim_ids":[r.get("claim_id") for r in ra],
            "full_relation_statuses":sorted({str(r.get("full_relation_status")) for r in ra if r.get("full_relation_status")}),
            "topology_states":sorted({str(r.get("topology_state")) for r in ra if r.get("topology_state")}),
            "component_backed_count":backed,
            "component_count":total,
            "evidence_depths":sorted({str(r.get("evidence_depth")) for r in ra if r.get("evidence_depth")}),
            "residual_states":sorted({str(r.get("aggregated_residual_state")) for r in ra if r.get("aggregated_residual_state")}),
            "final_epistemic_state":_state(ra),
            "ab_stable":sig(ra)==sig(rb),
            "authority_ready_candidate_shadow":bool(rr.get("authority_ready_candidate_shadow")),
            "authority_readiness_state":rr.get("state"),
        })
    counts={}
    for r in rows: counts[r["final_epistemic_state"]]=counts.get(r["final_epistemic_state"],0)+1
    return {
        "schema_version":"sers-closeout-freeze-v1",
        "scope":"sers_enhancement_reproducibility_tradeoff",
        "shadow_only":True,
        "selection_completeness_pass":bool(selection_audit.get("pass")),
        "review_stability_pass":bool(stability.get("authority_relevant_stable")),
        "diagnostic_sentinel":{
            "claim_id":sentinel.get("claim_id"),"doi":sentinel.get("doi"),
            "relationship":sentinel.get("relationship"),"confidence":sentinel.get("confidence"),
            "aggregation_eligible":bool(sentinel.get("aggregation_eligible",False)),
        },
        "state_counts":dict(sorted(counts.items())),
        "authority_ready_shadow_count":sum(r["authority_ready_candidate_shadow"] for r in rows),
        "hypotheses":rows,
        "novelty_authority_created":False,"n9_authority_created":False,
        "n10_authority_created":False,"production_selection_changed":False,
    }

def build_n9_projection(closeout):
    projected=[]; held=[]
    for r in closeout.get("hypotheses",[]):
        x={
            "hypothesis_id":r.get("hypothesis_id"),
            "final_epistemic_state":r.get("final_epistemic_state"),
            "authority_readiness_state":r.get("authority_readiness_state"),
            "component_backed_count":r.get("component_backed_count"),
            "component_count":r.get("component_count"),
            "residual_states":r.get("residual_states",[]),
            "projection_is_certification":False,
            "projection_is_novelty_proof":False,
        }
        if r.get("authority_ready_candidate_shadow"):
            x["projection_disposition"]="PROJECT_TO_N9_SHADOW"; projected.append(x)
        else:
            x["projection_disposition"]="HOLD_BEFORE_N9"; held.append(x)
    return {
        "schema_version":"residual-aware-n9-shadow-projection-v1","scope":"sers_only",
        "shadow_only":True,"projected_count":len(projected),"held_count":len(held),
        "projected":projected,"held":held,
        "novelty_authority_created":False,"n9_authority_created":False,
        "n10_authority_created":False,"production_selection_changed":False,
    }

def compare_existing_certification(projection, root:Path):
    pats=[re.compile(x,re.I) for x in (r"(^|[._-])n9([._-]|$)",r"scientific[_-]?verifier",r"certification",r"prospective[_-]?novelty")]
    artifacts=[]
    for p in sorted(root.rglob("*.json")):
        if not any(rx.search(p.name) for rx in pats): continue
        artifacts.append({"path":str(p),"name":p.name,"size_bytes":p.stat().st_size})
    ids={str(r.get("hypothesis_id")) for r in projection.get("projected",[]) if r.get("hypothesis_id")}
    overlaps=[]
    for a in artifacts:
        try: raw=Path(a["path"]).read_text(encoding="utf-8",errors="replace")
        except Exception: continue
        hit=sorted(h for h in ids if h in raw)
        if hit: overlaps.append({**a,"projected_hypothesis_overlap":hit,"comparison_level":"identifier_presence_only"})
    state=("BASELINE_ARTIFACTS_FOUND_IDENTIFIER_COMPARISON_ONLY" if overlaps else
           "BASELINE_ARTIFACTS_FOUND_NO_PROJECTED_ID_OVERLAP" if artifacts else
           "N9_BASELINE_UNAVAILABLE")
    return {
        "schema_version":"residual-aware-n9-baseline-comparison-v1","shadow_only":True,
        "comparison_state":state,"search_root":str(root),
        "candidate_artifact_count":len(artifacts),"overlapping_artifact_count":len(overlaps),
        "candidate_artifacts":artifacts[:100],"comparisons":overlaps[:100],
        "note":"No unknown local N9 runtime is invoked; persisted certification-like artifacts are inspected for lineage overlap only.",
        "n9_authority_created":False,"production_selection_changed":False,
    }

def render_md(report):
    L=["# SERS Closeout Report","",
       "Scope: Au/Ag plasmonic SERS enhancement–reproducibility trade-off.","",
       "## Global gates","",
       f"- Selection completeness: `{report['selection_completeness_pass']}`",
       f"- A/B review stability: `{report['review_stability_pass']}`",
       f"- Authority-ready shadow candidates: `{report['authority_ready_shadow_count']}`",
       f"- Sentinel: `{report['diagnostic_sentinel'].get('relationship')}` for `{report['diagnostic_sentinel'].get('doi')}`",
       "- Memory is not evidence.","- Diagnostic sentinel is not aggregation-eligible.",
       "- No novelty/N9/N10/production authority was created.","",
       "## Canonical SERS freeze","",
       "| Hypothesis | Final state | Components | Topology | Full relation | Readiness |",
       "|---|---|---:|---|---|---|"]
    for r in report.get("hypotheses",[]):
        label=r["hypothesis_id"]
        if r.get("display_text"):
            t=re.sub(r"\s+"," ",r["display_text"]).strip()
            if len(t)>80:t=t[:77]+"..."
            label+=f"<br>{t}"
        readiness=("AUTHORITY_READY_CANDIDATE_SHADOW" if r["authority_ready_candidate_shadow"] else str(r.get("authority_readiness_state") or "—"))
        L.append("| "+" | ".join([
            label.replace("|","\\|"),r["final_epistemic_state"],
            f'{r["component_backed_count"]}/{r["component_count"]}',
            ", ".join(r["topology_states"]) or "—",
            ", ".join(r["full_relation_statuses"]) or "—",readiness])+" |")
    L += ["","## Freeze semantics","",
          "- `PRIOR_ART_BACKED_OR_NO_RESIDUAL`: no residual higher-order novelty survives the current evidence accounting.",
          "- `RESIDUAL_AUTHORITY_CANDIDATE_SHADOW`: eligible only for shadow certification projection; not certified novelty.",
          "- `UNRESOLVED_EVIDENCE_GAP`: bounded SERS evidence accounting did not back every required lower-order component.",
          "- `UNRESOLVED_TOPOLOGY_GAP`: source decomposition does not justify component topology, so residual novelty is not assessed.",
          "","This closes the current SERS case only; cross-domain generalization is intentionally deferred.",""]
    return "\n".join(L)
