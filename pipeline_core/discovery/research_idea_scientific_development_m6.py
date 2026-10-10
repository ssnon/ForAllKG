"""M6 opt-in scientific-development loop, read-only to ResearchIdea.

No authority to select, delete, mutate, certify, or ground ResearchIdea nodes.
Automated checks validate *specification completeness*, never physical truth.
Import-free with respect to the existing SIS runtime: compatible with Python 3.11+.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "m6-scientific-development-shadow-v1"
ALLOWED_FIELDS = frozenset({
    "parent_idea_id", "local_id", "research_question", "mechanistic_hypothesis",
    "restricted_null", "known_physics_envelope", "independent_variables",
    "observable", "quantitative_or_ordinal_prediction", "assumptions",
    "controls", "intervention", "heldout_test", "local_falsifier",
    "surviving_uncertainty", "development_value", "novelty_certified",
    "scientific_truth_authority", "source_of_proposal",
})
REQUIRED_TEXT = ("parent_idea_id", "local_id", "research_question", "mechanistic_hypothesis",
                 "restricted_null", "known_physics_envelope", "observable",
                 "quantitative_or_ordinal_prediction", "intervention", "heldout_test",
                 "local_falsifier", "surviving_uncertainty", "development_value")
REQUIRED_LIST = ("independent_variables", "assumptions", "controls")


def canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(obj: Any) -> str:
    return hashlib.sha256(canon(obj)).hexdigest()


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def save_new(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:  # refuse overwrite
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def check_not_under(target: str | Path, input_files: list[str | Path]) -> None:
    out = Path(target).resolve()
    for filename in input_files:
        src = Path(filename).resolve()
        if out == src or out in src.parents or src in out.parents:
            raise ValueError("output must be separated from frozen source paths")


def validate_origin(packet: dict, audit: dict) -> tuple[dict, dict]:
    if (packet.get("status") != "NATIVE_SOURCE_EXCERPTS_UNREVIEWED" or
            not isinstance(packet.get("records"), list) or
            not packet["records"] or packet.get("idea_count") != len(packet["records"])):
        raise ValueError("requires complete frozen native source packet")
    if audit.get("status") != "SOURCE_EXCERPTS_ASSESSED_UNCERTIFIED":
        raise ValueError("requires M5-F uncertified analyst assessment")
    sources = packet["records"]
    assesses = audit["records"]
    sid = [x["idea_id"] for x in sources]
    aid = [x["idea_id"] for x in assesses]
    if len(sid) != len(set(sid)) or len(aid) != len(set(aid)) or set(sid) != set(aid):
        raise ValueError("non-bijective packet/assessment idea IDs")
    if any(x.get("scientific_novelty_certified") or not x.get("idea_preserved") for x in assesses):
        raise ValueError("source assessments claim prohibited authority")
    for x in sources:
        a = next(y for y in assesses if y["idea_id"] == x["idea_id"])
        if a["source_differential_prediction"] != x["differential_prediction"]:
            raise ValueError(f"source/assessment prediction mismatch {x['idea_id']}")
        if a["source_discriminating_observation"] != x["discriminating_observation"]:
            raise ValueError(f"source/assessment observation mismatch {x['idea_id']}")
        if a["source_falsification_condition"] != x["falsification_condition"]:
            raise ValueError(f"source/assessment falsifier mismatch {x['idea_id']}")
        if a["parent_idea_id"] not in x["parent_idea_ids"]:
            raise ValueError(f"parent mismatch {x['idea_id']}")
    return {x["idea_id"]: x for x in sources}, {a["idea_id"]: a for a in assesses}


def verify_native_sources(packet_path: str | Path, raw_g3: str | Path, raw_g4: str | Path) -> dict:
    """Optional local frozen-SIS link check, without modifying any input."""
    packet = load(packet_path)
    for gen, source in ((3, raw_g3), (4, raw_g4)):
        tag = f"G{gen}"
        if file_sha(source) != packet["source_shas"][tag]:
            raise ValueError(f"{tag} raw generation byte SHA mismatch")
        raw = load(source)
        nodes = {n["idea_id"]: n for n in raw["offspring_nodes"]}
        if len(nodes) != len(raw["offspring_nodes"]):
            raise ValueError(f"{tag} raw duplicate offspring nodes")
        sem = {x["idea_id"]: x for x in raw["semantic_records"]}
        if len(sem) != len(raw["semantic_records"]):
            raise ValueError(f"{tag} raw duplicate semantic records")
        for r in packet["records"]:
            if r["generation"] != gen:
                continue
            n = nodes.get(r["idea_id"])
            t = sem.get(r["idea_id"])
            if n is None or t is None:
                raise ValueError(f"{tag} source idea absent from raw")
            for f in ("kernel", "differential_prediction", "discriminating_observation", "falsification_condition", "parent_idea_ids"):
                if r[f] != n[f]:
                    raise ValueError(f"{tag} source mismatch in {f}: {r['idea_id']}")
            if r["conceptual_change_summary"] != t["conceptual_change_summary"]:
                raise ValueError(f"{tag} semantic summary mismatch: {r['idea_id']}")
    return {"status": "M6_NATIVE_G3_G4_SOURCE_VERIFIED", "original_inputs_modified": False,
            "source_packet_sha256": file_sha(packet_path),
            "raw_G3_sha256": file_sha(raw_g3), "raw_G4_sha256": file_sha(raw_g4)}


def prepare(packet_path: str | Path, assessment_path: str | Path, output: str | Path,
            select_idea_ids: list[str] | None = None) -> dict:
    check_not_under(output, [packet_path, assessment_path])
    packet, audit = load(packet_path), load(assessment_path)
    if audit.get("source_packet_sha256") != file_sha(packet_path):
        raise ValueError("M5-F assessment provenance source_packet_sha256 mismatch")
    sources, assessments = validate_origin(packet, audit)
    ids = select_idea_ids if select_idea_ids is not None else sorted(sources)
    if not ids or len(ids) != len(set(ids)) or any(i not in sources for i in ids):
        raise ValueError("selection must be distinct, nonempty subset of frozen ideas")
    tasks = []
    for iid in ids:
        s, a = sources[iid], assessments[iid]
        payload = {
            "schema_version": SCHEMA,
            "parent_idea_id": iid,
            "source_excerpt_sha256": sha(s),
            "audit_assessment_sha256": sha(a),
            "idea": {"generation": s["generation"], "kernel": s["kernel"],
                     "differential_prediction": s["differential_prediction"],
                     "discriminating_observation": s["discriminating_observation"],
                     "falsification_condition": s["falsification_condition"],
                     "conceptual_change_summary": s["conceptual_change_summary"]},
            "null_challenge": {
                "restricted_M0": a["restricted_orientation_only_null_M0"],
                "known_M1": a["known_physics_compatibility_envelope_M1"],
                "counterexample": a["mathematical_or_causal_counterexample"],
                "identifiability_controls": a["missing_independent_controls"],
                "bridge_to_test": a["minimum_conditionally_discriminating_bridge"],
                "falsifier_limit": a["falsifier_validity_scope"],
                "caution": a["scientific_caution"],
                "analysis_origin": "ANALYST_AUTHORED_M5F_UNCERTIFIED",
            },
        }
        payload["task_id"] = "m6_task:" + sha(payload)[:20]
        tasks.append(payload)
    out = {
        "schema_version": SCHEMA, "status": "M6_CHALLENGE_PREPARED_NO_MODEL_CALLS",
        "provenance": {"source_sha256": file_sha(packet_path),
                       "assessment_sha256": file_sha(assessment_path),
                       "raw_G3_G4_sha256_CLAIMED_NOT_RECHECKED": packet["source_shas"]},
        "tasks": tasks,
        "authority": {"scientific_truth_authority": False, "novelty_certified": False,
                      "production_selection_authority": False, "original_ideas_modified": False},
    }
    save_new(Path(output) / "M6_PREPARED_TASKS.json", out)
    return out


def validate_proposal(p: dict, parents: set[str]) -> None:
    if not isinstance(p, dict) or set(p) - ALLOWED_FIELDS:
        raise ValueError("proposal has unknown fields")
    if any(not isinstance(p.get(k), str) or not p[k].strip() for k in REQUIRED_TEXT):
        raise ValueError("missing required nonblank text")
    if any(not isinstance(p.get(k), list) or not p[k] or not all(isinstance(v, str) and v.strip() for v in p[k]) for k in REQUIRED_LIST):
        raise ValueError("missing nonblank list")
    if p["parent_idea_id"] not in parents:
        raise ValueError("unregistered source parent")
    if p.get("scientific_truth_authority") is not False or p.get("novelty_certified") is not False:
        raise ValueError("must explicitly deny truth/novelty authority")
    if p.get("source_of_proposal") not in {"ANALYST_SEED", "MODEL_GENERATED", "MODEL_REVISED", "EXTERNAL_REVIEWER"}:
        raise ValueError("unregistered proposal provenance")


def validate_drafts(prepared: dict, draft_doc: dict) -> list[dict]:
    if draft_doc.get("schema_version") != SCHEMA or not isinstance(draft_doc.get("proposals"), list):
        raise ValueError("bad proposal envelope")
    parents = {t["parent_idea_id"] for t in prepared["tasks"]}
    proposals = draft_doc["proposals"]
    if not proposals:
        raise ValueError("no proposals")
    keys = []
    for p in proposals:
        validate_proposal(p, parents)
        keys.append((p["parent_idea_id"], p["local_id"]))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicated parent/local proposal")
    if not parents.issubset({p["parent_idea_id"] for p in proposals}):
        raise ValueError("some selected parent ideas lack development proposals")
    return proposals


def critique(p: dict, task: dict) -> dict:
    """Deterministic design critique, NOT a scientific judge or novelty classifier."""
    defects = []
    # Syntactic checks cannot establish physical validity; all proposals remain unverified.
    if len(p["independent_variables"]) < 2:
        defects.append("NEEDS_MULTIVARIABLE_CONTRAST")
    if len(p["controls"]) < 3:
        defects.append("NEEDS_MULTIPLE_INDEPENDENT_CONTROLS")
    if len(p["assumptions"]) < 2:
        defects.append("NEEDS_EXPLICIT_NULL_ASSUMPTIONS")
    if len(p["quantitative_or_ordinal_prediction"]) < 40:
        defects.append("PREDICTION_TOO_VAGUE_TO_REVIEW")
    if len(p["heldout_test"]) < 40:
        defects.append("HOLDOUT_TOO_VAGUE_TO_REVIEW")
    if len(p["local_falsifier"]) < 45:
        defects.append("FALSIFIER_SCOPE_NEEDS_EXPLICIT_LIMIT")
    # The null challenge is always attached, never overridden by a generated answer.
    return {"parent_idea_id": p["parent_idea_id"], "local_id": p["local_id"],
            "review_status": "STRUCTURALLY_SPECIFIED_SCIENCE_UNREVIEWED" if not defects else "REVISION_RECOMMENDED_SCIENCE_UNREVIEWED",
            "specification_defects": defects,
            "known_physics_counterexample": task["null_challenge"]["counterexample"],
            "known_physics_null_not_rejected": True,
            "manual_scientific_review_required": True,
            "revision_request": "Make conditional null, observable, independent controls and heldout test more decisive without inventing evidence. Preserve parent. " + "; ".join(defects),
            "scientific_novelty_certified": False}


def develop(prepared_path: str | Path, drafts_path: str | Path, output: str | Path,
            revision_path: str | Path | None = None) -> dict:
    inputs = [prepared_path, drafts_path] + ([revision_path] if revision_path else [])
    check_not_under(output, inputs)
    prepared = load(prepared_path)
    if prepared.get("schema_version") != SCHEMA or prepared.get("status") != "M6_CHALLENGE_PREPARED_NO_MODEL_CALLS":
        raise ValueError("incorrect prepared task contract")
    tasks = {x["parent_idea_id"]: x for x in prepared["tasks"]}
    proposals = validate_drafts(prepared, load(drafts_path))
    revisions = []
    if revision_path:
        revisions_doc = load(revision_path)
        if revisions_doc.get("schema_version") != SCHEMA or not isinstance(revisions_doc.get("proposals"), list):
            raise ValueError("bad revision envelope")
        revisions = revisions_doc["proposals"]
        allowed = {(p["parent_idea_id"], p["local_id"]) for p in proposals}
        seen = set()
        for rev in revisions:
            validate_proposal(rev, set(tasks))
            key = (rev["parent_idea_id"], rev["local_id"])
            if key not in allowed or key in seen:
                raise ValueError("revision may only update existing branch, never add/delete")
            seen.add(key)
    bykey = {(p["parent_idea_id"], p["local_id"]): p for p in revisions}
    rows=[]
    for p in proposals:
        t = tasks[p["parent_idea_id"]]
        before = critique(p,t)
        rev = bykey.get((p["parent_idea_id"], p["local_id"]))
        after = critique(rev,t) if rev else None
        branch = {"branch_id": "m6_shadow_branch:"+sha([t["task_id"],p["local_id"]])[:20],
                  "parent_idea_id": p["parent_idea_id"], "source_generation": t["idea"]["generation"],
                  "task_id": t["task_id"],
                  "source_idea_excerpt_sha256": t["source_excerpt_sha256"],
                  "generation_kind": "SPECULATIVE_SHADOW_PROPOSAL_NOT_RESEARCH_IDEA_NODE",
                  "initial_proposal": p,"initial_critique":before,
                  "revision": rev,"revision_critique":after,
                  "parent_intact": True,
                  "scientific_truth_authority": False,
                  "novelty_certified": False,
                  "production_selection_authority": False}
        rows.append(branch)
    result={"schema_version":SCHEMA,"status":"M6_DEVELOPMENT_SHADOW_COMPLETED_SCIENCE_UNREVIEWED",
            "source_prepared_sha256":file_sha(prepared_path),
            "drafts_sha256":file_sha(drafts_path),
            "revisions_sha256":file_sha(revision_path) if revision_path else None,
            "branch_count":len(rows), "selected_parent_count":len(tasks),
            "development_branches":rows,
            "metrics_non_scientific":{
                "initial_structurally_specified":sum(not x["initial_critique"]["specification_defects"] for x in rows),
                "revised_structurally_specified":sum(x["revision_critique"] is not None and not x["revision_critique"]["specification_defects"] for x in rows),
                "baseline_science_capability": "NOT_MEASURED", "scientific_gain": "NOT_DEMONSTRATED"
            },
            "authority":{"scientific_truth_authority":False,"novelty_certified":False,
                         "production_selection_authority":False,"ResearchIdea_population_modified":False,
                         "may_deprioritize_parent":False}}
    save_new(Path(output)/"M6_DEVELOPED_BRANCHES.json",result)
    return result


def prompt_for_task(t: dict) -> tuple[str,str]:
    system=("You are a creative but epistemically conservative scientific research developer. "
            "Search aggressively; claim conservatively. Grounding controls claims, not imagination. "
            "Develop a speculative descendant, preserving the source ResearchIdea. "
            "Challenge known physics and identifiability; do not claim discovery, truth, or novelty. "
            "Supply a quantitative or ordinal conditional discriminator, independent measurements, "
            "a heldout experiment and a LOCAL falsifier. Output ONLY JSON with key 'proposals' "
            "containing exactly one proposal. Every proposal must contain all required schema fields "
            "and scientific_truth_authority=false, novelty_certified=false, "
            "source_of_proposal='MODEL_GENERATED'. "
            "Lists independent_variables, assumptions, controls must be arrays of nonempty strings. "
            "Do not confuse disproving restricted M0 with disproving broad known M1.")
    user=json.dumps({"task":t, "proposal_fields": sorted(ALLOWED_FIELDS),
                     "local_id":"generated_01", "required_text_fields":list(REQUIRED_TEXT),
                     "required_list_fields":list(REQUIRED_LIST)},ensure_ascii=False)
    return system,user


def prompt_for_revision(branch: dict, task: dict) -> tuple[str, str]:
    """Make a second, *new* development call that confronts the first proposal with M1."""
    system = (
        "You are revising a speculative science research proposal after a null/identifiability challenge. "
        "Search aggressively. Claim conservatively. Grounding controls claims, not imagination. "
        "Keep the exact parent_idea_id and local_id; never delete the original proposal. "
        "Compare against the BROAD known-physics envelope, not merely orientation-only M0. "
        "Do not fabricate measurements or novel physics. If no discriminator beyond known physics "
        "can be specified, present a useful predictive or measurement-method development within known physics. "
        "Improve intervention, independent measurements, quantitative/ordinal prediction, holdout, "
        "and local falsifier only where physically justified. "
        "Return ONLY JSON with key 'proposals' containing exactly one proposal with all required "
        "fields; source_of_proposal='MODEL_REVISED', novelty_certified=false, "
        "scientific_truth_authority=false."
    )
    user = json.dumps({
        "source_idea":task, "initial_proposal": branch["initial_proposal"],
        "first_pass_critique":branch["initial_critique"],
        "parent_idea_id":branch["parent_idea_id"],
        "local_id":branch["initial_proposal"]["local_id"],
        "proposal_fields":sorted(ALLOWED_FIELDS),
    }, ensure_ascii=False)
    return system,user
