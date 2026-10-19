"""M29 add-only, fixed-problem scientific rediscovery pilot.

Reuses Case B's proven per-stage paid-call ledger. DOES NOT edit production
FAKG, perform a web search, make scientific truth claims, or certify Level II.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pipeline_core.discovery.scientific_discovery_core import engine as core

PROTOCOL = "m29-problem-first-rediscovery-v1"
STAGES = ("scientific_work", "adversarial_review")
DEFAULT_MODEL = "openai/gpt-5.6-luna"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScientificWork(Strict):
    action: Literal["DERIVE", "MODEL", "COMPARE", "REFRAME", "ABSTAIN"]
    problem_understood: str = Field(min_length=25)
    observation_basis: list[str] = Field(min_length=1)
    substantive_reasoning: list[str] = Field(min_length=1)
    current_explanation_or_result: str = Field(min_length=25)
    strongest_conventional_null: str = Field(min_length=25)
    actual_null_test_or_obstruction: str = Field(min_length=25)
    new_testable_consequence_or_proof: str = Field(min_length=25)
    source_ids_used: list[str] = Field(default_factory=list)
    scientific_limits: list[str] = Field(min_length=1)
    scientific_delta: str = Field(min_length=25)
    next_action: Literal["CONTINUE", "REVISE", "GET_EVIDENCE", "ABSTAIN", "STOP"]


class AdversarialReview(Strict):
    original_work_accurately_summarized: str = Field(min_length=25)
    logically_supported_steps: list[str] = Field(min_length=1)
    independently_constructed_counterexample_or_explanation_of_failure: str = Field(min_length=25)
    strongest_known_physics_comparison: str = Field(min_length=25)
    measurement_or_causal_identifiability_limit: str = Field(min_length=25)
    revised_scientific_conclusion: str = Field(min_length=25)
    what_was_actually_learned: str = Field(min_length=25)
    still_unsupported_claims: list[str] = Field(min_length=1)
    disposition: Literal["CONDITIONAL_ADVANCE", "REDISCOVERY_ONLY", "INCONCLUSIVE", "ABSTAIN"]
    followup_if_any: str = Field(min_length=15)


def _packet_check(packet: dict) -> None:
    if packet.get("schema_version") != PROTOCOL:
        raise ValueError("BAD_M29_PACKET_SCHEMA")
    obs = packet.get("observations")
    if not isinstance(obs, list) or len(obs) < 3:
        raise ValueError("NEED_AT_LEAST_THREE_REPORTED_OBSERVATIONS")
    ids = [x.get("id") for x in obs if isinstance(x, dict)]
    if len(ids) != len(obs) or len(set(ids)) != len(ids) or not all(ids):
        raise ValueError("DUPLICATE_OR_MISSING_OBSERVATION_ID")
    if not packet.get("fixed_problem", {}).get("scientific_question"):
        raise ValueError("MISSING_FIXED_SCIENTIFIC_QUESTION")
    if packet.get("sealed_expert_interpretation") is not None:
        raise ValueError("EXPERT_INTERPRETATION_MUST_NOT_BE_IN_MODEL_INPUT")


def _model(stage: str):
    return ScientificWork if stage == STAGES[0] else AdversarialReview


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input(out: Path):
    manifest = core.read_json(out / "manifest.json")
    packet = core.read_json(out / "input.json")
    _packet_check(packet)
    if core.sha(packet) != manifest["input_sha256"]:
        raise ValueError("M29_FROZEN_INPUT_HASH_CHANGED")
    if _sha_file(out / "input.json") != manifest["input_bytes_sha256"]:
        raise ValueError("M29_FROZEN_INPUT_BYTES_CHANGED")
    if manifest["schema_version"] != PROTOCOL or manifest["budget_calls"] != 2:
        raise ValueError("M29_MANIFEST_INVALID")
    return manifest, packet


def _state(out: Path, stage: str):
    d = out / "calls" / stage
    r, a, raw = d / "receipt.json", d / "attempt.json", d / "raw_response.json"
    if r.exists():
        receipt = core.read_json(r)
        if not a.exists() or not raw.exists():
            raise ValueError(f"{stage}:RECEIPT_WITHOUT_RAW_AND_ATTEMPT")
        norm = core.read_json(d / "response.json")
        request = core.read_json(d / "request.json")
        if receipt.get("status") != "COMPLETED" or receipt.get("request_sha256") != core.sha(request):
            raise ValueError(f"{stage}:RECEIPT_REQUEST_CONFLICT")
        if receipt.get("original_raw_response_sha256") != core.sha(core.read_json(raw)):
            raise ValueError(f"{stage}:RAW_HASH_CONFLICT")
        if receipt.get("validated_response_sha256") != core.sha(norm):
            raise ValueError(f"{stage}:NORMALIZED_HASH_CONFLICT")
        _model(stage).model_validate(norm)
        return "VERIFIED"
    if raw.exists():
        return "RAW_SAVED_RECOVERY_REQUIRED"
    if a.exists():
        return "PAID_ATTEMPT_UNRESOLVED_NO_RETRY"
    if d.exists() and any(d.iterdir()):
        return "PARTIAL_RECORD_MANUAL_REVIEW"
    return "NOT_STARTED"


def verify(out: Path):
    m, packet = _input(out)
    states = {s: _state(out, s) for s in STAGES}
    if states[STAGES[1]] != "NOT_STARTED" and states[STAGES[0]] != "VERIFIED":
        raise ValueError("REVIEW_EXISTS_WITHOUT_VERIFIED_RESEARCH")
    return {
        "status": "VERIFIED_READ_ONLY",
        "problem": packet["fixed_problem"]["label"],
        "model": m["model"],
        "stages": states,
        "paid_attempts_recorded": len(list((out / "calls").glob("*/attempt.json"))),
        "completed_model_calls": sum(x == "VERIFIED" for x in states.values()),
        "source_history_preserved": True,
        "level_ii_certified": False,
    }


def _prompt(stage: str, packet: dict, work: dict | None):
    sys = (
        "You are a scientist given an externally selected, concrete scientific problem. "
        "The observed facts in the packet are source-paraphrased facts, not a prompt to invent a gap. "
        "Your job is substantive scientific reasoning: explanation, derivation, counterexample, "
        "constraint, alternative model, or rigorous abstention. Decide the direction yourself. "
        "Do NOT claim new scientific discovery, do NOT claim an experiment was run, and do NOT "
        "smuggle unverified ideas into the observations. Do not force mathematics or a new mechanism "
        "if another scientific approach is more appropriate. Published interpretations are deliberately "
        "held out; the study can be known to you from training, so acknowledge any remembered answer. "
        "Return a single JSON object exactly matching the requested schema; no markdown."
    )
    if stage == STAGES[0]:
        inst = (
            "Investigate the FIXED problem rather than searching for a different one. "
            "Actually reason from the evidence and identify a nontrivial conditional result or a rigorous "
            "obstruction. Distinguish the strongest conventional competing explanation from yours. "
            "A measurement proposal by itself is not a result; if evidence is insufficient, specify what "
            "can and cannot follow. You have no access to extra papers, computation, or live tools."
        )
    else:
        inst = (
            "Independently challenge the previous scientific_work. Build an explicit conventional "
            "countermodel or show why a claimed result does not logically follow from supplied "
            "observations. If a conclusion genuinely survives, preserve it with its limits. "
            "Admit whether the main result is likely already established rather than new. "
            "Do not reward attractive speculation or force a particular answer."
        )
    user = json.dumps({"stage":stage,"instructions":inst,"frozen_problem_and_observations":packet,
                       "previous_scientific_work":work,"response_json_schema":_model(stage).model_json_schema()},
                      ensure_ascii=False,sort_keys=True)
    return sys, user


def prepare(source: Path, out: Path, *, model: str, temperature: float, max_tokens: int):
    if out.exists():
        raise FileExistsError("M29_REFUSES_OVERWRITE_EXISTING_RUN")
    if not (0 <= temperature <= 1.5 and 1000 <= max_tokens <= 12000 and model.strip()):
        raise ValueError("INVALID_MODEL_SETTINGS")
    packet = core.read_json(source)
    _packet_check(packet)
    out.mkdir(parents=True, exist_ok=False)
    core.write_new(out / "input.json", packet)
    meta = {"schema_version":PROTOCOL,"model":model,"temperature":temperature,
            "max_tokens":max_tokens,"budget_calls":2,"input_sha256":core.sha(packet),
            "input_bytes_sha256":_sha_file(out / "input.json"),"paid_authorization_required":True,
            "autonomous_problem_selection":False,"novelty_or_truth_certified":False,
            "mode":"HISTORICAL_SCIENTIFIC_REDISCOVERY_CALIBRATION"}
    core.write_new(out / "manifest.json", meta)
    return verify(out)


def preview(out: Path):
    s = verify(out)
    next_stage = next((k for k,v in s["stages"].items() if v == "NOT_STARTED"), None)
    blocked = any(v not in {"NOT_STARTED", "VERIFIED"} for v in s["stages"].values())
    if blocked:
        next_stage = None
    return {"status":"PREVIEW_NO_CALL", "next_stage":next_stage,
            "paid_calls_if_next":int(next_stage is not None), "run_state":s}


def next_step(out: Path, allow_paid: bool):
    p = preview(out)
    stage = p["next_stage"]
    if stage is None:
        raise RuntimeError("NO_SAFE_NEW_STEP_CHECK_VERIFY_AND_RECOVER")
    if not allow_paid or os.getenv("FAKG_M29_AUTHORIZE_PAID") != "YES":
        raise PermissionError("EXPLICIT_M29_PAID_AUTHORIZATION_REQUIRED")
    settings, packet = _input(out)
    work = None
    if stage == STAGES[1]:
        work = core.read_json(out / "calls" / STAGES[0] / "response.json")
    system, user = _prompt(stage,packet,work)
    backend = core.OpenRouter(api_key_env="OPENROUTER_API_KEY")
    result = core._call(out, stage, _model(stage), system, user, backend,
                        max_calls=2, settings=settings)
    return {"status":"STAGE_COMMITTED_ONE_PAID_CALL","stage":stage,
            "action_or_disposition":getattr(result, "action", None) or getattr(result,"disposition",None),
            "current_state":verify(out)}


def recover(out: Path):
    states = verify(out)["stages"]
    stage = next((k for k in STAGES if states[k] == "RAW_SAVED_RECOVERY_REQUIRED"),None)
    if stage is None:
        raise RuntimeError("NO_SAVED_RAW_TO_RECOVER")
    d=out / "calls" / stage
    request=core.read_json(d/"request.json")
    attempt=core.read_json(d/"attempt.json")
    if attempt.get("request_sha256") != core.sha(request):
        raise ValueError("RECOVER_REQUEST_MISMATCH")
    raw=core.read_json(d/"raw_response.json")
    parsed=_model(stage).model_validate(raw)
    result=parsed.model_dump(mode="json")
    norm=d/"response.json"
    if norm.exists():
        if core.sha(core.read_json(norm)) != core.sha(result):
            raise ValueError("RECOVER_NORMALIZED_CONFLICT")
    else:
        core.write_new(norm,result)
    core.write_new(d/"receipt.json",{"status":"COMPLETED","request_sha256":core.sha(request),
                 "usage":{"recovery":"USAGE_UNKNOWN_RETAIN_PROVIDER_LOGS"},
                 "parsed_schema":_model(stage).__name__,"paid_attempt_count_for_stage":1,
                 "original_raw_response_sha256":core.sha(raw),
                 "validated_response_sha256":core.sha(result),"quarantined_matrix_count":0})
    return {"status":"RECOVERED_FROM_SAVED_RAW_NO_PAID_CALL", "stage":stage,"current_state":verify(out)}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    sub=cli.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare");prep.add_argument("--packet",type=Path,required=True)
    prep.add_argument("--out",type=Path,required=True);prep.add_argument("--model",default=DEFAULT_MODEL)
    prep.add_argument("--temperature",type=float,default=0.5);prep.add_argument("--max-tokens",type=int,default=6500)
    for name in ("verify","preview","next","recover"):
        x=sub.add_parser(name);x.add_argument("--out",type=Path,required=True)
        if name=="next":x.add_argument("--allow-paid",action="store_true")
    a=cli.parse_args()
    if a.command=="prepare": result=prepare(a.packet,a.out,model=a.model,temperature=a.temperature,max_tokens=a.max_tokens)
    elif a.command=="verify":result=verify(a.out)
    elif a.command=="preview":result=preview(a.out)
    elif a.command=="recover":result=recover(a.out)
    else:result=next_step(a.out,a.allow_paid)
    print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True))

if __name__=="__main__":main()
