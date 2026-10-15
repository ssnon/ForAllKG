#!/usr/bin/env python3
"""Case01 opt-in adapter: frozen native HypothesisCards + Ye evidence -> existing M7 scientific-development core.

This is NOT native production E2E; it is a controlled bridge to existing M7 create_model_prompt,
ScientificProgramDraft.parse, challenge_program, and retention_brief. Source authority never changes.
No paid call except explicit run-paid. Not an automatic hypothesis certification pipeline.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
from datetime import datetime, timezone
from typing import Any

YE_DOI = "10.1103/4yg7-1f85"
ANCHOR = "hypothesis:76c744b5c7b6c49096c0"
ARMS = ("ABSTRACT_ONLY", "YE_FULLTEXT_FIGURES")
FIGURE_PAGES = (3, 4, 5, 6)
SCHEMA = "case01-m7-source-bridge-v1"
REQUIRED_FILES = (
    "frozen/main/scientific_portfolio_shadow/materialized.shadow.portfolio.json",
    "frozen/main/hypothesis_axis_open_world.selected_works.json",
    "source/Ye_main_text_by_page.json",
    "source/Ye_2025_PRB_source_4yg7-1f85.pdf",
    *(f"source/Ye_page_{p:02d}.png" for p in FIGURE_PAGES),
)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_obj(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def verify_frozen(kit: Path) -> dict[str, Any]:
    """Offline provenance: selected Ye paper, eight unmodified native cards, 8 PDF pages."""
    kit = kit.resolve()
    for rel in REQUIRED_FILES:
        p = kit / rel
        if not p.is_file():
            raise FileNotFoundError(f"MISSING_FROZEN_SOURCE: {p}")
    port = read(kit / REQUIRED_FILES[0]); works = read(kit / REQUIRED_FILES[1]); pages = read(kit / REQUIRED_FILES[2])
    if len(port.get("hypotheses", [])) != 8 or len(works) != 12:
        raise ValueError("FROZEN_CARD_OR_WORK_COUNT_CHANGED")
    ids = [x["hypothesis_id"] for x in port["hypotheses"]]
    if len(ids) != len(set(ids)) or ANCHOR not in ids:
        raise ValueError("FROZEN_HYPOTHESIS_ID_CONTRACT_FAILED")
    ye = [x for x in works if str(x["work"].get("doi", "")).lower().strip() == YE_DOI]
    if len(ye) != 1:
        raise ValueError("YE_DOI_NOT_UNIQUELY_SELECTED")
    if len(pages) != 8 or [x.get("page") for x in pages] != list(range(1, 9)):
        raise ValueError("YE_PAGE_ORDER_CHANGED")
    combined = " ".join(str(p.get("text", "")) for p in pages).lower()
    for marker in ("umbelliferone", "4-ntp", "pva", "200", "2 h"):
        if marker not in combined:
            raise ValueError(f"YE_FULLTEXT_EVIDENCE_MARKER_MISSING: {marker}")
    return {"status": "SOURCE_PROVENANCE_PASS", "paper_doi": YE_DOI,
            "ye_work_id": ye[0]["work"]["work_id"], "native_card_count": len(ids),
            "page_count": len(pages), "anchor_card_id": ANCHOR,
            "file_sha256": {rel: sha_file(kit / rel) for rel in REQUIRED_FILES}}



def native_m7_module_sha() -> str:
    """Detect drift in the existing M7 scientific-development implementation."""
    try:
        import pipeline_core.discovery.scientific_development_m7 as native_m7
    except ImportError as exc:
        raise RuntimeError("NATIVE_M7_MODULE_UNAVAILABLE: run from ~/FAKG/ForAllKG with graphAgent311") from exc
    path = Path(native_m7.__file__)
    if not path.is_file():
        raise RuntimeError("NATIVE_M7_SOURCE_FILE_UNAVAILABLE")
    return sha_file(path)


def card_view(card: dict[str, Any]) -> dict[str, Any]:
    return {k: card.get(k) for k in ("hypothesis_id", "title", "hypothesis_statement", "inferential_bridge",
                    "predicted_observations", "falsification_criteria", "assumptions")}


def build_task(kit: Path, arm: str) -> dict[str, Any]:
    if arm not in ARMS: raise ValueError("INVALID_ARM")
    port = read(kit / REQUIRED_FILES[0])
    cards = port["hypotheses"]
    anchor = next(c for c in cards if c["hypothesis_id"] == ANCHOR)
    return {
        "label": "CASE01_FROZEN_PORTFOLIO_M7", "parent_idea_id": ANCHOR,
        "role": "OPEN_ALTERNATE_BRANCH",
        "population_lane": "EXPLICIT_FROZEN_HYPOTHESIS_CARD_ADAPTER_NOT_NATIVE_G5",
        "original_scientific_question": (
            "What causal mechanism produces intermittent mode-dependent Raman emission near "
            "individual gold nanorods in dye-rich solid samples, and what prediction "
            "would distinguish it from strong independent-site, chemical, and atomic-hotspot explanations?"
        ),
        "original_conceptual_change": anchor.get("inferential_bridge"),
        "original_differential_prediction": anchor.get("predicted_observations"),
        "original_discriminating_observation": anchor.get("predicted_observations"),
        "original_falsification_condition": anchor.get("falsification_criteria"),
        "original_kernel": {
            "primary_native_parent": card_view(anchor),
            "other_native_portfolio_cards_search_context_only": [card_view(c) for c in cards if c["hypothesis_id"] != ANCHOR],
            "all_cards_unverified": True,
            "never_claim_source_premise_authority_from_card": True,
        },
    }


def build_evidence(kit: Path, arm: str) -> dict[str, Any]:
    works = read(kit / REQUIRED_FILES[1]); pages = read(kit / REQUIRED_FILES[2]);
    refs = [{"title": w["work"].get("title"), "doi": w["work"].get("doi"),
             "abstract": w["work"].get("abstract"), "year": w["work"].get("year")}
            for w in works]
    # Same 12 prior-art abstracts in both arms. Ye FULLTEXT adds only original paper and its figures.
    evidence = {
        "research_case": "Ye et al., PRB 112, 075428 (2025)",
        "retrieved_prior_art_abstracts_same_in_both_arms": refs,
        "ye_paper_abstract": next(w["work"].get("abstract") for w in works if str(w["work"].get("doi", "")).lower() == YE_DOI),
        "sources_are_evidence_not_answers": True,
        "do_not_assume_supplemental_or_raw_traces": True,
        "no_externally_certified_novelty": True,
    }
    if arm == "YE_FULLTEXT_FIGURES":
        evidence["ye_main_paper_page_text"] = pages
        evidence["ye_original_pdf_figure_pages_attached_as_images"] = list(FIGURE_PAGES)
    return evidence


def build_messages(kit: Path, arm: str) -> list[dict[str, Any]]:
    try:
        from pipeline_core.discovery.scientific_development_m7 import create_model_prompt
    except ImportError as exc:
        raise RuntimeError("NATIVE_M7_MODULE_UNAVAILABLE: run from ~/FAKG/ForAllKG with graphAgent311") from exc
    messages = create_model_prompt(build_task(kit, arm))
    messages[0]["content"] += ("\nFor this source-controlled research continuation, treat native cards as "
        "unverified search ideas, not evidence. Examine whether your discriminating predictions "
        "could ALSO follow from strong independently switching local sites, collective common-mode "
        "enhancement, photochemical history, or transient atomic hotspots. If so, say so. "
        "Do not import any answer or human-written Case01 mechanism from outside the source bundle. "
        "If you cannot identify a genuinely distinct mechanism, state that explicitly in surviving_uncertainty. "
        "OUTPUT BUDGET: return one COMPLETE JSON object, no markdown, in approximately 1800-3000 output tokens. "
        "Do not drop required JSON fields; keep lists concise and physics concrete.")
    messages[1]["content"] += ("\n\nSOURCE EVIDENCE (frozen; NOT additional ground-truth authority):\n" +
        json.dumps(build_evidence(kit, arm), ensure_ascii=False, sort_keys=True))
    return messages


def prepare(kit: Path, out: Path) -> dict[str, Any]:
    if out.exists(): raise FileExistsError("OUTPUT_EXISTS_NO_OVERWRITE")
    provenance = verify_frozen(kit)
    # Fail BEFORE creating any output if installed native M7 is missing.
    prompts = {arm: build_messages(kit, arm) for arm in ARMS}
    out.mkdir(parents=True)
    for arm in ARMS:
        new_json(out / "prompts" / f"{arm}.json", {"messages": prompts[arm], "arm": arm, "source_authority": "RESEARCH_CONTEXT_ONLY"})
    new_json(out / "PROVENANCE.json", provenance)
    manifest = {
        "schema_version": SCHEMA, "status": "READY_NO_PROVIDER_CALLS", "kit_path": str(kit.resolve()),
        "source_file_sha256": provenance["file_sha256"], "prompt_sha256": {
            arm: sha_file(out / "prompts" / f"{arm}.json") for arm in ARMS},
        "study_design": "Same frozen eight native cards and 12 abstracts; M7 core in both; only source depth (Ye fulltext + figure images) differs.",
        "source_of_8_cards": "historical FAKG native scientific portfolio, not generated by this bridge",
        "actual_M7_core_reused": True,
        "native_M7_module_sha256": native_m7_module_sha(),
        "official_native_stage8_rewired": False,
        "official_SIS_scheduler_executed": False,
        "canonical_graph_mutated": False,
        "scientific_novelty_certified": False,
        "new_paid_calls": 0,
        "images_only_in_fulltext_arm": list(FIGURE_PAGES),
        "target_model_calls": 2,
        "blinding": "Candidate identities hidden in blind file; mapping stored privately after both calls",
    }
    new_json(out / "RUN_MANIFEST.json", manifest)
    return {"status": manifest["status"], "output": str(out),
            "arm_prompt_character_counts": {arm: sum(len(x["content"]) for x in prompts[arm]) for arm in ARMS},
            "native_M7_core_reused": True, "new_paid_calls": 0}


def check_frozen(out: Path) -> tuple[Path, dict[str, Any]]:
    m = read(out / "RUN_MANIFEST.json")
    if m.get("schema_version") != SCHEMA or m.get("status") != "READY_NO_PROVIDER_CALLS":
        raise ValueError("MANIFEST_SCHEMA_OR_STATUS_CHANGED")
    kit = Path(m["kit_path"])
    if native_m7_module_sha() != m.get("native_M7_module_sha256"):
        raise ValueError("NATIVE_M7_IMPLEMENTATION_CHANGED_SINCE_PREPARE")
    p = verify_frozen(kit)
    if p["file_sha256"] != m["source_file_sha256"]:
        raise ValueError("SOURCE_FILE_CHANGED_SINCE_PREPARE")
    for arm in ARMS:
        if sha_file(out / "prompts" / f"{arm}.json") != m["prompt_sha256"][arm]:
            raise ValueError("PROMPT_CHANGED_SINCE_PREPARE")
    return kit, m


def run_paid(out: Path, arm: str, model: str, max_tokens: int,
             api_key_env: str, base_url: str, temp: float) -> dict[str, Any]:
    if arm not in ARMS: raise ValueError("INVALID_ARM")
    if not (3000 <= max_tokens <= 9000): raise ValueError("TOKEN_CAP_OUTSIDE_3000_TO_9000")
    if not os.getenv(api_key_env): raise PermissionError(f"MISSING_PROVIDER_API_KEY: {api_key_env}")
    kit, _ = check_frozen(out)
    response_path = out / "responses" / f"{arm}.json"
    marker = out / "attempts" / f"{arm}.json"
    if response_path.exists() or marker.exists():
        raise FileExistsError("PAID_CALL_ALREADY_STARTED_OR_SAVED_NO_AUTOMATIC_RETRY")
    messages = read(out / "prompts" / f"{arm}.json")["messages"]
    if arm == "YE_FULLTEXT_FIGURES":
        content: list[dict[str, Any]] = [{"type": "text", "text": messages[-1]["content"]}]
        for pg in FIGURE_PAGES:
            path = kit / f"source/Ye_page_{pg:02d}.png"
            content += [
                {"type": "text", "text": f"Original Ye 2025 PDF page {pg}, figure/diagram image:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")}},
            ]
        messages[-1]["content"] = content
    from openai import OpenAI
    # Source and prompt checks above happen before marker; once marker exists, failures need manual reconciliation.
    new_json(marker, {"state": "STARTED_NO_AUTORETRY", "arm": arm, "model": model,
                      "model_temperature": temp, "max_tokens": max_tokens,
                      "timestamp_utc": datetime.now(timezone.utc).isoformat()})
    client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url,
                    max_retries=0, timeout=300)
    result = client.chat.completions.create(model=model, messages=messages,
                   response_format={"type": "json_object"},
                   temperature=temp, max_tokens=max_tokens)
    choice = result.choices[0]
    raw = choice.message.content or ""
    new_json(response_path, {"status": "SAVED_SCIENCE_UNREVIEWED", "arm": arm,
         "model": model, "finish_reason": choice.finish_reason,
         "response": result.model_dump(mode="json"),
         "source_prompt_sha256": sha_file(out / "prompts" / f"{arm}.json")})
    if choice.finish_reason != "stop":
        return {"status": "RAW_SAVED_BUT_INCOMPLETE_DO_NOT_RETRY", "arm": arm,
                "finish_reason": choice.finish_reason, "raw_response": str(response_path)}
    try:
        from pipeline_core.discovery.scientific_development_m7 import ScientificProgramDraft, challenge_program, retention_brief
        obj = json.loads(raw)
        if "program" in obj and isinstance(obj["program"], dict): obj = obj["program"]
        program = ScientificProgramDraft.parse(obj)
        task = build_task(kit, arm)
        if (program.idea_label != task["label"] or program.parent_idea_id != ANCHOR or
                program.source_role != task["role"]):
            raise ValueError("M7_PROGRAM_IDENTITY_CONTRACT_FAILURE")
        new_json(out / "parsed" / f"{arm}.json", program.to_dict())
        new_json(out / "m7_feedback" / f"{arm}.json", {
            "schema_version": SCHEMA, "status": "NATIVE_M7_STRUCTURAL_CHALLENGE_ONLY",
            "challenge": challenge_program(program), "retention_brief": retention_brief(program, task),
            "science_truth_certified": False, "novelty_certified": False})
    except Exception as exc:
        return {"status": "RAW_SAVED_BUT_M7_SCHEMA_INVALID_DO_NOT_RETRY", "arm": arm,
                "reason": f"{type(exc).__name__}: {exc}", "raw_response": str(response_path)}
    return {"status": "M7_PROGRAM_VALIDATED_SCIENCE_UNREVIEWED", "arm": arm,
            "finish_reason": choice.finish_reason,
            "parsed": str(out / "parsed" / f"{arm}.json"),
            "challenge": str(out / "m7_feedback" / f"{arm}.json")}


def blind(out: Path) -> dict[str, Any]:
    check_frozen(out)
    if (out / "BLIND_REVIEW_PUBLIC.json").exists() or (out / "BLIND_ASSIGNMENT_PRIVATE.json").exists():
        raise FileExistsError("BLIND_PACKET_ALREADY_EXISTS")
    rows = []
    for arm in ARMS:
        r = read(out / "responses" / f"{arm}.json")
        if r.get("finish_reason") != "stop":
            raise ValueError(f"INCOMPLETE_RESPONSE: {arm}")
        program = read(out / "parsed" / f"{arm}.json")
        # Structural validation at trust boundary, not scientific certification.
        from pipeline_core.discovery.scientific_development_m7 import ScientificProgramDraft
        obj = ScientificProgramDraft.parse(program)
        if obj.idea_label != "CASE01_FROZEN_PORTFOLIO_M7" or obj.parent_idea_id != ANCHOR:
            raise ValueError("BLIND_IDENTITY_MISMATCH")
        public = obj.to_dict()
        for key in ("idea_label", "parent_idea_id", "source_role", "source_of_draft"):
            public.pop(key, None)
        rows.append((arm, public))
    secrets.SystemRandom().shuffle(rows)
    pub = {"schema_version": SCHEMA, "status": "BLIND_SCIENCE_REVIEW_READY",
           "comparison": "source-depth ablation with native FAKG portfolio+M7 core fixed",
           "official_FAKG_E2E": False,
           "items": [{"id": f"B{i+1:02d}", "program": body} for i, (_, body) in enumerate(rows)],
           "rubric": "Assess actual causal novelty, formal entailment, constrained independent-site and coupled-physics strong nulls, full experimental identifiability, source fidelity. Do not reward length or mathematical decoration."}
    new_json(out / "BLIND_REVIEW_PUBLIC.json", pub)
    new_json(out / "BLIND_ASSIGNMENT_PRIVATE.json", {f"B{i+1:02d}": arm for i, (arm, _) in enumerate(rows)})
    return {"status": pub["status"], "file": str(out / "BLIND_REVIEW_PUBLIC.json"), "private_mapping_exposed": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight"); p.add_argument("--kit", type=Path, required=True)
    p = sub.add_parser("prepare"); p.add_argument("--kit", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("verify"); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run-paid"); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--model", default="openai/gpt-5.6-luna")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--max-tokens", type=int, default=7000)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--allow-paid", action="store_true")
    p.add_argument("--i-authorize-new-llm-calls", action="store_true")
    p = sub.add_parser("blind"); p.add_argument("--out", type=Path, required=True)
    a = parser.parse_args(argv)
    try:
        if a.command == "preflight":
            result = verify_frozen(a.kit)
        elif a.command == "prepare":
            result = prepare(a.kit, a.out)
        elif a.command == "verify":
            k, m = check_frozen(a.out)
            result = {"status": "FROZEN_INPUTS_VERIFIED_NO_CALLS", "kit": str(k),
                      "prompt_sha256": m["prompt_sha256"]}
        elif a.command == "run-paid":
            if not (a.allow_paid and a.i_authorize_new_llm_calls and os.getenv("CASE01_ALLOW_PAID") == "YES"):
                raise PermissionError("PAID_CALL_BLOCKED_REQUIRES_EXPLICIT_FLAGS_AND_CASE01_ALLOW_PAID=YES")
            result = run_paid(a.out, a.arm, a.model, a.max_tokens, a.api_key_env, a.base_url, a.temperature)
        else:
            result = blind(a.out)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
