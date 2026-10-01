
from __future__ import annotations

import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ARMS = (
    "LEGACY",
    "FRONTIER_BALANCED",
    "EVOLUTION_BALANCED",
    "PORTFOLIO_SELECTED",
)

RUBRIC = (
    {
        "id": "scientific_interest",
        "label": "Scientific interest",
        "low": "Routine / low research interest",
        "high": "Strongly worth scientific attention",
    },
    {
        "id": "conceptual_distinctiveness",
        "label": "Conceptual distinctiveness",
        "low": "Obvious reformulation",
        "high": "Genuinely different conceptual direction",
    },
    {
        "id": "mechanistic_plausibility",
        "label": "Mechanistic plausibility",
        "low": "Mechanistically weak or incoherent",
        "high": "Mechanistically coherent and plausible",
    },
    {
        "id": "falsifiability",
        "label": "Falsifiability",
        "low": "Hard to disconfirm",
        "high": "Clearly falsifiable",
    },
    {
        "id": "actionability",
        "label": "Experimental / computational actionability",
        "low": "Difficult to turn into a study",
        "high": "Clear path to a study",
    },
    {
        "id": "information_gain",
        "label": "Potential information gain",
        "low": "Likely low-value result",
        "high": "Either outcome would teach us something important",
    },
    {
        "id": "pursue_likelihood",
        "label": "Would you pursue this direction?",
        "low": "Definitely not",
        "high": "Definitely yes",
    },
)


class HumanReviewError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HumanReviewError(f"Required JSON artifact not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HumanReviewError(f"Expected JSON object: {path}")
    return value


def _load_json_if(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _stable_digest(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json_for_script(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")


def _portfolio_path(arm_dir: Path) -> Path | None:
    candidates = (
        arm_dir / "portfolio.json",
        arm_dir / "materialized.portfolio.json",
        arm_dir / "materialization.portfolio.json",
        arm_dir / "scientific_portfolio.portfolio.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    for path in sorted(arm_dir.glob("*portfolio*.json")):
        payload = _load_json_if(path)
        if isinstance(payload.get("hypotheses"), list):
            return path
    return None


def _context_path(prospective_root: Path, case_id: str) -> Path | None:
    case_dir = prospective_root / case_id
    for path in (
        case_dir / "hypothesis.context.json",
        case_dir / "context.json",
    ):
        if path.is_file():
            return path

    status = _load_json_if(prospective_root / "execution_status.json")
    for row in _list(status.get("cases")):
        if not isinstance(row, dict) or _text(row.get("case_id")) != case_id:
            continue
        run_text = _text(row.get("run_dir"))
        if not run_text:
            continue
        path = Path(run_text).expanduser() / "hypothesis.context.json"
        if path.is_file():
            return path
    return None


def _verification_summary(arm_dir: Path) -> dict[str, Any]:
    return _load_json_if(arm_dir / "verification" / "verification.summary.json")


def _external_cards(arm_dir: Path) -> dict[str, dict[str, Any]]:
    report = _load_json_if(
        arm_dir / "verification" / "external_novelty.report.json"
    )
    result: dict[str, dict[str, Any]] = {}
    for card in _list(report.get("cards")):
        if isinstance(card, dict):
            hid = _text(card.get("hypothesis_id"))
            if hid:
                result[hid] = dict(card)
    return result


def _feasibility_cards(arm_dir: Path) -> dict[str, dict[str, Any]]:
    payload: dict[str, Any] = {}
    for path in (
        arm_dir / "verification" / "feasibility" / "decision" / "portfolio.json",
        arm_dir / "verification" / "feasibility" / "portfolio.json",
    ):
        payload = _load_json_if(path)
        if payload:
            break
    result: dict[str, dict[str, Any]] = {}
    for card in _list(payload.get("cards")):
        if not isinstance(card, dict):
            continue
        hid = _text(card.get("hypothesis_id"))
        if hid:
            result[hid] = dict(card)
    return result


def _candidate_metadata_index(arm_dir: Path) -> dict[str, dict[str, Any]]:
    # Reveal-only best-effort metadata. Never used for sampling or scoring.
    index: dict[str, dict[str, Any]] = {}
    candidate_files: list[Path] = []
    for pattern in (
        "*candidate_pool*.json",
        "*selection*.json",
        "*materialization*.json",
        "*audit*.json",
    ):
        candidate_files.extend(sorted(arm_dir.glob(pattern)))

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            cid = _text(
                value.get("candidate_id")
                or value.get("source_candidate_id")
                or value.get("selected_candidate_id")
            )
            if cid:
                target = index.setdefault(cid, {})
                for key in (
                    "origin",
                    "source_kind",
                    "idea_form",
                    "operator_id",
                    "selection_profile",
                    "profile",
                    "task_relation_mode",
                    "scientific_intent",
                    "conceptual_change_summary",
                ):
                    if key in value and value.get(key) not in (None, "", [], {}):
                        target[key] = value.get(key)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in candidate_files:
        visit(_load_json_if(path))
    return index


def _hypothesis_candidate_links(arm_dir: Path) -> dict[str, str]:
    links: dict[str, str] = {}
    for pattern in ("*materialization*.json", "*selection*.json", "*audit*.json"):
        for path in sorted(arm_dir.glob(pattern)):
            payload = _load_json_if(path)

            def visit(value: Any) -> None:
                if isinstance(value, dict):
                    hid = _text(
                        value.get("hypothesis_id")
                        or value.get("materialized_hypothesis_id")
                    )
                    cid = _text(
                        value.get("candidate_id")
                        or value.get("source_candidate_id")
                        or value.get("selected_candidate_id")
                    )
                    if hid and cid:
                        links.setdefault(hid, cid)
                    for child in value.values():
                        visit(child)
                elif isinstance(value, list):
                    for child in value:
                        visit(child)

            visit(payload)
    return links


@dataclass(frozen=True)
class _ArmHypothesis:
    case_id: str
    arm: str
    context: dict[str, Any]
    portfolio_path: Path
    arm_dir: Path
    card: dict[str, Any]
    external: dict[str, Any]
    feasibility: dict[str, Any]
    verification: dict[str, Any]
    candidate_id: str | None
    candidate_meta: dict[str, Any]


def _load_arm_hypotheses(
    root: Path,
    case_id: str,
    arm: str,
) -> list[_ArmHypothesis]:
    arm_dir = root / case_id / arm
    portfolio_path = _portfolio_path(arm_dir)
    if portfolio_path is None:
        return []
    portfolio = _load_json(portfolio_path)
    context_path = _context_path(root, case_id)
    context = _load_json(context_path) if context_path else {}
    external = _external_cards(arm_dir)
    feasibility = _feasibility_cards(arm_dir)
    verification = _verification_summary(arm_dir)
    candidate_meta = _candidate_metadata_index(arm_dir)
    links = _hypothesis_candidate_links(arm_dir)

    result: list[_ArmHypothesis] = []
    for card in _list(portfolio.get("hypotheses")):
        if not isinstance(card, dict):
            continue
        hid = _text(card.get("hypothesis_id"))
        if not hid:
            continue
        cid = links.get(hid)
        result.append(
            _ArmHypothesis(
                case_id=case_id,
                arm=arm,
                context=context,
                portfolio_path=portfolio_path,
                arm_dir=arm_dir,
                card=dict(card),
                external=dict(external.get(hid, {})),
                feasibility=dict(feasibility.get(hid, {})),
                verification=dict(verification),
                candidate_id=cid,
                candidate_meta=dict(candidate_meta.get(cid or "", {})),
            )
        )
    return result


def discover_reviewable_cases(root: Path) -> list[str]:
    cases: list[str] = []
    for path in sorted(root.iterdir() if root.is_dir() else []):
        if path.is_dir() and any((path / arm).is_dir() for arm in ARMS):
            cases.append(path.name)
    return cases


def _evidence_index(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in _list(context.get("evidence_statements")):
        if isinstance(row, dict):
            sid = _text(row.get("statement_id"))
            if sid:
                result[sid] = dict(row)
    return result


def _blind_item(row: _ArmHypothesis, blind_id: str) -> dict[str, Any]:
    card = row.card
    return {
        "blind_id": blind_id,
        "case_id": row.case_id,
        "domain_profile_id": (
            _text(card.get("domain_profile_id"))
            or _text(row.context.get("domain_profile_id"))
        ),
        "research_question": _text(row.context.get("question")),
        "title": _text(card.get("title")),
        "hypothesis_statement": _text(card.get("hypothesis_statement")),
        "inferential_bridge": _text(card.get("inferential_bridge")),
        "predicted_observations": [
            {
                "observable": _text(x.get("observable")),
                "expected_direction": _text(x.get("expected_direction")),
                "rationale": _text(x.get("rationale")),
            }
            for x in _list(card.get("predicted_observations"))
            if isinstance(x, dict)
        ],
        "falsification_criteria": [
            {
                "observable": _text(x.get("observable")),
                "falsifying_outcome": _text(x.get("falsifying_outcome")),
            }
            for x in _list(card.get("falsification_criteria"))
            if isinstance(x, dict)
        ],
        "assumptions": [str(x) for x in _list(card.get("assumptions"))],
    }


def _reveal_item(row: _ArmHypothesis, blind_id: str) -> dict[str, Any]:
    card = row.card
    evidence = _evidence_index(row.context)
    premises = []
    for sid in _list(card.get("premise_statement_ids")):
        source = evidence.get(str(sid), {})
        premises.append(
            {
                "statement_id": str(sid),
                "text": _text(source.get("text")),
                "epistemic_role": _text(source.get("epistemic_role")),
                "claim_kind": _text(source.get("claim_kind")),
                "paper_ids": [str(x) for x in _list(source.get("paper_ids"))],
                "eligible_as_premise": source.get("eligible_as_premise"),
            }
        )

    verification = row.verification
    semantic_block = verification.get("semantic")
    external_block = verification.get("external_novelty")
    n9_block = verification.get("n9")
    feasibility_block = verification.get("feasibility")
    return {
        "blind_id": blind_id,
        "case_id": row.case_id,
        "arm": row.arm,
        "hypothesis_id": _text(card.get("hypothesis_id")),
        "hypothesis_type": _text(card.get("hypothesis_type")),
        "candidate_dependency": _text(card.get("candidate_dependency")),
        "cross_paper_synthesis": bool(card.get("cross_paper_synthesis", False)),
        "candidate_id": row.candidate_id,
        "candidate_metadata": row.candidate_meta,
        "premises": premises,
        "verification": {
            "semantic_status": _text(
                verification.get("semantic_status")
                or (
                    semantic_block.get("status")
                    if isinstance(semantic_block, dict)
                    else ""
                )
            ),
            "external_novelty_status": _text(
                verification.get("external_novelty_status")
                or (
                    external_block.get("status")
                    if isinstance(external_block, dict)
                    else ""
                )
            ),
            "n9_status": _text(
                verification.get("n9_status")
                or (
                    n9_block.get("status")
                    if isinstance(n9_block, dict)
                    else ""
                )
            ),
            "feasibility_status": _text(
                verification.get("feasibility_status")
                or (
                    feasibility_block.get("status")
                    if isinstance(feasibility_block, dict)
                    else ""
                )
            ),
            "operational_status": _text(verification.get("operational_status")),
        },
        "external_novelty_card": {
            "status": _text(row.external.get("status")),
            "interpretation": _text(row.external.get("interpretation")),
            "reason_codes": [str(x) for x in _list(row.external.get("reason_codes"))],
            "strongest_prior_art_work_ids": [
                str(x)
                for x in _list(row.external.get("strongest_prior_art_work_ids"))
            ],
            "search_limitations": [
                str(x) for x in _list(row.external.get("search_limitations"))
            ],
        },
        "feasibility_card": row.feasibility,
        "source_portfolio_path": str(row.portfolio_path),
        "diagnostic_only": True,
        "production_selection_authority": False,
    }


def build_review_bundle(
    prospective_output_root: Path,
    *,
    per_arm_per_case: int = 2,
    seed: str = "development-human-review-v1",
    paired_case_only: bool = True,
) -> dict[str, Any]:
    if per_arm_per_case < 1:
        raise ValueError("per_arm_per_case must be >= 1")

    root = prospective_output_root.resolve()
    case_ids = discover_reviewable_cases(root)
    selected_rows: list[_ArmHypothesis] = []
    included_cases: list[dict[str, Any]] = []
    excluded_cases: list[dict[str, Any]] = []

    for case_id in case_ids:
        by_arm = {
            arm: _load_arm_hypotheses(root, case_id, arm)
            for arm in ARMS
        }
        counts = {arm: len(rows) for arm, rows in by_arm.items()}
        if paired_case_only and any(counts[arm] == 0 for arm in ARMS):
            excluded_cases.append(
                {
                    "case_id": case_id,
                    "reason": "missing_nonempty_arm_for_paired_review",
                    "hypothesis_count_by_arm": counts,
                }
            )
            continue

        if paired_case_only:
            k = min(per_arm_per_case, *(counts[arm] for arm in ARMS))
        else:
            k = per_arm_per_case

        chosen_counts: dict[str, int] = {}
        for arm in ARMS:
            ranked = sorted(
                by_arm[arm],
                key=lambda row: _stable_digest(
                    seed,
                    "sample",
                    case_id,
                    arm,
                    _text(row.card.get("hypothesis_id")),
                ),
            )
            chosen = ranked[: min(k, len(ranked))]
            selected_rows.extend(chosen)
            chosen_counts[arm] = len(chosen)

        included_cases.append(
            {
                "case_id": case_id,
                "available_hypothesis_count_by_arm": counts,
                "sampled_hypothesis_count_by_arm": chosen_counts,
            }
        )

    selected_rows = sorted(
        selected_rows,
        key=lambda row: _stable_digest(
            seed,
            "global-order",
            row.case_id,
            row.arm,
            _text(row.card.get("hypothesis_id")),
        ),
    )

    blind_items: list[dict[str, Any]] = []
    reveal_items: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []

    for index, row in enumerate(selected_rows, start=1):
        blind_id = f"H-{index:03d}"
        blind_items.append(_blind_item(row, blind_id))
        reveal_items.append(_reveal_item(row, blind_id))
        key_rows.append(
            {
                "blind_id": blind_id,
                "case_id": row.case_id,
                "arm": row.arm,
                "hypothesis_id": _text(row.card.get("hypothesis_id")),
                "candidate_id": row.candidate_id,
            }
        )

    bundle_id = "development_human_review:" + _stable_digest(
        root,
        per_arm_per_case,
        seed,
        *(
            f"{row['blind_id']}:{row['hypothesis_id']}:{row['arm']}"
            for row in key_rows
        ),
    )[:20]

    public_manifest = {
        "schema_version": "development-human-review-manifest-v1",
        "bundle_id": bundle_id,
        "study_kind": "DEVELOPMENT_EXPLORATORY_HUMAN_REVIEW",
        "prospective_output_root": str(root),
        "paired_case_only": paired_case_only,
        "per_arm_per_case": per_arm_per_case,
        "seed_sha256": _stable_digest(seed),
        "sample_count": len(blind_items),
        "included_cases": included_cases,
        "excluded_cases": excluded_cases,
        "rubric": list(RUBRIC),
        "blind_phase_exposes_arm_identity": False,
        "blind_phase_exposes_provenance": False,
        "blind_phase_exposes_verification": False,
        "reveal_phase_is_post_blind_only": True,
        "diagnostic_only": True,
        "scientific_superiority_established": False,
        "production_selection_authority": False,
    }
    return {
        "manifest": public_manifest,
        "key": {
            "schema_version": "development-human-review-key-v1",
            "bundle_id": bundle_id,
            "rows": key_rows,
            "keep_private_until_blind_review_complete": True,
        },
        "blind": {
            "schema_version": "development-human-review-blind-v1",
            "bundle_id": bundle_id,
            "study_kind": "DEVELOPMENT_EXPLORATORY_HUMAN_REVIEW",
            "rubric": list(RUBRIC),
            "items": blind_items,
            "instructions": (
                "Rate scientific content without arm identity, provenance, "
                "prior-art, N9, or feasibility. Export the blind JSON before reveal."
            ),
        },
        "reveal": {
            "schema_version": "development-human-review-reveal-v1",
            "bundle_id": bundle_id,
            "items": reveal_items,
            "rubric": list(RUBRIC),
            "instructions": (
                "Import the completed blind JSON, then inspect provenance and "
                "verification without overwriting the blind ratings."
            ),
            "diagnostic_only": True,
            "production_selection_authority": False,
        },
    }


def _common_css() -> str:
    return r'''
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#17202a;background:#f5f7f9}
*{box-sizing:border-box} body{margin:0}.wrap{max-width:1100px;margin:auto;padding:24px}.hero,.card,.toolbar,.notice{background:white;border:1px solid #dde3e8;border-radius:14px;padding:18px;margin:0 0 16px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
h1{font-size:24px;margin:0 0 8px}h2{font-size:18px;margin:0 0 10px}h3{font-size:15px;margin:18px 0 8px}.muted{color:#66727d}.small{font-size:12px}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.rating{border-top:1px solid #edf0f2;padding:12px 0}.rating-head{display:flex;justify-content:space-between;gap:12px;margin-bottom:8px}.scale{display:flex;gap:8px;flex-wrap:wrap}.scale label{display:flex;align-items:center;gap:4px;border:1px solid #ccd5dc;border-radius:9px;padding:5px 8px;background:#fafbfc}
textarea,input[type=text],select{width:100%;border:1px solid #c9d2d9;border-radius:9px;padding:9px;font:inherit}textarea{min-height:72px}.btn{border:0;border-radius:9px;padding:9px 13px;background:#263746;color:white;cursor:pointer}.btn.secondary{background:#e9eef2;color:#263746}.btn:disabled{opacity:.45;cursor:not-allowed}.badge{display:inline-block;border-radius:999px;padding:3px 8px;background:#edf2f5;margin:2px;font-size:12px}.progress{height:8px;background:#e8edf0;border-radius:999px;overflow:hidden}.progress>div{height:100%;background:#536b7b}.nav{display:flex;justify-content:space-between;gap:10px;align-items:center}.hidden{display:none!important}.evidence{border-left:3px solid #a9b8c2;padding-left:10px;margin:8px 0}.warning{background:#fff8e6;border-color:#ead9a3}.success{background:#eef8f0;border-color:#b7dabd}
table{width:100%;border-collapse:collapse}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #e8ecef;padding:8px}
@media(max-width:760px){.grid{grid-template-columns:1fr}.wrap{padding:12px}}
'''


def render_blind_html(payload: dict[str, Any]) -> str:
    data = _json_for_script(payload)
    css = _common_css()
    title = "Development Human Review — Blind Phase"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{css}</style></head>
<body><div class="wrap">
<div class="hero"><h1>{html.escape(title)}</h1>
<p>Exploratory development review. Arm identity, provenance, prior-art, N9 and feasibility are intentionally hidden.</p>
<p class="muted">Complete the blind ratings and export the JSON before receiving the reveal viewer.</p></div>
<div id="setup" class="card"><h2>Reviewer setup</h2><label>Pseudonymous reviewer code</label><input id="reviewerCode" type="text" placeholder="e.g. reviewer-03">
<p class="small muted">Use a code rather than your name if anonymity is desired.</p><button class="btn" id="startBtn">Start / resume review</button></div>
<div id="reviewApp" class="hidden">
<div class="toolbar"><div class="nav"><button class="btn secondary" id="prevBtn">Previous</button><strong id="counter"></strong><button class="btn secondary" id="nextBtn">Next</button></div><div class="progress"><div id="progressBar"></div></div></div>
<div id="item"></div>
<div class="toolbar"><div class="nav"><span id="completion"></span><button class="btn" id="exportBtn">Export blind review JSON</button></div></div>
</div></div>
<script>
const DATA={data}; const SCALE=[1,2,3,4,5]; let reviewer=""; let idx=0; let state={{}};
function safe(s){{return String(s??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));}}
function storageKey(){{return `human-review-blind:${{DATA.bundle_id}}:${{reviewer}}`;}}
function loadState(){{try{{state=JSON.parse(localStorage.getItem(storageKey())||"{{}}");}}catch(e){{state={{}};}}}}
function saveState(){{localStorage.setItem(storageKey(),JSON.stringify(state));}}
function itemState(id){{return state[id]||(state[id]={{ratings:{{}},strengths:"",concerns:""}});}}
function render(){{
 const x=DATA.items[idx], s=itemState(x.blind_id);
 const preds=(x.predicted_observations||[]).map(p=>`<li><strong>${{safe(p.observable)}}</strong> — ${{safe(p.expected_direction)}}<br><span class="muted">${{safe(p.rationale)}}</span></li>`).join("");
 const fals=(x.falsification_criteria||[]).map(f=>`<li><strong>${{safe(f.observable)}}</strong>: ${{safe(f.falsifying_outcome)}}</li>`).join("");
 const ratings=DATA.rubric.map(r=>`<div class="rating"><div class="rating-head"><strong>${{safe(r.label)}}</strong><span class="small muted">${{safe(r.low)}} ↔ ${{safe(r.high)}}</span></div><div class="scale">${{SCALE.map(v=>`<label><input type="radio" name="${{safe(r.id)}}" value="${{v}}" ${{s.ratings[r.id]===v?"checked":""}}> ${{v}}</label>`).join("")}}</div></div>`).join("");
 document.getElementById("item").innerHTML=`<div class="card"><div><span class="badge">${{safe(x.blind_id)}}</span><span class="badge">${{safe(x.case_id)}}</span><span class="badge">${{safe(x.domain_profile_id)}}</span></div>
 <h2>${{safe(x.title)}}</h2><p class="muted"><strong>Research question:</strong> ${{safe(x.research_question)}}</p>
 <h3>Hypothesis</h3><p>${{safe(x.hypothesis_statement)}}</p><h3>Inferential bridge</h3><p>${{safe(x.inferential_bridge)}}</p>
 <h3>Predictions</h3><ul>${{preds}}</ul><h3>Falsification</h3><ul>${{fals}}</ul>
 ${{(x.assumptions||[]).length?`<h3>Assumptions</h3><ul>${{x.assumptions.map(a=>`<li>${{safe(a)}}</li>`).join("")}}</ul>`:""}}
 <h3>Blind ratings</h3>${{ratings}}<div class="grid"><div><label><strong>Main strength</strong></label><textarea id="strengths">${{safe(s.strengths)}}</textarea></div><div><label><strong>Main concern</strong></label><textarea id="concerns">${{safe(s.concerns)}}</textarea></div></div></div>`;
 document.querySelectorAll('input[type=radio]').forEach(el=>el.onchange=()=>{{s.ratings[el.name]=Number(el.value);saveState();updateProgress();}});
 document.getElementById("strengths").oninput=e=>{{s.strengths=e.target.value;saveState();}};
 document.getElementById("concerns").oninput=e=>{{s.concerns=e.target.value;saveState();}};
 document.getElementById("counter").textContent=`${{idx+1}} / ${{DATA.items.length}}`;
 document.getElementById("prevBtn").disabled=idx===0; document.getElementById("nextBtn").disabled=idx===DATA.items.length-1; updateProgress();
}}
function completeCount(){{return DATA.items.filter(x=>DATA.rubric.every(r=>itemState(x.blind_id).ratings[r.id]>=1)).length;}}
function updateProgress(){{const n=completeCount();document.getElementById("completion").textContent=`Fully rated: ${{n}} / ${{DATA.items.length}}`;document.getElementById("progressBar").style.width=`${{DATA.items.length?100*n/DATA.items.length:0}}%`;}}
function exportJson(){{
 const out={{schema_version:"development-human-review-response-v1",phase:"BLIND",bundle_id:DATA.bundle_id,reviewer_code:reviewer,completed_item_count:completeCount(),item_count:DATA.items.length,ratings:state,exported_at:new Date().toISOString()}};
 const blob=new Blob([JSON.stringify(out,null,2)],{{type:"application/json"}});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=`blind_review_${{reviewer}}.json`;a.click();URL.revokeObjectURL(a.href);
}}
document.getElementById("startBtn").onclick=()=>{{reviewer=document.getElementById("reviewerCode").value.trim();if(!reviewer){{alert("Enter a reviewer code.");return;}}loadState();document.getElementById("setup").classList.add("hidden");document.getElementById("reviewApp").classList.remove("hidden");render();}};
document.getElementById("prevBtn").onclick=()=>{{if(idx>0){{idx--;render();}}}};document.getElementById("nextBtn").onclick=()=>{{if(idx<DATA.items.length-1){{idx++;render();}}}};document.getElementById("exportBtn").onclick=exportJson;
</script></body></html>'''


def render_reveal_html(
    payload: dict[str, Any],
    blind_items: dict[str, dict[str, Any]],
) -> str:
    data = _json_for_script(payload)
    blind = _json_for_script(blind_items)
    css = _common_css()
    title = "Development Human Review — Reveal Phase"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{css}</style></head>
<body><div class="wrap"><div class="hero"><h1>{html.escape(title)}</h1><p>Use this only after completing and exporting the blind review.</p></div>
<div class="card"><input type="file" id="blindFile" accept=".json"><button class="btn" id="loadBtn">Load blind review</button><p id="importStatus" class="muted"></p></div>
<div id="app" class="hidden"><div id="items"></div><div class="toolbar"><button class="btn" id="exportBtn">Export reveal review JSON</button></div></div></div>
<script>
const DATA={data}; const BLIND_ITEMS={blind}; let blindResponse=null; let post={{}};
function safe(s){{return String(s??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));}}
function badge(label,value){{return value?`<span class="badge">${{safe(label)}}: ${{safe(value)}}</span>`:"";}}
function render(){{
 const root=document.getElementById("items");root.innerHTML="";
 DATA.items.forEach(r=>{{
  const b=BLIND_ITEMS[r.blind_id]||{{}}; const prior=(blindResponse.ratings||{{}})[r.blind_id]||{{ratings:{{}}}}; const p=post[r.blind_id]||(post[r.blind_id]={{trust_delta:0,changed:"",comment:""}});
  const ratings=DATA.rubric.map(x=>`${{safe(x.label)}}=${{prior.ratings?.[x.id]??"NA"}}`).join(" · ");
  const premises=(r.premises||[]).map(x=>`<div class="evidence"><strong>${{safe(x.text||x.statement_id)}}</strong><br><span class="small muted">${{safe(x.epistemic_role)}} · ${{safe((x.paper_ids||[]).join(", "))}}</span></div>`).join("");
  const ext=r.external_novelty_card||{{}}, v=r.verification||{{}}, meta=r.candidate_metadata||{{}};
  const c=document.createElement("div");c.className="card";c.innerHTML=`<div><span class="badge">${{safe(r.blind_id)}}</span><span class="badge">${{safe(r.case_id)}}</span><span class="badge">${{safe(r.arm)}}</span></div>
  <h2>${{safe(b.title)}}</h2><p>${{safe(b.hypothesis_statement)}}</p><div class="notice"><strong>Blind ratings</strong><p class="small">${{safe(ratings)}}</p></div>
  <h3>Lineage / selection</h3><p>${{badge("origin",meta.origin)}}${{badge("form",meta.idea_form)}}${{badge("operator",meta.operator_id)}}${{badge("profile",meta.selection_profile||meta.profile)}}${{badge("hypothesis type",r.hypothesis_type)}}</p>
  <h3>Grounded premises</h3>${{premises||'<p class="muted">No premise detail resolved.</p>'}}
  <h3>Verification</h3><p>${{badge("semantic",v.semantic_status)}}${{badge("external",v.external_novelty_status)}}${{badge("N9",v.n9_status)}}${{badge("feasibility",v.feasibility_status)}}</p>
  ${{ext.status?`<p><strong>Prior-art category:</strong> ${{safe(ext.status)}}</p>`:""}}${{ext.interpretation?`<p>${{safe(ext.interpretation)}}</p>`:""}}
  <div class="rating"><strong>After seeing provenance/verification, how did your confidence change?</strong><div class="scale">${{[-2,-1,0,1,2].map(n=>`<label><input type="radio" name="delta-${{r.blind_id}}" value="${{n}}" ${{p.trust_delta===n?"checked":""}}> ${{n>0?"+":""}}${{n}}</label>`).join("")}}</div></div>
  <label><strong>Did the additional information materially change your assessment?</strong></label><select data-changed="${{r.blind_id}}"><option value="">Choose</option><option value="NO" ${{p.changed==="NO"?"selected":""}}>No</option><option value="YES" ${{p.changed==="YES"?"selected":""}}>Yes</option></select>
  <p><label><strong>Post-reveal comment</strong></label><textarea data-comment="${{r.blind_id}}">${{safe(p.comment)}}</textarea></p>`;
  root.appendChild(c);
  c.querySelectorAll(`input[name="delta-${{CSS.escape(r.blind_id)}}"]`).forEach(el=>el.onchange=()=>{{p.trust_delta=Number(el.value);}});
  c.querySelector(`[data-changed="${{CSS.escape(r.blind_id)}}"]`).onchange=e=>{{p.changed=e.target.value;}};
  c.querySelector(`[data-comment="${{CSS.escape(r.blind_id)}}"]`).oninput=e=>{{p.comment=e.target.value;}};
 }});
}}
async function loadBlind(){{
 const file=document.getElementById("blindFile").files[0];if(!file){{alert("Choose the blind-review JSON.");return;}}
 try{{blindResponse=JSON.parse(await file.text());}}catch(e){{alert("Invalid JSON.");return;}}
 if(blindResponse.bundle_id!==DATA.bundle_id||blindResponse.phase!=="BLIND"){{alert("This blind review does not match this reveal bundle.");return;}}
 document.getElementById("importStatus").textContent=`Loaded reviewer ${{blindResponse.reviewer_code}} · ${{blindResponse.completed_item_count}}/${{blindResponse.item_count}} items complete`;document.getElementById("app").classList.remove("hidden");render();
}}
function exportJson(){{
 if(!blindResponse)return; const out={{schema_version:"development-human-review-response-v1",phase:"REVEAL",bundle_id:DATA.bundle_id,reviewer_code:blindResponse.reviewer_code,blind_response:blindResponse,post_reveal:post,exported_at:new Date().toISOString()}};
 const blob=new Blob([JSON.stringify(out,null,2)],{{type:"application/json"}});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=`reveal_review_${{blindResponse.reviewer_code}}.json`;a.click();URL.revokeObjectURL(a.href);
}}
document.getElementById("loadBtn").onclick=loadBlind;document.getElementById("exportBtn").onclick=exportJson;
</script></body></html>'''


def write_review_bundle(
    bundle: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output_dir / "review_manifest.json",
        "key": output_dir / "review_key.PRIVATE.json",
        "blind_json": output_dir / "blind_payload.json",
        "reveal_json": output_dir / "reveal_payload.json",
        "blind_html": output_dir / "blind_review.html",
        "reveal_html": output_dir / "reveal_review.html",
        "readme": output_dir / "README.txt",
    }
    paths["manifest"].write_text(
        json.dumps(bundle["manifest"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["key"].write_text(
        json.dumps(bundle["key"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["blind_json"].write_text(
        json.dumps(bundle["blind"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["reveal_json"].write_text(
        json.dumps(bundle["reveal"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["blind_html"].write_text(
        render_blind_html(bundle["blind"]),
        encoding="utf-8",
    )
    blind_lookup = {row["blind_id"]: row for row in bundle["blind"]["items"]}
    paths["reveal_html"].write_text(
        render_reveal_html(bundle["reveal"], blind_lookup),
        encoding="utf-8",
    )
    paths["readme"].write_text(
        "\n".join(
            [
                "Development Human Review Viewer",
                "===============================",
                "",
                "1. Send blind_review.html first.",
                "2. Reviewer uses a pseudonymous code and exports blind JSON.",
                "3. Collect blind JSON before sending reveal_review.html.",
                "4. Reveal phase imports blind JSON and shows provenance/verification.",
                "5. Keep review_key.PRIVATE.json private until blind review is complete.",
                "",
                "Exploratory development review only; not held-out validation.",
                "No production-selection or novelty authority is created.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


def aggregate_review_responses(
    *,
    key_path: Path,
    response_paths: Iterable[Path],
) -> dict[str, Any]:
    response_paths = list(response_paths)
    key = _load_json(key_path)
    by_blind = {
        _text(row.get("blind_id")): dict(row)
        for row in _list(key.get("rows"))
        if isinstance(row, dict) and _text(row.get("blind_id"))
    }

    # Deduplicate reviewer/item responses. If both the original blind JSON and
    # a later reveal JSON are present, the reveal response wins because it
    # embeds the frozen blind response and adds post-reveal fields.
    response_rows: dict[tuple[str, str], dict[str, Any]] = {}

    for path in response_paths:
        payload = _load_json(path)
        phase = _text(payload.get("phase"))
        if phase == "REVEAL":
            response = payload.get("blind_response")
            if not isinstance(response, dict):
                continue
            reveal = payload
            priority = 2
        elif phase == "BLIND":
            response = payload
            reveal = {}
            priority = 1
        else:
            continue

        if _text(response.get("bundle_id")) != _text(key.get("bundle_id")):
            raise HumanReviewError(f"Bundle mismatch in response: {path}")

        reviewer = _text(response.get("reviewer_code"), path.stem)
        ratings = response.get("ratings")
        if not isinstance(ratings, dict):
            ratings = {}
        post = reveal.get("post_reveal")
        if not isinstance(post, dict):
            post = {}

        for blind_id, answer in ratings.items():
            meta = by_blind.get(blind_id)
            if meta is None or not isinstance(answer, dict):
                continue
            key_tuple = (reviewer, blind_id)
            existing = response_rows.get(key_tuple)
            if existing is not None and existing["_priority"] > priority:
                continue
            response_rows[key_tuple] = {
                "_priority": priority,
                "reviewer_code": reviewer,
                "blind_id": blind_id,
                "answer": answer,
                "meta": meta,
                "post": post.get(blind_id) if isinstance(post.get(blind_id), dict) else {},
            }

    reviewer_rows: list[dict[str, Any]] = []
    arm_values: dict[str, dict[str, list[float]]] = {
        arm: {r["id"]: [] for r in RUBRIC} for arm in ARMS
    }
    case_arm_values: dict[str, dict[str, dict[str, list[float]]]] = {}

    for packed in response_rows.values():
        reviewer = packed["reviewer_code"]
        blind_id = packed["blind_id"]
        answer = packed["answer"]
        meta = packed["meta"]
        post = packed["post"]

        arm = _text(meta.get("arm"))
        case_id = _text(meta.get("case_id"))
        values = answer.get("ratings")
        if not isinstance(values, dict):
            values = {}

        row = {
            "reviewer_code": reviewer,
            "blind_id": blind_id,
            "case_id": case_id,
            "arm": arm,
            "hypothesis_id": meta.get("hypothesis_id"),
            "ratings": {},
            "trust_delta": post.get("trust_delta"),
        }
        for rubric in RUBRIC:
            rid = rubric["id"]
            value = values.get(rid)
            if isinstance(value, (int, float)) and 1 <= float(value) <= 5:
                fv = float(value)
                row["ratings"][rid] = fv
                if arm in arm_values:
                    arm_values[arm][rid].append(fv)
                case_arm_values.setdefault(case_id, {}).setdefault(
                    arm, {r["id"]: [] for r in RUBRIC}
                )[rid].append(fv)
        reviewer_rows.append(row)

    def summarize(values: dict[str, list[float]]) -> dict[str, Any]:
        return {
            rid: {
                "n": len(xs),
                "mean": (sum(xs) / len(xs)) if xs else None,
            }
            for rid, xs in values.items()
        }

    trust_by_arm: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for row in reviewer_rows:
        delta = row.get("trust_delta")
        arm = row.get("arm")
        if isinstance(delta, (int, float)) and arm in trust_by_arm:
            trust_by_arm[arm].append(float(delta))

    return {
        "schema_version": "development-human-review-aggregate-v1",
        "bundle_id": key.get("bundle_id"),
        "response_file_count": len(response_paths),
        "unique_reviewer_item_count": len(response_rows),
        "rating_row_count": len(reviewer_rows),
        "arm_summary": {
            arm: summarize(values) for arm, values in arm_values.items()
        },
        "case_arm_summary": {
            case_id: {
                arm: summarize(values) for arm, values in arms.items()
            }
            for case_id, arms in case_arm_values.items()
        },
        "post_reveal_trust_delta_by_arm": {
            arm: {
                "n": len(xs),
                "mean": (sum(xs) / len(xs)) if xs else None,
            }
            for arm, xs in trust_by_arm.items()
        },
        "rows": reviewer_rows,
        "development_exploratory_only": True,
        "scientific_superiority_established": False,
        "production_selection_authority": False,
    }
