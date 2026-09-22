from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


class ScientificCertificationViewerError(RuntimeError):
    pass


def _read_json(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ScientificCertificationViewerError(
                f"Required viewer artifact not found: {path}"
            )
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScientificCertificationViewerError(
            f"Invalid JSON in {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ScientificCertificationViewerError(
            f"Expected JSON object in {path}"
        )
    return value


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _semantic_rows(
    review: dict[str, Any],
    hypothesis_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _list(review.get("dimensions")):
        if not isinstance(row, dict):
            continue
        ids = [
            str(value)
            for value in _list(row.get("hypothesis_ids"))
            if str(value).strip()
        ]
        if ids and hypothesis_id not in ids:
            continue
        rows.append(dict(row))
    return rows


def _semantic_status(rows: list[dict[str, Any]]) -> str:
    verdicts = {
        _text(row.get("verdict")).lower()
        for row in rows
    }
    if "fail" in verdicts:
        return "fail"
    if "warning" in verdicts:
        return "pass_with_warnings"
    if rows:
        return "pass"
    return "not_assessed"


def _certification_by_hypothesis(
    report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _list(report.get("decisions")):
        if not isinstance(row, dict):
            continue
        hypothesis_id = (
            _text(row.get("hypothesis_id"))
            or _text(row.get("final_hypothesis_id"))
        )
        if hypothesis_id:
            result[hypothesis_id] = dict(row)
    return result


def _external_by_hypothesis(
    report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for card in _list(report.get("cards")):
        if not isinstance(card, dict):
            continue
        hypothesis_id = _text(card.get("hypothesis_id"))
        if hypothesis_id:
            result[hypothesis_id] = dict(card)
    return result


def _atomic_by_hypothesis(
    report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _list(report.get("hypotheses")):
        if not isinstance(row, dict):
            continue
        hypothesis_id = _text(row.get("hypothesis_id"))
        if hypothesis_id:
            result[hypothesis_id] = dict(row)
    return result


def load_scientific_certification_payload(
    *,
    run_dir: Path,
    candidate_portfolio: Path | None = None,
    certification_report: Path | None = None,
    semantic_review: Path | None = None,
    certified_portfolio: Path | None = None,
    atomic_report: Path | None = None,
    external_novelty_report: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()

    candidate_path = (
        candidate_portfolio
        or run_dir / "scientific_atomic_merged_candidates.portfolio.json"
    )
    certification_path = (
        certification_report
        or run_dir / "scientific_atomic_n10_certification.report.json"
    )
    semantic_path = (
        semantic_review
        or run_dir / "scientific_atomic_merged_final_semantic.review.json"
    )
    certified_path = (
        certified_portfolio
        or run_dir / "scientific_atomic_merged_certified.portfolio.json"
    )
    atomic_path = (
        atomic_report
        or run_dir / "scientific_atomic_cross_lane_v2.report.json"
    )
    external_path = (
        external_novelty_report
        or run_dir / "scientific_atomic_n10_external.report.json"
    )
    context_path = run_dir / "hypothesis.context.json"
    merge_report_path = (
        run_dir
        / "scientific_atomic_merged_certification_authority_report.json"
    )

    candidates = _read_json(candidate_path, required=True)
    certification = _read_json(certification_path, required=True)
    semantic = _read_json(semantic_path)
    certified = _read_json(certified_path)
    atomic = _read_json(atomic_path)
    external = _read_json(external_path)
    context = _read_json(context_path)
    merge_report = _read_json(merge_report_path)

    certification_by_id = _certification_by_hypothesis(certification)
    external_by_id = _external_by_hypothesis(external)
    atomic_by_id = _atomic_by_hypothesis(atomic)
    certified_ids = {
        _text(row.get("hypothesis_id"))
        for row in _list(certified.get("hypotheses"))
        if isinstance(row, dict)
        and _text(row.get("hypothesis_id"))
    }

    hypotheses: list[dict[str, Any]] = []
    for card in _list(candidates.get("hypotheses")):
        if not isinstance(card, dict):
            continue
        hypothesis_id = _text(card.get("hypothesis_id"))
        if not hypothesis_id:
            continue

        semantic_rows = _semantic_rows(
            semantic,
            hypothesis_id,
        )
        certification_row = certification_by_id.get(
            hypothesis_id,
            {},
        )

        if certification_row:
            certification_status = _text(
                certification_row.get("certification_status"),
                "not_assessed",
            )
        elif hypothesis_id in certified_ids:
            # Scientific candidates normally have explicit certification rows.
            # A card present only through strict legacy authority remains
            # distinguishable rather than being relabeled as scientific N10.
            certification_status = "LEGACY_STRICT_N10"
        else:
            certification_status = "not_assessed"

        predictions = [
            dict(row)
            for row in _list(card.get("predicted_observations"))
            if isinstance(row, dict)
        ]
        falsifiers = [
            dict(row)
            for row in _list(card.get("falsification_criteria"))
            if isinstance(row, dict)
        ]

        atomic_row = atomic_by_id.get(hypothesis_id, {})
        atomic_specs = [
            dict(row)
            for row in _list(atomic_row.get("atomic_specifications"))
            if isinstance(row, dict)
        ]

        hypotheses.append(
            {
                "hypothesis": {
                    "hypothesis_id": hypothesis_id,
                    "title": _text(card.get("title"), hypothesis_id),
                    "statement": _text(card.get("hypothesis_statement")),
                    "hypothesis_type": _text(
                        card.get("hypothesis_type"),
                        "hypothesis",
                    ),
                    "inferential_bridge": _text(
                        card.get("inferential_bridge")
                    ),
                    "assumptions": [
                        str(value)
                        for value in _list(card.get("assumptions"))
                        if str(value).strip()
                    ],
                    "premise_statement_ids": [
                        str(value)
                        for value in _list(
                            card.get("premise_statement_ids")
                        )
                        if str(value).strip()
                    ],
                    "gap_statement_ids": [
                        str(value)
                        for value in _list(
                            card.get("gap_statement_ids")
                        )
                        if str(value).strip()
                    ],
                    "source_paper_ids": [
                        str(value)
                        for value in _list(card.get("source_paper_ids"))
                        if str(value).strip()
                    ],
                    "predictions": predictions,
                    "falsifiers": falsifiers,
                    "evidence_profile": (
                        dict(card.get("evidence_profile"))
                        if isinstance(card.get("evidence_profile"), dict)
                        else {}
                    ),
                },
                "semantic": {
                    "status": _semantic_status(semantic_rows),
                    "dimensions": semantic_rows,
                },
                "certification": {
                    **dict(certification_row),
                    "certification_status": certification_status,
                },
                "external_novelty": dict(
                    external_by_id.get(hypothesis_id, {})
                ),
                "atomic": {
                    "atomic_specifications": atomic_specs,
                    "source_candidate_ids": list(
                        atomic_row.get("source_candidate_ids") or []
                    ),
                },
            }
        )

    return {
        "viewer_schema":
            "scientific-certification-viewer-v1",
        "question": _text(
            context.get("question"),
            "Scientific discovery candidates",
        ),
        "domain_profile_id": (
            _text(candidates.get("domain_profile_id"))
            or _text(context.get("domain_profile_id"))
            or "unknown"
        ),
        "context_id": (
            _text(candidates.get("source_context_id"))
            or _text(context.get("context_id"))
        ),
        "candidate_portfolio_id": _text(
            candidates.get("portfolio_id")
        ),
        "certified_portfolio_id": _text(
            certified.get("portfolio_id")
        ),
        "candidate_count": len(hypotheses),
        "certified_count": int(
            certification.get("certified_count")
            or merge_report.get("scientific_certified_count")
            or 0
        ),
        "unresolved_count": int(
            certification.get("unresolved_count") or 0
        ),
        "rejected_count": int(
            certification.get("rejected_count") or 0
        ),
        "legacy_strict_count": int(
            merge_report.get("legacy_strict_authority_count") or 0
        ),
        "semantic_available": bool(semantic),
        "semantic_overall_summary": _text(
            semantic.get("overall_summary")
        ),
        "artifact_paths": {
            "candidate_portfolio": str(candidate_path),
            "certification_report": str(certification_path),
            "certified_portfolio": str(certified_path),
            "semantic_review": str(semantic_path),
            "atomic_report": str(atomic_path),
            "external_novelty_report": str(external_path),
            "merge_authority_report": str(merge_report_path),
        },
        "hypotheses": hypotheses,
    }


def _json_for_script(value: dict[str, Any]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_scientific_certification_html(
    payload: dict[str, Any],
    *,
    title: str = "ForAllKG Scientific Candidate & Novelty Certification Viewer",
) -> str:
    safe_title = html.escape(title)
    data = _json_for_script(payload)

    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg:#f5f7fb; --panel:#fff; --text:#172033; --muted:#667085;
  --border:#d9e0ea; --accent:#315efb; --soft:#eef2ff;
  --good:#18794e; --warn:#9a6700; --bad:#b42318;
}
@media(prefers-color-scheme:dark){
  :root {
    --bg:#10131a; --panel:#171b24; --text:#eef2f7; --muted:#a8b0be;
    --border:#303848; --accent:#7c9cff; --soft:#202b4d;
    --good:#72d5a4; --warn:#f5c45b; --bad:#ff8c82;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{background:var(--panel);border-bottom:1px solid var(--border);padding:18px 22px}
h1{font-size:20px;margin:0 0 5px}.muted{color:var(--muted)}
.metrics{display:flex;flex-wrap:wrap;gap:8px;margin-top:13px}
.metric{border:1px solid var(--border);border-radius:10px;padding:7px 10px;min-width:115px}
.metric b{display:block;font-size:17px}
.layout{display:grid;grid-template-columns:320px minmax(0,1fr);min-height:calc(100vh - 125px)}
aside{background:var(--panel);border-right:1px solid var(--border);padding:14px}
main{padding:18px 20px 36px;min-width:0}
.hyp-btn{width:100%;text-align:left;border:1px solid var(--border);background:transparent;color:var(--text);border-radius:11px;padding:10px;margin-bottom:8px;cursor:pointer}
.hyp-btn.active{border-color:var(--accent);background:var(--soft)}
.badge{display:inline-block;border:1px solid var(--border);border-radius:999px;padding:2px 7px;margin:2px 4px 2px 0;font-size:11px}
.badge.good{color:var(--good)}.badge.warn{color:var(--warn)}.badge.bad{color:var(--bad)}
.card{background:var(--panel);border:1px solid var(--border);border-radius:13px;padding:14px;margin-bottom:12px}
.card h3{margin:0 0 9px;font-size:14px}.hero h2{font-size:19px;margin:0 0 6px}
.tabs{display:flex;gap:5px;border-bottom:1px solid var(--border);margin:14px 0;overflow:auto}
.tab{border:0;background:transparent;color:var(--muted);padding:9px 10px;cursor:pointer}
.tab.active{color:var(--text);border-bottom:2px solid var(--accent)}
.panel{display:none}.panel.active{display:block}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.quote{border-left:3px solid var(--accent);background:var(--soft);padding:8px 10px;border-radius:6px;margin:7px 0}
.section-label{text-transform:uppercase;font-size:11px;color:var(--muted);letter-spacing:.04em}
table{width:100%;border-collapse:collapse}th,td{border-bottom:1px solid var(--border);padding:8px;text-align:left;vertical-align:top}
th{font-size:11px;color:var(--muted)}
ul{padding-left:20px;margin:6px 0}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.notice{border:1px solid var(--border);background:var(--soft);border-radius:10px;padding:10px;margin-bottom:12px}
@media(max-width:900px){.layout{grid-template-columns:1fr}.grid2,.grid3{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid var(--border)}}
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div id="question"></div>
  <div id="meta" class="muted"></div>
  <div id="metrics" class="metrics"></div>
</header>
<div class="layout">
<aside>
  <div class="muted" style="margin-bottom:8px">Discovery candidates</div>
  <div id="hypList"></div>
</aside>
<main>
  <div class="notice">
    Candidate retention and novelty certification are separate authorities.
    A retained candidate is not automatically novelty-certified.
  </div>
  <div class="hero">
    <h2 id="title"></h2>
    <div id="statement"></div>
    <div id="badges" style="margin-top:9px"></div>
  </div>
  <div class="tabs">
    <button class="tab active" data-tab="overview">Overview</button>
    <button class="tab" data-tab="certification">N10 certification</button>
    <button class="tab" data-tab="semantic">Semantic review</button>
    <button class="tab" data-tab="atomic">Atomic specification</button>
    <button class="tab" data-tab="tests">Predictions & falsifiers</button>
    <button class="tab" data-tab="provenance">Provenance</button>
  </div>
  <section class="panel active" id="panel-overview"></section>
  <section class="panel" id="panel-certification"></section>
  <section class="panel" id="panel-semantic"></section>
  <section class="panel" id="panel-atomic"></section>
  <section class="panel" id="panel-tests"></section>
  <section class="panel" id="panel-provenance"></section>
</main>
</div>

<script id="viewer-data" type="application/json">__DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('viewer-data').textContent);
let selected=0;
const arr=v=>Array.isArray(v)?v:[];
const text=(v,d='')=>typeof v==='string'?v:d;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const human=s=>text(s).replaceAll('_',' ');
function cls(s){
  s=text(s).toLowerCase();
  if(s.includes('certified')||s==='pass'||s==='eligible')return'good';
  if(s.includes('reject')||s==='fail'||s==='ineligible')return'bad';
  if(s.includes('unresolved')||s.includes('warning')||s.includes('conditional')||s.includes('insufficient'))return'warn';
  return'';
}
const badge=(v,label=null)=>`<span class="badge ${cls(v)}">${esc(label??human(v||'not assessed'))}</span>`;
const list=xs=>arr(xs).length?`<ul>${arr(xs).map(x=>`<li>${esc(typeof x==='string'?x:JSON.stringify(x))}</li>`).join('')}</ul>`:'<span class="muted">None</span>';

function renderHeader(){
  document.getElementById('question').textContent=DATA.question||'Scientific discovery run';
  document.getElementById('meta').textContent=`Domain: ${DATA.domain_profile_id||'unknown'} · Candidate portfolio: ${DATA.candidate_portfolio_id||'n/a'}`;
  const metrics=[
    ['Candidates',DATA.candidate_count||0],
    ['Certified',DATA.certified_count||0],
    ['Unresolved',DATA.unresolved_count||0],
    ['Rejected',DATA.rejected_count||0],
    ['Legacy strict',DATA.legacy_strict_count||0],
  ];
  document.getElementById('metrics').innerHTML=metrics.map(([k,v])=>`<div class="metric"><b>${esc(v)}</b><span class="muted">${esc(k)}</span></div>`).join('');
}
function renderSidebar(){
  document.getElementById('hypList').innerHTML=arr(DATA.hypotheses).map((row,i)=>{
    const h=row.hypothesis||{}, s=row.semantic||{}, c=row.certification||{};
    return `<button class="hyp-btn ${i===selected?'active':''}" data-i="${i}">
      <div class="muted">H${i+1} · ${esc(h.hypothesis_type||'hypothesis')}</div>
      <b>${esc(h.title||h.hypothesis_id)}</b>
      <div>${badge(s.status,'semantic: '+human(s.status))}${badge(c.certification_status,'N10: '+human(c.certification_status))}</div>
    </button>`;
  }).join('');
  document.querySelectorAll('.hyp-btn').forEach(b=>b.onclick=()=>{selected=Number(b.dataset.i);render();});
}
function renderOverview(row){
  const h=row.hypothesis||{}, x=row.external_novelty||{};
  document.getElementById('panel-overview').innerHTML=`
    <div class="grid2">
      <div class="card">
        <h3>Scientific proposal</h3>
        <div class="quote">${esc(h.statement||'')}</div>
        <div class="section-label">Inferential bridge</div>
        <p>${esc(h.inferential_bridge||'')}</p>
        <div class="section-label">Assumptions</div>${list(h.assumptions)}
      </div>
      <div class="card">
        <h3>External novelty context</h3>
        ${badge(x.status||'not_assessed')}
        <p>${esc(x.interpretation||'No directly bound external-novelty card.')}</p>
        <div class="section-label">Reason codes</div>${list(x.reason_codes)}
      </div>
    </div>
    <div class="card">
      <h3>Evidence binding</h3>
      <div class="grid3">
        <div><div class="section-label">Premise statement IDs</div>${list(h.premise_statement_ids)}</div>
        <div><div class="section-label">Gap statement IDs</div>${list(h.gap_statement_ids)}</div>
        <div><div class="section-label">Source papers</div>${list(h.source_paper_ids)}</div>
      </div>
    </div>`;
}
function renderCertification(row){
  const c=row.certification||{};
  document.getElementById('panel-certification').innerHTML=`
    <div class="grid2">
      <div class="card">
        <h3>Novelty certification</h3>
        <div>${badge(c.certification_status)}</div>
        <table>
          <tbody>
            <tr><th>Selection class</th><td>${badge(c.selection_class)}</td></tr>
            <tr><th>Positive N10 authority</th><td>${esc(c.positive_nonobviousness_authority??false)}</td></tr>
            <tr><th>Candidate retained</th><td>${esc(c.candidate_retained??true)}</td></tr>
            <tr><th>Action</th><td>${esc(c.action||'')}</td></tr>
          </tbody>
        </table>
      </div>
      <div class="card">
        <h3>Unresolved dimensions</h3>
        ${list(c.unresolved_dimensions)}
        <div class="section-label" style="margin-top:12px">Reason codes</div>
        ${list(c.reason_codes)}
      </div>
    </div>
    <div class="notice">
      Semantic or feasibility acceptance does not upgrade this novelty status.
      Only positive N10 authority can enter the certified subset.
    </div>`;
}
function renderSemantic(row){
  const s=row.semantic||{};
  const body=arr(s.dimensions).map(r=>`<tr>
    <td>${esc(human(r.dimension||''))}</td>
    <td>${badge(r.verdict)}</td>
    <td>${esc(r.rationale||'')}</td>
  </tr>`).join('');
  document.getElementById('panel-semantic').innerHTML=`
    <div class="card">
      <h3>Semantic critic ${badge(s.status)}</h3>
      <div class="muted">${esc(DATA.semantic_overall_summary||'')}</div>
      <table>
        <thead><tr><th>Dimension</th><th>Verdict</th><th>Rationale</th></tr></thead>
        <tbody>${body||'<tr><td colspan="3">No semantic review rows</td></tr>'}</tbody>
      </table>
    </div>`;
}
function renderAtomic(row){
  const specs=arr((row.atomic||{}).atomic_specifications);
  const cards=specs.map((s,i)=>`<div class="card">
    <h3>Atomic claim ${i+1} ${badge(s.novelty_selection_role||s.importance||'')}</h3>
    <div class="quote">${esc(s.text||'')}</div>
    <div class="grid2">
      <div>
        <div class="section-label">Prior-art identity</div>${list(s.prior_art_identity_terms)}
        <div class="section-label">Relation endpoints</div>${list(s.relation_endpoint_anchors)}
        <div class="section-label">Relation nucleus</div>${list(s.relation_nucleus_terms)}
      </div>
      <div>
        <div class="section-label">Required bridge</div><p>${esc(s.required_bridge||'')}</p>
        <div class="section-label">Predicted observation</div><p>${esc(s.predicted_observation||'')}</p>
        <div class="section-label">Falsification condition</div><p>${esc(s.falsification_condition||'')}</p>
      </div>
    </div>
    <pre>${esc(JSON.stringify({claim_id:s.claim_id,source_candidate_ids:s.source_candidate_ids||[],scientific_structure:s.scientific_structure||{}},null,2))}</pre>
  </div>`).join('');
  document.getElementById('panel-atomic').innerHTML=cards||'<div class="card muted">No atomic synthesis report bound to this hypothesis.</div>';
}
function renderTests(row){
  const h=row.hypothesis||{};
  const preds=arr(h.predictions).map(p=>`<div class="card"><h3>${esc(p.observable||'Prediction')}</h3>${badge(p.expected_direction)}<p>${esc(p.rationale||'')}</p></div>`).join('');
  const fals=arr(h.falsifiers).map(f=>`<div class="card"><h3>${esc(f.observable||'Falsifier')}</h3><div class="quote">${esc(f.falsifying_outcome||'')}</div></div>`).join('');
  document.getElementById('panel-tests').innerHTML=`<div class="grid2"><div><h3>Predictions</h3>${preds||'<span class="muted">None</span>'}</div><div><h3>Falsifiers</h3>${fals||'<span class="muted">None</span>'}</div></div>`;
}
function renderProvenance(row){
  const h=row.hypothesis||{};
  document.getElementById('panel-provenance').innerHTML=`
    <div class="grid2">
      <div class="card"><h3>Hypothesis identity</h3><pre>${esc(JSON.stringify({hypothesis_id:h.hypothesis_id,evidence_profile:h.evidence_profile||{}},null,2))}</pre></div>
      <div class="card"><h3>Artifact paths</h3><pre>${esc(JSON.stringify(DATA.artifact_paths||{},null,2))}</pre></div>
    </div>`;
}
function renderMain(){
  const row=arr(DATA.hypotheses)[selected]; if(!row)return;
  const h=row.hypothesis||{}, s=row.semantic||{}, c=row.certification||{};
  document.getElementById('title').textContent=`H${selected+1} · ${h.title||h.hypothesis_id||'Hypothesis'}`;
  document.getElementById('statement').textContent=h.statement||'';
  document.getElementById('badges').innerHTML=
    badge(s.status,'semantic: '+human(s.status))+
    badge(c.certification_status,'N10: '+human(c.certification_status));
  renderOverview(row);renderCertification(row);renderSemantic(row);
  renderAtomic(row);renderTests(row);renderProvenance(row);
}
function setupTabs(){
  document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('panel-'+t.dataset.tab).classList.add('active');
  });
}
function render(){renderSidebar();renderMain();}
renderHeader();setupTabs();render();
</script>
</body>
</html>"""
    return (
        template
        .replace("__TITLE__", safe_title)
        .replace("__DATA__", data)
    )


def build_scientific_certification_viewer(
    *,
    run_dir: Path,
    output: Path,
    title: str = (
        "ForAllKG Scientific Candidate & Novelty Certification Viewer"
    ),
    candidate_portfolio: Path | None = None,
    certification_report: Path | None = None,
    semantic_review: Path | None = None,
    certified_portfolio: Path | None = None,
    atomic_report: Path | None = None,
    external_novelty_report: Path | None = None,
) -> Path:
    payload = load_scientific_certification_payload(
        run_dir=run_dir,
        candidate_portfolio=candidate_portfolio,
        certification_report=certification_report,
        semantic_review=semantic_review,
        certified_portfolio=certified_portfolio,
        atomic_report=atomic_report,
        external_novelty_report=external_novelty_report,
    )
    rendered = render_scientific_certification_html(
        payload,
        title=title,
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return output


__all__ = [
    "ScientificCertificationViewerError",
    "build_scientific_certification_viewer",
    "load_scientific_certification_payload",
    "render_scientific_certification_html",
]
