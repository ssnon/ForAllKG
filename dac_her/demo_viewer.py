from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


class DemoViewerError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DemoViewerError(f"Required demo artifact not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DemoViewerError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DemoViewerError(f"Expected a JSON object in {path}")
    return value


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DemoViewerError(f"Invalid JSON in {path}: {exc}") from exc
    return value if isinstance(value, dict) else None


def _artifact_by_hypothesis(directory: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.json")):
        value = _read_json_if_exists(path)
        if not value:
            continue
        hypothesis_id = value.get("hypothesis_id")
        if isinstance(hypothesis_id, str) and hypothesis_id:
            rows[hypothesis_id] = value
    return rows


def discover_feasibility_dir(run_dir: Path) -> Path:
    """Locate a v0.2 feasibility output directory from an E2E run root."""
    run_dir = run_dir.resolve()
    candidates = [
        run_dir,
        run_dir / "feasibility_v02",
        run_dir / "feasibility_v0.2",
        run_dir / "feasibility",
    ]
    candidates.extend(sorted(p for p in run_dir.glob("feasibility*") if p.is_dir()))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (
            (candidate / "feasibility" / "intake.json").exists()
            and (candidate / "decision" / "portfolio.json").exists()
        ):
            return candidate
    raise DemoViewerError(
        "Could not locate feasibility artifacts. Expected <dir>/feasibility/intake.json "
        "and <dir>/decision/portfolio.json. Pass --feasibility-dir explicitly."
    )


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _decision_by_hypothesis(portfolio: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for card in _list(portfolio.get("cards")):
        if not isinstance(card, dict):
            continue
        hypothesis_id = card.get("hypothesis_id")
        if isinstance(hypothesis_id, str):
            rows[hypothesis_id] = card
    return rows


def load_demo_payload(
    feasibility_dir: Path,
    *,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Load v2.7.0 feasibility artifacts into a schema-tolerant viewer payload.

    The viewer deliberately consumes JSON rather than importing Pydantic contracts.
    This keeps the demo usable across small contract-version changes while preserving
    the source artifact IDs verbatim.
    """
    feasibility_dir = feasibility_dir.resolve()
    intake = _read_json(feasibility_dir / "feasibility" / "intake.json")
    portfolio = _read_json(feasibility_dir / "decision" / "portfolio.json")
    manifest = _read_json_if_exists(feasibility_dir / "manifest.json") or {}

    scopes = _artifact_by_hypothesis(feasibility_dir / "scope")
    validations = _artifact_by_hypothesis(feasibility_dir / "validation")
    physics = _artifact_by_hypothesis(feasibility_dir / "physics")
    experimental = _artifact_by_hypothesis(feasibility_dir / "experimental")
    decisions = _decision_by_hypothesis(portfolio)

    source_manifest: dict[str, Any] = {}
    if run_dir:
        source_manifest = _read_json_if_exists(run_dir / "manifest.json") or {}

    hypotheses: list[dict[str, Any]] = []
    for hypothesis in _list(intake.get("hypotheses")):
        if not isinstance(hypothesis, dict):
            continue
        hypothesis_id = _text(hypothesis.get("hypothesis_id"))
        if not hypothesis_id:
            continue
        hypotheses.append(
            {
                "hypothesis": hypothesis,
                "scope": scopes.get(hypothesis_id, {}),
                "validation": validations.get(hypothesis_id, {}),
                "physics": physics.get(hypothesis_id, {}),
                "experimental": experimental.get(hypothesis_id, {}),
                "decision": decisions.get(hypothesis_id, {}),
            }
        )

    paper_ids = sorted(
        {
            paper_id
            for row in hypotheses
            for paper_id in _list(row["hypothesis"].get("source_paper_ids"))
            if isinstance(paper_id, str)
        }
    )

    return {
        "viewer_schema": "graphagentsdac-demo-viewer-v1",
        "question": _text(intake.get("question"), "GraphAgentsDAC demo"),
        "corpus_id": _text(intake.get("corpus_id"), "unknown"),
        "task_id": _text(intake.get("task_id")),
        "intake_id": _text(intake.get("intake_id")),
        "intake_sha256": _text(intake.get("intake_sha256")),
        "abstention_reason": intake.get("abstention_reason"),
        "paper_ids": paper_ids,
        "manifest": manifest,
        "source_manifest": source_manifest,
        "hypotheses": hypotheses,
    }


def _json_for_script(value: Any) -> str:
    # Avoid closing the script element from user/source text.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_demo_html(payload: dict[str, Any], *, title: str = "GraphAgentsDAC Demo Viewer") -> str:
    data_json = _json_for_script(payload)
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
:root {{
  color-scheme: light;
  --bg:#f5f7fb; --panel:#ffffff; --panel-2:#f8fafc; --text:#172033; --muted:#667085;
  --border:#d9e0ea; --accent:#315efb; --accent-soft:#eef2ff; --green:#18794e; --green-bg:#eaf8f1;
  --amber:#9a6700; --amber-bg:#fff6d8; --red:#b42318; --red-bg:#fff0ee; --blue:#175cd3;
  --blue-bg:#eef4ff; --gray-bg:#f2f4f7; --purple:#6941c6; --purple-bg:#f4f0ff;
  --shadow:0 8px 24px rgba(20,33,61,.08);
}}
@media (prefers-color-scheme: dark) {{
  :root {{ color-scheme:dark; --bg:#10131a; --panel:#171b24; --panel-2:#1d2330; --text:#eef2f7;
    --muted:#a8b0be; --border:#303848; --accent:#7c9cff; --accent-soft:#202b4d; --green:#72d5a4;
    --green-bg:#153428; --amber:#f5c45b; --amber-bg:#3b3017; --red:#ff8c82; --red-bg:#3c201f;
    --blue:#8db4ff; --blue-bg:#1d2d4e; --gray-bg:#242a35; --purple:#c0a4ff; --purple-bg:#2d2444;
    --shadow:none; }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
button {{ font:inherit; color:inherit; }}
.app {{ min-height:100vh; display:grid; grid-template-rows:auto 1fr; }}
header {{ background:var(--panel); border-bottom:1px solid var(--border); padding:18px 22px 16px; position:sticky; top:0; z-index:20; }}
.topline {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }}
h1 {{ font-size:18px; margin:0 0 5px; font-weight:700; }}
.question {{ font-size:15px; max-width:940px; }}
.meta {{ color:var(--muted); font-size:12px; margin-top:5px; }}
.metrics {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
.metric {{ border:1px solid var(--border); background:var(--panel-2); border-radius:10px; padding:7px 10px; min-width:105px; }}
.metric b {{ display:block; font-size:16px; }} .metric span {{ color:var(--muted); font-size:11px; }}
.layout {{ display:grid; grid-template-columns:300px minmax(0,1fr); min-height:0; }}
aside {{ background:var(--panel); border-right:1px solid var(--border); padding:14px; overflow:auto; }}
.side-title {{ font-weight:650; margin:2px 4px 10px; }}
.hyp-list {{ display:flex; flex-direction:column; gap:8px; }}
.hyp-btn {{ width:100%; text-align:left; border:1px solid var(--border); background:var(--panel-2); border-radius:12px; padding:10px; cursor:pointer; }}
.hyp-btn:hover {{ border-color:var(--accent); }} .hyp-btn.active {{ border-color:var(--accent); background:var(--accent-soft); }}
.hyp-index {{ color:var(--muted); font-size:11px; }} .hyp-title {{ font-weight:650; margin:3px 0 6px; }}
.badges {{ display:flex; flex-wrap:wrap; gap:5px; }}
.badge {{ display:inline-flex; align-items:center; gap:4px; padding:2px 7px; border-radius:999px; font-size:11px; border:1px solid transparent; white-space:nowrap; }}
.badge.good {{ color:var(--green); background:var(--green-bg); }} .badge.warn {{ color:var(--amber); background:var(--amber-bg); }}
.badge.bad {{ color:var(--red); background:var(--red-bg); }} .badge.info {{ color:var(--blue); background:var(--blue-bg); }}
.badge.purple {{ color:var(--purple); background:var(--purple-bg); }} .badge.muted {{ color:var(--muted); background:var(--gray-bg); }}
main {{ min-width:0; padding:18px 20px 36px; overflow:auto; }}
.hero {{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:16px; align-items:start; margin-bottom:14px; }}
.hero h2 {{ font-size:18px; line-height:1.35; margin:0 0 6px; }}
.statement {{ font-size:15px; max-width:1050px; }}
.final {{ text-align:right; }} .final-label {{ color:var(--muted); font-size:11px; }} .final-value {{ font-weight:750; margin-top:3px; }}
.tabs {{ display:flex; gap:5px; border-bottom:1px solid var(--border); margin-bottom:16px; overflow-x:auto; }}
.tab {{ border:0; border-bottom:2px solid transparent; padding:9px 11px; background:transparent; cursor:pointer; color:var(--muted); white-space:nowrap; }}
.tab.active {{ color:var(--text); border-bottom-color:var(--accent); font-weight:650; }}
.panel {{ display:none; }} .panel.active {{ display:block; }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:14px; box-shadow:var(--shadow); }}
.card + .card {{ margin-top:12px; }} .card h3 {{ margin:0 0 10px; font-size:14px; }}
.grid2 {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
.grid3 {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
@media (max-width:980px) {{ .layout {{ grid-template-columns:1fr; }} aside {{ border-right:0; border-bottom:1px solid var(--border); max-height:260px; }} .grid2,.grid3 {{ grid-template-columns:1fr; }} }}
.flow-wrap {{ overflow-x:auto; padding:8px 2px 16px; }}
.flow {{ min-width:1180px; display:grid; grid-template-columns:1.15fr 1.2fr .95fr 1fr 1.15fr 1.15fr; gap:22px; align-items:start; }}
.stage {{ position:relative; }} .stage:not(:last-child)::after {{ content:"→"; position:absolute; right:-17px; top:34px; color:var(--muted); font-size:20px; }}
.stage-label {{ text-transform:uppercase; letter-spacing:.07em; color:var(--muted); font-size:10px; margin:0 0 7px 4px; }}
.node {{ width:100%; border:1px solid var(--border); background:var(--panel); border-radius:12px; padding:10px; margin-bottom:8px; text-align:left; cursor:pointer; box-shadow:var(--shadow); }}
.node:hover {{ border-color:var(--accent); }} .node .k {{ color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.05em; }} .node .v {{ margin-top:3px; }}
.node.evidence {{ border-left:4px solid var(--blue); }} .node.hypothesis {{ border-left:4px solid var(--purple); }} .node.prediction {{ border-left:4px solid var(--green); }} .node.falsifier {{ border-left:4px solid var(--red); }} .node.validation {{ border-left:4px solid var(--accent); }}
.detail {{ margin-top:12px; border:1px dashed var(--border); background:var(--panel-2); border-radius:12px; padding:12px; }}
.detail h3 {{ margin:0 0 6px; font-size:13px; }} .detail pre {{ white-space:pre-wrap; word-break:break-word; margin:7px 0 0; font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; border-bottom:1px solid var(--border); padding:9px 8px; vertical-align:top; }} th {{ color:var(--muted); font-size:11px; font-weight:650; }} td.small {{ color:var(--muted); font-size:12px; }}
.section-label {{ color:var(--muted); font-size:11px; margin-bottom:6px; }}
ul.clean {{ margin:0; padding-left:18px; }} ul.clean li + li {{ margin-top:4px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:6px; }} .chip {{ background:var(--gray-bg); border-radius:8px; padding:4px 7px; font-size:12px; }}
.quote {{ border-left:3px solid var(--border); padding-left:10px; margin:8px 0; }}
.mono {{ font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; overflow-wrap:anywhere; color:var(--muted); }}
.empty {{ color:var(--muted); font-style:italic; }}
.warning {{ border:1px solid color-mix(in srgb, var(--amber) 40%, var(--border)); background:var(--amber-bg); color:var(--amber); border-radius:10px; padding:9px 10px; margin-bottom:12px; }}
</style>
</head>
<body>
<div class="app">
<header>
  <div class="topline"><div><h1>{safe_title}</h1><div class="question" id="question"></div><div class="meta" id="meta"></div></div></div>
  <div class="metrics" id="metrics"></div>
</header>
<div class="layout">
  <aside><div class="side-title">Hypotheses</div><div class="hyp-list" id="hypList"></div></aside>
  <main>
    <div id="abstention"></div>
    <div class="hero"><div><h2 id="hypTitle"></h2><div class="statement" id="hypStatement"></div><div class="badges" id="heroBadges" style="margin-top:9px"></div></div><div class="final"><div class="final-label">Final disposition</div><div class="final-value" id="finalDisposition"></div></div></div>
    <div class="tabs" id="tabs">
      <button class="tab active" data-tab="lineage">Lineage</button>
      <button class="tab" data-tab="verification">Verification matrix</button>
      <button class="tab" data-tab="design">Validation design</button>
      <button class="tab" data-tab="provenance">Provenance</button>
    </div>
    <section class="panel active" id="panel-lineage"><div id="lineage"></div></section>
    <section class="panel" id="panel-verification"><div id="verification"></div></section>
    <section class="panel" id="panel-design"><div id="design"></div></section>
    <section class="panel" id="panel-provenance"><div id="provenance"></div></section>
  </main>
</div>
</div>
<script id="demo-data" type="application/json">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById('demo-data').textContent);
let selected = 0;
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const arr = v => Array.isArray(v) ? v : [];
const text = (v, d='') => typeof v === 'string' ? v : d;
const human = s => text(s).replaceAll('_',' ').replace(/\\b\\w/g, m=>m.toUpperCase());
function statusClass(s) {{
  s = text(s).toLowerCase();
  if (['pass','eligible','physically_supported','experimentally_plausible','ready_for_experimental_validation'].includes(s)) return 'good';
  if (s.includes('fail') || s.includes('reject') || s.includes('implausible')) return 'bad';
  if (s.includes('conditional') || s.includes('warning') || s.includes('requires')) return 'warn';
  if (s.includes('unknown') || s.includes('insufficient') || s.includes('not_assessed')) return 'muted';
  return 'info';
}}
const badge = (s, label=null) => `<span class="badge ${{statusClass(s)}}">${{esc(label ?? human(s || 'unknown'))}}</span>`;
const chips = xs => arr(xs).length ? `<div class="chips">${{arr(xs).map(x=>`<span class="chip">${{esc(human(x))}}</span>`).join('')}}</div>` : '<span class="empty">None</span>';
const list = xs => arr(xs).length ? `<ul class="clean">${{arr(xs).map(x=>`<li>${{esc(x)}}</li>`).join('')}}</ul>` : '<span class="empty">None</span>';
function renderHeader() {{
  document.getElementById('question').textContent = DATA.question || 'GraphAgentsDAC demo';
  document.getElementById('meta').textContent = `Corpus: ${{DATA.corpus_id || 'unknown'}} · Task: ${{DATA.task_id || 'n/a'}}`;
  const hs = arr(DATA.hypotheses);
  const papers = arr(DATA.paper_ids);
  const design = hs.filter(x=>x.decision?.final_disposition === 'requires_validation_design').length;
  const compute = hs.filter(x=>x.physics?.disposition === 'requires_computation').length;
  document.getElementById('metrics').innerHTML = [
    ['Hypotheses', hs.length], ['Source papers', papers.length], ['Need validation design', design], ['Need computation', compute]
  ].map(([k,v])=>`<div class="metric"><b>${{v}}</b><span>${{k}}</span></div>`).join('');
  if (DATA.abstention_reason) document.getElementById('abstention').innerHTML = `<div class="warning">Abstention: ${{esc(DATA.abstention_reason)}}</div>`;
}}
function renderSidebar() {{
  document.getElementById('hypList').innerHTML = arr(DATA.hypotheses).map((row,i)=>{{
    const h=row.hypothesis||{{}}, d=row.decision||{{}}, s=row.scope||{{}};
    return `<button class="hyp-btn ${{i===selected?'active':''}}" data-index="${{i}}"><div class="hyp-index">H${{i+1}} · ${{esc(h.hypothesis_type||'hypothesis')}}</div><div class="hyp-title">${{esc(h.title||h.statement||h.hypothesis_id)}}</div><div class="badges">${{badge(s.catalyst_class)}}${{badge(d.final_disposition||'pending')}}</div></button>`;
  }}).join('');
  document.querySelectorAll('.hyp-btn').forEach(b=>b.addEventListener('click',()=>{{selected=Number(b.dataset.index); renderAll();}}));
}}
function node(type, kind, title, body, detail) {{
  return `<button class="node ${{kind}}" data-detail="${{esc(JSON.stringify(detail || {{title,body}}))}}"><div class="k">${{esc(type)}}</div><div class="v"><b>${{esc(title)}}</b>${{body?`<div style="margin-top:4px;color:var(--muted);font-size:12px">${{esc(body)}}</div>`:''}}</div></button>`;
}}
function renderLineage(row) {{
  const h=row.hypothesis||{{}}, s=row.scope||{{}}, v=row.validation||{{}}, p=row.physics||{{}}, e=row.experimental||{{}}, d=row.decision||{{}};
  const evidence = [];
  arr(h.source_paper_ids).forEach(pid=>evidence.push(node('source paper','evidence',pid,'reported evidence source',{{title:'Source paper',paper_id:pid}})));
  arr(h.premises).slice(0,4).forEach(pr=>evidence.push(node(pr.epistemic_role||'premise','evidence',pr.claim_kind||'evidence',pr.text,pr)));
  if (!evidence.length) evidence.push('<div class="empty">No premise artifacts</div>');
  const outcomes = arr(h.predictions).map(x=>node('prediction','prediction',x.expected_direction||'prediction',x.observable,x)).join('') + arr(h.falsifiers).map(x=>node('falsifier','falsifier','falsification criterion',x.falsifying_outcome,x)).join('');
  const semanticBody = [h.semantic_gate_status, ...arr(h.semantic_warning_dimensions)].filter(Boolean).join(' · ');
  const scopeBody = `${{human(s.catalyst_class||'unknown')}} · ${{human(s.hypothesis_level||'unknown')}}`;
  const valBody = `${{human(v.validation_strategy||'unknown')}}${{v.requires_candidate_concretization?' · concretization required':''}}`;
  const physBody = `${{human(p.disposition||'unknown')}} · ${{arr(p.unresolved_checks).length}} unresolved`;
  const expBody = `${{human(e.disposition||'unknown')}} · precedent ${{human(e.precedent_status||'not assessed')}}`;
  document.getElementById('lineage').innerHTML = `<div class="card"><h3>Hypothesis lineage and validation path</h3><div class="flow-wrap"><div class="flow">
    <div class="stage"><div class="stage-label">Evidence</div>${{evidence.join('')}}</div>
    <div class="stage"><div class="stage-label">Hypothesis</div>${{node('AI inference','hypothesis',h.title||'Hypothesis',h.statement,{{hypothesis:h,inferential_bridge:h.inferential_bridge}})}}</div>
    <div class="stage"><div class="stage-label">Testability</div>${{outcomes || '<div class="empty">No prediction/falsifier</div>'}}</div>
    <div class="stage"><div class="stage-label">Epistemic + scope</div>${{node('semantic gate','validation',human(h.semantic_gate_status||'unknown'),semanticBody,{{semantic_status:h.semantic_gate_status,warnings:h.semantic_warning_dimensions,failures:h.semantic_fail_dimensions}})}}${{node('scientific scope','validation',human(s.catalyst_class||'unknown'),scopeBody,s)}}</div>
    <div class="stage"><div class="stage-label">Validation design</div>${{node('validation specification','validation',human(v.validation_strategy||'unknown'),valBody,v)}}</div>
    <div class="stage"><div class="stage-label">Feasibility → decision</div>${{node('physics','validation',human(p.disposition||'unknown'),physBody,p)}}${{node('experiment','validation',human(e.disposition||'unknown'),expBody,e)}}${{node('final decision','hypothesis',human(d.final_disposition||'pending'),'next action',d)}}</div>
  </div></div><div class="detail" id="nodeDetail"><h3>Click a node</h3><div>Select any lineage node to inspect the exact artifact fields.</div></div></div>`;
  document.querySelectorAll('.node').forEach(n=>n.addEventListener('click',()=>{{
    let detail={{}}; try {{ detail=JSON.parse(n.dataset.detail||'{{}}'); }} catch (_) {{}}
    document.getElementById('nodeDetail').innerHTML=`<h3>${{esc(detail.title||detail.check_type||detail.hypothesis_id||'Artifact detail')}}</h3><pre>${{esc(JSON.stringify(detail,null,2))}}</pre>`;
  }}));
}}
function renderVerification(row) {{
  const p=row.physics||{{}}, e=row.experimental||{{}}, v=row.validation||{{}};
  const physicsRows = arr(p.checks).map(x=>`<tr><td>${{esc(human(x.check_type))}}</td><td>${{badge(x.status)}}</td><td>${{esc(human(x.basis||''))}}</td><td class="small">${{esc(x.rationale||'')}}</td></tr>`).join('');
  const naPhys = arr(p.not_applicable_checks).concat(arr(v.not_applicable_physics_checks));
  const naRows = [...new Set(naPhys)].map(x=>`<tr><td>${{esc(human(x))}}</td><td>${{badge('unknown','Not applicable')}}</td><td>scope rule</td><td class="small">Excluded by hypothesis/catalyst scope.</td></tr>`).join('');
  const expRows = arr(e.checks).map(x=>`<tr><td>${{esc(human(x.check_type))}}</td><td>${{badge(x.status)}}</td><td>${{esc(human(x.precedent_status||x.complexity||''))}}</td><td class="small">${{esc(x.rationale||'')}}</td></tr>`).join('');
  document.getElementById('verification').innerHTML = `<div class="grid2"><div class="card"><h3>Physics verification</h3><div class="badges" style="margin-bottom:9px">${{badge(p.disposition)}}${{badge(p.confidence||'unknown','Confidence: '+human(p.confidence||'unknown'))}}</div><table><thead><tr><th>Check</th><th>Status</th><th>Basis</th><th>Rationale</th></tr></thead><tbody>${{physicsRows+naRows || '<tr><td colspan="4" class="empty">No physics checks</td></tr>'}}</tbody></table></div>
  <div class="card"><h3>Experimental realizability</h3><div class="badges" style="margin-bottom:9px">${{badge(e.disposition)}}${{badge(e.precedent_status||'not_assessed','Precedent: '+human(e.precedent_status||'not assessed'))}}</div><table><thead><tr><th>Check</th><th>Status</th><th>Signal</th><th>Rationale</th></tr></thead><tbody>${{expRows || '<tr><td colspan="4" class="empty">No experimental checks</td></tr>'}}</tbody></table></div></div>`;
}}
function renderDesign(row) {{
  const s=row.scope||{{}}, v=row.validation||{{}}, d=row.decision||{{}}, h=row.hypothesis||{{}};
  document.getElementById('design').innerHTML = `<div class="grid3">
    <div class="card"><h3>Scientific scope</h3><div class="section-label">Catalyst / level / reaction</div><div class="badges">${{badge(s.catalyst_class)}}${{badge(s.hypothesis_level)}}${{badge(s.reaction||'unknown')}}</div><div style="margin-top:12px" class="section-label">Environments</div>${{chips(s.environments)}}<div style="margin-top:12px" class="section-label">Coordination variables</div>${{chips(s.coordination_variables)}}<div style="margin-top:12px" class="section-label">Observables</div>${{chips(s.dependent_observables)}}</div>
    <div class="card"><h3>Variable design</h3><div class="section-label">Controlled</div>${{chips(v.controlled_variables)}}<div style="margin-top:12px" class="section-label">Varied</div>${{chips(v.varied_variables)}}<div style="margin-top:12px" class="section-label">Required comparisons</div>${{list(v.required_comparisons)}}</div>
    <div class="card"><h3>Concretization</h3><div class="badges" style="margin-bottom:10px">${{badge(v.requires_candidate_concretization?'requires_validation_design':'ready',''+(v.requires_candidate_concretization?'Required':'Not required'))}}</div>${{list(v.candidate_concretization_requirements)}}<div style="margin-top:12px" class="section-label">Next actions</div>${{list(v.next_actions)}}</div>
  </div>
  <div class="grid2" style="margin-top:12px"><div class="card"><h3>Predicted success pattern</h3>${{list(v.success_patterns)}}<div style="margin-top:12px" class="section-label">Original predictions</div>${{arr(h.predictions).map(x=>`<div class="quote"><b>${{esc(human(x.expected_direction))}}</b><br>${{esc(x.observable||'')}}<div class="small">${{esc(x.rationale||'')}}</div></div>`).join('')||'<span class="empty">None</span>'}}</div><div class="card"><h3>Falsification pattern</h3>${{list(v.falsification_patterns)}}<div style="margin-top:12px" class="section-label">Original falsifiers</div>${{arr(h.falsifiers).map(x=>`<div class="quote">${{esc(x.falsifying_outcome||'')}}<div class="small">Observable: ${{esc(x.observable||'')}}</div></div>`).join('')||'<span class="empty">None</span>'}}</div></div>
  <div class="card"><h3>Decision-directed work</h3><div class="grid3"><div><div class="section-label">Required computations</div>${{list(d.required_computations)}}</div><div><div class="section-label">Required characterization</div>${{list(d.required_characterization)}}</div><div><div class="section-label">Required electrochemical tests</div>${{list(d.required_electrochemical_tests)}}</div></div></div>`;
}}
function renderProvenance(row) {{
  const h=row.hypothesis||{{}}, s=row.scope||{{}}, v=row.validation||{{}}, p=row.physics||{{}}, e=row.experimental||{{}}, d=row.decision||{{}};
  const premises = arr(h.premises).map(pr=>`<div class="card"><div class="badges">${{badge(pr.epistemic_role||'premise')}}${{badge(pr.claim_kind||'claim')}}</div><div class="quote">${{esc(pr.text||'')}}</div><div class="mono">statement_id: ${{esc(pr.statement_id||'')}}<br>paper_ids: ${{esc(arr(pr.paper_ids).join(', ')||'none')}}<br>requires_verification: ${{esc(pr.requires_verification)}}</div></div>`).join('');
  document.getElementById('provenance').innerHTML = `<div class="grid2"><div><div class="card"><h3>Inference boundary</h3><div class="section-label">Inferential bridge</div><div class="quote">${{esc(h.inferential_bridge||'')}}</div><div class="section-label" style="margin-top:12px">Assumptions</div>${{list(h.assumptions)}}</div>${{premises || '<div class="card empty">No premise records</div>'}}</div><div><div class="card"><h3>Artifact lineage</h3><div class="mono">hypothesis_id: ${{esc(h.hypothesis_id||'')}}<br>intake_id: ${{esc(DATA.intake_id||'')}}<br>scope_id: ${{esc(s.scope_id||'')}}<br>validation_specification_id: ${{esc(v.specification_id||'')}}<br>physics_report_id: ${{esc(p.report_id||'')}}<br>experimental_report_id: ${{esc(e.report_id||'')}}<br>decision_id: ${{esc(d.decision_id||'')}}</div></div><div class="card"><h3>Source papers</h3>${{chips(h.source_paper_ids)}}<div class="section-label" style="margin-top:12px">Semantic warnings</div>${{chips(h.semantic_warning_dimensions)}}<div class="section-label" style="margin-top:12px">Scope warnings</div>${{chips(s.scope_warnings)}}</div><div class="card"><h3>Key uncertainties</h3>${{list(d.key_uncertainties)}}</div></div></div>`;
}}
function renderMain() {{
  const row=arr(DATA.hypotheses)[selected]; if(!row) return;
  const h=row.hypothesis||{{}}, s=row.scope||{{}}, d=row.decision||{{}}, p=row.physics||{{}}, e=row.experimental||{{}};
  document.getElementById('hypTitle').textContent=`H${{selected+1}} · ${{h.title||h.hypothesis_id||'Hypothesis'}}`;
  document.getElementById('hypStatement').textContent=h.statement||'';
  document.getElementById('heroBadges').innerHTML=badge(h.semantic_gate_status)+badge(s.catalyst_class)+badge(s.hypothesis_level)+badge(p.disposition)+badge(e.disposition);
  document.getElementById('finalDisposition').innerHTML=badge(d.final_disposition||'pending');
  renderLineage(row); renderVerification(row); renderDesign(row); renderProvenance(row);
}}
function renderTabs() {{
  document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>{{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active')); tab.classList.add('active');
    document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active')); document.getElementById('panel-'+tab.dataset.tab).classList.add('active');
  }}));
}}
function renderAll() {{ renderSidebar(); renderMain(); }}
renderHeader(); renderTabs(); renderAll();
</script>
</body>
</html>"""


def build_demo_viewer(
    *,
    run_dir: Path,
    output: Path,
    feasibility_dir: Path | None = None,
    title: str = "GraphAgentsDAC Hypothesis Lineage & Validation Viewer",
) -> Path:
    run_dir = run_dir.resolve()
    feasibility_dir = (feasibility_dir or discover_feasibility_dir(run_dir)).resolve()
    payload = load_demo_payload(feasibility_dir, run_dir=run_dir)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_demo_html(payload, title=title), encoding="utf-8")
    return output
