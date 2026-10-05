from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_core.discovery.adaptive_yield_evaluation import (
    BASELINE_COMMIT,
    CONDITION_ORDER,
    AdaptiveYieldCaseAudit,
    AdaptiveYieldStageMetrics,
    build_blind_packet,
    count_llm_call_artifacts,
    count_retrieval_report_artifacts,
    elapsed_from_stage_rows,
    normalize_card,
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def statement_index(context_payload: dict[str, Any]) -> dict[str, str]:
    return {
        str(row.get("statement_id") or ""): str(row.get("text") or "")
        for row in context_payload.get("evidence_statements", []) or []
        if isinstance(row, dict) and row.get("statement_id")
    }


def all_context_statement_texts(run: Path, v2_dir: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    initial = run / "hypothesis.context.json"
    if initial.is_file():
        output.update(statement_index(load(initial)))
    for path in v2_dir.rglob("retraversal.context.json"):
        try:
            output.update(statement_index(load(path)))
        except Exception:
            continue
    return output


def portfolio_cards(path: Path) -> list[dict[str, Any]]:
    rows = load(path).get("hypotheses", []) or []
    return [dict(row) for row in rows if isinstance(row, dict)]


def attempt_cards(condition: str, d775: Path, d776: Path, d777: Path):
    paths: list[Path] = []
    if condition == "CLOSED_LOOP_775":
        paths = [d775 / "gen1.portfolio.json"]
    elif condition == "LOCAL_ADAPTIVE_776":
        paths = sorted(d776.glob("round_*/candidate.portfolio.json"))
    elif condition == "GRAPH_ADAPTIVE_777":
        paths = sorted(d777.glob("epoch_*/graph_retraversal/generated.portfolio.json"))
        paths += sorted(d777.glob("epoch_*/local_controller/round_*/candidate.portfolio.json"))
    seen: set[str] = set()
    output = []
    for path in paths:
        if not path.is_file():
            continue
        for card in portfolio_cards(path):
            hid = str(card.get("hypothesis_id") or "")
            key = json.dumps(
                {
                    "h": hid,
                    "s": card.get("hypothesis_statement"),
                    "p": card.get("premise_statement_ids", []),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if key in seen:
                continue
            seen.add(key)
            output.append((card, str(path), None))
    return output


def v2_cards(population_path: Path) -> list[tuple[dict[str, Any], str, int | None]]:
    population = load(population_path)
    output = []
    for entry in population.get("entries", []) or []:
        if not isinstance(entry, dict):
            continue
        ppath = Path(str(entry.get("portfolio_path") or "")).expanduser()
        if not ppath.is_file():
            raise FileNotFoundError(
                "adaptive v2 effective entry portfolio missing: " + str(ppath)
            )
        hid = str(entry.get("hypothesis_id") or "")
        matched = [
            row for row in portfolio_cards(ppath)
            if str(row.get("hypothesis_id") or "") == hid
        ]
        if len(matched) != 1:
            raise RuntimeError(
                f"adaptive v2 entry {hid} did not resolve uniquely in {ppath}"
            )
        output.append(
            (
                matched[0],
                str(ppath),
                int(entry["epoch_index"]) if entry.get("epoch_index") is not None else None,
            )
        )
    return output


def count_unresolved(summary: dict[str, Any]) -> int:
    for key in ("graph_retraversal_handoff_count", "graph_retraversal_request_count"):
        value = summary.get(key)
        if isinstance(value, int):
            return max(0, value)
    counts = summary.get("final_epistemic_state_counts") or {}
    if isinstance(counts, dict):
        return sum(
            int(v or 0)
            for k, v in counts.items()
            if str(k).startswith("UNRESOLVED_")
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Build an evaluation-only matched 7.75/7.76/7.77 adaptive "
            "scientific-yield case audit and blind comparison packet."
        )
    )
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--case-id", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--baseline-commit", default=BASELINE_COMMIT)
    args = p.parse_args()

    run = args.run_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    sp = run / "scientific_portfolio_shadow"
    d775 = sp / "closed_loop_shadow"
    d776 = sp / "adaptive_controller_shadow"
    d777 = sp / "adaptive_graph_retraversal_shadow"

    required = [
        d775 / "closed_loop.summary.json",
        d775 / "effective_gen1.portfolio.json",
        d776 / "adaptive_controller.summary.json",
        d776 / "adaptive_effective.portfolio.json",
        d777 / "adaptive_v2.summary.json",
        d777 / "adaptive_v2.effective_population.json",
        run / "hypothesis.context.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest_path = run / "e2e_runner.manifest.json"
    manifest = load(manifest_path) if manifest_path.is_file() else {}
    context = load(run / "hypothesis.context.json")
    question = str(
        context.get("question")
        or (context.get("task") or {}).get("question")
        or manifest.get("question")
        or ""
    )
    domain = str(
        context.get("domain_profile_id")
        or manifest.get("domain_profile_id")
        or ""
    )
    if not question:
        raise RuntimeError("could not resolve case question")

    statement_texts = all_context_statement_texts(run, d777)
    s775 = load(d775 / "closed_loop.summary.json")
    s776 = load(d776 / "adaptive_controller.summary.json")
    s777 = load(d777 / "adaptive_v2.summary.json")

    raw = {
        "CLOSED_LOOP_775": [
            (row, str(d775 / "effective_gen1.portfolio.json"), None)
            for row in portfolio_cards(d775 / "effective_gen1.portfolio.json")
        ],
        "LOCAL_ADAPTIVE_776": [
            (row, str(d776 / "adaptive_effective.portfolio.json"), 0)
            for row in portfolio_cards(d776 / "adaptive_effective.portfolio.json")
        ],
        "GRAPH_ADAPTIVE_777": v2_cards(
            d777 / "adaptive_v2.effective_population.json"
        ),
    }

    candidates = []
    previous_fingerprints: set[str] = set()
    stage_metrics = []
    summaries = {
        "CLOSED_LOOP_775": s775,
        "LOCAL_ADAPTIVE_776": s776,
        "GRAPH_ADAPTIVE_777": s777,
    }
    roots = {
        "CLOSED_LOOP_775": d775,
        "LOCAL_ADAPTIVE_776": d776,
        "GRAPH_ADAPTIVE_777": d777,
    }
    stage_prefix = {
        "CLOSED_LOOP_775": "[7.75/13]",
        "LOCAL_ADAPTIVE_776": "[7.76/13]",
        "GRAPH_ADAPTIVE_777": "[7.77/13]",
    }
    stage_rows = manifest.get("stages", []) if isinstance(manifest, dict) else []

    for condition in CONDITION_ORDER:
        normalized = []
        for card, portfolio_path, epoch_index in raw[condition]:
            row = normalize_card(
                condition=condition,
                card=card,
                portfolio_path=portfolio_path,
                premise_text_by_id=statement_texts,
                epoch_index=epoch_index,
                previous_fingerprints=previous_fingerprints,
            )
            normalized.append(row)
            candidates.append(row)

        current_fingerprints = {x.scientific_fingerprint for x in normalized}

        attempt_rows = []
        for card, portfolio_path, epoch_index in attempt_cards(
            condition, d775, d776, d777
        ):
            attempt = normalize_card(
                condition=condition,
                card=card,
                portfolio_path=portfolio_path,
                premise_text_by_id=statement_texts,
                epoch_index=epoch_index,
                previous_fingerprints=previous_fingerprints,
                record_kind="SEARCH_ATTEMPT",
            )
            attempt_rows.append(attempt)
            candidates.append(attempt)

        unique_attempt_fingerprints = {
            x.scientific_fingerprint for x in attempt_rows
        }
        summary = summaries[condition]
        root = roots[condition]
        graph_retraversal_count = (
            int(summary.get("graph_retraversal_execution_count") or 0)
            if condition == "GRAPH_ADAPTIVE_777"
            else 0
        )
        structurally_new_premise_count = 0
        if condition == "GRAPH_ADAPTIVE_777":
            lineage = d777 / "context_retraversal.lineage.json"
            if lineage.is_file():
                lp = load(lineage)
                structurally_new_premise_count = sum(
                    len(row.get("structurally_new_eligible_premise_ids", []) or [])
                    for row in lp.get("events", []) or []
                    if isinstance(row, dict)
                )

        stage_metrics.append(
            AdaptiveYieldStageMetrics(
                condition=condition,
                status=str(summary.get("status") or "UNKNOWN"),
                cumulative_effective_count=len(normalized),
                marginal_effective_count=sum(
                    x.marginal_from_previous_condition for x in normalized
                ),
                search_attempt_unique_count=len(unique_attempt_fingerprints),
                unresolved_count=count_unresolved(summary),
                graph_handoff_count=int(
                    summary.get("graph_retraversal_handoff_count")
                    or summary.get("graph_retraversal_request_count")
                    or 0
                ),
                graph_retraversal_count=graph_retraversal_count,
                new_context_count=int(summary.get("context_lineage_event_count") or 0),
                structurally_new_premise_count=structurally_new_premise_count,
                action_counts={
                    str(k): int(v)
                    for k, v in (summary.get("action_counts") or {}).items()
                },
                llm_call_artifact_count=count_llm_call_artifacts(root),
                retrieval_report_artifact_count=count_retrieval_report_artifacts(root),
                elapsed_seconds=elapsed_from_stage_rows(
                    stage_rows, stage_prefix[condition]
                ),
            )
        )
        previous_fingerprints |= current_fingerprints

    audit = AdaptiveYieldCaseAudit(
        case_id=args.case_id,
        run_dir=str(run),
        baseline_commit=str(args.baseline_commit),
        question=question,
        domain_profile_id=domain,
        stages=stage_metrics,
        candidates=candidates,
    )
    packet, key = build_blind_packet(audit)

    out.mkdir(parents=True, exist_ok=True)
    write(out / "adaptive_yield.case.json", audit)
    write(out / "adaptive_yield.blind_packet.json", packet)
    write(out / "adaptive_yield.blind_key.json", key)

    print("Adaptive Scientific Yield case audit complete")
    print("baseline:", audit.baseline_commit)
    for stage in audit.stages:
        print(
            stage.condition,
            "| cumulative=", stage.cumulative_effective_count,
            "| marginal=", stage.marginal_effective_count,
            "| attempts=", stage.search_attempt_unique_count,
            "| unresolved=", stage.unresolved_count,
            "| graph=", stage.graph_retraversal_count,
        )
    print("quality score computed: false")
    print("production selection changed: false")
    print("output:", out / "adaptive_yield.case.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
