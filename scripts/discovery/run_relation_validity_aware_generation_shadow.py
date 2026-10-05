from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.relation_validity_aware_generation import (
    RelationValidityAwareGenerator,
    compile_treatment,
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_rows(synthesis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = synthesis.get("candidates", [])
    if not isinstance(rows, list):
        raise ValueError("case synthesis candidates must be a list")
    return [row for row in rows if isinstance(row, dict)]


def _candidate_alias(row: dict[str, Any], index: int) -> str:
    validity = row.get("validity") or {}
    alias = str(validity.get("candidate_alias") or "").strip()
    return alias or f"V{index:02d}"


def _portfolio_paths(run_dir: Path) -> list[Path]:
    return sorted(set(run_dir.rglob("*portfolio*.json")))


def _context_paths(run_dir: Path) -> list[Path]:
    paths = set(run_dir.rglob("*context*.json"))
    explicit = run_dir / "hypothesis.context.json"
    if explicit.is_file():
        paths.add(explicit)
    return sorted(paths)


def _load_context(path: Path) -> HypothesisContext | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("grounded_context"), dict):
            payload = payload["grounded_context"]
        return HypothesisContext.model_validate(payload)
    except Exception:
        return None


def resolve_control(
    *, run_dir: Path, hypothesis_id: str
) -> tuple[Any, HypothesisContext, Path, Path]:
    matches: list[tuple[Any, Path]] = []
    for path in _portfolio_paths(run_dir):
        try:
            portfolio = HypothesisPortfolio.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            continue
        for card in portfolio.hypotheses:
            if str(card.hypothesis_id) == str(hypothesis_id):
                matches.append((card, path))
    if not matches:
        raise RuntimeError(
            f"could not resolve control hypothesis {hypothesis_id} from portfolio artifacts under {run_dir}"
        )

    contexts: dict[tuple[str, str], tuple[HypothesisContext, Path]] = {}
    for path in _context_paths(run_dir):
        context = _load_context(path)
        if context is None:
            continue
        contexts[(str(context.context_id), str(context.context_sha256))] = (context, path)

    for card, portfolio_path in matches:
        key = (str(card.source_context_id), str(card.source_context_sha256))
        if key in contexts:
            context, context_path = contexts[key]
            return card, context, portfolio_path, context_path
    raise RuntimeError(
        f"resolved hypothesis {hypothesis_id} but could not resolve its source context by id+sha"
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--case-synthesis", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=8192)
    return p


def main() -> int:
    args = parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    synthesis = load(args.case_synthesis.expanduser().resolve())
    generator = RelationValidityAwareGenerator(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        parse_retries=args.parse_retries,
        max_tokens=args.max_tokens,
        telemetry_path=output_dir / "generation.telemetry.jsonl",
        telemetry_context={"case_id": synthesis.get("case_id")},
    )

    records: list[dict[str, Any]] = []
    for index, row in enumerate(_candidate_rows(synthesis), start=1):
        alias = _candidate_alias(row, index)
        hid = str(row.get("hypothesis_id") or "")
        if not hid:
            raise RuntimeError(f"{alias} missing hypothesis_id")
        cdir = output_dir / f"candidate_{index:02d}"
        cdir.mkdir(parents=True, exist_ok=True)
        control, context, portfolio_path, context_path = resolve_control(
            run_dir=run_dir, hypothesis_id=hid
        )
        write(cdir / "control.card.json", control)
        write(cdir / "source.context.json", context)
        response = generator.generate(context=context, control=control)
        write(cdir / "identification_contract.json", response.identification_contract)
        write(cdir / "treatment.draft.json", response.draft)
        treatment, failures = compile_treatment(
            context=context, control=control, response=response
        )
        if failures:
            status = "REJECTED_TREATMENT_CONTRACT"
        elif treatment is None:
            status = "ABSTAINED_NOT_IDENTIFIABLE"
        else:
            status = "GENERATED_TREATMENT"
            write(cdir / "treatment.portfolio.json", treatment)
        treatment_id = (
            str(treatment.hypotheses[0].hypothesis_id)
            if treatment is not None and treatment.hypotheses
            else None
        )
        records.append(
            {
                "candidate_alias": alias,
                "control_hypothesis_id": hid,
                "control_title": str(control.title),
                "control_portfolio_path": str(portfolio_path),
                "source_context_path": str(context_path),
                "source_context_id": str(context.context_id),
                "treatment_status": status,
                "treatment_hypothesis_id": treatment_id,
                "treatment_portfolio_path": (
                    str(cdir / "treatment.portfolio.json") if treatment_id else None
                ),
                "identification_assessment": response.identification_contract.assessment,
                "identification_contract_path": str(cdir / "identification_contract.json"),
                "contract_failures": failures,
                "exact_premise_identity_required": True,
                "exact_gap_identity_required": True,
                "exact_hypothesis_type_required": True,
                "external_prior_art_as_positive_premise": False,
                "novelty_optimization_allowed": False,
                "production_selection_changed": False,
            }
        )
        print(
            alias,
            "|", status,
            "| assessment=", response.identification_contract.assessment,
            "| treatment=", treatment_id,
            "| failures=", failures,
        )

    summary = {
        "schema_version": "relation-validity-aware-generation-shadow-v1",
        "case_id": synthesis.get("case_id"),
        "control_candidate_count": len(records),
        "generated_treatment_count": sum(
            row["treatment_status"] == "GENERATED_TREATMENT" for row in records
        ),
        "abstained_treatment_count": sum(
            row["treatment_status"] == "ABSTAINED_NOT_IDENTIFIABLE" for row in records
        ),
        "rejected_treatment_count": sum(
            row["treatment_status"] == "REJECTED_TREATMENT_CONTRACT" for row in records
        ),
        "records": records,
        "same_grounded_premise_ablation": True,
        "external_prior_art_as_positive_premise": False,
        "novelty_authority_created": False,
        "generation_authority_created": False,
        "production_selection_changed": False,
    }
    write(output_dir / "treatment_generation.summary.json", summary)
    print()
    print("===== RELATION-VALIDITY-AWARE GENERATION SHADOW V1 =====")
    print("controls:", summary["control_candidate_count"])
    print("generated:", summary["generated_treatment_count"])
    print("abstained:", summary["abstained_treatment_count"])
    print("rejected:", summary["rejected_treatment_count"])
    print("SAME_GROUNDED_PREMISE_ABLATION=True")
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
