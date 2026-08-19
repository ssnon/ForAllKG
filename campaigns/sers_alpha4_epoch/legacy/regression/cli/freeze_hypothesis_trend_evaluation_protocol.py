from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.hypothesis_trend_directional_compiler import (
    DirectionAwareTrendHypothesisCompiler,
)
from dac_her.hypothesis_trend_directional_contracts import (
    DirectionAwareTrendHypothesisPortfolio,
    DirectionAwareTrendHypothesisPortfolioDraft,
)
from dac_her.hypothesis_trend_directional_exposure import (
    build_directional_trend_maker_exposure,
)
from dac_her.hypothesis_trend_directional_prompt import (
    PROMPT_VERSION,
    DirectionAwareTrendHypothesisPromptAssembler,
)
from dac_her.hypothesis_trend_directional_run_record import (
    DirectionAwareTrendHypothesisMakerRunRecord,
)
from dac_her.hypothesis_trend_directional_validator import (
    DirectionAwareTrendHypothesisValidator,
)
from dac_her.hypothesis_trend_evaluation import (
    FATAL_RULE_CODES,
    HYPOTHESIS_TREND_EVALUATION_PROTOCOL_SEMANTICS_ID,
    HYPOTHESIS_TREND_EVALUATOR_SEMANTICS_ID,
    NONFATAL_OBSERVATION_CODES,
    SeenSmokeAnchor,
    TrendHypothesisEvaluationPolicy,
    TrendHypothesisEvaluationProtocol,
    TrendHypothesisMakerSettings,
    canonical_json,
    current_component_hashes,
    detect_claim_scope_issues,
    sha256_json,
    stable_id,
)
from dac_her.hypothesis_trend_input import (
    TrendAwareHypothesisInput,
    verify_trend_aware_input_sources,
)


EXPECTED_SEEN = {'source_input_sha256': 'e103a995336d660d0ba6cf6c2504b981fa7e7b721330b2b7f4b2e32e9393232e', 'directional_exposure_sha256': '62bccbf939fcd769b3c29402dd8d4dd3502ea7878a497068ab3e599beb698dbe', 'prompt_sha256': 'c1252106ac49f35a7c5a89f381284d18a0323b19ddff00c2dae3a180ffcfbbca', 'portfolio_sha256': 'c6c613f5ab08c1b4a4346a6703a37b299c5b5a5cac68b0bbd7a0874ad821daed', 'model': 'openai/gpt-5.6-luna', 'backend': 'instructor_openai_compatible_direction_aware_trend_hypothesis', 'generation_attempts': 1, 'repair_attempts': 0}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        value.model_dump(mode="json")
        if hasattr(value, "model_dump")
        else value
    )
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    if path.exists():
        observed = path.read_text(encoding="utf-8")
        if observed != text:
            raise RuntimeError(
                f"Refusing to replace non-identical frozen 5e file: {path}"
            )
        return
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen-input", required=True, type=Path)
    parser.add_argument("--seen-run", required=True, type=Path)
    parser.add_argument("--seen-final-draft", required=True, type=Path)
    parser.add_argument("--seen-portfolio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd()

    source = TrendAwareHypothesisInput.model_validate_json(
        args.seen_input.read_text(encoding="utf-8")
    )
    run = DirectionAwareTrendHypothesisMakerRunRecord.model_validate_json(
        args.seen_run.read_text(encoding="utf-8")
    )
    draft = DirectionAwareTrendHypothesisPortfolioDraft.model_validate_json(
        args.seen_final_draft.read_text(encoding="utf-8")
    )
    portfolio = DirectionAwareTrendHypothesisPortfolio.model_validate_json(
        args.seen_portfolio.read_text(encoding="utf-8")
    )

    verify_trend_aware_input_sources(source)

    if source.input_sha256 != EXPECTED_SEEN["source_input_sha256"]:
        raise RuntimeError(
            "5e freeze requires the reviewed alpha4c.5d.1 seen input."
        )
    if run.directional_exposure_sha256 != (
        EXPECTED_SEEN["directional_exposure_sha256"]
    ):
        raise RuntimeError(
            "5e freeze directional exposure anchor mismatch."
        )
    if run.prompt_sha256 != EXPECTED_SEEN["prompt_sha256"]:
        raise RuntimeError("5e freeze prompt anchor mismatch.")
    if run.portfolio_sha256 != EXPECTED_SEEN["portfolio_sha256"]:
        raise RuntimeError("5e freeze portfolio anchor mismatch.")
    if run.model != EXPECTED_SEEN["model"]:
        raise RuntimeError("5e freeze seen model mismatch.")
    if run.backend != EXPECTED_SEEN["backend"]:
        raise RuntimeError("5e freeze seen backend mismatch.")
    if run.generation_attempts != EXPECTED_SEEN["generation_attempts"]:
        raise RuntimeError("5e freeze generation-attempt anchor mismatch.")
    if run.repair_attempts != EXPECTED_SEEN["repair_attempts"]:
        raise RuntimeError("5e freeze repair-attempt anchor mismatch.")
    if (
        not run.final_validation_passed
        or run.failure_stage != "none"
        or run.validation_errors != 0
    ):
        raise RuntimeError(
            "5e freeze requires a clean accepted 5d.1 seen smoke."
        )

    exposure = build_directional_trend_maker_exposure(source)
    prompt = DirectionAwareTrendHypothesisPromptAssembler(
        max_hypotheses=1
    ).build(source, exposure=exposure)
    if exposure.exposure_sha256 != run.directional_exposure_sha256:
        raise RuntimeError("seen exposure cannot be reproduced.")
    if prompt.prompt_sha256 != run.prompt_sha256:
        raise RuntimeError("seen prompt cannot be reproduced.")

    recompiled = DirectionAwareTrendHypothesisCompiler().compile(
        source, draft
    )
    if (
        recompiled.model_dump(mode="json")
        != portfolio.model_dump(mode="json")
    ):
        raise RuntimeError(
            "seen portfolio is not exact deterministic compilation."
        )
    validation = DirectionAwareTrendHypothesisValidator().validate(
        source, portfolio
    )
    if not validation.passes:
        raise RuntimeError(
            "reviewed seen portfolio no longer revalidates."
        )
    for card in portfolio.hypotheses:
        text = "\n".join(
            [
                card.title,
                card.hypothesis_statement,
                card.inferential_bridge,
                *[
                    row.observable + "\n" + row.rationale
                    for row in card.predicted_observations
                ],
                *[
                    row.observable + "\n" + row.falsifying_outcome
                    for row in card.falsification_criteria
                ],
                *card.assumptions,
            ]
        )
        claim_issues = detect_claim_scope_issues(
            text,
            cross_paper_synthesis=card.cross_paper_synthesis,
        )
        if claim_issues:
            raise RuntimeError(
                "reviewed seen portfolio has 5e claim-scope issues: "
                f"{sorted(claim_issues)}"
            )

    component_hashes = current_component_hashes(root)
    settings = TrendHypothesisMakerSettings(
        max_hypotheses=1,
        max_repairs=1,
        temperature=0.0,
        prompt_version=PROMPT_VERSION,
        backend=run.backend,
        model=run.model,
        parse_retries=run.parse_retries,
        backend_mode=run.backend_mode,
        base_url=run.base_url,
    )
    anchor = SeenSmokeAnchor(
        source_input_id=source.input_id,
        source_input_sha256=source.input_sha256,
        directional_exposure_id=run.directional_exposure_id,
        directional_exposure_sha256=
            run.directional_exposure_sha256,
        prompt_sha256=run.prompt_sha256,
        run_id=run.run_id,
        portfolio_id=portfolio.portfolio_id,
        portfolio_sha256=run.portfolio_sha256,
        paper_ids=list(source.trend_corpus_binding.paper_ids),
        generation_attempts=run.generation_attempts,
        repair_attempts=run.repair_attempts,
        validation_errors=run.validation_errors,
        validation_warnings=run.validation_warnings,
    )
    protocol_id = stable_id(
        "trend_hypothesis_evaluation_protocol",
        HYPOTHESIS_TREND_EVALUATION_PROTOCOL_SEMANTICS_ID,
        source.domain_profile_id,
        anchor.source_input_sha256,
        anchor.portfolio_sha256,
        sha256_json(component_hashes),
        ",".join(FATAL_RULE_CODES),
        ",".join(NONFATAL_OBSERVATION_CODES),
    )
    payload = {
        "schema_version":
            "trend-hypothesis-evaluation-protocol-v1",
        "protocol_id": protocol_id,
        "semantics_id":
            HYPOTHESIS_TREND_EVALUATION_PROTOCOL_SEMANTICS_ID,
        "evaluator_semantics_id":
            HYPOTHESIS_TREND_EVALUATOR_SEMANTICS_ID,
        "domain_profile_id": source.domain_profile_id,
        "maker_settings": settings.model_dump(mode="json"),
        "seen_smoke_anchor": anchor.model_dump(mode="json"),
        "frozen_component_sha256": component_hashes,
        "fatal_rule_codes": list(FATAL_RULE_CODES),
        "nonfatal_observation_codes":
            list(NONFATAL_OBSERVATION_CODES),
        "policy":
            TrendHypothesisEvaluationPolicy().model_dump(
                mode="json"
            ),
        "reserve_campaign_prefix": "sers_alpha4c5e_reserve_",
        "reserve_not_yet_registered": True,
        "reserve_consumed_by_protocol_freeze": False,
    }
    payload["protocol_sha256"] = sha256_json(payload)
    protocol = TrendHypothesisEvaluationProtocol.model_validate(
        payload
    )
    _write_json(args.output, protocol)

    print("alpha4c.5e evaluation protocol freeze")
    print("Protocol ID:", protocol.protocol_id)
    print("Protocol SHA256:", protocol.protocol_sha256)
    print("Fatal rules:", len(protocol.fatal_rule_codes))
    print(
        "Nonfatal observations:",
        len(protocol.nonfatal_observation_codes),
    )
    print("Minimum hypothesis count:", None)
    print("Count thresholds used for acceptance: False")
    print("Abstention is failure: False")
    print("One bounded repair then valid is failure: False")
    print("Canonical independent-variable frame: increase")
    print("Positive Trend binding: increase -> increase")
    print("Negative Trend binding: increase -> decrease")
    print("LLM sign transformation allowed: False")
    print("Reserve registered: False")
    print("Reserve consumed: False")
    print("LLM calls: 0")
    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
