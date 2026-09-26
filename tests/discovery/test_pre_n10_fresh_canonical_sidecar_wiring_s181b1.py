from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

from pipeline_core.discovery.atomic_scientific_source_provenance import (
    AtomicScientificSourceBindingBundle,
)
from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.pre_n10_canonical_source_reference_v1 import (
    PreN10CanonicalSourceReferenceReportV1,
)
from pipeline_core.discovery.pre_n10_prospective_campaign_v1 import (
    execute_pre_n10_prospective_campaign_v1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    PreN10ScientificContractReportV1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v2 import (
    PreN10ScientificContractReportV2,
)
from scripts.discovery import run_pre_n10_scientific_contract_v1 as cli


class _Backend:
    def decompose(
        self,
        hypothesis: HypothesisCard,
        *,
        max_claims: int,
    ) -> NoveltyClaimDecompositionDraft:
        return NoveltyClaimDecompositionDraft(
            claims=[
                NoveltyClaimDraft(
                    local_id="atomic:1",
                    kind="mechanistic_link",
                    importance="core",
                    novelty_selection_role="NOVELTY_BEARING",
                    text="Factor X changes response Y.",
                    rationale="fixture",
                    search_concepts=["Factor X", "response Y"],
                    search_queries=["Factor X response Y"],
                    prior_art_identity_terms=["response Y"],
                    relation_nucleus_terms=["Factor X", "response Y"],
                    required_bridge="Factor X changes response Y.",
                    predicted_observation="response Y source observable",
                    falsification_condition=(
                        "response Y source falsifying outcome"
                    ),
                    semantic_fidelity_binding=(
                        NoveltyClaimSemanticFidelityBindingDraft(
                            proposition_basis=(
                                "Factor X changes response Y."
                            ),
                            relation_endpoint_anchors=[
                                "Factor X",
                                "response Y",
                            ],
                            prediction_observation_id="prediction:1",
                            falsification_criterion_id="falsifier:1",
                        )
                    ),
                )
            ]
        )


def _portfolio() -> HypothesisPortfolio:
    observable = "response Y source observable"
    card = HypothesisCard(
        hypothesis_id="hypothesis:1",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        title="Fixture",
        hypothesis_statement="Factor X changes response Y.",
        hypothesis_type="mechanistic_extension",
        premise_statement_ids=["statement:1"],
        inferential_bridge="Factor X changes response Y.",
        predicted_observations=[
            PredictedObservation(
                observation_id="prediction:1",
                observable=observable,
                expected_direction="increase",
                rationale="fixture",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:1",
                observable=observable,
                falsifying_outcome=(
                    "response Y source falsifying outcome"
                ),
            )
        ],
        source_paper_ids=["paper:1"],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )
    return HypothesisPortfolio(
        portfolio_id="portfolio:1",
        domain_profile_id=card.domain_profile_id,
        source_context_id=card.source_context_id,
        source_context_sha256=card.source_context_sha256,
        source_report_id=card.source_report_id,
        source_report_sha256=card.source_report_sha256,
        hypotheses=[card],
    )


def _write(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_fresh_cli_materializes_canonical_sidecars_when_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    query_path = tmp_path / "claims_queries.json"
    contract_v1_path = tmp_path / "contract_v1.json"
    bundle_path = tmp_path / "source_binding.bundle.json"
    canonical_path = tmp_path / "canonical_source.report.json"
    contract_v2_path = tmp_path / "contract_v2.json"
    _write(portfolio_path, _portfolio())

    monkeypatch.setattr(
        cli,
        "InstructorOpenAICompatibleExternalNoveltyBackend",
        lambda **_kwargs: _Backend(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pre_n10_scientific_contract_v1",
            "--portfolio",
            str(portfolio_path),
            "--model",
            "fixture",
            "--query-plan-output",
            str(query_path),
            "--contract-output",
            str(contract_v1_path),
            "--source-binding-bundle-output",
            str(bundle_path),
            "--canonical-source-reference-output",
            str(canonical_path),
            "--contract-v2-output",
            str(contract_v2_path),
        ],
    )

    assert cli.main() == 0

    bundle = AtomicScientificSourceBindingBundle.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    canonical = PreN10CanonicalSourceReferenceReportV1.model_validate_json(
        canonical_path.read_text(encoding="utf-8")
    )
    v1 = PreN10ScientificContractReportV1.model_validate_json(
        contract_v1_path.read_text(encoding="utf-8")
    )
    v2 = PreN10ScientificContractReportV2.model_validate_json(
        contract_v2_path.read_text(encoding="utf-8")
    )

    assert bundle.claim_count == 1
    assert bundle.prediction_source_id_present_count == 1
    assert bundle.falsifier_source_id_present_count == 1
    assert canonical.source_binding_bundle_id == bundle.bundle_id
    assert canonical.stable_ready_count == 1
    assert v1.disposition == "READY_FOR_N10"
    assert v2.disposition == "READY_FOR_N10"
    assert v2.source_binding_bundle_id == bundle.bundle_id
    assert v2.canonical_source_reference_report_id == canonical.report_id
    assert v2.stable_source_ids_used_for_readiness is True
    assert v2.exact_text_reconstruction_used_for_readiness is False


def test_fresh_cli_requires_canonical_sidecar_outputs_all_or_none(
    tmp_path: Path,
    monkeypatch,
) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    _write(portfolio_path, _portfolio())

    monkeypatch.setattr(
        cli,
        "InstructorOpenAICompatibleExternalNoveltyBackend",
        lambda **_kwargs: _Backend(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pre_n10_scientific_contract_v1",
            "--portfolio",
            str(portfolio_path),
            "--model",
            "fixture",
            "--query-plan-output",
            str(tmp_path / "claims_queries.json"),
            "--contract-output",
            str(tmp_path / "contract_v1.json"),
            "--source-binding-bundle-output",
            str(tmp_path / "source_binding.bundle.json"),
        ],
    )

    try:
        cli.main()
    except ValueError as exc:
        assert "must be supplied all-or-none" in str(exc)
    else:
        raise AssertionError(
            "partial canonical sidecar outputs must fail closed"
        )


def test_prospective_initial_vpre_requests_all_canonical_sidecars() -> None:
    source = inspect.getsource(execute_pre_n10_prospective_campaign_v1)

    for flag in (
        "--source-binding-bundle-output",
        "--canonical-source-reference-output",
        "--contract-v2-output",
    ):
        assert flag in source

    for artifact in (
        "source_binding_bundle_path",
        "canonical_source_reference_path",
        "contract_v2_path",
    ):
        assert artifact in source

    assert "AtomicScientificSourceBindingBundle.model_validate_json" in source
    assert "PreN10CanonicalSourceReferenceReportV1.model_validate_json" in source
    assert "PreN10ScientificContractReportV2.model_validate_json" in source
