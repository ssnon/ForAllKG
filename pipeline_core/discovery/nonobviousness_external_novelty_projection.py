from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisContext,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.nonobviousness_canonical_candidate import (
    N11CanonicalCandidate,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class N11ExternalNoveltyProjection(
    StrictModel
):
    schema_version: Literal[
        "n11-external-novelty-projection-v1"
    ] = "n11-external-novelty-projection-v1"

    source_candidate_id: str

    projection_role: Literal[
        "EXTERNAL_NOVELTY_INPUT_ONLY"
    ] = "EXTERNAL_NOVELTY_INPUT_ONLY"

    supplemental_provenance_retained_upstream: Literal[
        True
    ] = True

    supplemental_promoted_to_positive_premise: Literal[
        False
    ] = False

    gap_promoted_to_positive_premise: Literal[
        False
    ] = False

    production_authority: Literal[
        False
    ] = False

    portfolio: HypothesisPortfolio


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(part)
        for part in parts
    ).encode(
        "utf-8"
    )

    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


class N11ExternalNoveltyProjectionCompiler:
    def compile(
        self,
        *,
        context: HypothesisContext,
        candidate: N11CanonicalCandidate,
    ) -> N11ExternalNoveltyProjection:
        if candidate.production_authority:
            raise ValueError(
                "pre-N10 N11 candidate must not "
                "have production authority"
            )

        if (
            candidate.novelty_status
            != "NOT_ASSESSED"
        ):
            raise ValueError(
                "external-novelty projection requires "
                "an unassessed canonical candidate"
            )

        if (
            candidate.n10_status
            != "NOT_ASSESSED"
        ):
            raise ValueError(
                "external-novelty projection must occur "
                "before N10 adjudication"
            )

        if (
            candidate.source_context_id
            != context.context_id
        ):
            raise ValueError(
                "candidate/context ID mismatch"
            )

        if (
            candidate.source_context_sha256
            != context.context_sha256
        ):
            raise ValueError(
                "candidate/context SHA mismatch"
            )

        if (
            candidate.source_report_id
            != context.source_report_id
        ):
            raise ValueError(
                "candidate/report ID mismatch"
            )

        if (
            candidate.source_report_sha256
            != context.source_report_sha256
        ):
            raise ValueError(
                "candidate/report SHA mismatch"
            )

        statements = {
            row.statement_id: row
            for row
            in context.evidence_statements
        }

        premises = []

        for statement_id in (
            candidate
            .baseline_premise_statement_ids
        ):
            row = statements.get(
                statement_id
            )

            if row is None:
                raise ValueError(
                    "unknown canonical baseline premise: "
                    + statement_id
                )

            if not row.eligible_as_premise:
                raise ValueError(
                    "canonical baseline premise became "
                    "ineligible: "
                    + statement_id
                )

            if row.alignment_path_ids:
                raise ValueError(
                    "alignment-dependent baseline premise "
                    "cannot enter novelty projection: "
                    + statement_id
                )

            premises.append(
                row
            )

        gaps = []

        for statement_id in (
            candidate.gap_statement_ids
        ):
            row = statements.get(
                statement_id
            )

            if row is None:
                raise ValueError(
                    "unknown canonical gap: "
                    + statement_id
                )

            if not row.eligible_as_gap:
                raise ValueError(
                    "canonical gap became ineligible: "
                    + statement_id
                )

            gaps.append(
                row
            )

        premise_papers = sorted(
            {
                paper_id
                for row in premises
                for paper_id in row.paper_ids
            }
        )

        gap_papers = sorted(
            {
                paper_id
                for row in gaps
                for paper_id in row.paper_ids
            }
        )

        candidate_count = sum(
            bool(
                row.requires_verification
            )
            for row in premises
        )

        if candidate_count == 0:
            candidate_dependency = "none"

        elif candidate_count == len(
            premises
        ):
            candidate_dependency = "essential"

        else:
            candidate_dependency = "supporting"

        prediction_by_id = {
            row.prediction_id: row
            for row
            in candidate.predictions
        }

        predictions = [
            PredictedObservation(
                observation_id=
                    row.prediction_id,

                observable=
                    row.observable,

                expected_direction=
                    row.expected_direction,

                rationale=
                    row.rationale,
            )
            for row
            in candidate.predictions
        ]

        falsifiers = []

        for row in candidate.falsifiers:
            prediction = (
                prediction_by_id.get(
                    row.prediction_id
                )
            )

            if prediction is None:
                raise ValueError(
                    "canonical falsifier references "
                    "unknown prediction"
                )

            # Standard HypothesisCard requires the observable
            # to be repeated. Reuse the exact referenced
            # prediction observable rather than asking an LLM
            # to paraphrase it.
            falsifiers.append(
                FalsificationCriterion(
                    criterion_id=
                        row.falsifier_id,

                    observable=
                        prediction.observable,

                    falsifying_outcome=
                        row.falsifying_outcome,
                )
            )

        hypothesis_id = _stable_id(
            "hypothesis",
            "n11_external_novelty_projection",
            candidate.candidate_id,
        )

        card = HypothesisCard(
            hypothesis_id=
                hypothesis_id,

            domain_profile_id=
                candidate.domain_profile_id,

            source_context_id=
                candidate.source_context_id,

            source_context_sha256=
                candidate.source_context_sha256,

            source_report_id=
                candidate.source_report_id,

            source_report_sha256=
                candidate.source_report_sha256,

            title=
                candidate.title,

            hypothesis_statement=
                candidate.hypothesis_statement,

            hypothesis_type=
                candidate.hypothesis_type,

            # CRITICAL:
            # supplemental evidence is deliberately absent here.
            premise_statement_ids=list(
                candidate
                .baseline_premise_statement_ids
            ),

            gap_statement_ids=list(
                candidate.gap_statement_ids
            ),

            inferential_bridge=
                candidate.inferential_bridge,

            predicted_observations=
                predictions,

            falsification_criteria=
                falsifiers,

            assumptions=list(
                candidate.assumptions
            ),

            # Standard HypothesisCard semantics:
            # source paper scope derives ONLY from positive premises.
            source_paper_ids=
                premise_papers,

            gap_paper_ids=
                gap_papers,

            cross_paper_synthesis=(
                len(
                    premise_papers
                )
                >= 2
            ),

            candidate_dependency=
                candidate_dependency,

            evidence_profile=
                HypothesisEvidenceProfile(
                    premise_count=
                        len(
                            premises
                        ),

                    gap_count=
                        len(
                            gaps
                        ),

                    source_paper_count=
                        len(
                            premise_papers
                        ),

                    candidate_premise_count=
                        candidate_count,

                    reported_premise_count=
                        sum(
                            row.epistemic_role
                            == "reported"
                            for row
                            in premises
                        ),

                    synthesis_premise_count=
                        sum(
                            row.epistemic_role
                            == "evidence_synthesis"
                            for row
                            in premises
                        ),
                ),

            status=
                "hypothesized",

            novelty_status=
                "not_assessed",
        )

        portfolio = HypothesisPortfolio(
            portfolio_id=_stable_id(
                "hypothesis_portfolio",
                "n11_external_novelty_projection",
                candidate.candidate_id,
                hypothesis_id,
            ),

            domain_profile_id=
                candidate.domain_profile_id,

            source_context_id=
                candidate.source_context_id,

            source_context_sha256=
                candidate.source_context_sha256,

            source_report_id=
                candidate.source_report_id,

            source_report_sha256=
                candidate.source_report_sha256,

            hypotheses=[
                card
            ],

            abstention_reason=
                None,
        )

        return N11ExternalNoveltyProjection(
            source_candidate_id=
                candidate.candidate_id,

            portfolio=
                portfolio,
        )
