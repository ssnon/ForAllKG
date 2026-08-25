from __future__ import annotations

import pytest

from domains.sers.context_contracts import (
    SERSContextFact,
    SERSContextProvenance,
    SERSContextSignature,
)
from domains.sers.hypothesis_context_contracts import (
    HypothesisContextAssertionDraft,
    HypothesisContextInterpretationDraft,
    HypothesisContextMentionDraft,
    expected_hypothesis_context_assertions,
)
from domains.sers.hypothesis_context_interpreter import (
    HypothesisContextInterpreterValidationError,
    SERSHypothesisContextInterpreter,
)
from domains.sers.hypothesis_context_llm import (
    HypothesisContextGeneration,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    PredictedObservation,
)


VALID_FACT = "sers_context_fact:valid"
BAD_FACT = "sers_context_fact:hallucinated"


def _card() -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="hypothesis:test",
        domain_profile_id="sers_au_ag",
        source_context_id="context:test",
        source_context_sha256="context-sha",
        source_report_id="report:test",
        source_report_sha256="report-sha",
        title="Nanogap response",
        hypothesis_statement=(
            "A nanogap controls SERS enhancement."
        ),
        hypothesis_type="mechanistic_extension",
        premise_statement_ids=["stmt:1"],
        inferential_bridge=(
            "The nanogap changes the local response."
        ),
        predicted_observations=[
            PredictedObservation(
                observation_id="obs:1",
                observable="SERS intensity",
                expected_direction="increase",
                rationale=(
                    "Higher local enhancement is expected."
                ),
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="crit:1",
                observable="SERS intensity",
                falsifying_outcome=(
                    "No increase is observed."
                ),
            )
        ],
        source_paper_ids=["paper:1"],
        candidate_dependency="supporting",
        evidence_profile=
            HypothesisEvidenceProfile(
                premise_count=1,
                gap_count=0,
                source_paper_count=1,
                candidate_premise_count=1,
                reported_premise_count=0,
                synthesis_premise_count=0,
            ),
    )


def _signature() -> SERSContextSignature:
    return SERSContextSignature(
        signature_id="signature:test",
        domain_profile_id="sers_au_ag",
        scope="axis_inspiration",
        source_ref_id="axis:test",
        facts=[
            SERSContextFact(
                fact_id=VALID_FACT,
                dimension="gap_regime",
                scientific_role="gap_regime",
                knowledge_state="explicit",
                value="nanogap",
                normalized_value="nanogap",
                provenance=[
                    SERSContextProvenance(
                        kind="axis_anchor",
                        candidate_unit_ids=[
                            "candidate_unit:test"
                        ],
                    )
                ],
            )
        ],
    )


def _draft(
    fact_id: str,
    *,
    corrupt_central_text: bool = False,
) -> HypothesisContextInterpretationDraft:
    card = _card()

    assertions = []

    for row in (
        expected_hypothesis_context_assertions(
            card
        )
    ):
        text = row["assertion_text"]

        if (
            corrupt_central_text
            and row["assertion_kind"]
            == "central"
        ):
            text = "Different assertion text."

        mentions = []

        if row["assertion_kind"] == "central":
            mentions = [
                HypothesisContextMentionDraft(
                    mention_id="central_m1",
                    mention_text="nanogap",
                    source_fact_ids=[
                        fact_id
                    ],
                    asserted_dimension=
                        "gap_regime",
                    asserted_role=
                        "gap_regime",
                    treatment="preserve",
                    experimental_role=
                        "experimental_variable",
                    rationale=(
                        "Maps the nanogap phrase "
                        "to the supplied gap context."
                    ),
                )
            ]

        assertions.append(
            HypothesisContextAssertionDraft(
                assertion_id=
                    row["assertion_id"],
                assertion_kind=
                    row["assertion_kind"],
                assertion_text=text,
                mentions=mentions,
            )
        )

    return HypothesisContextInterpretationDraft(
        hypothesis_id=card.hypothesis_id,
        source_signature_ids=[
            "signature:test"
        ],
        assertions=assertions,
    )


class SequenceBackend:
    backend_name = "test"
    model_name = "test"

    def __init__(
        self,
        drafts: list[
            HypothesisContextInterpretationDraft
        ],
    ) -> None:
        self.drafts = list(drafts)
        self.prompts = []

    def interpret(self, prompt):
        self.prompts.append(prompt)

        if not self.drafts:
            raise AssertionError(
                "unexpected extra backend call"
            )

        return HypothesisContextGeneration(
            draft=self.drafts.pop(0)
        )


def test_unknown_fact_gets_one_bounded_repair() -> None:
    backend = SequenceBackend([
        _draft(BAD_FACT),
        _draft(VALID_FACT),
    ])

    outcome = (
        SERSHypothesisContextInterpreter(
            backend
        ).interpret(
            card=_card(),
            source_signatures=[
                _signature()
            ],
        )
    )

    assert len(backend.prompts) == 2

    repair_prompt = (
        backend.prompts[1].user_prompt
    )

    assert (
        "VALIDATION REPAIR INPUT"
        in repair_prompt
    )

    assert VALID_FACT in repair_prompt
    assert BAD_FACT in repair_prompt

    assert (
        outcome.canonical_draft
        .assertions[0]
        .mentions[0]
        .source_fact_ids
        == [VALID_FACT]
    )


def test_second_unknown_fact_failure_remains_fail_closed() -> None:
    backend = SequenceBackend([
        _draft(BAD_FACT),
        _draft(BAD_FACT),
    ])

    with pytest.raises(
        HypothesisContextInterpreterValidationError
    ):
        SERSHypothesisContextInterpreter(
            backend
        ).interpret(
            card=_card(),
            source_signatures=[
                _signature()
            ],
        )

    assert len(backend.prompts) == 2


def test_non_reference_validation_failure_is_not_retried() -> None:
    backend = SequenceBackend([
        _draft(
            VALID_FACT,
            corrupt_central_text=True,
        )
    ])

    with pytest.raises(
        HypothesisContextInterpreterValidationError
    ):
        SERSHypothesisContextInterpreter(
            backend
        ).interpret(
            card=_card(),
            source_signatures=[
                _signature()
            ],
        )

    assert len(backend.prompts) == 1
