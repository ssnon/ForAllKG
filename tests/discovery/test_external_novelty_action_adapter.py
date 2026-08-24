from types import SimpleNamespace

import pytest

from pipeline_core.discovery.external_novelty_action_adapter import (
    ExternalNoveltyActionBindingError,
    _claim_authority,
)


@pytest.mark.parametrize(
    (
        "status",
        "importance",
        "authority",
    ),
    [
        (
            "COMPONENTS_ONLY",
            "core",
            "informational",
        ),
        (
            "PARTIAL_PRIOR_ART",
            "core",
            "actionable",
        ),
        (
            "PARTIAL_PRIOR_ART",
            "supporting",
            "advisory",
        ),
        (
            "DIRECT_PRIOR_ART",
            "core",
            "terminal_candidate",
        ),
        (
            "DIRECT_PRIOR_ART",
            "supporting",
            "actionable",
        ),
        (
            "INSUFFICIENT_METADATA",
            "core",
            "advisory",
        ),
    ],
)
def test_claim_authority_mapping(
    status,
    importance,
    authority,
):
    assert (
        _claim_authority(
            status=status,
            importance=importance,
        )
        == authority
    )


def test_unknown_claim_status_is_rejected():
    with pytest.raises(
        ExternalNoveltyActionBindingError,
        match="unsupported",
    ):
        _claim_authority(
            status="NOT_A_STATUS",
            importance="core",
        )


def test_pre_refinement_novelty_cannot_cross_accepted_refinement():
    adapter = __import__(
        "pipeline_core.discovery.external_novelty_action_adapter",
        fromlist=[
            "ExternalNoveltyFindingActionAdapter",
        ],
    ).ExternalNoveltyFindingActionAdapter()

    base = SimpleNamespace(
        epistemic_usage=(
            "prior_art_only_not_positive_premise"
        ),
        external_novelty_claim_scope=(
            "search-bounded_prior-art_assessment_not_literature-wide_proof"
        ),
        source_portfolio_id="portfolio:source",
        report_id="external:test",
        cards=[
            SimpleNamespace(
                hypothesis_id="hypothesis:source",
            )
        ],
    )

    corrected = {
        "schema_version": (
            "n1-4b-relation-nucleus-"
            "hybrid-reaggregation-corrected-v1"
        ),
        "cards": [
            {
                "hypothesis_id":
                    "hypothesis:source",

                "new_status":
                    "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",

                "claims": [],
            }
        ],
    }

    refinement = SimpleNamespace(
        source_portfolio_id="portfolio:source",
        final_portfolio_id="portfolio:final",
        report_id="refinement:test",
        attempts=[
            SimpleNamespace(
                original_hypothesis_id=
                    "hypothesis:source",

                decision=
                    "accepted_refinement",

                final_hypothesis_id=
                    "hypothesis:final",
            )
        ],
    )

    final = SimpleNamespace(
        portfolio_id="portfolio:final",
        hypotheses=[
            SimpleNamespace(
                hypothesis_id=
                    "hypothesis:final",
            )
        ],
    )

    with pytest.raises(
        ExternalNoveltyActionBindingError,
        match=(
            "fresh final novelty assessment "
            "is required"
        ),
    ):
        adapter.normalize(
            base_report=base,
            corrected_overlay=corrected,
            refinement_report=refinement,
            final_portfolio=final,
        )
