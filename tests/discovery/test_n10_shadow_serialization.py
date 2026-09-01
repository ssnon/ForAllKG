import json

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
    NoveltyStructureBasis,
)
from pipeline_core.discovery.nonobviousness_shadow import (
    compile_shadow_claim,
)
from pipeline_core.discovery.novelty_residue import (
    NoveltyResidueClaim,
)


def test_n10_scientific_structure_is_json_serializable_in_shadow():
    statement = (
        "A critical laser power Pc separates two distinct "
        "spacing-to-SERS regimes."
    )

    claim = NoveltyResidueClaim(
        hypothesis_id="hypothesis:h",
        claim_id="claim:c",
        claim_text=statement,
        claim_kind="distinctive_prediction",
        prior_art_status="COMPONENTS_ONLY",
        disposition="RESIDUAL",
        is_residue=True,
        distinguishing_terms=(
            "critical laser power",
        ),
        prior_art_identity_terms=(
            "laser power",
        ),
        relation_nucleus_terms=(
            "interparticle spacing",
            "SERS enhancement",
            "dependence",
        ),
        required_bridge=(
            "Laser power drives a transition at Pc that changes "
            "how spacing maps to measured SERS enhancement."
        ),
        predicted_observation=(
            "Below and above Pc, the spacing-to-SERS response "
            "occupies two distinguishable regimes."
        ),
        falsification_condition=(
            "The spacing-to-SERS response varies smoothly with "
            "power and shows no reproducible regime boundary."
        ),
        direct_or_partial_work_ids=(),
        lower_order_work_ids=(),
        component_work_ids=(),
        scientific_structure=NoveltyClaimScientificStructure(
            introduces_threshold=True,
            introduces_regime_change=True,
            inferential_distance="NEW_REGIME_STRUCTURE",
            mechanistic_necessity="NEW_BRIDGE_REQUIRED",
            regime_specificity="THRESHOLD",
            counterintuitiveness="NONTRIVIAL",
            testable_distinctiveness="QUANTITATIVE",
            basis=[
                NoveltyStructureBasis(
                    feature="threshold",
                    source_text=statement,
                ),
            ],
        ),
    )

    artifact = compile_shadow_claim(
        claim
    )

    encoded = json.dumps(
        artifact
    )

    assert encoded

    structure = (
        artifact["claim"][
            "scientific_structure"
        ]
    )

    assert isinstance(
        structure,
        dict,
    )

    assert (
        structure["introduces_threshold"]
        is True
    )

    assert (
        structure["regime_specificity"]
        == "THRESHOLD"
    )

    assert (
        structure["basis"][0]["feature"]
        == "threshold"
    )
