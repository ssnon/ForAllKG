from __future__ import annotations

from domains.extraction_registry import (
    get_extraction_adapter,
)
from pipeline_core.corpus.extraction.draft_schema import (
    KnowledgeGraphDraft,
)
from pipeline_core.corpus.strict_validation import (
    validate_draft,
)
from pipeline_core.runtime.validation_issues import (
    IssueCode,
)


def _draft(
    *,
    target_type: str,
    target_label: str,
    target_description: str,
    relation: str,
    evidence: str,
) -> KnowledgeGraphDraft:
    return KnowledgeGraphDraft.model_validate({
        "paper_id": "P",
        "chunk_id": "P:main:c",
        "section": "Methods",
        "document_id": "main",
        "document_role": "main",
        "page_ids": [3],
        "asset_ids": [],
        "entities": [
            {
                "id": "method",
                "type": "SynthesisMethod",
                "label": "Synthesis",
                "description": "Source-supported synthesis method.",
            },
            {
                "id": "input",
                "type": target_type,
                "label": target_label,
                "description": target_description,
            },
        ],
        "experiments": [],
        "calculations": [],
        "measurements": [],
        "measurement_groups": [],
        "observation_claims": [],
        "mechanism_claims": [],
        "edges": [
            {
                "source": "method",
                "relation": relation,
                "target": "input",
                "evidence_type": "synthesis_procedure",
                "evidence_strength": "direct",
                "evidence_text": evidence,
                "confidence": "high",
                "evidence_pointers": [
                    {
                        "document_id": "main",
                        "document_role": "main",
                        "page_id": 3,
                        "asset_ids": [],
                        "locator_text": "Methods",
                    }
                ],
                "subsection": "Methods",
            }
        ],
    })


def _report(draft: KnowledgeGraphDraft):
    adapter = get_extraction_adapter(
        "sers_au_ag"
    )

    return validate_draft(
        draft,
        relation_constraints=(
            adapter.strict_relation_constraints
        ),
        semantic_issue_collector=(
            adapter.strict_semantic_issue_collector
        ),
    )


def test_semantic_contract_is_fingerprinted():
    adapter = get_extraction_adapter(
        "sers_au_ag"
    )

    payload = (
        adapter
        .strict_semantic_contract_payload()
    )

    assert payload is not None
    assert (
        payload["contract_id"]
        == "sers_au_ag_precursor_role_v2"
    )
    assert len(payload["rules"]) == 7
    assert payload["collector"].endswith(
        ".collect_sers_strict_semantic_issues"
    )


def test_explicit_reductant_cannot_be_laundered_as_precursor():
    report = _report(
        _draft(
            target_type="Precursor",
            target_label="NaBH4",
            target_description=(
                "Aqueous reductant used in "
                "the gold-seed synthesis."
            ),
            relation="USES_PRECURSOR",
            evidence=(
                "500 μL of 10 mM NaBH4 aqueous "
                "solution was added as a reductant."
            ),
        )
    )

    assert not report.valid
    assert report.count(
        IssueCode.DOMAIN_SEMANTIC_ROLE_CONTRADICTION
    ) == 1


def test_actual_gold_precursor_remains_valid():
    report = _report(
        _draft(
            target_type="Precursor",
            target_label="HAuCl4",
            target_description=(
                "Gold precursor used in seed synthesis."
            ),
            relation="USES_PRECURSOR",
            evidence=(
                "HAuCl4 aqueous solution was used "
                "to prepare the gold seeds."
            ),
        )
    )

    assert report.valid


def test_generic_reagent_is_review_only_not_hard_error():
    report = _report(
        _draft(
            target_type="Precursor",
            target_label="Potassium carbonate (K2CO3)",
            target_description=(
                "Reagent used in the gold growth solution."
            ),
            relation="USES_PRECURSOR",
            evidence=(
                "50 mg of K2CO3 was added to "
                "the gold growth solution."
            ),
        )
    )

    assert report.valid


def test_seed_feedstock_is_not_hard_gated_yet():
    report = _report(
        _draft(
            target_type="Precursor",
            target_label="Gold seed solution",
            target_description=(
                "Gold seed solution used to couple "
                "gold seeds to SiO2 nanoparticles."
            ),
            relation="USES_PRECURSOR",
            evidence=(
                "The wafer was soaked in 10 mL "
                "of gold seed solution."
            ),
        )
    )

    assert report.valid


def test_solvent_word_about_environment_does_not_retype_precursor():
    report = _report(
        _draft(
            target_type="Precursor",
            target_label="Metal precursor",
            target_description=(
                "Metal precursor used in polyol synthesis."
            ),
            relation="USES_PRECURSOR",
            evidence=(
                "The polyol method consists of suspending "
                "the metal precursor in a glycol solvent."
            ),
        )
    )

    assert report.valid


def test_correct_material_reductant_is_valid():
    report = _report(
        _draft(
            target_type="Material",
            target_label="NaBH4",
            target_description=(
                "Aqueous reductant used in "
                "the gold-seed synthesis."
            ),
            relation="USES_MATERIAL",
            evidence=(
                "500 μL of 10 mM NaBH4 aqueous "
                "solution was added as a reductant."
            ),
        )
    )

    assert report.valid

def test_physical_laser_ablation_target_cannot_be_laundered_as_precursor():
    report = _report(
        _draft(
            target_type="Precursor",
            target_label="Bulk Au-Ag targets",
            target_description=(
                "Bulk Au-Ag targets submerged in distilled "
                "water for pulsed laser ablation."
            ),
            relation="USES_PRECURSOR",
            evidence=(
                "Bulk Au-Ag targets were submerged in distilled "
                "water and subjected to pulsed laser ablation."
            ),
        )
    )

    assert not report.valid
    assert report.count(
        IssueCode.DOMAIN_SEMANTIC_ROLE_CONTRADICTION
    ) == 1


def test_physical_laser_ablation_target_as_material_is_valid():
    report = _report(
        _draft(
            target_type="Material",
            target_label="Bulk Au-Ag targets",
            target_description=(
                "Bulk Au-Ag targets submerged in distilled "
                "water for pulsed laser ablation."
            ),
            relation="USES_MATERIAL",
            evidence=(
                "Bulk Au-Ag targets were submerged in distilled "
                "water and subjected to pulsed laser ablation."
            ),
        )
    )

    assert report.valid


def test_generic_au_powder_feedstock_remains_review_only():
    report = _report(
        _draft(
            target_type="Precursor",
            target_label="Au powder",
            target_description=(
                "99.99% Au powder used in the Au "
                "nanoplate synthesis."
            ),
            relation="USES_PRECURSOR",
            evidence=(
                "Au powder was placed in the center "
                "of the heating zone."
            ),
        )
    )

    assert report.valid


def test_explicit_agi_precursor_remains_valid_under_v2():
    report = _report(
        _draft(
            target_type="Precursor",
            target_label="AgI powder",
            target_description=(
                "AgI powder used as the precursor "
                "for AuAg alloy nanoplate synthesis."
            ),
            relation="USES_PRECURSOR",
            evidence=(
                "AgI powder was employed as a precursor "
                "for synthesis of AuAg alloy nanoplates."
            ),
        )
    )

    assert report.valid


def test_physical_target_prompt_guidance_is_fingerprinted():
    from domains.sers.prompts import (
        SERS_PATCH_SYSTEM_PROMPT,
        SERS_PROMPT_VERSION,
        SERS_SYSTEM_PROMPT,
    )

    assert SERS_PROMPT_VERSION.endswith(
        "alpha4a5"
    )

    for prompt in (
        SERS_SYSTEM_PROMPT,
        SERS_PATCH_SYSTEM_PROMPT,
    ):
        assert "physical target or foil" in prompt
        assert "USES_MATERIAL" in prompt
