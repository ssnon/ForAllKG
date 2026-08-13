import pytest

from dac_her.reproducibility_domain import ReproducibilityEvidence


def test_reproducibility_contract_preserves_numeric_text_xor():
    with pytest.raises(ValueError):
        ReproducibilityEvidence(
            evidence_id="repro:x",
            domain_profile_id="sers_au_ag",
            reproducibility_semantics_id="sem",
            paper_id="P1",
            evidence_kind="relative_standard_deviation",
            reproducibility_scope="batch_to_batch",
            value_numeric=5.0,
            value_text="five percent",
            source_node_ids=("m1",),
        )


def test_reproducibility_contract_requires_grounded_source():
    with pytest.raises(ValueError):
        ReproducibilityEvidence(
            evidence_id="repro:x",
            domain_profile_id="sers_au_ag",
            reproducibility_semantics_id="sem",
            paper_id="P1",
            evidence_kind="repeatability_statement",
            reproducibility_scope="unknown",
        )
