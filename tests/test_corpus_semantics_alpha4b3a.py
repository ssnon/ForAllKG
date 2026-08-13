from __future__ import annotations

import pytest

from dac_her.domain_profile import CorpusSemantics
from dac_her.domains.catalysis_mechanism import CATALYSIS_MECHANISM_PROFILE
from dac_her.domains.dac_her import DAC_HER_PROFILE
from dac_her.domains.sers_au_ag import SERS_AU_AG_PROFILE
from dac_her.resolution_candidates import normalize_scientific_text


def test_alpha4b3a_profiles_expose_explicit_corpus_semantics() -> None:
    assert DAC_HER_PROFILE.corpus is not None
    assert SERS_AU_AG_PROFILE.corpus is not None
    assert CATALYSIS_MECHANISM_PROFILE.corpus is not None

    assert DAC_HER_PROFILE.corpus.semantics_id == "dac_her_corpus_v1_alpha4b3a"
    assert SERS_AU_AG_PROFILE.corpus.semantics_id == "sers_au_ag_corpus_v1_alpha4b3a"
    assert (
        CATALYSIS_MECHANISM_PROFILE.corpus.semantics_id
        == "catalysis_mechanism_corpus_v1_alpha4b3a"
    )


def test_alpha4b3a_registry_alignment_source_of_truth_is_resolution_semantics() -> None:
    assert DAC_HER_PROFILE.resolution.auto_merge_types == frozenset(
        {"Metal", "Reaction"}
    )
    assert SERS_AU_AG_PROFILE.resolution.auto_merge_types == frozenset({"Metal"})
    assert CATALYSIS_MECHANISM_PROFILE.resolution.auto_merge_types == frozenset(
        {"Metal", "Reaction"}
    )


def test_alpha4b3a_review_and_pattern_capabilities_are_domain_owned() -> None:
    assert SERS_AU_AG_PROFILE.corpus is not None
    assert SERS_AU_AG_PROFILE.corpus.review_candidate_types == frozenset(
        {
            "PlasmonicSubstrate",
            "Nanostructure",
            "Support",
            "Material",
            "StructuralMotif",
            "Morphology",
            "Analyte",
            "RamanReporter",
        }
    )
    assert SERS_AU_AG_PROFILE.corpus.pattern_alignment_mode == "confirmed_exact"

    assert CATALYSIS_MECHANISM_PROFILE.corpus is not None
    assert CATALYSIS_MECHANISM_PROFILE.corpus.pattern_alignment_mode == "disabled"


def test_alpha4b3a_corpus_semantics_validates_contract() -> None:
    with pytest.raises(ValueError, match="semantics_id"):
        CorpusSemantics(
            semantics_id=" ",
            review_candidate_types=frozenset(),
        )

    with pytest.raises(ValueError, match="review candidate"):
        CorpusSemantics(
            semantics_id="demo",
            review_candidate_types=frozenset({"", "Material"}),
        )

    with pytest.raises(ValueError, match="pattern_alignment_mode"):
        CorpusSemantics(
            semantics_id="demo",
            review_candidate_types=frozenset({"Material"}),
            pattern_alignment_mode="fuzzy",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="subset"):
        CorpusSemantics(
            semantics_id="demo",
            review_candidate_types=frozenset({"Material"}),
            high_priority_review_types_override=frozenset({"Support"}),
        )


def test_alpha4b3a_normalization_can_be_bound_to_domain_without_global_mutation() -> None:
    sers = normalize_scientific_text(
        "surface-enhanced Raman scattering nanoparticles",
        domain_profile=SERS_AU_AG_PROFILE,
    )
    her = normalize_scientific_text(
        "surface-enhanced Raman scattering nanoparticles",
        domain_profile=DAC_HER_PROFILE,
    )
    assert sers == "sers nanoparticle"
    assert her != sers
