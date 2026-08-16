from __future__ import annotations

from types import SimpleNamespace

from dac_her.external_novelty_contracts import ExternalNoveltyPolicy
from dac_her.hypothesis_novelty_status_lattice_v2 import (
    CLAIM_STATUSES,
    _synthetic_coverage,
    characterize_status_lattice,
    make_assessor,
)


def _reviews(*statuses: str):
    return [
        SimpleNamespace(
            importance="core",
            status=status,
            claim_id=f"c{i}",
        )
        for i, status in enumerate(statuses, start=1)
    ]


def _status(*statuses: str, sufficient: bool = True) -> str:
    assessor = make_assessor(ExternalNoveltyPolicy())
    return assessor._status(
        _reviews(*statuses),
        _synthetic_coverage(sufficient),
    )[0]


def test_exhaustive_lattice_case_count():
    audit = characterize_status_lattice(max_core_claims=4)
    expected = 2 * sum(
        len(CLAIM_STATUSES) ** n
        for n in range(1, 5)
    )
    assert audit["case_count"] == expected
    assert len(audit["rows"]) == expected


def test_positive_relation_plus_gap_is_new_combination():
    assert _status(
        "PARTIAL_PRIOR_ART",
        "COMPONENTS_ONLY",
    ) == "NEW_COMBINATION_OF_KNOWN_EFFECTS"
    assert _status(
        "DIRECT_PRIOR_ART",
        "NO_DIRECT_MATCH_FOUND",
    ) == "NEW_COMBINATION_OF_KNOWN_EFFECTS"


def test_all_components_is_relational_gap():
    assert _status(
        "COMPONENTS_ONLY",
        "COMPONENTS_ONLY",
    ) == "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"


def test_components_plus_no_direct_without_positive_relation_is_relational_gap():
    assert _status(
        "COMPONENTS_ONLY",
        "NO_DIRECT_MATCH_FOUND",
    ) == "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"


def test_title_only_core_fails_closed_even_with_coverage():
    assert _status(
        "TITLE_ONLY_NEIGHBORS",
        "COMPONENTS_ONLY",
    ) == "INSUFFICIENT_SEARCH_EVIDENCE"


def test_insufficient_metadata_core_fails_closed():
    assert _status(
        "COMPONENTS_ONLY",
        "INSUFFICIENT_METADATA",
    ) == "INSUFFICIENT_SEARCH_EVIDENCE"


def test_all_no_direct_requires_coverage():
    assert _status(
        "NO_DIRECT_MATCH_FOUND",
        "NO_DIRECT_MATCH_FOUND",
        sufficient=False,
    ) == "INSUFFICIENT_SEARCH_EVIDENCE"
    assert _status(
        "NO_DIRECT_MATCH_FOUND",
        "NO_DIRECT_MATCH_FOUND",
        sufficient=True,
    ) == "PLAUSIBLY_NOVEL"


def test_direct_partial_semantics_unchanged():
    assert _status(
        "DIRECT_PRIOR_ART",
        "DIRECT_PRIOR_ART",
    ) == "WELL_ESTABLISHED"
    assert _status(
        "DIRECT_PRIOR_ART",
        "PARTIAL_PRIOR_ART",
    ) == "LITERATURE_SUPPORTED_EXTENSION"
