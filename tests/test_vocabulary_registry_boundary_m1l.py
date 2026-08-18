from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from dac_her.vocab_registry import (
    ParameterizedVocabularyMatch,
    VocabularyEntry,
    VocabularyRegistry,
    load_default_registries,
    normalize_vocab_text,
    slugify,
)


def _entry(
    entry_id: str,
    label: str,
    *,
    aliases: tuple[str, ...] = (),
    metadata: dict | None = None,
) -> VocabularyEntry:
    return VocabularyEntry(
        entry_id=entry_id,
        label=label,
        aliases=aliases,
        metadata=metadata or {},
    )


def test_vocabulary_dataclass_surfaces_are_frozen():
    assert [
        field.name
        for field in fields(
            VocabularyEntry
        )
    ] == [
        "entry_id",
        "label",
        "aliases",
        "metadata",
    ]

    assert [
        field.name
        for field in fields(
            ParameterizedVocabularyMatch
        )
    ] == [
        "entry",
        "parameters",
        "matched_pattern",
        "matched_text",
    ]


def test_normalization_and_slug_contract():
    assert (
        normalize_vocab_text(
            "  HER—Overpotential_at/10 mA cm^-2  "
        )
        == "her overpotential at 10 ma cm 2"
    )

    assert (
        normalize_vocab_text(
            "η10 / ΔG_H"
        )
        == "η10 δg h"
    )

    assert (
        slugify(
            "η10 / ΔG_H"
        )
        == "eta10_deltag_h"
    )

    assert slugify(None) == "unknown"


def test_registry_resolves_ids_labels_and_aliases():
    entry = _entry(
        "tafel_slope",
        "Tafel slope",
        aliases=(
            "Tafel slopes",
            "TAFEL-SLOPE",
        ),
    )

    registry = VocabularyRegistry(
        kind="metrics",
        version="test-v1",
        entries={
            entry.entry_id: entry,
        },
    )

    assert (
        registry.resolve(
            "tafel_slope"
        )
        is entry
    )

    assert (
        registry.resolve(
            None,
            "Tafel slopes",
        )
        is entry
    )

    assert (
        registry.resolve(
            None,
            "tafel slope",
        )
        is entry
    )

    assert (
        registry.resolve(
            None,
            "unknown metric",
        )
        is None
    )


def test_registry_rejects_alias_collision():
    first = _entry(
        "first",
        "First metric",
        aliases=(
            "shared alias",
        ),
    )

    second = _entry(
        "second",
        "Second metric",
        aliases=(
            "Shared_Alias",
        ),
    )

    with pytest.raises(
        ValueError,
        match="Vocabulary alias collision",
    ):
        VocabularyRegistry(
            kind="metrics",
            version="test-v1",
            entries={
                "first": first,
                "second": second,
            },
        )


def test_from_yaml_is_generic_and_preserves_metadata(
    tmp_path: Path,
):
    path = tmp_path / "custom.yaml"

    payload = {
        "version": "custom-v7",
        "things": {
            "alpha": {
                "label": "Alpha thing",
                "aliases": [
                    "A",
                    "alpha alias",
                ],
                "family": "demo",
                "canonical_unit": "arb",
            },
            "beta": {
                "preferred_label": (
                    "Preferred beta"
                ),
            },
        },
    }

    path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    registry = (
        VocabularyRegistry.from_yaml(
            path,
            root_key="things",
        )
    )

    assert registry.kind == "things"
    assert registry.version == "custom-v7"

    assert (
        registry.entries["alpha"].label
        == "Alpha thing"
    )

    assert (
        registry.entries["alpha"].aliases
        == (
            "A",
            "alpha alias",
        )
    )

    assert (
        registry.entries["alpha"].metadata
        == {
            "family": "demo",
            "canonical_unit": "arb",
        }
    )

    assert (
        registry.entries["beta"].label
        == "Preferred beta"
    )


def test_parameterized_resolution_prefers_exact_then_extracts_groups():
    entry = _entry(
        "adsorption_energy",
        "Adsorption energy",
        aliases=(
            "binding energy",
        ),
        metadata={
            "match_patterns": [
                (
                    r"(?P<species>H|CO)"
                    r"\s+adsorption energy"
                ),
            ],
        },
    )

    registry = VocabularyRegistry(
        kind="metrics",
        version="test-v1",
        entries={
            entry.entry_id: entry,
        },
    )

    match = (
        registry.resolve_parameterized(
            entry_id="adsorption_energy",
            label=None,
            source_texts=(
                "H adsorption energy",
            ),
        )
    )

    assert match.registered is True
    assert match.entry is entry

    assert match.parameters == {
        "species": "H",
    }

    assert (
        match.matched_pattern
        == (
            r"(?P<species>H|CO)"
            r"\s+adsorption energy"
        )
    )


def test_parameterized_resolution_can_resolve_by_pattern():
    generic = _entry(
        "orbital_energy",
        "Orbital energy",
        metadata={
            "match_patterns": [
                (
                    r"(?P<orbital>d|p)"
                    r"-band center"
                ),
            ],
        },
    )

    registry = VocabularyRegistry(
        kind="metrics",
        version="test-v1",
        entries={
            generic.entry_id: generic,
        },
    )

    match = (
        registry.resolve_parameterized(
            entry_id=None,
            label="d-band center",
        )
    )

    assert match.registered is True

    assert (
        match.entry
        is generic
    )

    assert match.parameters == {
        "orbital": "d",
    }


def test_canonical_or_unregistered_contract():
    entry = _entry(
        "tafel_slope",
        "Tafel slope",
        aliases=("Tafel slopes",),
    )

    registry = VocabularyRegistry(
        kind="metrics",
        version="test-v1",
        entries={
            entry.entry_id: entry,
        },
    )

    assert (
        registry.canonical_or_unregistered(
            entry_id=None,
            label="Tafel slopes",
        )
        == (
            "tafel_slope",
            "Tafel slope",
            True,
        )
    )

    assert (
        registry.canonical_or_unregistered(
            entry_id=None,
            label="η custom metric",
        )
        == (
            "unregistered_eta_custom_metric",
            "η custom metric",
            False,
        )
    )


def test_prompt_lines_are_sorted_and_metadata_selected():
    registry = VocabularyRegistry(
        kind="methods",
        version="test-v1",
        entries={
            "zeta": _entry(
                "zeta",
                "Zeta method",
                metadata={
                    "family": "spectroscopy",
                    "ignored": "x",
                },
            ),
            "alpha": _entry(
                "alpha",
                "Alpha method",
                metadata={
                    "family": "microscopy",
                },
            ),
        },
    )

    assert registry.prompt_lines(
        metadata_keys=("family",)
    ) == [
        (
            "- alpha: Alpha method "
            "(family=microscopy)"
        ),
        (
            "- zeta: Zeta method "
            "(family=spectroscopy)"
        ),
    ]


def test_default_loader_policy_is_legacy_owned(
    tmp_path: Path,
):
    vocab_dir = (
        tmp_path
        / "configs"
        / "vocabularies"
    )

    vocab_dir.mkdir(
        parents=True
    )

    (
        vocab_dir
        / "experiment_methods.yaml"
    ).write_text(
        yaml.safe_dump(
            {
                "version": "methods-v1",
                "methods": {
                    "method_a": {
                        "label": "Method A",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    (
        vocab_dir
        / "metrics.yaml"
    ).write_text(
        yaml.safe_dump(
            {
                "version": "metrics-v1",
                "metrics": {
                    "metric_a": {
                        "label": "Metric A",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    experiments, metrics = (
        load_default_registries(
            tmp_path
        )
    )

    assert experiments.kind == "methods"
    assert experiments.version == "methods-v1"

    assert metrics.kind == "metrics"
    assert metrics.version == "metrics-v1"

    assert list(
        experiments.entries
    ) == [
        "method_a"
    ]

    assert list(
        metrics.entries
    ) == [
        "metric_a"
    ]


def test_current_engine_and_default_policy_share_one_legacy_module():
    path = Path(
        "dac_her/vocab_registry.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    top_level = {
        node.name
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.ClassDef,
            ),
        )
    }

    assert {
        "normalize_vocab_text",
        "slugify",
        "VocabularyEntry",
        "ParameterizedVocabularyMatch",
        "VocabularyRegistry",
        "load_default_registries",
    }.issubset(
        top_level
    )

    assert not Path(
        "pipeline_core/"
        "vocabulary_registry.py"
    ).exists()
