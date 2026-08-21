"""DAC vocabulary compatibility facade and default vocabulary policy."""

from __future__ import annotations

from pathlib import Path

from pipeline_core.corpus.extraction.vocabulary_registry import (
    ParameterizedVocabularyMatch,
    VocabularyEntry,
    VocabularyRegistry,
    normalize_vocab_text,
    slugify,
)


def load_default_registries(
    project_root: str | Path,
) -> tuple[VocabularyRegistry, VocabularyRegistry]:
    root = Path(project_root)
    vocab_dir = root / "configs" / "vocabularies"
    experiments = VocabularyRegistry.from_yaml(
        vocab_dir / "experiment_methods.yaml",
        root_key="methods",
    )
    metrics = VocabularyRegistry.from_yaml(
        vocab_dir / "metrics.yaml",
        root_key="metrics",
    )
    return experiments, metrics
