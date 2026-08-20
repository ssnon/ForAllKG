from __future__ import annotations

from pipeline_core.corpus.vocab_registry import VocabularyRegistry


BROAD_METHODS_ONLY_CONTEXT_ID = "broad-experiment-methods-only-v1"


def build_broad_experiment_methods_vocabulary_context(
    experiment_registry: VocabularyRegistry,
) -> str:
    """Serialize only registry-backed experiment methods for Broad abstracts.

    Broad abstract extraction forbids Measurement/MeasurementGroup generation,
    so metric IDs are not needed on the LLM prompt surface. The metric registry
    remains loaded and is still passed to finalization/validation elsewhere.
    """
    return "\n".join([
        "REGISTERED EXPERIMENT METHODS:",
        *experiment_registry.prompt_lines(metadata_keys=("family",)),
    ])
