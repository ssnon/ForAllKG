from __future__ import annotations

import pipeline_core.corpus.schemas as legacy
import pipeline_core.corpus.extraction.evidence_schema as core


def test_legacy_evidence_schema_symbols_are_core_objects():
    assert legacy.DocumentRole is core.DocumentRole
    assert legacy.EvidenceType is core.EvidenceType
    assert legacy.EvidenceStrength is core.EvidenceStrength
    assert legacy.ConfidenceLevel is core.ConfidenceLevel
    assert legacy.RelationType is core.RelationType
    assert legacy.EvidencePointer is core.EvidencePointer
    assert legacy.KGEdge is core.KGEdge


def test_evidence_models_are_owned_by_pipeline_core():
    assert (
        core.EvidencePointer.__module__
        == "pipeline_core.corpus.extraction.evidence_schema"
    )
    assert core.KGEdge.__module__ == "pipeline_core.corpus.extraction.evidence_schema"


def test_known_relation_vocabulary_remains_legacy_owned():
    assert hasattr(legacy, "KnownRelationType")
    assert not hasattr(core, "KnownRelationType")
