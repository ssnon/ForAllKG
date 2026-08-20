from pipeline_core.literature.discovery.contracts import LiteratureRecord
from pipeline_core.literature.discovery.relevance import assess_candidate
from pipeline_core.literature.discovery.selection_plan import (
    BucketSelectionRule,
    LiteratureSelectionPlan,
)


def _plan():
    return LiteratureSelectionPlan(
        schema_version="v1",
        plan_id="selection",
        query_plan_id="query",
        target_count=2,
        min_abstract_chars=50,
        min_total_score=5.0,
        allowed_languages=("en",),
        global_context_terms=("electrocatalysis", "catalyst"),
        global_mechanism_terms=("reconstruction", "mechanism"),
        bucket_rules=(
            BucketSelectionRule("working", ("reconstruction", "active site evolution")),
            BucketSelectionRule("kinetics", ("activation barrier", "microkinetic")),
        ),
        max_abstract_chars=500,
    )


def _record(*, abstract, metadata=None, bucket="working"):
    return LiteratureRecord.from_provider_result(
        provider="openalex",
        provider_id="W1",
        title="Operando reconstruction of an electrocatalysis active site",
        abstract=abstract,
        doi="10.1/demo",
        mechanism_bucket=bucket,
        metadata=metadata or {"language": "en"},
    )


def test_relevance_accepts_mechanism_bearing_catalysis_abstract():
    record = _record(
        abstract=(
            "This catalyst study examines electrocatalysis under working conditions. "
            "Adsorbate-driven reconstruction causes active site evolution and changes "
            "the catalytic mechanism during operation."
        )
    )
    assessment = assess_candidate(record, _plan())
    assert assessment.eligible is True
    assert assessment.best_bucket == "working"
    assert assessment.bucket_scores["working"] > assessment.bucket_scores["kinetics"]


def test_relevance_rejects_missing_abstract_and_retracted_work():
    record = _record(abstract=None, metadata={"language": "en", "is_retracted": True})
    assessment = assess_candidate(record, _plan())
    assert assessment.eligible is False
    assert "missing_abstract" in assessment.exclusion_reasons
    assert "retracted" in assessment.exclusion_reasons


def test_relevance_rejects_fulltext_like_abstract_outlier():
    record = _record(
        abstract=(
            "This catalyst study examines electrocatalysis and reconstruction. "
            + ("mechanism reconstruction catalyst " * 60)
        )
    )
    assessment = assess_candidate(record, _plan())
    assert assessment.eligible is False
    assert "abstract_length_outlier" in assessment.exclusion_reasons
