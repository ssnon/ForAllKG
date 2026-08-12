from dac_her.literature_discovery.contracts import LiteratureRecord
from dac_her.literature_discovery.query_plan import LiteratureQueryPlan, QueryBucket
from dac_her.literature_discovery.selection import select_literature
from dac_her.literature_discovery.selection_plan import (
    BucketSelectionRule,
    LiteratureSelectionPlan,
)


def _query_plan():
    return LiteratureQueryPlan(
        schema_version="v1",
        plan_id="query",
        description="",
        buckets=(
            QueryBucket("working", "Working", 50, ("q1",)),
            QueryBucket("kinetics", "Kinetics", 50, ("q2",)),
        ),
    )


def _selection_plan():
    return LiteratureSelectionPlan(
        schema_version="v1",
        plan_id="selection",
        query_plan_id="query",
        target_count=4,
        min_abstract_chars=50,
        min_total_score=5.0,
        allowed_languages=("en",),
        global_context_terms=("electrocatalysis", "catalyst"),
        global_mechanism_terms=("mechanism", "reconstruction", "activation barrier"),
        bucket_rules=(
            BucketSelectionRule("working", ("reconstruction",)),
            BucketSelectionRule("kinetics", ("activation barrier",)),
        ),
    )


def _record(index: int, bucket: str, phrase: str):
    return LiteratureRecord.from_provider_result(
        provider="openalex",
        provider_id=f"W{index}",
        title=f"Catalyst {phrase} electrocatalysis {index}",
        abstract=(
            f"This electrocatalysis catalyst paper studies the {phrase} mechanism "
            "under controlled reaction conditions and reports mechanistic evidence "
            "for changes in the active catalytic state."
        ),
        doi=f"10.1/{index}",
        mechanism_bucket=bucket,
        metadata={"language": "en"},
    )


def test_balanced_selection_respects_quota_and_is_deterministic():
    records = [
        _record(1, "working", "reconstruction"),
        _record(2, "working", "reconstruction"),
        _record(3, "working", "reconstruction"),
        _record(4, "kinetics", "activation barrier"),
        _record(5, "kinetics", "activation barrier"),
        _record(6, "kinetics", "activation barrier"),
    ]
    first = select_literature(
        records, query_plan=_query_plan(), selection_plan=_selection_plan()
    )
    second = select_literature(
        list(reversed(records)), query_plan=_query_plan(), selection_plan=_selection_plan()
    )
    assert first.selected_count == 4
    assert [row.record.paper_id for row in first.selected] == [
        row.record.paper_id for row in second.selected
    ]
    counts = {}
    for row in first.selected:
        counts[row.assigned_bucket] = counts.get(row.assigned_bucket, 0) + 1
    assert counts == {"kinetics": 2, "working": 2}
