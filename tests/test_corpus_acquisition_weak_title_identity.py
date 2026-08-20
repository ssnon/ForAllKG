from dac_her.corpus_acquisition.catalog_expansion import (
    _identity_keys,
)
from pipeline_core.literature.catalog import (
    canonicalize_catalog_works,
    normalize_title,
)
from pipeline_core.literature.catalog_contracts import (
    CatalogWork,
)


def _work(
    *,
    work_id: str,
    provider_id: str,
    title: str,
    query_id: str,
) -> CatalogWork:
    return CatalogWork(
        work_id=work_id,
        title=title,
        year=2024,
        publication_date=None,
        doi=None,
        url=None,
        open_access_url=None,
        abstract="SERS metadata candidate",
        authors=[],
        venue=None,
        citation_count=None,
        publication_types=[],
        providers=["openalex"],
        provider_ids={
            "openalex": provider_id,
        },
        retrieval_query_ids=[
            query_id,
        ],
        retrieval_axis_ids=[
            "nanogap",
        ],
    )


def test_non_latin_weak_titles_do_not_merge() -> None:
    first = _work(
        work_id="catalog_work:provider_w1",
        provider_id="W7146188176",
        title=(
            "金属ナノ粒子配列 : 光学評価、光化学"
            "リアクター及びSERS型センサーへの応用"
        ),
        query_id="q1",
    )

    second = _work(
        work_id="catalog_work:provider_w2",
        provider_id="W7999999999",
        title=(
            "電気化学反応解析に向けたその場"
            "表面増強ラマン分光(SERS)センサの研究"
        ),
        query_id="q2",
    )

    assert normalize_title(first.title) == "sers"
    assert normalize_title(second.title) == "sers"

    rows, _ = canonicalize_catalog_works(
        [
            first,
            second,
        ]
    )

    assert len(rows) == 2

    assert {
        row.work_id
        for row in rows
    } == {
        first.work_id,
        second.work_id,
    }


def test_same_provider_work_still_merges_across_queries() -> None:
    first = _work(
        work_id="catalog_work:provider_w1",
        provider_id="W7146188176",
        title="日本語SERS研究",
        query_id="q1",
    )

    second = _work(
        work_id="catalog_work:provider_w1",
        provider_id="W7146188176",
        title="日本語SERS研究",
        query_id="q2",
    )

    rows, _ = canonicalize_catalog_works(
        [
            first,
            second,
        ]
    )

    assert len(rows) == 1

    assert set(
        rows[0].retrieval_query_ids
    ) == {
        "q1",
        "q2",
    }


def test_openalex_native_id_is_strong_expansion_identity() -> None:
    row = _work(
        work_id="catalog_work:any",
        provider_id="W7146188176",
        title=(
            "金属ナノ粒子配列 : 光学評価、光化学"
            "リアクター及びSERS型センサーへの応用"
        ),
        query_id="q1",
    )

    keys = set(
        _identity_keys(row)
    )

    assert (
        "provider:openalex:W7146188176"
        in keys
    )

    assert "title:sers" not in keys


def test_different_openalex_ids_remain_distinct_for_weak_title() -> None:
    one = _work(
        work_id="catalog_work:a",
        provider_id="W1",
        title="日本語SERS研究",
        query_id="q1",
    )

    two = _work(
        work_id="catalog_work:b",
        provider_id="W2",
        title="別の日本語SERS研究",
        query_id="q2",
    )

    assert (
        set(_identity_keys(one))
        != set(_identity_keys(two))
    )
