import ast
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import pipeline_core.discovery.external_novelty_llm as llm
from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReviewDraft,
    NoveltyClaim,
)

FIELDS = (
    "prior_art_identity_terms",
    "relation_nucleus_terms",
    "distinguishing_terms",
)


def _capture(empty=False):
    claim = NoveltyClaim(
        claim_id="claim:relation-context",
        hypothesis_id="hypothesis:relation-context",
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        text="Synthetic relation transport fixture.",
        rationale="Prompt contract only.",
        prior_art_identity_terms=["identity alpha", "identity beta"],
        relation_nucleus_terms=["relation gamma", "relation delta"],
        distinguishing_terms=["condition epsilon"],
    )
    if empty:
        claim = claim.model_copy(update={name: [] for name in FIELDS})
    works = [
        {"work_id": "prior_art_work:one", "title": "First fixture",
         "abstract": "First bounded abstract."},
        {"work_id": "prior_art_work:two", "title": "Second fixture",
         "abstract": "Second bounded abstract."},
    ]
    original_claim = claim.model_dump()
    original_works = deepcopy(works)
    backend = llm.InstructorOpenAICompatibleExternalNoveltyBackend(
        model="OFFLINE_TEST_ONLY",
        api_key="OFFLINE_TEST_ONLY",
        base_url="http://127.0.0.1:1",
    )
    draft = ClaimPriorArtReviewDraft(
        matches=[], interpretation="Mock response only."
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=object()))
    with (
        patch.object(backend, "_get_client", return_value=client),
        patch.object(
            llm, "run_instructor_structured_call",
            return_value=(draft, None),
        ) as call,
        patch.object(llm, "record_prior_art_review_call"),
    ):
        result = backend.review_claim(claim, works)
    assert call.call_count == 1
    assert result is draft
    assert claim.model_dump() == original_claim
    assert works == original_works
    return claim, call.call_args.kwargs


def _field(user, name):
    values = [
        line[len(name) + 2:] for line in user.splitlines()
        if line.startswith(name + ": ")
    ]
    assert len(values) == 1, f"missing/duplicate field: {name}"
    return ast.literal_eval(values[0])


@pytest.mark.parametrize("name", FIELDS)
def test_canonical_relation_field_reaches_reviewer_unchanged(name):
    claim, request = _capture()
    user = request["messages"][1]["content"]
    assert _field(user, name) == getattr(claim, name)


def test_relation_specification_is_not_presented_as_evidence():
    _, request = _capture()
    user = request["messages"][1]["content"]
    assert "claim specification only, not prior-art evidence" in user
    assert "they do not establish that relation" in user
    assert (
        "Do not infer missing scientific links from separately mentioned components"
        in user
    )
    boundary = user.index("RETRIEVED PRIOR-ART CANDIDATES")
    assert user.index("claim specification only") < boundary
    for name in FIELDS:
        assert user.index(name + ": ") < boundary


def test_empty_relation_fields_are_not_inferred():
    _, request = _capture(empty=True)
    for name in FIELDS:
        assert _field(request["messages"][1]["content"], name) == []


def test_relation_metadata_only_changes_specification_lines():
    _, filled = _capture()
    _, empty = _capture(empty=True)

    def strip_fields(user):
        return "\n".join(
            line for line in user.splitlines()
            if not any(line.startswith(name + ": ") for name in FIELDS)
        )

    assert filled["messages"][0] == empty["messages"][0]
    assert strip_fields(filled["messages"][1]["content"]) == strip_fields(
        empty["messages"][1]["content"]
    )
    assert (
        {k: v for k, v in filled.items() if k != "messages"}
        == {k: v for k, v in empty.items() if k != "messages"}
    )
    user = filled["messages"][1]["content"]
    assert "[1] work_id=prior_art_work:one" in user
    assert "[2] work_id=prior_art_work:two" in user
    assert "abstract: First bounded abstract." in user
    assert "abstract: Second bounded abstract." in user
    assert (
        "ALLOWED_WORK_IDS\n================\n"
        "prior_art_work:one\nprior_art_work:two\n"
    ) in user
