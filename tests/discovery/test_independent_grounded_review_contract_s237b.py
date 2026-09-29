from pipeline_core.discovery.independent_evidence_grounded_review import (
    build_single_work_user_prompt,
)


def test_s237b_uses_full_abstract_without_truncation():
    abstract = "A" * 5000
    prompt = build_single_work_user_prompt(
        claim_text="X affects Y.",
        work={
            "work_id": "w1",
            "title": "t",
            "doi": "10.1/x",
            "abstract": abstract,
        },
    )
    assert abstract in prompt
    assert "…" not in prompt


def test_s237b_prompt_excludes_extra_decomposition_hints():
    prompt = build_single_work_user_prompt(
        claim_text="X affects Y.",
        work={
            "work_id": "w1",
            "title": "t",
            "doi": "10.1/x",
            "abstract": "X is associated with Y.",
        },
    )
    assert "prior_art_identity_terms" not in prompt
    assert "relation_nucleus_terms" not in prompt
    assert "distinguishing_terms" not in prompt
    assert "importance:" not in prompt
    assert "kind:" not in prompt
