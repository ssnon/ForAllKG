from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.higher_order_tension_extractor import (
    extract_scientific_tension_candidates,
)


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _premise(
    pid: str,
    role: str,
    relation: str,
    subject: str,
    obj: str,
    authority: str = "confirmed_known",
):
    return ns(
        premise_id=pid,
        premise_role=role,
        relation=relation,
        subject=subject,
        object=obj,
        authority=ns(value=authority),
    )


def _arm(
    *,
    context_id: str,
    premises,
    modifier_text: str,
    anchor_text: str,
    anchor_role: str = "mediator",
):
    context = ns(
        context_id=context_id,
        requested_source="nanostructure shape",
        requested_target="electromagnetic hotspot location and intensity",
        premises=list(premises),
        structural_opportunity=ns(
            modifier_text=modifier_text,
            modifier_anchor_text=anchor_text,
            modifier_anchor_role=anchor_role,
        ),
    )
    return ns(higher_order_context=context)


def _critique(*rows):
    return ns(arms=list(rows))


def _critic_row(index: int, *codes):
    return ns(
        arm_index=index,
        issues=[ns(code=code) for code in codes],
    )


def test_anchor_omission_becomes_proxy_decoupling_tension_only():
    arm = _arm(
        context_id="ctx:air",
        premises=[
            _premise(
                "p:air",
                "modifier_relation",
                "VARIES_WITH",
                "SERS enhancement factor",
                "air exposure time",
                "candidate_inspiration",
            )
        ],
        modifier_text="air exposure time",
        anchor_text="SERS enhancement factor",
    )
    result = extract_scientific_tension_candidates(
        outcome=ns(arms=(arm,)),
        semantic_critique=_critique(
            _critic_row(1, "MODIFIER_ANCHOR_BRIDGE_OMITTED")
        ),
    )

    assert result.candidate_count == 1
    row = result.candidates[0]
    assert row.tension_type == "proxy_decoupling"
    assert row.candidate_inspiration_involved is True
    assert row.competing_explanation_generation_authorized is False
    assert row.positive_premise_authority is False
    assert row.novelty_authority is False


def test_explicit_tradeoff_relation_becomes_tradeoff_tension():
    arm = _arm(
        context_id="ctx:tradeoff",
        premises=[
            _premise(
                "p:trade",
                "target_backbone_relation",
                "IMPOSES_TRADEOFF",
                "hotspot intensity",
                "hotspot size",
            )
        ],
        modifier_text="particle spacing",
        anchor_text="field enhancement",
    )
    result = extract_scientific_tension_candidates(
        outcome=ns(arms=(arm,)),
        semantic_critique=_critique(_critic_row(1)),
    )

    assert result.candidate_count == 1
    assert result.candidates[0].tension_type == "tradeoff_pareto"


def test_explicit_contrast_is_not_promoted_to_conflicting_literature():
    arm = _arm(
        context_id="ctx:contrast",
        premises=[
            _premise(
                "p:contrast",
                "modifier_relation",
                "CONTRASTS_WITH",
                "electromagnetic enhancement",
                "decoration material",
            )
        ],
        modifier_text="decoration material",
        anchor_text="electromagnetic enhancement",
    )
    result = extract_scientific_tension_candidates(
        outcome=ns(arms=(arm,)),
        semantic_critique=_critique(_critic_row(1)),
    )

    row = result.candidates[0]
    assert row.tension_type == "contrasting_relations"
    assert "conflicting literature" in row.rationale.lower()
    assert row.gap_authority is False


def test_restatement_or_duplicate_issue_alone_does_not_create_tension():
    arm = _arm(
        context_id="ctx:restatement",
        premises=[
            _premise(
                "p:mod",
                "modifier_relation",
                "VARIES_WITH",
                "plasmonic properties",
                "nanostructure shape",
                "candidate_inspiration",
            )
        ],
        modifier_text="plasmonic properties",
        anchor_text="nanostructure shape",
        anchor_role="source",
    )
    result = extract_scientific_tension_candidates(
        outcome=ns(arms=(arm,)),
        semantic_critique=_critique(
            _critic_row(
                1,
                "BACKBONE_RESTATEMENT_RISK",
                "KNOWN_CONTEXT_DUPLICATE",
            )
        ),
    )

    assert result.candidate_count == 0
    assert result.competing_explanation_generation_authorized is False
