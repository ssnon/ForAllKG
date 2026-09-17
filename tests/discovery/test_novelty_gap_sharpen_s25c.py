from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.novelty_gap_analysis import NoveltyGapAnalyzer


class _Novelty:
    def targeted_query_variants(self, core):
        return [core + " variant"]

    def contextual_conflict_query_variants(self, core):
        return [core + " scope"]


class _Domain:
    novelty = _Novelty()


def _profile(*, relation_backed_ids=None, gap_like_ids=None, complete=True):
    return SimpleNamespace(
        role_binding_complete=complete,
        novelty_bearing_claim_count=1,
        novelty_bearing_relation_backed_claim_ids=list(
            relation_backed_ids or []
        ),
        novelty_bearing_gap_like_claim_ids=list(gap_like_ids or []),
        novelty_bearing_prior_art_state="X",
    )


def test_relation_backed_novelty_claim_triggers_gap_sharpen():
    analyzer = NoveltyGapAnalyzer(domain_profile=_Domain())
    card = SimpleNamespace(
        status="NEW_COMBINATION_OF_KNOWN_EFFECTS",
        novelty_depth_profile=_profile(
            relation_backed_ids=["c2"],
        ),
    )
    assert analyzer._action(card) == "gap_sharpen"


def test_gap_like_novelty_claim_preserves_legacy_keep():
    analyzer = NoveltyGapAnalyzer(domain_profile=_Domain())
    card = SimpleNamespace(
        status="NEW_COMBINATION_OF_KNOWN_EFFECTS",
        novelty_depth_profile=_profile(
            gap_like_ids=["c2"],
        ),
    )
    assert analyzer._action(card) == "keep"


def test_missing_or_incomplete_profile_preserves_legacy_keep():
    analyzer = NoveltyGapAnalyzer(domain_profile=_Domain())

    missing = SimpleNamespace(
        status="NEW_COMBINATION_OF_KNOWN_EFFECTS",
        novelty_depth_profile=None,
    )
    incomplete = SimpleNamespace(
        status="NEW_COMBINATION_OF_KNOWN_EFFECTS",
        novelty_depth_profile=_profile(
            relation_backed_ids=["c2"],
            complete=False,
        ),
    )

    assert analyzer._action(missing) == "keep"
    assert analyzer._action(incomplete) == "keep"


def test_operator_set_is_fixed_and_domain_neutral():
    analyzer = NoveltyGapAnalyzer(domain_profile=_Domain())
    assert analyzer.GAP_SHARPENING_OPERATORS == (
        "MODERATOR",
        "INTERACTION",
        "RESIDUAL",
        "BOUNDARY",
        "PROXY_DECOUPLING",
        "COMPENSATION_LIMIT",
    )
