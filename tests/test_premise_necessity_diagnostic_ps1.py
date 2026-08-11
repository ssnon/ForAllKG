from __future__ import annotations

from dac_her.premise_necessity_diagnostic import (
    PremiseSemanticScores,
    _pareto_dominators,
    _rank_descending,
)


def _scores(
    h: float,
    b: float,
    a: float,
    p: float,
) -> PremiseSemanticScores:
    return PremiseSemanticScores(
        hypothesis_score=h,
        bridge_score=b,
        axis_score=a,
        prediction_score=p,
        core_score=(h + b) / 2.0,
        hypothesis_rank=1,
        bridge_rank=1,
        axis_rank=1,
        prediction_rank=1,
        core_rank=1,
    )


def test_ps1_rank_is_descending_and_deterministic():
    ranks = _rank_descending(
        {
            "b": 0.4,
            "a": 0.4,
            "c": 0.2,
        }
    )
    assert ranks == {
        "a": 1,
        "b": 2,
        "c": 3,
    }


def test_ps1_pareto_domination_requires_no_worse_all_axes():
    selected = _scores(
        0.40,
        0.40,
        0.40,
        0.40,
    )
    candidates = {
        "dominates": _scores(
            0.50,
            0.45,
            0.42,
            0.41,
        ),
        "tradeoff": _scores(
            0.80,
            0.20,
            0.80,
            0.20,
        ),
    }

    result = _pareto_dominators(
        selected,
        candidates,
        epsilon=0.01,
    )

    assert result == ["dominates"]


def test_ps1_pareto_epsilon_avoids_noise_only_dominance():
    selected = _scores(
        0.40,
        0.40,
        0.40,
        0.40,
    )
    candidates = {
        "noise": _scores(
            0.405,
            0.405,
            0.405,
            0.405,
        ),
    }

    result = _pareto_dominators(
        selected,
        candidates,
        epsilon=0.01,
    )

    assert result == []
