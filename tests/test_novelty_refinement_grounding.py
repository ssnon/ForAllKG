from types import SimpleNamespace

from dac_her.novelty_refinement_runtime import TargetedNoveltyRefinementRuntime


def test_grounding_preservation_requires_exact_lineage_and_type():
    original = SimpleNamespace(
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        hypothesis_type="mechanistic_extension",
    )
    same = SimpleNamespace(
        premise_statement_ids=["p2", "p1"],
        gap_statement_ids=["g1"],
        hypothesis_type="mechanistic_extension",
    )
    drift = SimpleNamespace(
        premise_statement_ids=["p1", "p3"],
        gap_statement_ids=["g1"],
        hypothesis_type="mechanistic_extension",
    )
    assert TargetedNoveltyRefinementRuntime._grounding_preserved(original, same)
    assert not TargetedNoveltyRefinementRuntime._grounding_preserved(original, drift)
