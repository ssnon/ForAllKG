from pathlib import Path


def test_full_shadow_uses_evidence_relationship_compiler():
    text = Path(
        "scripts/discovery/"
        "run_nonobviousness_full_shadow.py"
    ).read_text()

    assert (
        "review_and_compile_closure_relationships"
        in text
    )

    assert (
        "relationship_outcome.compiled.bridge_kind"
        in text
    )

    assert (
        "relationship_outcome.compiled.scope_compatible"
        in text
    )

    # The old hypothesis-side placeholders must no longer control
    # closure compilation in the production-shadow runner.
    closure_call = text.split(
        "compile_nonobviousness_evidence_closure(",
        1,
    )[1].split(
        ")",
        1,
    )[0]

    assert "inputs.bridge_kind" not in closure_call
    assert "inputs.scope_compatible" not in closure_call
