from pipeline_core.discovery.external_novelty_llm import _DECOMPOSE_SYSTEM
from scripts.discovery.run_external_novelty import parse_args


def test_decomposition_prompt_has_strict_composite_shape_invariant():
    assert "STRICT STRUCTURED-SHAPE INVARIANT" in _DECOMPOSE_SYSTEM
    assert (
        "If either higher_order_relation_basis OR "
        "higher_order_component_local_ids is non-empty, kind MUST be composite."
        in _DECOMPOSE_SYSTEM
    )
    assert (
        "For EVERY non-composite claim, higher_order_relation_basis=[] "
        "AND higher_order_component_local_ids=[]."
        in _DECOMPOSE_SYSTEM
    )


def test_external_novelty_cli_exposes_structured_parse_retries(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_external_novelty",
            "--portfolio", "p.json",
            "--domain-profile", "sers_au_ag",
            "--model", "m",
            "--output-prefix", "out",
        ],
    )
    args = parse_args()
    assert args.parse_retries == 3
