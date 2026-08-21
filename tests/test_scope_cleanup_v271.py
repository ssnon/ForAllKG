from __future__ import annotations

from pipeline_core.discovery.feasibility.feasibility_contracts import FeasibilityHypothesis, FeasibilityIntake
from pipeline_core.discovery.feasibility.scope_compiler import HypothesisScopeCompiler
from pipeline_core.runtime.validation_specification import ValidationSpecificationCompiler


def _intake(*hypotheses: FeasibilityHypothesis) -> FeasibilityIntake:
    return FeasibilityIntake(
        intake_id="intake:v271",
        intake_sha256="isha",
        source_context_id="ctx",
        source_context_sha256="csha",
        source_portfolio_id="portfolio",
        source_portfolio_sha256="psha",
        source_semantic_review_id="review",
        task_id="task",
        question="q",
        corpus_id="corpus",
        hypotheses=list(hypotheses),
    )


def _hypothesis(
    hypothesis_id: str,
    *,
    title: str,
    statement: str,
    hypothesis_type: str = "mechanistic_extension",
) -> FeasibilityHypothesis:
    return FeasibilityHypothesis(
        hypothesis_id=hypothesis_id,
        title=title,
        statement=statement,
        hypothesis_type=hypothesis_type,
        inferential_bridge="Explicit test target.",
        source_paper_ids=["fixture"],
        candidate_dependency="none",
        semantic_gate_status="eligible",
    )


def test_explicit_fe_ru_dac_is_reachable_as_candidate_specific():
    h = _hypothesis(
        "h:feru",
        title="Fe-Ru DAC candidate",
        statement=(
            "A Fe-Ru dual-atom catalyst on N-doped graphene may improve HER "
            "through coupled hydrogen adsorption and charge transfer."
        ),
    )
    scope = HypothesisScopeCompiler().compile(h)
    spec = ValidationSpecificationCompiler().compile(h, scope)

    assert scope.catalyst_class == "dual_atom"
    assert scope.hypothesis_level == "candidate_specific"
    assert scope.metals == ["Fe", "Ru"]
    assert scope.requires_candidate_concretization is False
    assert spec.validation_strategy == "candidate_specific_computation"


def test_explicit_pt_sac_is_reachable_and_single_metal_is_extracted():
    h = _hypothesis(
        "h:pt-sac",
        title="Pt single-atom candidate",
        statement=(
            "A Pt single-atom site on N4 carbon may alter hydrogen adsorption "
            "and HER activity."
        ),
    )
    scope = HypothesisScopeCompiler().compile(h)
    spec = ValidationSpecificationCompiler().compile(h, scope)

    assert scope.catalyst_class == "single_atom"
    assert scope.hypothesis_level == "candidate_specific"
    assert scope.metals == ["Pt"]
    assert scope.requires_candidate_concretization is False
    assert "isolated_site_stability" in spec.required_physics_checks
    assert "pair_stability" in spec.not_applicable_physics_checks


def test_full_element_names_are_normalized_to_symbols():
    h = _hypothesis(
        "h:named-metals",
        title="Platinum-ruthenium DAC",
        statement=(
            "A platinum-ruthenium dual-atom catalyst may modify HER kinetics."
        ),
    )
    scope = HypothesisScopeCompiler().compile(h)

    assert scope.metals == ["Pt", "Ru"]
    assert scope.hypothesis_level == "candidate_specific"


def test_explicit_metals_do_not_override_comparative_or_family_scope():
    family = _hypothesis(
        "h:family-explicit",
        title="Fe-Ru coordination series",
        statement=(
            "Within Fe-Ru dual-atom catalysts, HER activity may vary non-monotonically "
            "with nitrogen coordination number and local geometry."
        ),
    )
    comparative = _hypothesis(
        "h:comparative-explicit",
        title="Fe-Ru geometry comparison",
        statement=(
            "At a given nitrogen coordination number in Fe-Ru dual-atom catalysts, "
            "different local geometries may alter HER activity."
        ),
        hypothesis_type="descriptor_mediation",
    )
    scopes = {
        row.hypothesis_id: row
        for row in HypothesisScopeCompiler().compile_intake(_intake(family, comparative))
    }

    assert scopes["h:family-explicit"].hypothesis_level == "material_family"
    assert scopes["h:comparative-explicit"].hypothesis_level == "comparative_study"
