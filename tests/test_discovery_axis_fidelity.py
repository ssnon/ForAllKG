import numpy as np
from types import SimpleNamespace

from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxis
from dac_her.discovery_axis_fidelity import DiscoveryAxisFidelityCritic


class KeywordEncoder:
    keys = ("charge", "donation", "adsorption", "volcano", "adjacent", "tafel")

    def encode_query(self, text: str) -> np.ndarray:
        lower = text.lower()
        values = [float(key in lower) for key in self.keys]
        values.append(1.0)
        return np.asarray(values, dtype=np.float32)


def axis() -> DiscoveryAxis:
    return DiscoveryAxis(
        axis_id="axis:1",
        axis_rank=1,
        inspiration_id="insp:1",
        source_path_id="path:1",
        candidate_unit_id="unit:1",
        label="nitrogen-coordination charge-donation correlation",
        entry_anchor_label="Pt2N4",
        exit_anchor_label="PtN4",
        proposed_subject="nitrogen coordination",
        proposed_relation="charge donation",
        proposed_object="paired metal centers",
        rendered_path="Pt2N4 -> charge donation -> PtN4",
        source_mode="exploratory",
        exploration_score=0.55,
        candidate_unit_score=0.52,
        planner_score=0.50,
        mechanistic_continuity_band="high",
    )


def card(*, uses_axis: bool):
    if uses_axis:
        statement = "Nitrogen coordination may change inter-metal charge donation and thereby alter HER response."
        bridge = "The proposed mediator is coordination-dependent charge donation between paired metal centers."
        observable = "charge donation and hydrogen adsorption across the coordination series"
    else:
        statement = "Nitrogen coordination shifts hydrogen adsorption toward a volcano optimum."
        bridge = "Hydrogen adsorption is the dominant mediator of the HER volcano response."
        observable = "hydrogen adsorption free energy"
    return SimpleNamespace(
        hypothesis_id="h:1",
        title="test",
        hypothesis_statement=statement,
        inferential_bridge=bridge,
        predicted_observations=[SimpleNamespace(observable=observable, rationale=bridge)],
        falsification_criteria=[
            SimpleNamespace(observable=observable, falsifying_outcome="no corresponding change")
        ],
    )


def test_axis_fidelity_rejects_decorative_lineage():
    critic = DiscoveryAxisFidelityCritic()
    bad = critic.review(axis(), card(uses_axis=False), KeywordEncoder())
    good = critic.review(axis(), card(uses_axis=True), KeywordEncoder())

    assert bad.status == "fail"
    assert "no_axis_distinctive_terms_used" in bad.reason_codes
    assert good.status in {"pass", "warning"}
    assert "charge" in good.matched_signature_tokens
