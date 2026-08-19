from __future__ import annotations

import networkx as nx

from domains.sers.trend_precision import SERS_AU_AG_TREND_PRECISION_ADAPTER
from dac_her.trend_precision import audit_trend_precision


def _claim(tid, cid, sid):
    return {"trend_id":tid,"domain_profile_id":"sers_au_ag","trend_semantics_id":"sers_au_ag_trend_v2_alpha4c21","paper_id":"P1","independent_variable_key":"nanogap_size","dependent_observable_key":"sers_enhancement_factor","direction":"negative","shape":"monotonic","evidence_basis":"reported_directional_claim","source_expression":"The SERS enhancement factor increases as the interior gap size decreases.","source_expressions":[],"source_claim_ids":[cid],"source_measurement_ids":[],"source_measurement_result_ids":[],"source_calculation_ids":[],"source_node_ids":[cid],"subject_ids":[sid]}


def test_duplicate_claim_mentions_collapse_to_one_local_result():
    g=nx.MultiDiGraph()
    g.add_node("c1",type="ObservationClaim"); g.add_node("c2",type="ObservationClaim")
    g.add_node("substrate_double_shelled_au_ag_nanobox",type="PlasmonicSubstrate",label="double-shelled Au/Ag nanobox")
    g.add_node("substrate_double_shelled_au_ag_nanoboxes",type="PlasmonicSubstrate",label="double-shelled Au/Ag nanoboxes")
    rows=[_claim("t1","c1","substrate_double_shelled_au_ag_nanobox"),_claim("t2","c2","substrate_double_shelled_au_ag_nanoboxes")]
    anns=[SERS_AU_AG_TREND_PRECISION_ADAPTER.annotate(r,g) for r in rows]
    results=SERS_AU_AG_TREND_PRECISION_ADAPTER.consolidate(rows,anns,{"P1":g})
    assert len(results)==1 and results[0].support_mention_count==2
    audit=audit_trend_precision(evidence_rows=rows,annotations=anns,results=results,adapter=SERS_AU_AG_TREND_PRECISION_ADAPTER)
    assert audit.structural_gate and audit.duplicate_claim_mentions_collapsed==1


def test_dda_numeric_is_calculated_and_ratio_landmark_preserves_orientation():
    g=nx.MultiDiGraph(); g.add_node("group_dda_gap_comparison",type="MeasurementGroup",label="DDA gap comparison"); g.add_node("m2",type="Measurement"); g.add_node("m8",type="Measurement")
    numeric={"trend_id":"tn","domain_profile_id":"sers_au_ag","trend_semantics_id":"sers_au_ag_trend_v2_alpha4c21","paper_id":"P1","independent_variable_key":"nanogap_size","dependent_observable_key":"sers_enhancement_factor","direction":"negative","shape":"monotonic","evidence_basis":"controlled_numeric_pair","source_expression":"corresponding to an EF of 1.3 x 10^8","source_expressions":[],"source_claim_ids":[],"source_measurement_ids":["m2","m8"],"source_measurement_result_ids":["r2","r8"],"source_calculation_ids":[],"source_node_ids":["m2","m8","group_dda_gap_comparison"],"subject_ids":[]}
    ann=SERS_AU_AG_TREND_PRECISION_ADAPTER.annotate(numeric,g)
    assert ann.evidence_kind=="calculated_numeric"
    assert ann.classification_basis=="explicit_calculation_lineage_text"
    g.add_node("c",type="ObservationClaim")
    ratio={"trend_id":"tr","domain_profile_id":"sers_au_ag","trend_semantics_id":"sers_au_ag_trend_v2_alpha4c21","paper_id":"P1","independent_variable_key":"ag_to_au_ratio","dependent_observable_key":"raman_intensity","direction":"non_monotonic","shape":"single_optimum","evidence_basis":"reported_directional_claim","source_expression":"Among the tested Au-Ag ratios, the 10:7 bimetallic nanoparticle substrate produced the strongest SERRS signal.","source_expressions":[],"source_claim_ids":["c"],"source_measurement_ids":[],"source_measurement_result_ids":[],"source_calculation_ids":[],"source_node_ids":["c"],"subject_ids":[]}
    rann=SERS_AU_AG_TREND_PRECISION_ADAPTER.annotate(ratio,g)
    assert abs(rann.canonical_control_value_numeric-0.7)<1e-12
    assert rann.source_control_value_text=="10:7"
    assert rann.normalization_transform=="au_ag_to_ag_over_au"


def test_numeric_and_claim_lanes_never_merge():
    g=nx.MultiDiGraph(); g.add_node("c",type="ObservationClaim"); g.add_node("m",type="Measurement")
    claim=_claim("tc","c",""); claim["subject_ids"]=[]
    numeric={"trend_id":"tn","domain_profile_id":"sers_au_ag","trend_semantics_id":"sers_au_ag_trend_v2_alpha4c21","paper_id":"P1","independent_variable_key":"nanogap_size","dependent_observable_key":"sers_enhancement_factor","direction":"negative","shape":"monotonic","evidence_basis":"controlled_numeric_pair","source_expression":"","source_expressions":[],"source_claim_ids":[],"source_measurement_ids":["m"],"source_measurement_result_ids":["r"],"source_calculation_ids":[],"source_node_ids":["m"],"subject_ids":[]}
    rows=[claim,numeric]; anns=[SERS_AU_AG_TREND_PRECISION_ADAPTER.annotate(r,g) for r in rows]
    results=SERS_AU_AG_TREND_PRECISION_ADAPTER.consolidate(rows,anns,{"P1":g})
    assert {x.result_lane for x in results}=={"claim","numeric"}
