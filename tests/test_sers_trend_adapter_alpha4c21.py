from __future__ import annotations

import json
import networkx as nx

from dac_her.domains.sers_au_ag_trend_alpha4c21 import (
    SERS_AU_AG_TREND_ADAPTER,
    SERS_AU_AG_TREND_SEMANTICS_ID,
)
from dac_her.trend_domain import TrendEvidenceSource


def _dimension(name: str, value: str):
    return {"name":name,"status":"known","normalized_value":value,
            "source_values":[value],"source_node_ids":["e1"],
            "provenance_scopes":["experiment_conditions_json"]}


def _method(mid: str, *, concentration: str = "", reporter: str = "r6g"):
    dims=[_dimension("analyte","r6g"),_dimension("reporter",reporter),_dimension("excitation_wavelength","633 nm")]
    if concentration:
        dims.append(_dimension("analyte_concentration", concentration))
    else:
        dims.append({"name":"analyte_concentration","status":"unknown","normalized_value":"","source_values":[],"source_node_ids":[],"provenance_scopes":[]})
    for name in ("laser_power","integration_time","sample_preparation","preparation_medium","measurement_environment","sample_state","substrate_condition"):
        dims.append({"name":name,"status":"unknown","normalized_value":"","source_values":[],"source_node_ids":[],"provenance_scopes":[]})
    return {"method_context_id":f"method:{mid}","paper_id":"P1","measurement_id":mid,"producer_ids":["e1"],"subject_ids":["s1"],"dimensions":dims,"source_node_ids":[mid,"e1"]}


def _context(mid: str, value: float, observable="raman_intensity"):
    return {"context_id":f"ctx:{mid}","paper_id":"P1","measurement_id":mid,
            "observable_key":observable,"observable_label":observable,"value_numeric":value,
            "value_text":"","unit":"" if observable=="sers_enhancement_factor" else "a.u.",
            "source_expression":f"{observable} {value}","subject_ids":["s1"],
            "source_node_ids":[mid,"g1","e1"],"method_context_id":f"method:{mid}"}


def test_varied_analyte_concentration_is_not_a_method_mismatch():
    g=nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    g.add_node("g1",type="MeasurementGroup"); g.add_node("e1",type="Experiment"); g.add_node("s1",type="PlasmonicSubstrate")
    identities=[]; methods=[]; contexts=[]
    for i,(c,y) in enumerate([(1e-8,1.0),(1e-7,2.0),(1e-6,3.0)],1):
        mid=f"m{i}"; g.add_node(mid,type="Measurement",conditions_json=json.dumps([{"name":"ATP concentration","value_numeric":c,"unit":"M"}]))
        g.add_edge("e1",mid,relation="HAS_MEASUREMENT"); g.add_edge(mid,"g1",relation="IN_MEASUREMENT_GROUP")
        identities.append({"identity_id":f"id:{mid}","representative_measurement_id":mid,"source_mention_ids":[mid]})
        methods.append(_method(mid, concentration=f"{c} M")); contexts.append(_context(mid,y))
    src=TrendEvidenceSource(graph=g,paper_id="P1",measurement_result_rows=tuple(identities),method_context_rows=tuple(methods),comparison_context_rows=tuple(contexts))
    ev=SERS_AU_AG_TREND_ADAPTER.extract_evidence(src)
    trend=next(x for x in ev if x.is_quantitative and x.independent_variable_key=="analyte_concentration")
    assert trend.direction=="positive" and len(trend.series_points)==3


def test_dda_pair_retains_calculation_provenance():
    g=nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    g.add_node("g1",type="MeasurementGroup",label="DDA gap comparison"); g.add_node("e1",type="Experiment"); g.add_node("calc1",type="Calculation",label="DDA calculation"); g.add_node("s1",type="PlasmonicSubstrate")
    g.add_edge("g1","calc1",relation="SIMULATED_BY")
    identities=[]; methods=[]; contexts=[]
    for mid,gap,ef in (("m2",2.0,1.3e8),("m8",8.0,4.8e6)):
        g.add_node(mid,type="Measurement",conditions_json=json.dumps([{"name":"interior gap size","value_numeric":gap,"unit":"nm"}]))
        g.add_edge("e1",mid,relation="HAS_MEASUREMENT"); g.add_edge(mid,"g1",relation="IN_MEASUREMENT_GROUP")
        identities.append({"identity_id":f"id:{mid}","representative_measurement_id":mid,"source_mention_ids":[mid]})
        methods.append(_method(mid)); contexts.append(_context(mid,ef,"sers_enhancement_factor"))
    src=TrendEvidenceSource(graph=g,paper_id="P1",measurement_result_rows=tuple(identities),method_context_rows=tuple(methods),comparison_context_rows=tuple(contexts))
    trend=next(x for x in SERS_AU_AG_TREND_ADAPTER.extract_evidence(src) if x.is_quantitative)
    assert trend.source_calculation_ids == ("calc1",)
    assert "calc1" in trend.source_node_ids


def test_relative_5_8_detail_does_not_promote_directional_intensity_to_formal_ef():
    g=nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    g.add_node("c",type="ObservationClaim",statement=("SERS intensity increased with Ag shell thickness from 3.6 to 8.4 nm and reached an approximately optimal value at 8.4 nm; increasing the shell to 10.0 nm produced essentially the same enhancement factor of 5.8 relative to Au nanocubes."))
    ev=SERS_AU_AG_TREND_ADAPTER.extract_evidence(TrendEvidenceSource(graph=g,paper_id="P1",measurement_result_rows=({"dummy":1},),method_context_rows=({"dummy":1},),comparison_context_rows=({"dummy":1},)))
    assert len(ev)==1
    assert ev[0].dependent_observable_key=="raman_intensity"
    assert ev[0].shape=="saturating"


def test_expanded_control_claims_and_semantics_id():
    g=nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    g.add_node("a",type="ObservationClaim",statement="Raman intensity increased as ATP concentration increased.")
    g.add_node("b",type="ObservationClaim",statement="There was a highly linear relationship between nanoparticle concentration and SERS signal intensity.")
    g.add_node("c",type="ObservationClaim",statement="The SERS signal intensity increases with increasing Au content.")
    ev=SERS_AU_AG_TREND_ADAPTER.extract_evidence(TrendEvidenceSource(graph=g,paper_id="P1",measurement_result_rows=({"dummy":1},),method_context_rows=({"dummy":1},),comparison_context_rows=({"dummy":1},)))
    by={x.independent_variable_key:x for x in ev}
    assert by["analyte_concentration"].direction=="positive"
    assert by["particle_concentration"].evidence_basis=="reported_correlation"
    assert by["particle_concentration"].direction=="unspecified"
    assert by["au_content"].direction=="positive"
    assert SERS_AU_AG_TREND_SEMANTICS_ID=="sers_au_ag_trend_v2_alpha4c21"
