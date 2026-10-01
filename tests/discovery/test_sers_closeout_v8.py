
from pipeline_core.discovery.sers_closeout_v8 import build_closeout,build_n9_projection
def test_closeout_and_projection_fail_closed():
    agg={"composites":[
        {"hypothesis_id":"hypothesis:a","claim_id":"a","full_relation_status":"COMPONENTS_ONLY","topology_state":"EXPLICIT","aggregated_component_claim_ids":["x"],"aggregated_relation_backed_component_claim_ids":["x"],"evidence_depth":"METADATA_ONLY","aggregated_residual_state":"HIGHER_ORDER_RESIDUAL_SAME_WORK_KNOWN_BASE","aggregation_disposition":"RESIDUAL_CANDIDATE_SHADOW"},
        {"hypothesis_id":"hypothesis:b","claim_id":"b","full_relation_status":"COMPONENTS_ONLY","topology_state":"NO_COMPONENT_TOPOLOGY","aggregated_component_claim_ids":[],"aggregated_relation_backed_component_claim_ids":[],"evidence_depth":"NOT_APPLICABLE","aggregated_residual_state":"NOT_ASSESSABLE_NO_COMPONENT_TOPOLOGY","aggregation_disposition":"HOLD_FOR_TOPOLOGY"}]}
    ready={"rows":[{"hypothesis_id":"hypothesis:a","state":"AUTHORITY_READY_CANDIDATE_SHADOW","authority_ready_candidate_shadow":True},{"hypothesis_id":"hypothesis:b","state":"NOT_RESIDUAL_CANDIDATE","authority_ready_candidate_shadow":False}]}
    c=build_closeout({"hypotheses":[]},agg,agg,ready,{"pass":True},{"authority_relevant_stable":True},{"records":[]},{"relationship":"DIRECT_PRIOR_ART","aggregation_eligible":False})
    by={r["hypothesis_id"]:r for r in c["hypotheses"]}
    assert by["hypothesis:a"]["final_epistemic_state"]=="RESIDUAL_AUTHORITY_CANDIDATE_SHADOW"
    assert by["hypothesis:b"]["final_epistemic_state"]=="UNRESOLVED_TOPOLOGY_GAP"
    p=build_n9_projection(c)
    assert p["projected_count"]==1 and p["held_count"]==1
    assert p["n9_authority_created"] is False
