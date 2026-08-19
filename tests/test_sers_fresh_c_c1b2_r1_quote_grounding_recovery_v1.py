import copy
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b1_reviewer_contract_v1 import FreshCPaperReview,FreshCFinalAdjudication
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_scientific_adjudication_v1 import openai_strict_transport_schema
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_r1_quote_grounding_recovery_v1 import (
 DEFAULT_PROTOCOL_PATH,paper_recovery_transport_schema,final_recovery_transport_schema,validate_protocol,
)
def test_only_quote_leaf_changes():
 parent=openai_strict_transport_schema(FreshCPaperReview);recovery=paper_recovery_transport_schema()
 q=recovery["$defs"]["EvidenceLocator"]["properties"]["verbatim_quote"];test=copy.deepcopy(parent)
 test["$defs"]["EvidenceLocator"]["properties"]["verbatim_quote"]=q
 assert test==recovery
def test_quote_is_null_only_and_required():
 s=paper_recovery_transport_schema();loc=s["$defs"]["EvidenceLocator"]
 assert "verbatim_quote" in loc["required"];assert loc["properties"]["verbatim_quote"]["type"]=="null"
def test_final_schema_unchanged():
 assert final_recovery_transport_schema()==openai_strict_transport_schema(FreshCFinalAdjudication)
def test_scientific_semantics_unchanged():
 p=validate_protocol(DEFAULT_PROTOCOL_PATH)
 assert p["raw_reviewer_models_changed"] is False
 assert p["scientific_system_prompts_changed"] is False
 assert p["scientific_target_boundaries_changed"] is False
 assert p["relation_label_vocabulary_changed"] is False
 assert p["verdict_lattice_changed"] is False
 assert p["reuse_failed_parent_response_allowed"] is False
def test_recovery_is_not_new_fresh_reserve():
 p=validate_protocol(DEFAULT_PROTOCOL_PATH)
 assert p["recovery_result_may_claim_new_fresh_reserve"] is False
 assert p["same_recovery_epoch_rerun_allowed_after_start"] is False
 assert p["failure_restores_freshness"] is False
 assert p["failure_authorizes_tuning_on_fresh_c"] is False
