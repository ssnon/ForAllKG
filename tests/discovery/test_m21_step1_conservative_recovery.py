"""One saved-response recovery only; never pay or alter original research."""
import json
import hashlib
from pathlib import Path
import pytest

from pipeline_core.discovery.scientific_research_control import scientific_execution_work as work
from pipeline_core.discovery.scientific_research_control import open_literature_research as opening_mod
from scripts.discovery import run_m21_step1_conservative_recovery as recovery
from test_open_literature_research import opening, backend_of, scenario
from test_scientific_execution_work import step


def invalid():
    obj=step(op='INVESTIGATE')
    obj['scientific_work_content']['work_conclusion']='DERIVATION_PROPOSED'
    obj['scientific_work_content']['candidate_consequence']=None
    obj['scientific_work_content']['reasoning_or_operations']=[]
    return obj


def run_with_invalid_paid_response(scenario,tmp_path):
    opening_mod.choose(scenario,backend=backend_of(opening()))
    source=tmp_path/'original'
    work.prepare(scenario,source)
    bad=invalid()
    with pytest.raises(RuntimeError,match='RAW_SAVED_USE_RECOVER_NO_RETRY'):
        work.next_step(source,backend=backend_of(bad))
    return source,bad


def test_fork_preserves_original_and_reclassifies_only_one_field(scenario,tmp_path):
    source,bad=run_with_invalid_paid_response(scenario,tmp_path)
    before={p.name:recovery.h(p) for p in (source/'step_0001').iterdir()}
    assert recovery.inspect(source)['status']=='M21_STEP1_RECOVERY_ELIGIBLE_NO_CALL'
    fork=tmp_path/'separate_fork'
    out=recovery.fork(source,fork)
    assert out['steps']==1 and out['epistemic_downgrade']=='DERIVATION_PROPOSED -> INCONCLUSIVE'
    assert out['tool_status'] is None
    assert {p.name:recovery.h(p) for p in (source/'step_0001').iterdir()}==before
    assert work.verify(source)['incomplete']
    norm=work.read(fork/'step_0001'/'normalized.json')
    expected=json.loads(json.dumps(bad))
    expected['scientific_work_content']['work_conclusion']='INCONCLUSIVE'
    assert norm==work.Step.model_validate(expected).model_dump(mode='json')
    with pytest.raises(ValueError,match='INVALID_SAVED_SCIENCE_JSON'):
        work.verify(fork)
    assert recovery.run_compatible('verify',fork)['steps']==1
    assert recovery.run_compatible('preview',fork)['step']==2


def test_next_step_remains_strict_and_paid_raw_still_protected(scenario,tmp_path):
    source,_=run_with_invalid_paid_response(scenario,tmp_path)
    fork=tmp_path/'fork';recovery.fork(source,fork)
    with recovery.recovery_scope(recovery.check_audit(fork)['saved_raw_text_sha256']):
        result=work.next_step(fork,backend=backend_of(step(action='RETURN',op='REVISE')))
    assert result['next_action']=='RETURN'
    assert recovery.run_compatible('verify',fork)['steps']==2
    assert recovery.run_compatible('finish',fork)['status']=='SCIENTIFIC_EXECUTION_FROZEN_NO_CALL'
    assert recovery.run_compatible('verify',fork)['finalized']
    # Scope is SHA-specific: must not normalize another future invalid output.
    raw=json.dumps(invalid()).replace('DERIVATION_PROPOSED','COUNTEREXAMPLE_PROPOSED')
    with recovery.recovery_scope(recovery.check_audit(fork)['saved_raw_text_sha256']):
        with pytest.raises(ValueError):work.parse(raw)


def test_rejects_other_invalid_scientific_content_and_detects_tampering(scenario,tmp_path):
    source,_=run_with_invalid_paid_response(scenario,tmp_path)
    fork=tmp_path/'fork';recovery.fork(source,fork)
    audit=recovery.check_audit(fork)
    assert audit['paid_calls_for_recovery']==0
    with pytest.raises(FileExistsError):recovery.fork(source,fork)
    (fork/'step_0001'/'normalized.json').write_text('{}')
    with pytest.raises(ValueError,match='RECOVERY_NORMALIZATION_CHANGED'):
        recovery.run_compatible('verify',fork)


def test_strict_diagnosis_refuses_mismatched_errors():
    good=step()
    with pytest.raises(ValueError,match='RAW_IS_STRICTLY_VALID'):recovery.strict_diagnosis(json.dumps(good))
    bad=invalid();bad['scientific_work_content']['reasoning_or_operations']=['derivation argument is meaningful'];bad['scientific_work_content']['candidate_consequence']='A sufficiently concrete consequence'
    with pytest.raises(ValueError,match='RAW_IS_STRICTLY_VALID'):recovery.strict_diagnosis(json.dumps(bad))
    bad=invalid();bad['operation']='UNKNOWN'
    with pytest.raises(ValueError):recovery.strict_diagnosis(json.dumps(bad))
    with pytest.raises(ValueError):recovery.strict_diagnosis('not JSON')
