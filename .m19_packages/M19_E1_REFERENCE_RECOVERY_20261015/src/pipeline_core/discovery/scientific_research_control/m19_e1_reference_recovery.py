"""M19-E1 v1.0 additive, offline citation quarantine/recovery overlay.

Does not change the hash-pinned original runner, original paid raw, original prompts,
or scientific text. Rejects all unapproved references. Quarantine is NOT citation
validation and cannot upgrade an ungrounded source claim into an accepted fact.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from . import m19_deep_work_e1 as legacy

QUARANTINE_SCHEMA = 'm19-e1-invalid-reference-quarantine-v1'
SIDECAR = 'reference_quarantine.json'


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def _step_dir_checked(out: Path, arm: str, step: int) -> Path:
    legacy._safe_repo_root(out)
    if arm not in legacy.ARMS or not (1 <= step <= legacy.MAX_PER_ARM):
        raise ValueError('INVALID_ARM_OR_STEP')
    sd = legacy._step_dir(out, arm, step)
    if not (sd/'raw_response.json').is_file():
        raise FileNotFoundError('NO_SAVED_RAW_RESPONSE: '+str(sd))
    return sd


def _read_output(out: Path, sd: Path):
    raw = legacy.base.read(sd/'raw_response.json')
    raw_text = raw['raw_text']
    obj = legacy.StepOutput.model_validate(legacy.base._raw_obj(raw_text)).model_dump(mode='json')
    allowed = legacy._source_references(legacy.base.read(out/'frozen_input.json'))
    original_refs = obj['source_or_prior_work_refs']
    unknown = sorted(set(original_refs)-allowed)
    retained = [r for r in original_refs if r in allowed]
    return raw_text, obj, unknown, retained, allowed


def _content_occurrences(obj: dict, invalid: list[str]) -> dict[str, int]:
    # Check scientific prose separately from the metadata list. If it mentions
    # a quarantined source, that is a scientific limitation, not a repair.
    content = json.dumps({'research_delta':obj['research_delta'],
                          'provisional_work_content':obj['provisional_work_content'],
                          'reason_for_choice':obj['reason_for_choice'],
                          'next_question_or_test':obj['next_question_or_test']},ensure_ascii=False)
    return {ref: content.count(ref) for ref in invalid}


def audit(out: Path, arm: str, step: int) -> dict:
    sd = _step_dir_checked(out,arm,step)
    raw, obj, invalid, retained, allowed = _read_output(out,sd)
    q=sd/SIDECAR
    return {'status':'OFFLINE_RAW_REFERENCE_AUDIT','arm':arm,'step':step,
        'raw_text_sha256':_digest(raw),'original_model_refs':obj['source_or_prior_work_refs'],
        'invalid_refs':invalid,'retained_authorized_refs':retained,
        'invalid_refs_mentioned_in_science_prose':_content_occurrences(obj,invalid),
        'quarantine_sidecar_exists':q.exists(),'raw_preserved':True,
        'warning':'The original science text is NOT corrected. Review invalid citations as possible hallucinations; no claims are certified.',
        'allowed_ref_count':len(allowed),'paid_calls':0}


def _validate_quarantine(out:Path,sd:Path, *, expected_raw:str | None =None)->dict:
    p=sd/SIDECAR
    q=legacy.base.read(p)
    raw_text,obj,invalid,retained,allowed=_read_output(out,sd)
    if expected_raw is not None and raw_text!=expected_raw:
        raise ValueError('QUARANTINE_RAW_DOES_NOT_MATCH_INVOKING_PARSE')
    if set(q) != {'schema','arm','step','raw_text_sha256','request_file_sha256',
                  'attempt_file_sha256','overlay_source_sha256','original_model_refs','invalid_refs',
                  'retained_authorized_refs','science_prose_invalid_ref_mentions',
                  'disposition','scientific_validation'}:
        raise ValueError('QUARANTINE_WRONG_SCHEMA_FIELDS')
    expected={'schema':QUARANTINE_SCHEMA,'arm':sd.parent.parent.name,
        'step':int(sd.name.split('_')[-1]),'raw_text_sha256':_digest(raw_text),
        'request_file_sha256':legacy.base.file_sha(sd/'request.json'),
        'attempt_file_sha256':legacy.base.file_sha(sd/'attempt.json'),
        'overlay_source_sha256':legacy.base.file_sha(Path(__file__)),
        'original_model_refs':obj['source_or_prior_work_refs'],
        'invalid_refs':invalid,'retained_authorized_refs':retained,
        'science_prose_invalid_ref_mentions':_content_occurrences(obj,invalid),
        'disposition':'QUARANTINED_UNAUTHORIZED_REFERENCE_NOT_EVIDENCE',
        'scientific_validation':False}
    if q!=expected:raise ValueError('QUARANTINE_CONTENT_OR_SOURCE_CHANGED')
    if not invalid:raise ValueError('QUARANTINE_WITH_NO_INVALID_REFS')
    if any(ref not in allowed for ref in retained):raise ValueError('QUARANTINE_ALLOWED_REF_MISMATCH')
    return q


def quarantine(out:Path,arm:str,step:int,*,accept_invalid_refs:bool=False)->dict:
    if not accept_invalid_refs:raise PermissionError('EXPLICIT_REFERENCE_QUARANTINE_APPROVAL_REQUIRED')
    sd=_step_dir_checked(out,arm,step)
    if (sd/'normalized.json').exists() or (sd/'record.json').exists():
        raise ValueError('REFUSE_SALVAGE_AFTER_RECORD_COMMIT')
    raw_text,obj,invalid,retained,allowed=_read_output(out,sd)
    if not invalid:raise ValueError('NO_UNAUTHORIZED_REFERENCES_TO_QUARANTINE')
    q={'schema':QUARANTINE_SCHEMA,'arm':arm,'step':step,
        'raw_text_sha256':_digest(raw_text),
        'request_file_sha256':legacy.base.file_sha(sd/'request.json'),
        'attempt_file_sha256':legacy.base.file_sha(sd/'attempt.json'),
        'overlay_source_sha256':legacy.base.file_sha(Path(__file__)),
        'original_model_refs':obj['source_or_prior_work_refs'],
        'invalid_refs':invalid,'retained_authorized_refs':retained,
        'science_prose_invalid_ref_mentions':_content_occurrences(obj,invalid),
        'disposition':'QUARANTINED_UNAUTHORIZED_REFERENCE_NOT_EVIDENCE',
        'scientific_validation':False}
    legacy.base.write_new(sd/SIDECAR,q) # refuse overwrite
    _validate_quarantine(out,sd)
    return {'status':'REFERENCE_QUARANTINED_OFFLINE_NO_CALL','arm':arm,
        'step':step,'invalid_refs':invalid,'retained_refs':retained,
        'scientific_text_unchanged':True,'saved_raw_unchanged':True,'paid_calls':0}


def _matching_quarantine(out:Path,raw_text:str):
    matches=[]
    for arm in legacy.ARMS:
        for step in range(1,legacy.MAX_PER_ARM+1):
            sd=legacy._step_dir(out,arm,step)
            p=sd/SIDECAR
            if p.exists():
                q=legacy.base.read(p)
                if q.get('raw_text_sha256')==_digest(raw_text):
                    _validate_quarantine(out,sd,expected_raw=raw_text)
                    matches.append(q)
    if len(matches)>1:raise ValueError('AMBIGUOUS_QUARANTINE_FOR_RAW')
    return matches[0] if matches else None


def _parse_with_quarantine(out:Path,raw_text:str)->dict:
    obj=legacy.StepOutput.model_validate(legacy.base._raw_obj(raw_text)).model_dump(mode='json')
    allowed=legacy._source_references(legacy.base.read(out/'frozen_input.json'))
    invalid=sorted(set(obj['source_or_prior_work_refs'])-allowed)
    if not invalid:return obj
    q=_matching_quarantine(out,raw_text)
    if q is None:
        raise ValueError('UNKNOWN_SOURCE_REF_UNQUARANTINED: '+str(invalid))
    if q['invalid_refs']!=invalid:raise ValueError('QUARANTINE_INVALID_REF_LIST_MISMATCH')
    obj['source_or_prior_work_refs']=q['retained_authorized_refs']
    return obj


# Original implementation/source-hash stays untouched. Only users of this opt-in
# entrypoint install the stricter quarantine-aware parser for this interpreter.
legacy._parse=_parse_with_quarantine


def verify(out:Path)->dict:
    # Validate ALL existing sidecars, even from incomplete steps, before replay.
    for arm in legacy.ARMS:
        for step in range(1,legacy.MAX_PER_ARM+1):
            sd=legacy._step_dir(out,arm,step)
            if (sd/SIDECAR).exists():_validate_quarantine(out,sd)
    return legacy.verify(out)


def inspect(out:Path)->dict:
    v=verify(out)
    q=[]
    for arm in legacy.ARMS:
        for step in range(1,legacy.MAX_PER_ARM+1):
            sd=legacy._step_dir(out,arm,step)
            if (sd/SIDECAR).is_file():
                side=_validate_quarantine(out,sd)
                q.append({'arm':arm,'step':step,'invalid_refs':side['invalid_refs'],
                          'science_prose_invalid_ref_mentions':side['science_prose_invalid_ref_mentions'],
                          'sidecar_file_sha256':legacy.base.file_sha(sd/SIDECAR)})
    result=legacy.inspect(out)
    result['reference_quarantine']=q
    result['reference_warning']='Quarantined source IDs are NOT evidence; inspect original scientific claims.' if q else None
    return result


def recover(out:Path)->dict:
    verify(out)
    rec=legacy.recover(out)
    verify(out)
    return rec


def preview(out:Path,arm:str)->dict:
    verify(out)
    return legacy.preview(out,arm)


def next_step(out:Path,arm:str,*,allow_paid:bool=False,max_total_paid_attempts:int=6)->dict:
    verify(out)
    return legacy.next_step(out,arm,allow_paid=allow_paid,
                            max_total_paid_attempts=max_total_paid_attempts)


def finalize(out:Path,arm:str)->dict:
    verify(out)
    return legacy.finalize(out,arm)
