"""Read-only, complete validation and work-payload preview for a saved Decision 2.

Checks the real state/event chain and the exact saved model request before any
committing operation. Never edits or normalizes the model response and never
calls a provider. Installed in the repository for regular Python imports.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import research_work_v11_runner as rw
from pipeline_core.discovery.scientific_research_control.research_work_v11_contracts import ResearchDecision


def report(out: Path) -> dict:
    raw_file=out/'calls/decision_0002/raw_response.json'
    result={'status':'BLOCKED_INSPECT_SAVED_RESPONSE','paid_call_allowed':False,
            'raw_response_file_sha256': hashlib.sha256(raw_file.read_bytes()).hexdigest()
                if raw_file.is_file() else None}
    if not raw_file.is_file():
        result['status']='BLOCKED_NO_SAVED_DECISION_2'
        return result
    try:
        c=rw.verify(out)
        s=c['state']
        result['decision_count']=s['decision_count']
        result['pending_work_status']=s['pending_work'] and s['pending_work']['status']
        if s['decision_count']!=1 or not s['pending_work'] or s['pending_work']['status']!='EXECUTED':
            raise ValueError('expected Decision 1 followed by executed Work 1')
        call=raw_file.parent
        request=rw.read(call/'request.json')
        attempt=rw.read(call/'attempt.json')
        if request != rw._make_call_request(c,'decision') or attempt['request_sha256']!=rw.sha(request):
            raise ValueError('saved paid request differs from the replayed prompt/state')
        saved=rw.read(raw_file)
        d=ResearchDecision.model_validate(rw._raw_obj(saved['raw_text']))
        result['action']=d.action
        if d.work_request:
            result['work_kind']=d.work_request.work_kind
            result['source_ids']=d.work_request.source_ids
            result['available_inputs']=d.work_request.available_inputs
            allrefs=list(dict.fromkeys([*d.work_request.available_inputs,*d.work_request.source_ids]))
            result['read_source_ids']=[ref for ref in allrefs if ref in set(s['read_ids'])]
            result['artifact_refs']=[ref for ref in allrefs if ref.endswith(':product')]
        interp=d.interpretation
        prev=s['pending_work']
        # Make exact citation classification visible even if the subsequent
        # full state validation fails. Do not rewrite or auto-accept bad refs.
        if interp is not None:
            refs=list(interp.supported_by_artifact_refs)
            prior_artifact=f"{prev['work_id']}:product"
            read_ids=set(s['read_ids'])
            valid={prior_artifact, *read_ids, *(f'source:{x}' for x in read_ids)}
            result['interpretation_supported_refs']=refs
            result['interpretation_ref_classification']=[
                {'ref':ref, 'classification':(
                    'EXACT_PRIOR_PRODUCT' if ref==prior_artifact else
                    'READ_SOURCE_ID' if ref in read_ids else
                    'READ_SOURCE_TYPED_ALIAS' if ref.startswith('source:') and ref[7:] in read_ids else
                    'NOT_AUTHORIZED')}
                for ref in refs
            ]
            result['unauthorized_interpretation_refs']=[ref for ref in refs if ref not in valid]
        result['prior_work_interpretation_references_match']=bool(
            interp and interp.work_id==prev['work_id']
            and interp.product_sha256==prev['product_sha256']
            and f"{prev['work_id']}:product" in interp.supported_by_artifact_refs)
        # Validates all constraints, including artifact legitimacy, source
        # read-status, target IDs, final-candidate rules and budget.
        future=rw.apply_event(s,'DECISION',d.model_dump(mode='json'),
                              manifest=c['manifest'],sources=c['sources'])
        if d.work_request and d.work_request.work_kind=='REASON':
            preview_c=dict(c)
            preview_c['state']=future
            preview=rw.work_payload(preview_c)
            result['reason_source_ids_supplied']=[r['id'] for r in preview['read_scientific_sources']]
            result['reason_artifact_refs_supplied']=[r['artifact_ref'] for r in preview['read_work_products']]
            result['product_content_supplied']=bool(preview['read_work_products'])
        result['status']='SAVED_DECISION_COMPATIBLE_FOR_OFFLINE_RECOVERY'
    except Exception as e:
        result['blocker']={'type':type(e).__name__,'message':str(e)}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    r=report(a.out)
    print(json.dumps(r,ensure_ascii=False,indent=2))
    if r['status']!='SAVED_DECISION_COMPATIBLE_FOR_OFFLINE_RECOVERY':
        raise SystemExit(2)

if __name__=='__main__':main()
