"""Read-only scientific discovery evidence audit; zero model/solver/network calls.

Not a novelty detector. It prevents prior evidence from being promoted to new
scientific discoveries merely because a text quality score improved.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

SCHEMA='for_all_kg_discovery_capability_retrospective_v1'
STATUS='EXPLORATORY_INFRASTRUCTURE_READY_DISCOVERY_UNCERTIFIED'


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_snapshot(path:Path)->dict:
    obj=json.loads(path.read_text(encoding='utf-8'))
    if obj.get('schema')!=SCHEMA:raise ValueError('SNAPSHOT_SCHEMA_MISMATCH')
    studies=obj.get('studies')
    if not isinstance(studies,list) or len(studies)!=6:raise ValueError('SIX_STUDIES_REQUIRED')
    by={s['study']:s for s in studies}
    if set(by)!={'M13','M14','M14C4','M15_R9','M15_R10','M16'}:raise ValueError('STUDY_IDS_MISMATCH')
    if by['M13']['budget_parity_certified'] is not False or by['M14']['budget_parity_certified'] is not False:
        raise ValueError('UNSUPPORTED_BUDGET_PARITY_CLAIM')
    if by['M14C4']['admitted_problem_count']!=0:raise ValueError('M14C4_NOT_ABSTAIN')
    if any(s['novelty_certified'] is not False for s in studies):raise ValueError('UNSUPPORTED_NOVELTY_CLAIM')
    if by['M16']['numeric_over_known']!=0:raise ValueError('M16_DELTA_NOT_ZERO')
    m16=by['M16']['arm_means']
    if set(m16)!={'BASELINE','KNOWN_PHYSICS','NUMERIC_PHENOMENON'}:raise ValueError('M16_ARMS_CHANGED')
    if m16['KNOWN_PHYSICS']!=m16['NUMERIC_PHENOMENON']:raise ValueError('M16_MEANS_CHANGED')
    if len({p['problem_id'] for p in obj['candidate_questions']})!=len(obj['candidate_questions']):raise ValueError('REPEATED_PROBLEM_ID')
    if any(p['claim_status']!='UNVERIFIED_PRIOR_ART_GAP' for p in obj['candidate_questions']):
        raise ValueError('UNSUPPORTED_PRIOR_ART_CERTIFICATION')
    return obj


def verify_sources(observations:dict, source_dir:Path|None)->dict:
    result={}
    for item in observations['source_file_index']:
        file=item['file']
        if source_dir is None or not (source_dir/file).is_file():
            result[file]='NOT_PRESENT_FOR_INDEPENDENT_HASH_CHECK'
        else:
            result[file]='HASH_MATCH' if sha(source_dir/file)==item['sha256'] else 'HASH_MISMATCH'
    return result


def nonidentifiability_counterexample()->dict:
    """Synthetic multiplicative gauge freedom, not an Au/Ag empirical result.

    Observed band intensities: I_b = N * F_b * T_b * C_b. If only I_b is
    observed, T_b and C_b can be reassigned without changing predictions.
    """
    a={'N':3,'F':[2,2],'T':[5,4],'C':[1,1]}
    b={'N':3,'F':[2,2],'T':[1,1],'C':[5,4]}
    def intensities(z):return [z['N']*z['F'][i]*z['T'][i]*z['C'][i] for i in range(2)]
    assert a!=b and intensities(a)==intensities(b)==[30,24]
    return {'toy_only':True,'equation':'I_b=N*F_b*T_b*C_b','parameters_A':a,'parameters_B':b,'same_observations':intensities(a),'conclusion':'intensities alone cannot uniquely identify tensor and chemical factors without constraints','scientific_novelty_certified':False}


def markdown_report(o:dict,src:dict,claim:dict)->str:
    by={s['study']:s for s in o['studies']}
    nums=by['M16']['arm_means']
    lines=[
        '# ForAllKG — Discovery Capability Evidence Audit',
        '',
        '**판정: `'+STATUS+'`**',
        '',
        '본 문서는 M13~M16에서 **실제로 확인된 증거와 미확인 주장**을 분리한다. 새로운 가설 생성이나 외부 문헌 검증은 실행하지 않았다.',
        '',
        '## 고정된 증거',
        '',
        '| 실험 | 실제 기록 | 주장할 수 없는 것 |',
        '|---|---|---|',
        '| M13 | 후보 '+str(by['M13']['n_candidates'])+'개; 실사용 모델 호출수 불균형 | feedback의 공정한 우월성, 신규성 |',
        '| M14 | 후보 '+str(by['M14']['n_candidates'])+'개; 저장된 응답은 대조되나 실제 유료 호출 동등성 불인증 | adaptive 전략의 인과적 우월성 |',
        '| M14C4 | 경험적 연구 문제 인정 '+str(by['M14C4']['admitted_problem_count'])+'건, ABSTAIN | 독립 검증된 발견 사례 |',
        '| M15-R9 | 진단 피드백의 주관 심사 평균 차이 +'+str(by['M15_R9']['subjective_mean_delta'])+'/10 | 과학적 신규성 개선 |',
        '| M15-R10 | 알려진 Mie 현상의 주관 심사 평균 차이 +'+str(by['M15_R10']['subjective_mean_delta'])+'/12; 고정 지점 holdout 방향 유지 | 신규 물리, Au/Ag 증거 |',
        '| M16 | Baseline '+str(nums['BASELINE'])+', Known '+str(nums['KNOWN_PHYSICS'])+', Numeric '+str(nums['NUMERIC_PHENOMENON'])+'/12 | 수치 피드백의 추가적 효용, 자율적 문제 발견 |',
        '',
        '### 결론의 권한',
        '- **확보**: 후보 생성·진화·문헌 비판·계산 검증 도구의 실행 기록.',
        '- **조건부**: 외부 정보를 입력하면 후속 연구 계획의 서술과 질문 연결성이 달라질 수 있음.',
        '- **미확보**: 기존보다 가치 있는 미해결 문제의 독립적 발견, 전문가가 인증한 신규성, 새 현상의 독립 검증.',
        '',
        '## 실험 설계의 편향 및 병목',
    ]
    for x in o['diagnosed_confounders']:lines+=['- **'+x['id']+'** — '+x['severity']+': '+x['basis']]
    lines+=['','## 기존 후보 등록부 — 모두 미확인','', '| 문제 ID | 주제 | 상태 |','|---|---|---|']
    for p in o['candidate_questions']: lines.append('| `'+p['problem_id']+'` | '+p['family']+' | '+p['claim_status']+' |')
    lines+=['', '이 5개 기록은 다섯 개의 독립적인 발견을 뜻하지 않는다. 특히 M16 식별성 계열은 M14의 배향/선택 규칙 후보와 개념적으로 겹칠 수 있다.',
            '', '## 객관적 음성 대조: 관측적 비식별성',
            '', '합성 데이터에서 `I_b = N × F_b × T_b × C_b`를 사용하면 서로 다른 두 파라미터 해가 `[30, 24]`라는 동일한 관측을 만든다. 이 수학적 반례는 **신규 과학적 현상도 Au/Ag의 실측 결과도 아니다.** 제한 없는 곱셈형 분해에서는 강한 직교 제약 없이는 각 원인을 유일하게 복원할 수 없음을 확인한다.',
            '', '## 입력 원본 무결성', '', '| 파일 | 확인 상태 |','|---|---|']
    for name,status in src.items():lines.append('| `'+name+'` | '+status+' |')
    lines+=['','모든 입력 파일이 HASH_MATCH인 경우에만 원본을 현재 환경에서 다시 검증했다고 간주한다. 스냅샷만으로는 그 주장을 하지 않는다.',
            '', '## 다음 행동 결정',
            '', '1. **기존 5개 후보에 신규성 배지를 부여하지 않는다.** 독립 1차 논문 확인과 경쟁 null model의 관측적 구별을 통과해야 승격한다.',
            '2. **생성기의 새 구조·솔버·광범위한 A/B 실험 추가를 보류한다.** 먼저 pre-registration 기준을 적용해 실제 평가 가능한 문제 세트를 수집한다.',
            '3. 동일 모델·동일 지식 예산에서 기존 ForAllKG와 강한 직접 LLM baseline을 비교한다. 외부 심사자가 연구 공백과 잠재적 반증 조건을 평가해야 한다.',
            '4. 발견 증거가 없는 동안 신규성 주장 대신 `UNVERIFIED_GAP`, `KNOWN_NULL`, `INDETERMINATE`, `EMPIRICALLY_TESTED`를 사용한다.',
            '', '## 산출물 무결성', '', '감사 결과 JSON이 핵심 판정을 고정한다. 로컬 소스 유무 및 해시 불일치는 별도로 기록된다.', '']
    return '\n'.join(lines)


def audit(snapshot:Path, out:Path, sources:Path|None=None, require_sources:bool=False)->dict:
    if out.exists():raise FileExistsError('OUTPUT_DIRECTORY_EXISTS_NO_OVERWRITE')
    o=read_snapshot(snapshot)
    verified=verify_sources(o,sources)
    if 'HASH_MISMATCH' in verified.values():raise ValueError('SOURCE_CONTENT_CHANGED_OR_WRONG_DIRECTORY')
    if require_sources and any(v!='HASH_MATCH' for v in verified.values()):raise ValueError('REQUIRED_SOURCE_MISSING')
    by={s['study']:s for s in o['studies']}
    toy=nonidentifiability_counterexample()
    audit_obj={'schema':'for_all_kg_discovery_capability_audit_result_v1','status':STATUS,
      'snapshot_sha256':sha(snapshot), 'sources_verified_count':sum(x=='HASH_MATCH' for x in verified.values()),
      'sources_total':len(verified),'source_checks':verified,'studies':o['studies'],
      'verified_new_discoveries':0,'independent_novelty_certifications':0,
      'claim_limit':'NOT_OBSERVED_IN_AVAILABLE_EVIDENCE, NOT AN IMPOSSIBILITY RESULT',
      'known_literature_candidate_count':len(o['candidate_questions']),
      'candidate_status':'UNVERIFIED_PRIOR_ART_GAP', 'bias_flags':o['diagnosed_confounders'],
      'negative_control':toy,'recommendation':'FREEZE_OLD_RESULTS_AND_RUN_PREREGISTERED_DISCOVERY_BENCHMARK',
      'no_model_or_solver_called':True}
    out.mkdir(parents=True)
    p=out/'DISCOVERY_CAPABILITY_AUDIT.json';p.write_text(json.dumps(audit_obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    md=out/'DISCOVERY_CAPABILITY_AUDIT.md';md.write_text(markdown_report(o,verified,audit_obj),encoding='utf-8')
    with (out/'UNVERIFIED_PROBLEM_REGISTER.csv').open('w',encoding='utf-8',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=['problem_id','family','question','claim_status','action','source']);w.writeheader()
        for c in o['candidate_questions']:w.writerow({k:c[k] for k in w.fieldnames})
    man={'schema':'for_all_kg_discovery_audit_files_v1','sha256':{q.name:sha(q) for q in (p,md,out/'UNVERIFIED_PROBLEM_REGISTER.csv')}}
    (out/'OUTPUT_MANIFEST.json').write_text(json.dumps(man,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return audit_obj


def check(out:Path)->None:
    man=json.loads((out/'OUTPUT_MANIFEST.json').read_text())
    for fn,digest in man['sha256'].items():
        if sha(out/fn)!=digest:raise ValueError('OUTPUT_TAMPER_DETECTED:'+fn)
    doc=json.loads((out/'DISCOVERY_CAPABILITY_AUDIT.json').read_text())
    if doc['verified_new_discoveries']!=0 or not doc['no_model_or_solver_called']:raise ValueError('AUTHORITY_MISMATCH')
    if not doc['negative_control']['parameters_A']!=doc['negative_control']['parameters_B']:raise ValueError('INVALID_NEGATIVE_CONTROL')
    if not doc['negative_control']['same_observations']==[30,24]:raise ValueError('NEGATIVE_CONTROL_CHANGED')


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit');a.add_argument('--snapshot',type=Path,required=True);a.add_argument('--out-dir',type=Path,required=True);a.add_argument('--sources-dir',type=Path);a.add_argument('--require-sources',action='store_true')
    c=sub.add_parser('check');c.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='audit':
        result=audit(args.snapshot,args.out_dir,args.sources_dir,args.require_sources)
        print(result['status']);print('SOURCE_HASH_MATCH',result['sources_verified_count'],'/',result['sources_total']);print('NEW_DISCOVERIES_VERIFIED',result['verified_new_discoveries'])
    else:check(args.out_dir);print('DISCOVERY_CAPABILITY_AUDIT_CHECK_OK')


if __name__=='__main__':main()
