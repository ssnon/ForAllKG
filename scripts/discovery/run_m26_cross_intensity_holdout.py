"""M26: real-data cross-intensity time-curve holdout for Au nanorod photocharging.

Add-only, no background network. The measurement source must be supplied locally.
The model sees training traces + held-out intensity/time grid, but NEVER held-out
values. Numerical scoring is not mechanistic or discovery certification.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import statistics
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares

PROTOCOL='m26-cross-intensity-holdout-v1'
SOURCE_DOI='10.1038/s41467-025-67130-8'
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def file_sha(p):return sha_bytes(Path(p).read_bytes())
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def root(p):return Path(p).expanduser().resolve()


def _colnum(ref):
    out=0
    for c in ''.join(x for x in ref if x.isalpha()).upper():out=out*26+(ord(c)-64)
    return out-1


def _xlsx_tables(p):
    """Read-only pure stdlib XLSX cached numeric data, no macro/external formula execution."""
    with zipfile.ZipFile(p) as z:
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        rel=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        targets={e.attrib['Id']:e.attrib['Target'].lstrip('/') for e in rel}
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            ss=ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in ss.findall('m:si',NS):shared.append(''.join(t.text or '' for t in si.findall('.//m:t',NS)))
        tables={}
        for s in wb.findall('.//m:sheet',NS):
            name=s.attrib['name'];rid=s.attrib['{%s}id'%NS['r']]
            target=targets[rid]; target=target if target.startswith('xl/') else 'xl/'+target
            if '..' in target.split('/') or target not in z.namelist():raise ValueError('INVALID_XLSX_SHEET_PATH')
            xml=ET.fromstring(z.read(target)); grid=[]
            for row in xml.findall('.//m:sheetData/m:row',NS):
                v={}
                for c in row.findall('m:c',NS):
                    i=_colnum(c.attrib.get('r',''))
                    if i<0 or i>150:continue
                    typ=c.attrib.get('t',''); val=c.find('m:v',NS)
                    if typ=='inlineStr':
                        part=c.find('m:is',NS);value=''.join(x.text or '' for x in part.findall('.//m:t',NS)) if part is not None else ''
                    elif val is None:value=''
                    elif typ=='s':value=shared[int(val.text)]
                    elif typ in ('str','e'):value=val.text or ''
                    else:value=val.text or ''
                    v[i]=value
                if v:
                    end=min(max(v)+1,151)
                    grid.append([v.get(i,'') for i in range(end)])
                if len(grid)>=100000:raise ValueError('XLSX_TOO_MANY_ROWS')
            tables[name]=grid
        return tables


def tables(p):
    p=root(p)
    if not p.is_file():raise FileNotFoundError(p)
    if p.stat().st_size>100_000_000:raise ValueError('DATA_FILE_OVER_100_MB')
    suff=p.suffix.lower()
    if suff in ('.csv','.tsv'):
        text=p.read_text(encoding='utf-8-sig')
        return {'CSV':list(csv.reader(io.StringIO(text),delimiter='\t' if suff=='.tsv' else ','))}
    if suff=='.xlsx':return _xlsx_tables(p)
    if suff=='.zip':
        with zipfile.ZipFile(p) as z:
            return {'ZIP_ENTRIES':[[x.filename,str(x.file_size)] for x in z.infolist() if not x.is_dir()][:100]}
    raise ValueError('SOURCE_MUST_BE_CSV_TSV_XLSX_OR_ZIP_TO_INSPECT')


def inspect(p):
    t=tables(p)
    return {'status':'M26_SOURCE_INSPECTED_NO_CALL','file':str(root(p)),'sha256':file_sha(p),
      'tables':[{'name':k,'rows':len(v),'header':v[0][:25] if v else [],'preview':v[1:4] if len(v)>1 else []}for k,v in t.items()],
      'warning':'INSPECTION_ONLY. Identify the figure-4 time-vs-LSPR curves, units, and powers before prepare. ZIP: extract the original source file first.'}


def number(x):
    try:
        if x is None or str(x).strip()=='':return None
        v=float(str(x).strip())
        return v if math.isfinite(v) else None
    except (TypeError,ValueError):return None


def _index(h,field):
    if field is None:raise ValueError('EXPLICIT_COLUMN_MAPPING_REQUIRED')
    h=[str(x).strip() for x in h]
    if field in h:
        if h.count(field)>1:raise ValueError('AMBIGUOUS_DUPLICATE_HEADER')
        return h.index(field)
    # #1, #2: numeric zero-based column selectors for awkward spreadsheet headers.
    if field.startswith('#') and field[1:].isdigit() and int(field[1:])<len(h):return int(field[1:])
    raise ValueError('COLUMN_NOT_FOUND: '+field+'; AVAILABLE='+repr(h[:30]))


def convert(p,sheet,time_col=None,power_col=None,value_col=None,wide_spec=None,pairs_spec=None):
    all_tables=tables(p)
    if sheet not in all_tables:raise ValueError('UNKNOWN_SHEET '+repr(sheet)+' available='+repr(list(all_tables)))
    tbl=all_tables[sheet]
    if len(tbl)<5:raise ValueError('TOO_FEW_SOURCE_ROWS')
    hdr=tbl[0];rows=[]
    if pairs_spec:
        if time_col or power_col or value_col or wide_spec:raise ValueError('PAIRS_SPEC_EXCLUDES_OTHER_COLUMN_MODES')
        series=[]
        for v in pairs_spec.split(','):
            P_s,t_s,y_s=v.split(':',2);P=number(P_s)
            if P is None or P<=0:raise ValueError('INVALID_PAIR_POWER')
            series.append((P,_index(hdr,t_s),_index(hdr,y_s)))
        if len(set(v[0] for v in series))!=len(series):raise ValueError('DUPLICATE_PAIR_POWER')
        for r in tbl[1:]:
            for P,ti,yi in series:
                if max(ti,yi)>=len(r):continue
                t,y=number(r[ti]),number(r[yi])
                if t is not None and y is not None:rows.append([P,t,y])
    elif wide_spec:
        ti=_index(hdr,time_col)
        if power_col or value_col:raise ValueError('WIDE_EXCLUDES_LONG_COLUMNS')
        pairs=[]
        for v in wide_spec.split(','):
            a,b=v.split(':',1);P=number(a)
            if P is None or P<=0:raise ValueError('INVALID_DECLARED_POWER')
            pairs.append((P,_index(hdr,b)))
        if len(set(a for a,b in pairs))!=len(pairs):raise ValueError('DUPLICATE_WIDE_POWER')
        for r in tbl[1:]:
            if ti>=len(r):continue
            t=number(r[ti])
            if t is None:continue
            for P,si in pairs:
                y=number(r[si]) if si<len(r) else None
                if y is not None:rows.append([P,t,y])
    else:
        ti=_index(hdr,time_col);pi=_index(hdr,power_col);si=_index(hdr,value_col)
        for r in tbl[1:]:
            if max(ti,pi,si)>=len(r):continue
            P,t,y=number(r[pi]),number(r[ti]),number(r[si])
            if None not in (P,t,y):rows.append([P,t,y])
    if len(rows)>75000:raise ValueError('TOO_MANY_POINTS')
    if any(P<=0 or t<0 or abs(y)>1e8 for P,t,y in rows):raise ValueError('INVALID_PHYSICAL_DATA_RANGE')
    # Equal power/time pairs must be unique: no silent replicate pooling.
    keys=[(p,t) for p,t,y in rows]
    if len(keys)!=len(set(keys)):raise ValueError('DUPLICATE_POWER_TIME: EXPLICITLY_REDUCE_REPLICATES_FIRST')
    return rows


def _split(rows):
    groups={}
    for P,t,y in rows:groups.setdefault(float(P),[]).append((float(t),float(y)))
    powers=sorted(groups)
    if len(powers)<4:raise ValueError('NEED_AT_LEAST_FOUR_DISTINCT_INTENSITIES')
    for p,g in groups.items():
        g.sort()
        if len(g)<8 or len({t for t,y in g})<8:raise ValueError(f'INSUFFICIENT_TIMEPOINTS_FOR_POWER_{p}')
        if max(y for t,y in g)-min(y for t,y in g)<1e-12:raise ValueError(f'CONSTANT_TRACE_{p}')
    hold=powers[(len(powers)-1)//2]   # one predetermined interior power
    if hold in (powers[0],powers[-1]):raise ValueError('NEED_INTERIOR_HOLDOUT_POWER')
    train=[p for p in powers if p!=hold]
    lo=max(min(t for t,y in groups[p]) for p in powers)
    hi=min(max(t for t,y in groups[p]) for p in powers)
    if hi-lo<=0 or hi<=0:raise ValueError('NO_COMMON_NONEMPTY_TIME_WINDOW')
    # Timings from training coverage alone; no target values or target timestamps in request.
    grid=[float(x) for x in np.linspace(lo,hi,12)]
    # Refuse massive gaps in held-out response curve to avoid misleading interpolation.
    hv=np.array([t for t,y in groups[hold]])
    if any(min(abs(hv-t))>(hi-lo)/5 for t in grid):raise ValueError('HOLDOUT_SPARSE_NEAR_EVALUATION_GRID')
    # Training min/max coverage also checked; no extrapolation beyond observed training or holdout time.
    if any(len([t for t,y in groups[p] if lo<=t<=hi])<5 for p in powers):raise ValueError('TOO_FEW_SAMPLES_IN_COMMON_WINDOW')
    return groups,hold,train,grid


def _sample(g,limit=22):
    arr=np.array(g,dtype=float)
    idx=sorted(set(np.linspace(0,len(g)-1,min(len(g),limit)).round().astype(int).tolist()))
    return [[round(float(arr[i,0]),9),round(float(arr[i,1]),12)]for i in idx]


def _model_value(name,p,t):
    t=np.array(t,dtype=float)
    y0=float(p.get('y0',0.0));A=float(p.get('A',0.0));tau=float(p.get('tau_min',1.0))
    if name=='EXP_SAT':return y0+A*(-np.expm1(-t/tau))
    if name=='STRETCHED_EXP':return y0+A*(-np.expm1(-np.power(t/tau,float(p['beta']))))
    if name=='BIPHASIC':
        w=float(p['fast_fraction']);slow=float(p['tau_slow_min'])
        return y0+A*(w*(-np.expm1(-t/tau))+(1-w)*(-np.expm1(-t/slow)))
    if name=='POWER_SAT':return y0+A*t/(tau+t)
    if name=='LINEAR':return y0+float(p['slope_per_min'])*t
    raise ValueError('UNKNOWN_FORM')


def _safe_model(obj):
    if not isinstance(obj,dict) or set(obj)!={'research_question','model_family','parameters','mechanistic_rationale','training_evidence_used','cross_intensity_inference','falsification_conditions','limitations_and_alternatives','source_transfer_warning'}:
        raise ValueError('INVALID_SCIENCE_RESPONSE_KEYS')
    for s in ('research_question','mechanistic_rationale','training_evidence_used','cross_intensity_inference','falsification_conditions','limitations_and_alternatives','source_transfer_warning'):
        if not isinstance(obj[s],str) or len(obj[s].strip())<24:raise ValueError('MISSING_EXPLANATION '+s)
    name=obj['model_family'];params=obj['parameters']
    schema={'EXP_SAT':{'y0','A','tau_min'},'STRETCHED_EXP':{'y0','A','tau_min','beta'},'BIPHASIC':{'y0','A','tau_min','tau_slow_min','fast_fraction'},'POWER_SAT':{'y0','A','tau_min'},'LINEAR':{'y0','slope_per_min'}}
    if name not in schema or not isinstance(params,dict) or set(params)!=schema[name]:raise ValueError('INVALID_PARAMETERS')
    if any(type(x) not in (float,int) or not math.isfinite(x) or abs(x)>1e7 for x in params.values()):raise ValueError('NONFINITE_OR_HUGE_PARAMETERS')
    for k in ('tau_min','tau_slow_min'):
        if k in params and not 1e-6<=params[k]<=1e7:raise ValueError('NONPOSITIVE_OR_UNPHYSICAL_TAU')
    if 'beta' in params and not .15<=params['beta']<=4:raise ValueError('BETA_OUT_OF_RANGE')
    if 'fast_fraction' in params and not 0<=params['fast_fraction']<=1:raise ValueError('FRACTION_OUT_OF_RANGE')
    return obj

SYSTEM='''You are an autonomous scientific modeller. Choose a meaningful, explicit kinetic interpretation and predict an ENTIRE unseen light-intensity time curve using observations ONLY from the visible training intensities. The hidden curve has NO observed values in your packet. No browser or external tools available. Independent baselines include power interpolation and saturation-exponential fitting, so do not present simple curve fitting as scientific discovery. Interpret physical assumptions, units and identifiability honestly. Output ONE JSON object with exactly these keys: research_question, model_family, parameters, mechanistic_rationale, training_evidence_used, cross_intensity_inference, falsification_conditions, limitations_and_alternatives, source_transfer_warning. model_family is one of EXP_SAT, STRETCHED_EXP, BIPHASIC, POWER_SAT, LINEAR. Parameters for EXP_SAT/POWER_SAT: {"y0":number,"A":number,"tau_min":positive_number}; STRETCHED_EXP adds beta; BIPHASIC: {"y0":number,"A":number,"tau_min":positive_number,"tau_slow_min":positive_number,"fast_fraction":0..1}; LINEAR: {"y0":number,"slope_per_min":number}. Their time response definitions are in the input. Report why the held-out intensity parameters can be transferred from training intensities; declare extrapolation risks. No claims about Ye SERS are licensed.''' 


def _origin(m25):
    if m25 is None:return {}
    from scripts.discovery import run_m25_two_intervention_holdout as previous
    state=previous.verify(m25)
    if state['evaluation']!='REVEALED':raise ValueError('M25_NOT_REVEALED')
    p=m25/'evaluation.json';return {str(p):file_sha(p)}


def prepare(out,source,table,time_col=None,power_col=None,signal_col=None,wide_spec=None,pairs_spec=None,model=None,m25=None):
    out=root(out);source=root(source)
    if out.exists():raise FileExistsError('FRESH_RUN_DIRECTORY_REQUIRED')
    origins=_origin(root(m25) if m25 else None)
    if model is None:
        if not m25:raise ValueError('PROVIDE_MODEL_OR_M25')
        from scripts.discovery import run_m25_two_intervention_holdout as p25
        cfg=read(root(m25)/'lock.json')['model_settings'];model=cfg['model']
    rows=convert(source,table,time_col,power_col,signal_col,wide_spec,pairs_spec)
    groups,held,train,grid=_split(rows)
    public={
        'subject':'LSPR time-dependent response to illumination power for solvated gold nanorods',
        'source_of_measurements':'research article source-data file, manually selected columns; not Ye SERS',
        'units_note':'power_mW and elapsed_minutes are required; response retains user-selected source unit and sign',
        'training_intensities_mW':train,'held_out_intensity_mW':held,
        'heldout_time_grid_minutes':grid,
        'training_measured_traces':{str(p):_sample(groups[p]) for p in train},
        'training_all_point_counts':{str(p):len(groups[p]) for p in train},
        'model_definitions':{
          'EXP_SAT':'y0+A*(1-exp(-t/tau_min))',
          'STRETCHED_EXP':'y0+A*(1-exp(-(t/tau_min)^beta))',
          'BIPHASIC':'y0+A*[fast_fraction*(1-exp(-t/tau_min))+(1-fast_fraction)*(1-exp(-t/tau_slow_min))]',
          'POWER_SAT':'y0+A*t/(tau_min+t)',
          'LINEAR':'y0+slope_per_min*t'},
        'scientific_task':'Select, justify, and parameterize one kinetic model for the held-out illumination intensity. Do not assume other intensities have identical parameters, and distinguish physical interpretation from empirical fitting.',
        'no_measured_target_in_request':True}
    private={'source_doi':SOURCE_DOI,'source_path':str(source),'source_sha256':file_sha(source),
             'table':table,'mapping':{'time_col':time_col,'power_col':power_col,'signal_col':signal_col,'wide_spec':wide_spec,'pairs_spec':pairs_spec},
             'all_rows_sha256':sha_bytes(canon(rows)),'holdout_intensity_mW':held,
             'holdout_trace':groups[held],'time_grid_min':grid,
             'train_full_traces':{str(p):groups[p] for p in train},
             'data_semantics_unverified':'Confirm spreadsheet columns are the published Fig4 kinetic time-series rather than spectra/fit values.'}
    out.mkdir(parents=True)
    try:
        put(out/'public_input.json',public);put(out/'private_holdout.json',private)
        put(out/'lock.json',{'protocol':PROTOCOL,'script_sha256':file_sha(__file__),
          'public_sha256':file_sha(out/'public_input.json'),'private_sha256':file_sha(out/'private_holdout.json'),
          'origin_hashes':origins,'model':model,'temperature':0.2,'max_tokens':4800,
          'source_hash':file_sha(source),'paid_target_values_in_request':False})
        verify(out)
    except BaseException:
        shutil.rmtree(out);raise
    return {'status':'M26_REAL_DATA_PREPARED_NO_CALL','out':str(out),
        'train_powers':train,'held_out_power':held,'heldout_curve_points':len(groups[held]),
        'raw_source_verified_hash':file_sha(source),'paid_calls':0}


def _guard(out):
    out=root(out);lock=read(out/'lock.json');private=read(out/'private_holdout.json')
    if lock['protocol']!=PROTOCOL or lock['script_sha256']!=file_sha(__file__):raise ValueError('EXECUTOR_CHANGED')
    if file_sha(out/'public_input.json')!=lock['public_sha256'] or file_sha(out/'private_holdout.json')!=lock['private_sha256']:raise ValueError('PUBLIC_PRIVATE_MUTATION')
    if file_sha(Path(private['source_path']))!=lock['source_hash']:raise ValueError('SOURCE_FILE_MUTATED')
    for f,h in lock['origin_hashes'].items():
        if file_sha(f)!=h:raise ValueError('PRIOR_RESEARCH_ORIGIN_CHANGED')
    return out,lock


def request(out):
    out,lock=_guard(out)
    return {'model':lock['model'],'temperature':lock['temperature'],'max_tokens':lock['max_tokens'],
        'messages':[{'role':'system','content':SYSTEM},
                    {'role':'user','content':json.dumps(read(out/'public_input.json'),ensure_ascii=False,sort_keys=True)}]}


def verify(out):
    out,lock=_guard(out)
    d=out/'propose'
    if not d.exists():
        if (out/'evaluation.json').exists():raise ValueError('SCORED_WITHOUT_PREDICTION')
        return {'status':'M26_VERIFIED_NO_CALL','prediction':'NOT_STARTED','evaluation':'NOT_STARTED'}
    req=request(out)
    if read(d/'request.json')!=req or read(d/'attempt.json')['request_sha256']!=sha_bytes(canon(req)):raise ValueError('REQUEST_MISMATCH')
    if not (d/'raw_response.json').exists():return {'status':'M26_VERIFIED_NO_CALL','prediction':'ATTEMPT_BILLING_UNKNOWN_DO_NOT_RETRY','evaluation':'NOT_STARTED'}
    if not (d/'prediction.json').exists() or not (d/'receipt.json').exists():return {'status':'M26_VERIFIED_NO_CALL','prediction':'RAW_SAVED_RECOVER_NO_RETRY','evaluation':'NOT_STARTED'}
    norm=_safe_model(json.loads(read(d/'raw_response.json')['raw_text']))
    if read(d/'prediction.json')!=norm:raise ValueError('PREDICTION_MUTATED')
    receipt={'request_sha256':sha_bytes(canon(req)),'raw_sha256':file_sha(d/'raw_response.json'),'prediction_sha256':file_sha(d/'prediction.json')}
    if receipt!=read(d/'receipt.json'):raise ValueError('RECEIPT_CHANGED')
    ev=out/'evaluation.json'
    if ev.exists():
        expected=evaluate(out,norm,receipt)
        if read(ev)!=expected:raise ValueError('EVALUATION_MUTATED')
        return {'status':'M26_VERIFIED_NO_CALL','prediction':'READY','evaluation':'REVEALED'}
    return {'status':'M26_VERIFIED_NO_CALL','prediction':'READY','evaluation':'SEALED'}


def preview(out):
    s=verify(out)
    if s['prediction']!='NOT_STARTED':raise ValueError('NOT_FRESH_NO_PAID_RETRY')
    req=request(out);encoded=canon(req)
    private=read(root(out)/'private_holdout.json')
    # Only public training curves and target time/power; hidden target measured y's never fed.
    public=read(root(out)/'public_input.json')
    assert 'holdout_trace' not in json.dumps(public)
    return {'status':'M26_PREVIEW_NO_CALL','next_call_is_paid':True,'request_sha256':sha_bytes(encoded),
        'unseen_power':public['held_out_intensity_mW'],'training_powers':public['training_intensities_mW'],
        'private_target_not_in_request':True,'physical_source_units_require_human_review':True}


def recover(out):
    out=root(out);s=verify(out)
    if s['prediction'] not in ('RAW_SAVED_RECOVER_NO_RETRY','READY'):raise ValueError('NO_SAVED_RAW')
    d=out/'propose';old=file_sha(d/'raw_response.json')
    norm=_safe_model(json.loads(read(d/'raw_response.json')['raw_text']))
    if not (d/'prediction.json').exists():put(d/'prediction.json',norm)
    elif read(d/'prediction.json')!=norm:raise ValueError('PREDICTION_CONFLICT')
    receipt={'request_sha256':sha_bytes(canon(request(out))),'raw_sha256':file_sha(d/'raw_response.json'),'prediction_sha256':file_sha(d/'prediction.json')}
    if not (d/'receipt.json').exists():put(d/'receipt.json',receipt)
    elif read(d/'receipt.json')!=receipt:raise ValueError('RECEIPT_CONFLICT')
    if old!=file_sha(d/'raw_response.json'):raise ValueError('PAID_RAW_CHANGED')
    return {'status':'M26_SAVED_RESPONSE_RECOVERED_NO_REPAY','verification':verify(out)}


def propose(out,allow_paid=False,backend=None):
    p=preview(out)
    if backend is None:
        from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as base
        base._require_paid(backend,allow_paid,1)
    out=root(out);d=out/'propose';d.mkdir()
    req=request(out)
    put(d/'request.json',req);put(d/'attempt.json',{'request_sha256':p['request_sha256'],'billing_status':'UNCERTAIN_UNTIL_RAW_SAVED'})
    if backend is None:raw,usage,rid=base._invoke(req,backend)
    else:raw,usage,rid=backend(req)
    put(d/'raw_response.json',{'raw_text':raw,'usage':usage,'response_id':rid})
    try:r=recover(out)
    except Exception as e:raise RuntimeError('PAID_RAW_SAVED_USE_RECOVER_NO_REPAY') from e
    return r|{'new_paid_calls':int(backend is None)}


def _baseline_curves(private,t):
    p=float(private['holdout_intensity_mW']);tr={float(k):np.array(v,dtype=float) for k,v in private['train_full_traces'].items()}
    powers=sorted(tr);below=max(q for q in powers if q<p);above=min(q for q in powers if q>p)
    def yi(q):return np.interp(t,tr[q][:,0],tr[q][:,1])
    near=min(powers,key=lambda q:(abs(q-p),q))
    near_curve=yi(near)
    w=(p-below)/(above-below)
    power_interp=(1-w)*yi(below)+w*yi(above)
    fit_params={}
    for q,arr in tr.items():
        x=arr[:,0];y=arr[:,1]
        tspan=max(float(x.max()-x.min()),.01)
        amp0=float(y[-1]-y[0]);base0=float(y[0])
        scales=max(float(np.std(y)),1e-6)
        def residual(v):
            b,A,logtau=v
            return (b+A*(-np.expm1(-x/np.exp(logtau)))-y)/scales
        r=least_squares(residual,[base0,amp0,math.log(tspan/2)],bounds=([-np.inf,-np.inf,math.log(1e-6)],[np.inf,np.inf,math.log(1e7)]),max_nfev=1200)
        fit_params[q]=[float(r.x[0]),float(r.x[1]),float(r.x[2])]
    interp=[(1-w)*fit_params[below][i]+w*fit_params[above][i] for i in range(3)]
    baseline_exp=interp[0]+interp[1]*(-np.expm1(-t/np.exp(interp[2])))
    return {'nearest_power_trace':near_curve,'bracketed_power_linear_interpolation':power_interp,
            'per_power_exponential_parameter_interpolation':baseline_exp}


def _metrics(y,guess):
    y=np.asarray(y,dtype=float);guess=np.asarray(guess,dtype=float)
    if len(y)!=len(guess) or not np.all(np.isfinite(guess)):raise ValueError('NONFINITE_MODEL_PREDICTION')
    err=guess-y;scale=max(float(np.ptp(y)),float(np.max(np.abs(y)))*.1,1e-10)
    return {'rmse':float(np.sqrt(np.mean(err**2))),'mae':float(np.mean(abs(err))),
      'nrmse_range_or_scale':float(np.sqrt(np.mean(err**2))/scale),'scale':scale}


def evaluate(out,norm,receipt):
    p=read(root(out)/'private_holdout.json');t=np.array(p['time_grid_min'],dtype=float)
    gt=np.array(p['holdout_trace'],dtype=float)
    y=np.interp(t,gt[:,0],gt[:,1])
    candidate=_model_value(norm['model_family'],norm['parameters'],t)
    candidates=_baseline_curves(p,t);candidates['FAKG']=candidate
    sc={k:_metrics(y,v) for k,v in candidates.items()}
    sorted_baselines=sorted((v['nrmse_range_or_scale'],k) for k,v in sc.items() if k!='FAKG')
    bestscore,bestname=sorted_baselines[0]
    result={'status':'M26_REVEALED_EMPIRICAL_FORECAST_AUDIT',
      'source_type':'USER_SUPPLIED_SOURCE_DATA_FILE_NOT_INDEPENDENTLY_AUTHENTICATED',
      'provenance':{'doi':SOURCE_DOI,'local_sha256':p['source_sha256'],'table':p['table'],'mapping':p['mapping']},
      'held_out_power_mW':p['holdout_intensity_mW'],'hidden_points_count':len(gt),'forecast_grid_count':len(t),
      'forecast_time_grid_min':[float(v) for v in t],
      'ground_truth_interpolated':[float(v) for v in y],
      'model_forecast':[float(v) for v in candidate],
      'baseline_model_forecasts':{k:[float(x) for x in v] for k,v in candidates.items() if k!='FAKG'},
      'metrics':sc,'best_baseline_name':bestname,
      'FAKG_beats_best_baseline':bool(sc['FAKG']['nrmse_range_or_scale']<bestscore),
      'model_family':norm['model_family'],'prediction_receipt':receipt,
      'limitations':['One held-out power and one published experiment','Source-data column semantics manually selected; not scientifically authenticated by machine',
       'Public source could occur in model pretraining','Only phenomenological fitting, not a novel mechanism or Ye/PVA validation',
       'Baseline interpolation assumes fixed apparatus; uncertainty intervals not estimated'],
      'scientific_validation':False,'level_ii_certified':False}
    return result


def reveal(out):
    s=verify(out)
    if s['prediction']!='READY' or s['evaluation']!='SEALED':raise ValueError('NEED_FROZEN_UNREVEALED_PREDICTION')
    out=root(out)
    pred=read(out/'propose/prediction.json');receipt=read(out/'propose/receipt.json')
    result=evaluate(out,pred,receipt);put(out/'evaluation.json',result)
    verify(out)
    return {'status':'M26_HOLDOUT_REVEALED_NO_PAID_CALL',
        'beats_best_baseline':result['FAKG_beats_best_baseline'],
        'model_nrmse':result['metrics']['FAKG']['nrmse_range_or_scale'],
        'best_baseline':result['best_baseline_name']}


def report(out):
    st=verify(out);r={'status':'M26_REPORT_NO_CALL','run_status':st,'scientific_validation':False}
    if st['prediction']=='READY':
        p=read(root(out)/'propose/prediction.json');r['chosen_model']=p['model_family'];r['mechanistic_claim_UNVERIFIED']=p['mechanistic_rationale']
    if st['evaluation']=='REVEALED':
        x=read(root(out)/'evaluation.json');r['independent_prediction_score']=x['metrics'];r['FAKG_beats_best_baseline']=x['FAKG_beats_best_baseline']
    return r


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    sub=pa.add_subparsers(dest='command',required=True)
    ins=sub.add_parser('inspect');ins.add_argument('--data',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--data',required=True);prep.add_argument('--sheet',required=True);prep.add_argument('--time-col')
    prep.add_argument('--power-col');prep.add_argument('--signal-col');prep.add_argument('--wide-spec');prep.add_argument('--pairs-spec');prep.add_argument('--out',required=True)
    prep.add_argument('--m25');prep.add_argument('--model')
    for name in ('verify','preview','propose','recover','reveal','report'):
        s=sub.add_parser(name);s.add_argument('--out',required=True)
        if name=='propose':s.add_argument('--allow-paid',action='store_true')
    a=pa.parse_args()
    if a.command=='inspect':result=inspect(a.data)
    elif a.command=='prepare':result=prepare(a.out,a.data,a.sheet,a.time_col,a.power_col,a.signal_col,a.wide_spec,a.pairs_spec,a.model,a.m25)
    else:result=globals()[a.command](a.out,**({'allow_paid':a.allow_paid} if a.command=='propose' else {}))
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))

if __name__=='__main__':main()
