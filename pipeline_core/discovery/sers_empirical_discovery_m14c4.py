"""M14-C4: reproducible *empirical* screening, never novelty certification.

The analysis is an observational, preregistered re-analysis of published spectra.
Only standard-library dependencies. No scientific assertions are created from text.
"""
from __future__ import annotations
import csv, hashlib, io, json, math, re, statistics
from collections import defaultdict
from pathlib import Path

SCHEMA='m14c4-empirical-gap-discovery-v1'
DATASETS={
 'ILS_ADENINE_2020': {
  'record':'https://zenodo.org/records/3572359',
  'source_url':'https://zenodo.org/records/3572359/files/ILSdata.csv?download=1',
  'filename':'ILSdata.csv','md5':'0f77151643c10f53d871e58bd973e485',
  'doi':'10.5281/zenodo.3572359','paper_doi':'10.1021/acs.analchem.9b05658',
  'description':'Published multi-instrument interlaboratory adenine SERS spectra with concentration, instrument, and batch metadata',
  'metadata':['labcode','substrate','laser','method','sample','type','conc','batch','replica'],
  'analysis_ready':True,
 },
 'WATER_TITRATION_638_2022': {
  'record':'https://zenodo.org/records/6980619',
  'source_url':'https://zenodo.org/records/6980619/files/MD150%20to%20155_Master_df_638_corrected_mod_v02.csv?download=1',
  'filename':'MD150 to 155_Master_df_638_corrected_mod_v02.csv',
  'md5':'a2c7b7f5e6ac2a3a9e36df7c6b7c9a84','doi':'10.5281/zenodo.6980619',
  'description':'Independent mineral water titration SERS spectra, 638 nm, semicolon-delimited',
  'analysis_ready':False,'reason':'Requires separate titration-step metadata mapping and preregistered chemistry target; never force adenine concentration model',
 },
}
COHORT_FIELDS=('method','substrate','laser','sample','type')
GROUP_FIELD='labcode'
BAND=(715.0,750.0)   # specified in original Fornasaro et al. work
MIN_LABS=3
MIN_SAMPLES=24
MIN_TRAIN_LEVELS=3
MAX_COHORTS=3

def canonical(x):return (json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def digest_file(path,method='sha256'):
 h=hashlib.new(method)
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def numeric_header(s):
 t=s.strip().lstrip('\ufeff')
 if t.startswith('X'):t=t[1:]
 try: return float(t)
 except (ValueError,TypeError):return None

def concentration(s):
 s=str(s).strip().replace('−','-')
 if not re.fullmatch(r'(?:\+|-)?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?',s):return None
 try:v=float(s)
 except ValueError:return None
 return v if math.isfinite(v) and v>0 else None

def parse_csv(path:Path):
 """Strict ILS table parse; metadata may be lower/uppercase, spectra NA are missing."""
 with path.open('r',encoding='utf-8-sig',errors='replace',newline='') as f:
  header=next(csv.reader(f))
  delimiter=';' if len(header)<3 and ';' in header[0] else ','
  f.seek(0);r=csv.reader(f,delimiter=delimiter);header=next(r)
  ix={h.strip().lower():i for i,h in enumerate(header)}
  absent=[k for k in DATASETS['ILS_ADENINE_2020']['metadata'] if k not in ix]
  if absent:raise ValueError('M14C4_REQUIRED_METADATA_MISSING: '+str(absent)+'; header='+repr(header[:15]))
  band=sorted((w,i) for i,h in enumerate(header) if (w:=numeric_header(h)) is not None and BAND[0]<=w<=BAND[1])
  if len(band)<2:raise ValueError('M14C4_NO_NUMERIC_715_750_SPECTRAL_AXIS; header_sample='+repr(header[:20]))
  rows=[];reason=defaultdict(int);count=0
  for line in r:
   count+=1
   if len(line)!=len(header):reason['bad_width']+=1;continue
   meta={k:line[i].strip() for k,i in ix.items() if k in DATASETS['ILS_ADENINE_2020']['metadata']}
   v=concentration(meta['conc'])
   if v is None:reason['invalid_or_unlabeled_concentration']+=1;continue
   vals=[]
   for w,i in band:
    s=line[i].strip()
    try:y=float(s)
    except (ValueError,TypeError):break
    if not math.isfinite(y):break
    vals.append((w,y))
   if len(vals)!=len(band):reason['missing_or_invalid_band']+=1;continue
   area=sum((w1-w0)*(y0+y1)/2 for (w0,y0),(w1,y1) in zip(vals,vals[1:]))
   if not math.isfinite(area):reason['nonfinite_area']+=1;continue
   ctuple=tuple(meta[k] for k in COHORT_FIELDS)
   lab=meta[GROUP_FIELD]
   if not lab:reason['empty_lab']+=1;continue
   rows.append({'cohort':ctuple,'lab':lab,'batch':meta['batch'],'x':math.log10(v),'conc':v,'y':area})
 return {'rows':rows,'raw_rows':count,'rejections':dict(reason),'spectral_columns':len(band),
         'band_observed_range':[band[0][0],band[-1][0]],'metadata_columns':list(ix.keys())}

def screening(rows):
 grouped=defaultdict(list)
 for r in rows:grouped[r['cohort']].append(r)
 candidates=[]
 for coh,pts in grouped.items():
  labs={p['lab'] for p in pts};levels={p['x'] for p in pts}
  eligible=len(labs)>=MIN_LABS and len(pts)>=MIN_SAMPLES and len(levels)>=MIN_TRAIN_LEVELS
  candidates.append({'cohort':list(coh),'cohort_key':sha(canonical(list(coh)))[:16],
    'n':len(pts),'n_labs':len(labs),'n_concentrations':len(levels),'eligible':eligible})
 candidates.sort(key=lambda z:(not z['eligible'],-z['n_labs'],-z['n'],z['cohort_key']))
 selected=[x['cohort_key'] for x in candidates if x['eligible']][:MAX_COHORTS]
 return {'candidate_cohorts':candidates,'selected_cohort_keys':selected,
  'selection_policy':'Group and size eligibility from metadata and band completeness; no intensity magnitudes or fitted residuals used for selection',
  'unselected_eligible_cohorts':[x['cohort_key'] for x in candidates if x['eligible'] and x['cohort_key'] not in selected]}

def solve_system(a,b):
 n=len(b);z=[row[:] + [b[i]] for i,row in enumerate(a)]
 for c in range(n):
  pivot=max(range(c,n),key=lambda r:abs(z[r][c]))
  if abs(z[pivot][c])<1e-12:return None
  z[c],z[pivot]=z[pivot],z[c]
  scale=z[c][c]
  for k in range(c,n+1):z[c][k]/=scale
  for r in range(n):
   if r==c:continue
   mul=z[r][c]
   for k in range(c,n+1):z[r][k]-=mul*z[c][k]
 return [z[i][n] for i in range(n)]

def fit(train,degree):
 xs=[p['x'] for p in train];ys=[p['y'] for p in train]
 mu=statistics.mean(xs);scale=max(statistics.pstdev(xs),1e-8)
 def phi(x):return [((x-mu)/scale)**i for i in range(degree+1)]
 feat=[phi(x) for x in xs];m=degree+1
 gram=[[sum(v[i]*v[j] for v in feat) for j in range(m)] for i in range(m)]
 rhs=[sum(v[i]*y for v,y in zip(feat,ys)) for i in range(m)]
 coef=solve_system(gram,rhs)
 if coef is None:return None
 return lambda x:sum(a*t for a,t in zip(coef,phi(x)))

def rmse(values):return math.sqrt(sum(d*d for d in values)/len(values)) if values else None

def compare(parsed,screen):
 rows=parsed['rows'];chosen=set(screen['selected_cohort_keys']);groups=defaultdict(list)
 for r in rows:
  key=sha(canonical(list(r['cohort'])))[:16]
  if key in chosen:groups[key].append(r)
 out=[];skipped=[]
 for key in sorted(chosen):
  pts=groups[key];labs=sorted({p['lab'] for p in pts})
  for lab in labs:
   train=[p for p in pts if p['lab']!=lab];test=[p for p in pts if p['lab']==lab]
   if len(train)<12 or len(test)<3 or len(set(p['x'] for p in train))<MIN_TRAIN_LEVELS or len(set(p['x'] for p in test))<2:
    skipped.append({'cohort_key':key,'heldout_lab':lab,'reason':'Insufficient training/test levels or samples'});continue
   if min(p['x'] for p in test)<min(p['x'] for p in train)-1e-8 or max(p['x'] for p in test)>max(p['x'] for p in train)+1e-8:
    skipped.append({'cohort_key':key,'heldout_lab':lab,'reason':'Test concentrations outside training support; do not conflate transfer with extrapolation'});continue
   for model,degree in [('NO_CONCENTRATION',0),('LINEAR_LOG_CONC',1),('QUADRATIC_LOG_CONC',2)]:
    predictor=fit(train,degree)
    if predictor is None:
     skipped.append({'cohort_key':key,'heldout_lab':lab,'model':model,'reason':'Rank deficient design'});continue
    residuals=[predictor(p['x'])-p['y'] for p in test]
    # RMSE is conditional on the same lab context; not an experiment and not mechanistic inference.
    out.append({'cohort_key':key,'heldout_lab':lab,'model':model,'train_n':len(train),
      'test_n':len(test),'test_rmse':rmse(residuals),'test_mean_y':statistics.mean(p['y'] for p in test),
      'test_sd_y':statistics.pstdev(p['y'] for p in test)})
 return out,skipped

def infer(folds,screen):
 by=defaultdict(dict)
 for r in folds:by[(r['cohort_key'],r['heldout_lab'])][r['model']]=r
 cohort_results=[]
 for ck in screen['selected_cohort_keys']:
  matches=[v for (k,_),v in by.items() if k==ck and len(v)==3]
  if len(matches)<3:
   cohort_results.append({'cohort_key':ck,'status':'EVIDENCE_INSUFFICIENT','action':'SUSPEND',
      'reason':'Fewer than three fully evaluated leave-one-lab-out folds; cannot assess transfer',
      'novelty_certified':False});continue
  wins=0;wins_lin=0;improvements=[];linear_vs_null=[]
  for r in matches:
   l=r['LINEAR_LOG_CONC']['test_rmse'];q=r['QUADRATIC_LOG_CONC']['test_rmse'];n=r['NO_CONCENTRATION']['test_rmse']
   wins+=q<.9*l
   wins_lin+=l<.9*n
   improvements.append((l-q)/max(abs(l),1e-12))
   linear_vs_null.append((n-l)/max(abs(n),1e-12))
  stable_q=wins>=max(3,math.ceil(.75*len(matches)))
  stable_l=wins_lin>=max(3,math.ceil(.75*len(matches)))
  if stable_q:
   action='BRANCH';status='EMPIRICAL_NONLINEAR_PREDICTIVE_SIGNAL_REQUIRES_VALIDATION'
   gap='Does the 715-750 cm-1 adenine band show concentration-dependent nonlinear response that transfers across independent labs after controlling batch and instrument effects?'
  elif stable_l:
   action='REFINE';status='CONCENTRATION_RESPONSE_PARTLY_TRANSFERRABLE'
   gap='Which instrument and substrate conditions limit transfer of the established adenine SERS concentration-response relation?'
  else:
   action='SUSPEND';status='TRANSFERABILITY_NOT_ESTABLISHED'
   gap='Which unmeasured calibration factors explain the failure of the tested global concentration-response regressions to transfer between laboratories?'
  cohort_results.append({'cohort_key':ck,'status':status,'action':action,'problem_question':gap,
     'source_of_question':'Empirical leave-one-laboratory-out predictive errors in preregistered 715-750 cm-1 band',
     'cohort_fold_count':len(matches),'quadratic_beats_linear_by_10_percent_folds':wins,
     'linear_beats_constant_by_10_percent_folds':wins_lin,
     'quadratic_relative_rmse_gain_by_fold':improvements,
     'linear_relative_rmse_gain_by_fold':linear_vs_null,
     'independent_intervention_present':False,'measured_uncertainty_beyond_between_lab_variability':False,
     'prior_art_warning':'Original study already quantified adenine using the 715-750 cm-1 band, and addressed interlaboratory reproducibility.',
     'followup_evidence_needed':['Independent instrument response calibration and blank spectra',
         'Held-out labs/batches not used in model selection',
         'Replicate-level noise characterization and sample-preparation provenance'],
     'novelty_certified':False,'new_mechanism_identified':False})
 return cohort_results

def csv_bytes(rows):
 if not rows:return b''
 b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return b.getvalue().encode()
