"""M15-R8A: provenance-pinned Johnson-Christy Au/Ag SCUFF exploratory execution.

Two explicit phases: prepare (no solver) and run (one explicitly authorized solver).
No nanorod science qualification or hypothesis authority is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

VERSION = 'm15r8a-auag-scuff-pilot-v1'
# Git blob SHA-1 for source data in polyanskiy/refractiveindex.info-database,
# database/data/main/{Au,Ag}/nk/Johnson.yml, retrieved on 2026-10-11.
# A changed upstream version must be investigated, not silently used.
SOURCE_BLOB_SHA1 = {
    'Au': 'cf2f1b2c490d60bea818096920e323da94bbb540',
    'Ag': 'fcace717cd05616f344cf732784253260d5c3769',
}
SOURCE_URL = {
    material: 'https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/'
              f'database/data/main/{material}/nk/Johnson.yml'
    for material in ('Au', 'Ag')
}
C_M_PER_S = 299792458.0
ROW = re.compile(r'^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def new_text(path: Path, body: str) -> None:
    with path.open('x', encoding='utf-8') as fh:
        fh.write(body)


def new_json(path: Path, obj: dict) -> None:
    new_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def executable(spec: str) -> str:
    path = (shutil.which(spec) if '/' not in spec else str(Path(spec).resolve()))
    if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
        raise ValueError('SCUFF_EXECUTABLE_NOT_FOUND')
    return path


def read_nk(raw: bytes) -> list[tuple[float, float, float]]:
    txt = raw.decode('utf-8')
    if txt.count('type: tabulated nk') != 1 or 'data: |' not in txt:
        raise ValueError('NOT_SINGLE_TABULATED_NK')
    records = [tuple(float(x) for x in m.groups()) for line in txt.splitlines()
               if (m := ROW.fullmatch(line))]
    if len(records) < 40:
        raise ValueError('NK_DATA_TOO_SHORT')
    if not all(0.15 < x[0] < 2.5 and x[1] >= 0 and x[2] >= 0 and all(map(math.isfinite,x))
               for x in records):
        raise ValueError('INVALID_NK_VALUE')
    if any(b[0] <= a[0] for a,b in zip(records,records[1:])):
        raise ValueError('NK_WAVELENGTH_NOT_INCREASING')
    return records


def source_to_eps(raw:bytes, material:str, wl_nm:float) -> tuple[str,dict]:
    if material not in SOURCE_BLOB_SHA1:
        raise ValueError('UNKNOWN_MATERIAL')
    if git_blob_sha1(raw) != SOURCE_BLOB_SHA1[material]:
        raise ValueError('UPSTREAM_JOHNSON_FILE_BLOB_MISMATCH_'+material)
    data = read_nk(raw)
    wl_um = wl_nm/1000.0
    if not data[0][0] <= wl_um <= data[-1][0]:
        raise ValueError('REQUESTED_WAVELENGTH_OUTSIDE_NK_RANGE_'+material)
    # SCUFF FILE_ materials: omega in physical rad/s, complex epsilon in eps1+eps2*i.
    # Reverse wavelength order to make omega strictly ascending.
    # Only pure numeric rows in FILE_ material files; provenance is stored in the plan.
    lines = []
    for wl, n, k in reversed(data):
        omega=2*math.pi*C_M_PER_S/(wl*1e-6)
        real=n*n-k*k
        imag=2*n*k
        lines.append(f'{omega:.15e} {real:.15e}{imag:+.15e}i')
    meta={'material':material,'git_blob_sha1':git_blob_sha1(raw),
          'source_url':SOURCE_URL[material],
          'source_sha256':hashlib.sha256(raw).hexdigest(),
          'data_points':len(data), 'wavelength_min_nm':data[0][0]*1000,
          'wavelength_max_nm':data[-1][0]*1000,
          'model':'Johnson-Christy 1972 bulk thin film optical constants; not calibrated to current nanoparticles',
          'formula':'epsilon=(n+i*k)^2; omega=2*pi*c/lambda_vacuum_m; linear interpolation of epsilon(omega) by SCUFF'}
    return '\n'.join(lines)+'\n',meta


def _load_sources(r4:Path,r5:Path,r7b:Path)->tuple[dict,dict,dict]:
    from scripts.discovery.run_m15_r5_mesh_verify import check as check_r5, load_r4, read_msh2
    r4plan = load_r4(r4)
    r5status = check_r5(r5)
    if r5status['mesh_topology'] != 'MESH_TOPOLOGY_PASS_PRELIMINARY':
        raise ValueError('R5_MESH_NOT_PRELIMINARILY_VALID')
    gate = json.loads((r5/'M15R5_MESH_GATE.json').read_text())
    # Verify source R4->R5 relation (check_r5 does too) and exactly known mesh filenames.
    if gate['r4_plan_sha256'] != sha256(r4/'M15R4_BEM_CANDIDATE_PLAN.json'):
        raise ValueError('R4_R5_MISMATCH')
    if not r7b.is_file():
        raise ValueError('R7B_REPORT_NOT_FOUND')
    r7baudit=json.loads(r7b.read_text())
    if (r7baudit.get('metrics',{}).get('screen_status')!='SPHERE_3MESH_TREND_SUPPORTIVE'
        or r7baudit.get('metrics',{}).get('au_ag_science_gate')!='BLOCKED'):
        raise ValueError('R7B_EXPECTED_SCREEN_MISSING')
    if r4plan['core_material'] not in SOURCE_BLOB_SHA1 or r4plan['shell_material'] not in SOURCE_BLOB_SHA1:
        raise ValueError('UNKNOWN_METAL')
    if r4plan['medium'] not in ('air','water'):
        raise ValueError('UNSUPPORTED_MEDIUM')
    for name in ('inner_core.msh','outer_shell.msh'):
        if not (r5/name).is_file():
            raise ValueError('R5_MESH_MISSING_'+name)
        read_msh2(r5/name)
    return r4plan,gate,r7baudit


def _clearance_screen(mesh: Path, outer_length_um:float,outer_diam_um:float)->dict:
    from scripts.discovery.run_m15_r5_mesh_verify import read_msh2
    nodes,tris=read_msh2(mesh)
    ids={i for tri in tris for i in tri}
    half_barrel=(outer_length_um-outer_diam_um)/2
    radius=outer_diam_um/2
    vals=[]
    for idx in ids:
        x,y,z=nodes[idx]
        axial=max(abs(x)-half_barrel,0.0)
        distance=math.hypot(axial, math.hypot(y,z))
        vals.append(radius-distance)
    # This compares core vertices against exact outer analytic capsule only.
    # Does NOT prove that independent triangulated boundaries are disjoint.
    return {'core_vertices':len(vals), 'min_core_vertex_to_analytic_outer_boundary_um':min(vals),
            'core_vertices_inside_outer_analytic_capsule':min(vals)>1e-5,
            'triangle_triangle_collision_verified':False}


def prepare(r4:Path,r5:Path,r7b:Path,au:Path,ag:Path,out:Path)->dict:
    r4,r5,r7b,au,ag,out=(x.resolve() for x in (r4,r5,r7b,au,ag,out))
    if out.exists():
        raise FileExistsError('OUTPUT_DIRECTORY_EXISTS_NO_OVERWRITE')
    plan,gate,audit=_load_sources(r4,r5,r7b)
    wavelength=plan['first_test_wavelength_nm']
    if not 500 <= wavelength <= 850:
        raise ValueError('PILOT_WAVELENGTH_OUTSIDE_500_850_NM')
    optical={}
    for mat,path in (('Au',au),('Ag',ag)):
        optical[mat]=source_to_eps(path.read_bytes(),mat,wavelength)
    shape=plan['geometry_um']
    clearance=_clearance_screen(r5/'inner_core.msh',shape['outer_length'],shape['outer_diameter'])
    if not clearance['core_vertices_inside_outer_analytic_capsule']:
        raise ValueError('CORE_VERTICES_NOT_IN_ANALYTIC_OUTER_CAPSULE')
    if out.exists():
        raise FileExistsError('OUTPUT_DIRECTORY_EXISTS_NO_OVERWRITE')
    out.mkdir(parents=True)
    for name in ('inner_core.msh','outer_shell.msh'):
        shutil.copyfile(r5/name,out/name)
    for mat,(converted,_) in optical.items():
        new_text(out/f'{mat}_eps.dat',converted)
    bg='VACUUM' if plan['medium']=='air' else 'CONST_EPS_1.7689'
    # Nested OBJECT syntax is supported by SCUFF-EM; outer object before inner.
    geo=(f'# M15-R8A Au/Ag, exploratory only; source R4 {sha256(r4/"M15R4_BEM_CANDIDATE_PLAN.json")}\n'
         f'REGION Exterior MATERIAL {bg}\n'
         f'OBJECT OuterShell\n  MESHFILE outer_shell.msh\n  MATERIAL FILE_{plan["shell_material"]}_eps.dat\nENDOBJECT\n'
         f'OBJECT InnerCore\n  MESHFILE inner_core.msh\n  MATERIAL FILE_{plan["core_material"]}_eps.dat\nENDOBJECT\n')
    new_text(out/'nanorod.scuffgeo',geo)
    pol=(1,0,0) if plan['field_component']=='Ex' else (0,1,0)
    new_text(out/'scuff.args',f'geometry nanorod.scuffgeo\nLambda {wavelength/1000:.12g}\n'
             f'pwDirection 0 0 1\npwPolarization {pol[0]} {pol[1]} {pol[2]}\nPFTFile nanorod.PFT\n')
    infiles=('inner_core.msh','outer_shell.msh','Au_eps.dat','Ag_eps.dat','nanorod.scuffgeo','scuff.args')
    receipt={'schema_version':VERSION,'status':'AU_AG_SCUFF_INPUT_PREPARED_UNVALIDATED',
             'input_hashes':{name:sha256(out/name) for name in infiles},
             'source_files':{mat:optical[mat][1] for mat in ('Au','Ag')},
             'r4':str(r4),'r5':str(r5),'r7b':str(r7b),
             'r4_plan_sha256':sha256(r4/'M15R4_BEM_CANDIDATE_PLAN.json'),
             'r5_gate_sha256':sha256(r5/'M15R5_MESH_GATE.json'),
             'r7b_audit_sha256':sha256(r7b),
             'material_assignment':{'outer':plan['shell_material'],'inner':plan['core_material'], 'exterior':bg},
             'wavelength_nm':wavelength,'polarization':plan['field_component'],
             'incident_direction':[0,0,1],'clearance_screen':clearance,
             'model_limitations':['Bulk-film nk from Johnson-Christy not nanoparticle- or interface-specific',
                                  'Source data frequency interpolation / dielectric dispersion model mismatch with Meep possible',
                                  'Discrete triangle-triangle shell clearance unverified',
                                  'Single mesh; no Au/Ag mesh or material dataset convergence',
                                  'PFT components are not yet converted to nanoparticle scattering cross section',
                                  'SCUFF result alone does not predict SERS molecular band ratios'],
             'au_ag_science_gate':'BLOCKED','science_backend_qualified':False,
             'physics_claim_authority':False,'novelty_claim_authority':False}
    new_json(out/'M15R8A_PLAN.json',receipt)
    return {'status':receipt['status'],'material_samples':{k:v[1]['data_points'] for k,v in optical.items()},
            'core':plan['core_material'],'shell':plan['shell_material'],'lambda_nm':wavelength,
            'science_gate':'BLOCKED','scuff_runs':0,'out_dir':str(out)}


def verify(out:Path)->dict:
    out=out.resolve()
    plan=json.loads((out/'M15R8A_PLAN.json').read_text())
    if (plan.get('schema_version')!=VERSION or plan.get('science_backend_qualified') is not False
        or plan.get('au_ag_science_gate')!='BLOCKED'):
        raise ValueError('BAD_PLAN')
    for n,digest in plan['input_hashes'].items():
        if sha256(out/n)!=digest:
            raise ValueError('INPUT_MODIFIED_'+n)
    for folder,fn,key in (('r4','M15R4_BEM_CANDIDATE_PLAN.json','r4_plan_sha256'),
                          ('r5','M15R5_MESH_GATE.json','r5_gate_sha256')):
        if sha256(Path(plan[folder])/fn)!=plan[key]:
            raise ValueError('PROVENANCE_CHANGED_'+folder)
    if sha256(Path(plan['r7b']))!=plan['r7b_audit_sha256']:
        raise ValueError('R7B_PROVENANCE_CHANGED')
    return plan


def run(out:Path,scuff:str,timeout:int,threads:int)->dict:
    if not 30 <= timeout <= 3600:raise ValueError('TIMEOUT_OUT_OF_RANGE')
    if not 1 <= threads <= 8:raise ValueError('THREAD_COUNT_OUT_OF_RANGE')
    out=out.resolve();plan=verify(out)
    binary=executable(scuff)
    if (out/'M15R8A_RUN.json').exists() or (out/'nanorod.PFT').exists():
        raise FileExistsError('SOLVER_ALREADY_RUN_NO_RETRY_OR_OVERWRITE')
    cmd=[binary,'--geometry','nanorod.scuffgeo','--Lambda',f'{plan["wavelength_nm"]/1000:.12g}',
         '--pwDirection','0','0','1','--pwPolarization',
         '1' if plan['polarization']=='Ex' else '0',
         '0' if plan['polarization']=='Ex' else '1','0','--PFTFile','nanorod.PFT']
    env=os.environ.copy()
    env['OMP_NUM_THREADS']=str(threads)
    env['SCUFF_MATPROP_PATH']=str(out) + (':' + env['SCUFF_MATPROP_PATH'] if env.get('SCUFF_MATPROP_PATH') else '')
    env['SCUFF_MESH_PATH']=str(out) + (':' + env['SCUFF_MESH_PATH'] if env.get('SCUFF_MESH_PATH') else '')
    try:
        proc=subprocess.run(cmd,cwd=out,env=env,capture_output=True,text=True,timeout=timeout,check=False)
        rc,timedout,stdout,stderr=proc.returncode,False,proc.stdout,proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc,timedout=None,True
        stdout=(exc.stdout or b'').decode(errors='replace') if isinstance(exc.stdout,bytes) else (exc.stdout or '')
        stderr=(exc.stderr or b'').decode(errors='replace') if isinstance(exc.stderr,bytes) else (exc.stderr or '')
    new_text(out/'scuff.stdout.log',stdout)
    new_text(out/'scuff.stderr.log',stderr)
    pft=out/'nanorod.PFT'
    success=(rc==0 and not timedout and pft.exists() and pft.stat().st_size>0)
    report={'schema_version':VERSION,'status':'AU_AG_PFT_GENERATED_UNCALIBRATED' if success else 'SCUFF_RUN_FAILED_REVIEW_LOGS',
            'exit_code':rc,'timed_out':timedout,'command':cmd,'threads':threads,
            'input_plan_sha256':sha256(out/'M15R8A_PLAN.json'),
            'pft_sha256':sha256(pft) if success else None,
            'pft_preview':pft.read_text(errors='replace')[:2400] if success else None,
            'au_ag_science_gate':'BLOCKED','science_backend_qualified':False,
            'quantitative_scattering_qualified':False,'scientific_novelty_certified':False}
    new_json(out/'M15R8A_RUN.json',report)
    return {'status':report['status'],'returncode':rc,'timed_out':timedout,
            'pft_present':success,'science_gate':'BLOCKED','report':str(out/'M15R8A_RUN.json')}


def check(out:Path)->dict:
    plan=verify(out)
    report=out/'M15R8A_RUN.json'
    state='NOT_RUN'
    if report.exists():
        data=json.loads(report.read_text())
        if (data.get('input_plan_sha256')!=sha256(out/'M15R8A_PLAN.json')
            or data.get('au_ag_science_gate')!='BLOCKED' or data.get('science_backend_qualified') is not False):
            raise ValueError('RUN_RECEIPT_MISMATCH')
        if data['pft_sha256'] and sha256(out/'nanorod.PFT')!=data['pft_sha256']:
            raise ValueError('PFT_MODIFIED')
        state=data['status']
    return {'status':'M15R8A_CHECK_OK','solver':state,
            'material_source_pinned':True,'science_gate':plan['au_ag_science_gate']}


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    s=ap.add_subparsers(dest='cmd',required=True)
    p=s.add_parser('prepare');p.add_argument('--r4-dir',type=Path,required=True)
    p.add_argument('--r5-dir',type=Path,required=True);p.add_argument('--r7b-audit',type=Path,required=True)
    p.add_argument('--au-yml',type=Path,required=True);p.add_argument('--ag-yml',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p=s.add_parser('run');p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--scuff-scatter',required=True);p.add_argument('--timeout-seconds',type=int,default=1200)
    p.add_argument('--threads',type=int,default=2)
    p=s.add_parser('check');p.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args()
    try:
        if args.cmd=='prepare':
            result=prepare(args.r4_dir,args.r5_dir,args.r7b_audit,args.au_yml,args.ag_yml,args.out_dir)
        elif args.cmd=='run':
            result=run(args.out_dir,args.scuff_scatter,args.timeout_seconds,args.threads)
        else:result=check(args.out_dir)
        print(json.dumps(result,indent=2,ensure_ascii=False))
    except (ValueError,FileExistsError,KeyError,OSError,AssertionError,subprocess.TimeoutExpired,json.JSONDecodeError) as exc:
        ap.exit(2,'M15R8A_ERROR: '+str(exc)+'\n')

if __name__=='__main__':main()
