"""M15-R6: execute a separate SCUFF-EM *sphere* positive control before Au/Ag runs.

This is a controlled source/mesh/runtime handoff, NOT science-qualification.
- Only the dielectric sphere benchmark is executable here.
- Uses R5's immutable geometry provenance, never runs the Au/Ag mesh.
- Does NOT convert SCUFF's scattered power into scattering cross section.
- Never supplies fabricated metal dispersion data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

VERSION = "m15r6-scuff-sphere-control-v1"
RADIUS_UM = 0.05
LAMBDA_UM = 0.633
EPSILON = 2.25
EDGE_UM = 0.012


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_write(path: Path, content: str) -> None:
    with path.open('x',encoding='utf-8') as f:
        f.write(content)


def json_new(path: Path, data: object) -> None:
    safe_write(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)+'\n')


def executable(name: str) -> str:
    which = shutil.which(name) if '/' not in name else str(Path(name).resolve())
    if not which or not Path(which).is_file() or not Path(which).stat().st_mode & 0o111:
        raise ValueError('EXECUTABLE_NOT_FOUND_'+name)
    return which


def mie_sphere_scattering(radius: float=RADIUS_UM, wavelength: float=LAMBDA_UM, epsilon: float=EPSILON) -> float:
    """Independent (analytic spherical-wave series) Mie cross section in um^2."""
    from scipy.special import spherical_jn, spherical_yn
    x=2*math.pi*radius/wavelength
    m=math.sqrt(epsilon)
    mx=m*x
    nmax=max(6,int(math.ceil(x+4*x**(1/3)+2)))
    q=0.0
    for n in range(1,nmax+1):
        jx=spherical_jn(n,x);jmx=spherical_jn(n,mx)
        djx=spherical_jn(n,x,derivative=True);djmx=spherical_jn(n,mx,derivative=True)
        yx=spherical_yn(n,x);dyx=spherical_yn(n,x,derivative=True)
        psix=x*jx; dpsix=jx+x*djx
        psimx=mx*jmx; dpsimx=jmx+mx*djmx
        xix=x*complex(jx,yx); dxix=complex(jx+x*djx,yx+x*dyx)
        an=(m*psimx*dpsix-psix*dpsimx)/(m*psimx*dxix-xix*dpsimx)
        bn=(psimx*dpsix-m*psix*dpsimx)/(psimx*dxix-m*xix*dpsimx)
        q+=(2*n+1)*(abs(an)**2+abs(bn)**2)
    return math.pi*radius**2*2*q/x**2


def validate_r5(r5: Path) -> dict:
    # Existing R5 check checks original R4 request/mesh hashes and recomputes topology.
    from scripts.discovery.run_m15_r5_mesh_verify import check as check_r5
    outcome=check_r5(r5)
    if outcome['mesh_topology']!='MESH_TOPOLOGY_PASS_PRELIMINARY':
        raise ValueError('R5_MESH_GATE_NOT_PASSED')
    gate=json.loads((r5/'M15R5_MESH_GATE.json').read_text())
    if gate.get('science_backend_qualified') is not False or gate.get('nanorod_science_gate')!='BLOCKED':
        raise ValueError('R5_SCIENCE_GATE_CHANGED')
    return gate


def prepare(r5: Path,out: Path, gmsh: str, timeout: int) -> dict:
    if not 5<=timeout<=1800: raise ValueError('TIMEOUT_OUT_OF_RANGE')
    r5=r5.resolve();out=out.resolve()
    gate=validate_r5(r5)
    exe=executable(gmsh)
    if out.exists(): raise FileExistsError('R6_OUTPUT_ALREADY_EXISTS_NO_OVERWRITE')
    out.mkdir(parents=True)
    geo=(f'// M15-R6 dielectric sphere, um units; analytic positive control\n'
         'SetFactory("OpenCASCADE");\n'
         f'Mesh.MeshSizeMin={EDGE_UM:.12g};\nMesh.MeshSizeMax={EDGE_UM:.12g};\n'
         'Mesh.MshFileVersion=2.2;\n'
         f'Sphere(1)={{0,0,0,{RADIUS_UM:.12g}}};\n'
         'boundary[]=Boundary{Volume{1};};\n'
         'Physical Surface(101)={boundary[]};\n')
    safe_write(out/'sphere.geo',geo)
    safe_write(out/'sphere.scuffgeo',
        '# Dielectric-sphere control ONLY. Does not use Au/Ag optical constants.\n'
        f'OBJECT TestSphere\n MESHFILE sphere.msh\n MATERIAL CONST_EPS_{EPSILON:g}\nENDOBJECT\n')
    safe_write(out/'scuff.args',
        'geometry sphere.scuffgeo\n'
        f'Lambda {LAMBDA_UM:.12g}\n'
        'pwDirection 0 0 1\n'
        'pwPolarization 1 0 0\n'
        'PFTFile sphere.PFT\n')
    target=out/'sphere.msh'
    cmd=[exe,'-2','-format','msh2','-v','2','-o',str(target),str(out/'sphere.geo')]
    proc=subprocess.run(cmd,cwd=out,capture_output=True,text=True,timeout=timeout,check=False)
    (out/'gmsh.stdout.log').write_text(proc.stdout)
    (out/'gmsh.stderr.log').write_text(proc.stderr)
    if proc.returncode !=0 or not target.is_file():
        raise ValueError('R6_GMSH_FAILED_SEE_LOGS')
    # Reuse actual R5 triangle parser & audit. Validate closed shell, approximate volume.
    from scripts.discovery.run_m15_r5_mesh_verify import read_msh2, audit
    vertices,faces=read_msh2(target)
    sphere_audit=audit(vertices,faces,2*RADIUS_UM,2*RADIUS_UM,EDGE_UM)
    if sphere_audit['status']!='MESH_TOPOLOGY_PASS_PRELIMINARY':
        raise ValueError('R6_SPHERE_TOPOLOGY_INVALID')
    data={
       'schema_version':VERSION,
       'status':'SPHERE_POSITIVE_CONTROL_PREPARED_UNCALIBRATED',
       'r5_dir':str(r5), 'r5_gate_sha256':digest(r5/'M15R5_MESH_GATE.json'),
       'nanorod_science_gate':'BLOCKED', 'au_ag_solver_runs':0,
       'sphere_parameters':{'radius_um':RADIUS_UM,'lambda_vacuum_um':LAMBDA_UM,'epsilon_relative':EPSILON,
                            'mesh_edge_um':EDGE_UM},
       'sphere_analytic_mie_scattering_um2':mie_sphere_scattering(),
       'sphere_mesh_audit':sphere_audit,
       'input_sha256':{name:digest(out/name) for name in ('sphere.geo','sphere.scuffgeo','scuff.args','sphere.msh')},
       'scuff_pft_to_cross_section_calibrated':False,
       'model_validated':False, 'scientific_novelty_certified':False,
       'caveat':'SCUFF PFT reports powers, not directly a scattering cross-section in um^2; normalization and mesh convergence require independent review.'}
    json_new(out/'M15R6_CONTROL_PLAN.json',data)
    return {'status':data['status'],'triangle_count':len(faces),'mie_reference_um2':data['sphere_analytic_mie_scattering_um2'],
            'scuff_scatter_available':shutil.which('scuff-scatter') is not None,
            'nanorod_science_gate':'BLOCKED','out_dir':str(out)}


def verify(out: Path) -> dict:
    out=out.resolve();p=out/'M15R6_CONTROL_PLAN.json'
    data=json.loads(p.read_text())
    if data.get('schema_version')!=VERSION or data.get('nanorod_science_gate')!='BLOCKED' or data.get('model_validated') is not False:
        raise ValueError('R6_PLAN_INVALID')
    r5=Path(data['r5_dir']);validate_r5(r5)
    if digest(r5/'M15R5_MESH_GATE.json')!=data['r5_gate_sha256']:
        raise ValueError('R5_GATE_MODIFIED')
    for name,sha in data['input_sha256'].items():
        if name not in ('sphere.geo','sphere.scuffgeo','scuff.args','sphere.msh') or digest(out/name)!=sha:
            raise ValueError('R6_INPUT_CHANGED_'+name)
    if not math.isclose(mie_sphere_scattering(),data['sphere_analytic_mie_scattering_um2'],rel_tol=1e-12):
        raise ValueError('MIE_REFERENCE_CHANGED')
    return data


def run_scuff(out:Path, scuff:str, timeout:int) -> dict:
    if not 10<=timeout<=3600:raise ValueError('TIMEOUT_OUT_OF_RANGE')
    out=out.resolve();verify(out)
    exe=executable(scuff)
    result_file=out/'M15R6_SCUFF_RUN.json'
    if result_file.exists() or (out/'sphere.PFT').exists():
        raise FileExistsError('SCUFF_ALREADY_RUN_DO_NOT_OVERWRITE')
    # Argument-file mode is documented in scuff-scatter examples: '< Args'.
    # Use actual CLI options without shell to ensure no injection.
    cmd=[exe,'--geometry','sphere.scuffgeo','--Lambda',str(LAMBDA_UM),
         '--pwDirection','0','0','1','--pwPolarization','1','0','0','--PFTFile','sphere.PFT']
    try:
        proc=subprocess.run(cmd,cwd=out,capture_output=True,text=True,timeout=timeout,check=False)
        code=proc.returncode;timedout=False
        stdout=proc.stdout;stderr=proc.stderr
    except subprocess.TimeoutExpired as exc:
        code=None;timedout=True
        stdout=(exc.stdout or b'').decode(errors='replace') if isinstance(exc.stdout,bytes) else (exc.stdout or '')
        stderr=(exc.stderr or b'').decode(errors='replace') if isinstance(exc.stderr,bytes) else (exc.stderr or '')
    (out/'scuff.stdout.log').write_text(stdout)
    (out/'scuff.stderr.log').write_text(stderr)
    success=(not timedout and code==0 and (out/'sphere.PFT').is_file() and (out/'sphere.PFT').stat().st_size>0)
    status='SCUFF_PFT_PRODUCED_UNCALIBRATED' if success else 'SCUFF_EXECUTION_FAILED_REVIEW_LOGS'
    receipt={'schema_version':VERSION,'status':status,'command':cmd,'executable':exe,
             'exit_code':code,'timed_out':timedout,'pft_sha256':digest(out/'sphere.PFT') if success else None,
             'pft_preview':(out/'sphere.PFT').read_text(errors='replace')[:2500] if success else None,
             'input_plan_sha256':digest(out/'M15R6_CONTROL_PLAN.json'),
             'nanorod_science_gate':'BLOCKED','pft_to_cross_section_calibrated':False,
             'science_backend_qualified':False,'model_validated':False}
    json_new(result_file,receipt)
    return {'status':status,'scuff_returncode':code,'timed_out':timedout,'pft_present':success,
            'nanorod_science_gate':'BLOCKED','report':str(result_file)}


def check(out:Path) -> dict:
    data=verify(out)
    report=out/'M15R6_SCUFF_RUN.json'
    output={'status':'M15R6_PLAN_CHECK_OK','sphere_mesh_triangles':data['sphere_mesh_audit']['faces'],
            'sphere_mie_reference_um2':data['sphere_analytic_mie_scattering_um2'],
            'nanorod_science_gate':'BLOCKED','scuff_run':'NOT_RUN'}
    if report.is_file():
        r=json.loads(report.read_text())
        if r['input_plan_sha256']!=digest(out/'M15R6_CONTROL_PLAN.json') or r.get('science_backend_qualified') is not False:
            raise ValueError('R6_SCUFF_RECEIPT_INVALID')
        if r['pft_sha256'] and digest(out/'sphere.PFT')!=r['pft_sha256']:
            raise ValueError('PFT_RESULT_CHANGED')
        output['scuff_run']=r['status']
    return output


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--r5-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--gmsh',default='gmsh');p.add_argument('--timeout-seconds',type=int,default=180)
    p=sub.add_parser('run-control');p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--scuff-scatter',default='scuff-scatter');p.add_argument('--timeout-seconds',type=int,default=900)
    p=sub.add_parser('check');p.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args()
    try:
        if args.command=='prepare':result=prepare(args.r5_dir,args.out_dir,args.gmsh,args.timeout_seconds)
        elif args.command=='run-control':result=run_scuff(args.out_dir,args.scuff_scatter,args.timeout_seconds)
        else:result=check(args.out_dir)
        print(json.dumps(result,indent=2,ensure_ascii=False))
    except (ValueError,AssertionError,FileNotFoundError,FileExistsError,KeyError,TimeoutError,OSError,ImportError,subprocess.SubprocessError,json.JSONDecodeError) as exc:
        ap.exit(2,f'M15R6_ERROR: {exc}\n')

if __name__=='__main__':
    main()
