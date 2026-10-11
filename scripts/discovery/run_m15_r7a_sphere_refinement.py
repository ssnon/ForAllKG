"""M15-R7A: independently re-mesh an R6 dielectric sphere and compare BEM to Mie.

Explicit steps: prepare -> run -> audit -> check. No automatic expensive solver
runs and no Au/Ag science qualification. Requires prior immutable R6/R6B results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from scripts.discovery import run_m15_r6_scuff_control as r6
from scripts.discovery import run_m15_r6b_pft_calibration as r6b
from scripts.discovery import run_m15_r5_mesh_verify as r5

VERSION = 'm15r7a-sphere-refinement-v1'
_ALLOWED_EDGE_NM = {8.0, 6.0}
_FILES = ('sphere.geo', 'sphere.scuffgeo', 'sphere.msh')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jwrite(path: Path, obj: dict) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')


def _baseline(r6b_dir: Path) -> tuple[dict, dict]:
    """Use R6B verified report; do not trust injected or altered baselines."""
    b = r6b.check(r6b_dir)
    if b['science_gate'] != 'BLOCKED' or b['mesh_convergence_demonstrated'] is not False:
        raise ValueError('R6B_AUTHORITY_INVALID')
    base = json.loads((r6b_dir / 'M15R6B_SINGLE_MESH_GATE.json').read_text())
    if base['sphere_triangle_count'] <= 0 or base['mesh_edge_um'] != r6.EDGE_UM:
        raise ValueError('R6B_EXPECTED_12NM_BASELINE')
    if base['comparison']['relative_error'] < 0 or not math.isfinite(base['comparison']['relative_error']):
        raise ValueError('INVALID_BASELINE_ERROR')
    return b, base


def _load_plan(out: Path) -> dict:
    p = json.loads((out / 'M15R7A_PLAN.json').read_text())
    if p.get('schema_version') != VERSION or p.get('science_gate') != 'BLOCKED':
        raise ValueError('INVALID_R7A_PLAN')
    if p['mesh_edge_nm'] not in _ALLOWED_EDGE_NM:
        raise ValueError('R7A_EDGE_NOT_ALLOWED')
    _baseline(Path(p['r6b_dir']))
    if sha(Path(p['r6b_dir']) / 'M15R6B_SINGLE_MESH_GATE.json') != p['r6b_sha256']:
        raise ValueError('BASELINE_CHANGED')
    for name in _FILES:
        if sha(out / name) != p['input_sha256'][name]:
            raise ValueError('R7A_INPUT_CHANGED_' + name)
    nodes, tris = r5.read_msh2(out / 'sphere.msh')
    candidate = r5.audit(nodes, tris, 2*r6.RADIUS_UM, 2*r6.RADIUS_UM,
                          p['mesh_edge_nm']/1000.0)
    if candidate != p['mesh_audit'] or candidate['status'] != 'MESH_TOPOLOGY_PASS_PRELIMINARY':
        raise ValueError('R7A_MESH_AUDIT_CHANGED_OR_FAILED')
    if candidate['faces'] <= p['baseline_triangles']:
        raise ValueError('REFINEMENT_DID_NOT_ADD_FACES')
    return p


def prepare(r6b_dir: Path, out: Path, gmsh: str, edge_nm: float, timeout: int) -> dict:
    if edge_nm not in _ALLOWED_EDGE_NM:
        raise ValueError('PREDECLARED_EDGE_MUST_BE_8_OR_6_NM')
    if not 10 <= timeout <= 600:
        raise ValueError('MESH_TIMEOUT_RANGE')
    r6b_dir, out = r6b_dir.resolve(), out.resolve()
    _b, b = _baseline(r6b_dir)
    exe = r6.executable(gmsh)
    if out.exists():
        raise FileExistsError('OUTPUT_DIR_EXISTS_NO_OVERWRITE')
    out.mkdir(parents=True)
    edge = edge_nm/1000.0
    geo = (f'// M15-R7A: dielectric sphere radius {r6.RADIUS_UM} um, edge {edge_nm} nm\n'
           'SetFactory("OpenCASCADE");\n'
           f'Mesh.MeshSizeMin={edge:.12g};\nMesh.MeshSizeMax={edge:.12g};\n'
           'Mesh.CharacteristicLengthFromCurvature=0;\n'
           'Mesh.CharacteristicLengthFromPoints=0;\n'
           'Mesh.MshFileVersion=2.2;\n'
           f'Sphere(1)={{0,0,0,{r6.RADIUS_UM:.12g}}};\n'
           'boundary[]=Boundary{Volume{1};};\n'
           'Physical Surface(101)={boundary[]};\n')
    (out/'sphere.geo').write_text(geo)
    # Retain SCUFF geometry/source conventions exactly to isolate the mesh variable.
    source = Path(b['r6_dir'])
    (out/'sphere.scuffgeo').write_bytes((source/'sphere.scuffgeo').read_bytes())
    cmd = [exe, '-2', '-format', 'msh2', '-v', '2', '-o', str(out/'sphere.msh'), str(out/'sphere.geo')]
    proc = subprocess.run(cmd, cwd=out, capture_output=True, text=True, timeout=timeout, check=False)
    (out/'gmsh.stdout.log').write_text(proc.stdout)
    (out/'gmsh.stderr.log').write_text(proc.stderr)
    if proc.returncode or not (out/'sphere.msh').is_file():
        raise ValueError('GMSH_FAILED_CHECK_LOGS')
    nodes, tris = r5.read_msh2(out/'sphere.msh')
    topo = r5.audit(nodes, tris, 2*r6.RADIUS_UM, 2*r6.RADIUS_UM, edge)
    if topo['status'] != 'MESH_TOPOLOGY_PASS_PRELIMINARY':
        raise ValueError('REFINED_TOPOLOGY_NOT_VALID')
    if topo['faces'] <= b['sphere_triangle_count']:
        raise ValueError('REFINEMENT_DID_NOT_ADD_FACES')
    plan = {'schema_version':VERSION, 'status':'REFINED_MESH_PREPARED_NO_SCUFF',
            'science_gate':'BLOCKED', 'nanorod_science_gate':'BLOCKED',
            'r6b_dir':str(r6b_dir),'r6b_sha256':sha(r6b_dir/'M15R6B_SINGLE_MESH_GATE.json'),
            'mesh_edge_nm':edge_nm, 'baseline_mesh_edge_nm':b['mesh_edge_um']*1000,
            'baseline_triangles':b['sphere_triangle_count'],
            'baseline_reference_error_pct':b['comparison']['relative_error_pct'],
            'mesh_audit':topo,
            'input_sha256':{name:sha(out/name) for name in _FILES},
            'gmsh_version':subprocess.run([exe,'-version'],capture_output=True,text=True,timeout=30,check=False).stdout.strip(),
            'limitations': ['Sphere benchmark only', 'single refinement does not demonstrate asymptotic convergence',
                            'fixed source, wavelength and material', 'no Au/Ag dispersion or SERS claims']}
    jwrite(out/'M15R7A_PLAN.json', plan)
    return {'status':plan['status'],'baseline_triangles':b['sphere_triangle_count'],
            'refined_triangles':topo['faces'],'edge_nm':edge_nm,'scuff_runs':0,'science_gate':'BLOCKED'}


def run(out: Path, scuff: str, timeout: int) -> dict:
    if not 30 <= timeout <= 3600:
        raise ValueError('SCUFF_TIMEOUT_RANGE')
    out = out.resolve()
    plan = _load_plan(out)
    if (out/'M15R7A_RUN.json').exists() or (out/'sphere.PFT').exists():
        raise FileExistsError('SOLUTION_ALREADY_EXISTS_NO_OVERWRITE')
    exe = r6.executable(scuff)
    cmd = [exe, '--geometry', 'sphere.scuffgeo', '--Lambda', str(r6.LAMBDA_UM),
           '--pwDirection','0','0','1', '--pwPolarization','1','0','0', '--PFTFile','sphere.PFT']
    timed_out = False
    try:
        proc = subprocess.run(cmd,cwd=out,capture_output=True,text=True,timeout=timeout,check=False)
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or b''; stderr = exc.stderr or b''; returncode = None
        if isinstance(stdout,bytes): stdout=stdout.decode(errors='replace')
        if isinstance(stderr,bytes): stderr=stderr.decode(errors='replace')
    (out/'scuff.stdout.log').write_text(stdout)
    (out/'scuff.stderr.log').write_text(stderr)
    ok = not timed_out and returncode == 0 and (out/'sphere.PFT').is_file() and (out/'sphere.PFT').stat().st_size > 0
    rec = {'schema_version':VERSION,'status':'REFINED_SCUFF_PFT_PRODUCED_UNCALIBRATED' if ok else 'SCUFF_FAILED',
           'exit_code':returncode,'timed_out':timed_out,'command':cmd,'executable':exe,
           'input_plan_sha256':sha(out/'M15R7A_PLAN.json'),
           'pft_sha256':sha(out/'sphere.PFT') if ok else None,
           'science_gate':'BLOCKED','au_ag_solver_runs':0,'novelty_certified':False}
    jwrite(out/'M15R7A_RUN.json',rec)
    return {'status':rec['status'],'returncode':returncode,'timed_out':timed_out,
            'refined_mesh_triangles':plan['mesh_audit']['faces'],'science_gate':'BLOCKED'}


def audit(out: Path) -> dict:
    out = out.resolve();plan = _load_plan(out)
    if (out/'M15R7A_AUDIT.json').exists():
        raise FileExistsError('AUDIT_ALREADY_EXISTS_NO_OVERWRITE')
    rec = json.loads((out/'M15R7A_RUN.json').read_text())
    if rec.get('status') != 'REFINED_SCUFF_PFT_PRODUCED_UNCALIBRATED' or rec.get('exit_code') != 0:
        raise ValueError('SCUFF_RUN_NOT_SUCCESSFUL')
    if rec.get('science_gate') != 'BLOCKED' or rec.get('input_plan_sha256') != sha(out/'M15R7A_PLAN.json'):
        raise ValueError('RUN_RECEIPT_INVALID')
    if rec.get('pft_sha256') != sha(out/'sphere.PFT'):
        raise ValueError('PFT_CHANGED')
    cmd=rec['command']; expected=['--pwPolarization','1','0','0']
    if '--pwPolarization' not in cmd or cmd[cmd.index('--pwPolarization'):cmd.index('--pwPolarization')+4]!=expected:
        raise ValueError('SOURCE_NORMALIZATION_NOT_CONFIRMED')
    pft=r6b.parse_pft((out/'sphere.PFT').read_text(),expected_omega=2*math.pi/r6.LAMBDA_UM)
    mie=r6.mie_sphere_scattering()
    comp=r6b.compare(pft,mie)
    _, b=_baseline(Path(plan['r6b_dir']))
    base=b['comparison']; change=abs(comp['scuff_sigma_sca_um2']-base['scuff_sigma_sca_um2'])/mie
    improved=bool(comp['relative_error']<base['relative_error'])
    result={'schema_version':VERSION,'status':'REFINED_MIE_COMPARISON_RECORDED',
            'science_gate':'BLOCKED','nanorod_science_gate':'BLOCKED',
            'baseline_triangles':plan['baseline_triangles'],'refined_triangles':plan['mesh_audit']['faces'],
            'baseline_edge_nm':plan['baseline_mesh_edge_nm'],'refined_edge_nm':plan['mesh_edge_nm'],
            'mie_reference_um2':mie,'baseline_scuff_um2':base['scuff_sigma_sca_um2'],
            'refined_scuff_um2':comp['scuff_sigma_sca_um2'],
            'baseline_reference_error_pct':base['relative_error_pct'],
            'refined_reference_error_pct':comp['relative_error_pct'],
            'normalized_coarse_fine_difference_pct':100*change,
            'refinement_reduced_mie_error':improved,
            'pabs_over_psca_absolute':comp['pabs_over_psca_absolute'],
            'refinement_trend':'TOWARD_MIE' if improved else 'NOT_TOWARD_MIE',
            'mesh_convergence_demonstrated':False,
            'science_backend_qualified':False,'scientific_novelty_certified':False,
            'plan_sha256':sha(out/'M15R7A_PLAN.json'),
            'run_sha256':sha(out/'M15R7A_RUN.json'),
            'pft_sha256':sha(out/'sphere.PFT'),
            'next_action':'Review refinement trend and decide whether to run a third independent mesh; do not authorize Au/Ag from sphere result.'}
    jwrite(out/'M15R7A_AUDIT.json',result)
    report=('# M15-R7A: SCUFF dielectric sphere mesh refinement\n\n'
            f'- Coarse: {plan["baseline_mesh_edge_nm"]:.3g} nm, {plan["baseline_triangles"]} triangles, error {base["relative_error_pct"]:.5g}%\n'
            f'- Refined: {plan["mesh_edge_nm"]:.3g} nm, {plan["mesh_audit"]["faces"]} triangles, error {comp["relative_error_pct"]:.5g}%\n'
            f'- Change / Mie reference: {100*change:.5g}%\n'
            f'- Lossless residual |Pabs|/Pscat: {comp["pabs_over_psca_absolute"]:.5g}\n'
            f'- Error trend: {result["refinement_trend"]}\n\n'
            '**Neither an asymptotic convergence proof nor Au/Ag qualification.**\n')
    (out/'M15R7A_REPORT.md').write_text(report)
    return {'status':result['status'],'coarse_error_pct':base['relative_error_pct'],
            'refined_error_pct':comp['relative_error_pct'],'refinement_trend':result['refinement_trend'],
            'mesh_convergence_demonstrated':False,'nanorod_science_gate':'BLOCKED'}


def check(out: Path) -> dict:
    out=out.resolve();plan=_load_plan(out)
    response={'status':'M15R7A_CHECK_OK','mesh_triangles':plan['mesh_audit']['faces'],
              'science_gate':'BLOCKED','solver_run':'NOT_RUN'}
    rp=out/'M15R7A_RUN.json'
    if rp.exists():
        run_data=json.loads(rp.read_text())
        if run_data.get('input_plan_sha256')!=sha(out/'M15R7A_PLAN.json') or run_data.get('science_gate')!='BLOCKED':
            raise ValueError('RUN_PROVENANCE_INVALID')
        if run_data['pft_sha256'] is not None and sha(out/'sphere.PFT')!=run_data['pft_sha256']:
            raise ValueError('PFT_PROVENANCE_CHANGED')
        response['solver_run']=run_data['status']
    a=out/'M15R7A_AUDIT.json'
    if a.exists():
        result=json.loads(a.read_text())
        if result.get('schema_version')!=VERSION or result.get('science_backend_qualified') is not False:
            raise ValueError('AUDIT_AUTHORITY_INVALID')
        for name,key in [('M15R7A_PLAN.json','plan_sha256'),('M15R7A_RUN.json','run_sha256'),('sphere.PFT','pft_sha256')]:
            if sha(out/name)!=result[key]: raise ValueError('AUDIT_SOURCE_CHANGED')
        pft=r6b.parse_pft((out/'sphere.PFT').read_text(),expected_omega=2*math.pi/r6.LAMBDA_UM)
        comp=r6b.compare(pft,r6.mie_sphere_scattering())
        if not math.isclose(comp['scuff_sigma_sca_um2'],result['refined_scuff_um2'],rel_tol=1e-12):
            raise ValueError('REFINED_RESULT_NOT_REPRODUCIBLE')
        response['refined_error_pct']=comp['relative_error_pct']
    return response


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare');p.add_argument('--r6b-dir',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path);p.add_argument('--gmsh',required=True)
    p.add_argument('--mesh-edge-nm',type=float,default=8.0,choices=[8.0,6.0]);p.add_argument('--timeout-seconds',type=int,default=180)
    p=sub.add_parser('run');p.add_argument('--out-dir',required=True,type=Path)
    p.add_argument('--scuff-scatter',required=True);p.add_argument('--timeout-seconds',type=int,default=900)
    p=sub.add_parser('audit');p.add_argument('--out-dir',required=True,type=Path)
    p=sub.add_parser('check');p.add_argument('--out-dir',required=True,type=Path)
    args=ap.parse_args()
    try:
        if args.cmd=='prepare':ans=prepare(args.r6b_dir,args.out_dir,args.gmsh,args.mesh_edge_nm,args.timeout_seconds)
        elif args.cmd=='run':ans=run(args.out_dir,args.scuff_scatter,args.timeout_seconds)
        elif args.cmd=='audit':ans=audit(args.out_dir)
        else:ans=check(args.out_dir)
        print(json.dumps(ans,indent=2,ensure_ascii=False))
    except (ValueError,FileNotFoundError,FileExistsError,OSError,subprocess.TimeoutExpired,KeyError,IndexError,ImportError) as exc:
        ap.exit(2,f'M15R7A_ERROR: {exc}\n')

if __name__=='__main__':main()
