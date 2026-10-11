"""M15-R7B: compare independent 12/8/6 nm SCUFF sphere benchmark results.

No solver / paid model calls. Fail closed on missing or inconsistent provenance.
Three-mesh *screen* is not a proof of convergence or Au/Ag authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.discovery import run_m15_r6b_pft_calibration as r6b
from scripts.discovery import run_m15_r6_scuff_control as r6
from scripts.discovery import run_m15_r7a_sphere_refinement as r7a

VERSION = 'm15r7b-sphere-3-mesh-screen-v1'
# Declared BEFORE the 6nm experiment. They are engineering screen thresholds only.
MAX_FINE_MIE_ERR_PCT = 2.0
MAX_FINE_MIDDLE_DELTA_MIE_PCT = 1.0
MAX_LOSSLESS_RESIDUAL = 1e-3


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_snapshot(r6b_dir: Path, mid_dir: Path, fine_dir: Path) -> dict:
    r6b_dir, mid_dir, fine_dir = map(lambda p: p.resolve(), (r6b_dir, mid_dir, fine_dir))
    for directory in (r6b_dir, mid_dir, fine_dir):
        if not directory.is_dir():
            raise FileNotFoundError('SOURCE_DIR_MISSING_' + str(directory))
    if len({str(r6b_dir),str(mid_dir),str(fine_dir)}) != 3:
        raise ValueError('SOURCE_DIRS_MUST_BE_DISTINCT')
    checked_base = r6b.check(r6b_dir)
    if checked_base['science_gate'] != 'BLOCKED':
        raise ValueError('R6B_GATE_NOT_BLOCKED')
    base = json.loads((r6b_dir/'M15R6B_SINGLE_MESH_GATE.json').read_text())
    if base.get('science_backend_qualified') is not False:
        raise ValueError('BASELINE_SCIENCE_AUTHORITY_INVALID')
    if not math.isclose(base['mesh_edge_um']*1000,12.0,rel_tol=0,abs_tol=1e-7):
        raise ValueError('BASELINE_NOT_12NM')
    base_comp = base['comparison']
    mie = float(base_comp['mie_sigma_sca_um2'])
    if not (math.isfinite(mie) and mie>0):
        raise ValueError('MIE_REFERENCE_INVALID')
    if not math.isclose(mie,r6.mie_sphere_scattering(),rel_tol=1e-10):
        raise ValueError('MIE_REFERENCE_CHANGED')
    b_sigma=float(base_comp['scuff_sigma_sca_um2'])
    if not math.isclose(base_comp['relative_error_pct'],abs(b_sigma-mie)/mie*100,rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError('BASELINE_ERROR_INCONSISTENT')

    inputs={'M15R6B_SINGLE_MESH_GATE.json':sha(r6b_dir/'M15R6B_SINGLE_MESH_GATE.json')}
    data=[]
    for expected, path, label in ((8.0,mid_dir,'middle'),(6.0,fine_dir,'fine')):
        chk=r7a.check(path)
        if chk.get('solver_run') != 'REFINED_SCUFF_PFT_PRODUCED_UNCALIBRATED':
            raise ValueError(label.upper()+'_RUN_NOT_SUCCESSFUL')
        plan=r7a._load_plan(path)
        a=json.loads((path/'M15R7A_AUDIT.json').read_text())
        rec=json.loads((path/'M15R7A_RUN.json').read_text())
        if not math.isclose(plan['mesh_edge_nm'],expected,rel_tol=0,abs_tol=1e-9):
            raise ValueError(label.upper()+'_EXPECTED_MESH_SIZE')
        if Path(plan['r6b_dir']).resolve()!=r6b_dir or plan['r6b_sha256']!=inputs['M15R6B_SINGLE_MESH_GATE.json']:
            raise ValueError(label.upper()+'_BASELINE_MISMATCH')
        if a['schema_version']!=r7a.VERSION or a['science_gate']!='BLOCKED' or a['science_backend_qualified'] is not False or a['mesh_convergence_demonstrated'] is not False:
            raise ValueError(label.upper()+'_AUTHORITY_INVALID')
        if rec['status']!='REFINED_SCUFF_PFT_PRODUCED_UNCALIBRATED' or rec['exit_code'] != 0 or rec['timed_out'] is not False:
            raise ValueError(label.upper()+'_RECEIPT_INVALID')
        if a['plan_sha256'] != sha(path/'M15R7A_PLAN.json') or a['run_sha256'] != sha(path/'M15R7A_RUN.json') or a['pft_sha256'] != sha(path/'sphere.PFT'):
            raise ValueError(label.upper()+'_AUDIT_HASH_MISMATCH')
        # The same SCUFF geometry, wavelength and incident polarization must be present.
        if sha(mid_dir/'sphere.scuffgeo') != sha(fine_dir/'sphere.scuffgeo'):
            raise ValueError('SCUFF_PHYSICAL_MODEL_CHANGED')
        cmd=rec['command']
        for key, expected_args in (('--Lambda',[str(r6.LAMBDA_UM)]), ('--pwDirection',['0','0','1']),('--pwPolarization',['1','0','0'])):
            if key not in cmd or cmd[cmd.index(key)+1:cmd.index(key)+1+len(expected_args)]!=expected_args:
                raise ValueError(label.upper()+'_SCUFF_SOURCE_CHANGED_'+key)
        pft = r6b.parse_pft((path/'sphere.PFT').read_text(),expected_omega=2*math.pi/r6.LAMBDA_UM)
        comp=r6b.compare(pft,mie)
        for field, value in (('refined_scuff_um2',comp['scuff_sigma_sca_um2']),('refined_reference_error_pct',comp['relative_error_pct']),('mie_reference_um2',mie),('pabs_over_psca_absolute',comp['pabs_over_psca_absolute'])):
            if not math.isclose(a[field],value,rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError(label.upper()+'_AUDIT_VALUE_INCONSISTENT_'+field)
        if plan['mesh_audit']['faces'] <= 0:
            raise ValueError(label.upper()+'_BAD_MESH')
        row={'label':label,'edge_nm':expected,'triangles':plan['mesh_audit']['faces'],
             'sigma_um2':comp['scuff_sigma_sca_um2'],'mie_error_pct':comp['relative_error_pct'],
             'lossless_residual':comp['pabs_over_psca_absolute'], 'directory':str(path)}
        data.append(row)
        for filename in ('M15R7A_PLAN.json','M15R7A_RUN.json','M15R7A_AUDIT.json','sphere.PFT','sphere.msh'):
            inputs[f'{label}/{filename}']=sha(path/filename)
    if not (base['sphere_triangle_count'] < data[0]['triangles'] < data[1]['triangles']):
        raise ValueError('THREE_MESH_TRIANGLE_COUNT_NOT_MONOTONE')
    m=data[0]; f=data[1]
    coarse_err=abs(b_sigma-mie)/mie*100
    delta_coarse_mid=abs(m['sigma_um2']-b_sigma)/mie*100
    delta_mid_fine=abs(f['sigma_um2']-m['sigma_um2'])/mie*100
    improved=bool(coarse_err>m['mie_error_pct']>f['mie_error_pct'])
    reduced_increment=bool(delta_mid_fine<delta_coarse_mid)
    passed=bool(improved and reduced_increment
                and f['mie_error_pct'] <= MAX_FINE_MIE_ERR_PCT
                and delta_mid_fine <= MAX_FINE_MIDDLE_DELTA_MIE_PCT
                and f['lossless_residual'] <= MAX_LOSSLESS_RESIDUAL)
    metrics={'reference_mie_um2':mie,
             'coarse':{'label':'coarse','edge_nm':12.0,'triangles':base['sphere_triangle_count'],
                       'sigma_um2':b_sigma,'mie_error_pct':coarse_err,
                       'lossless_residual':base_comp['pabs_over_psca_absolute'],'directory':str(Path(base['r6_dir']))},
             'middle':m,'fine':f,
             'coarse_to_middle_difference_over_mie_pct':delta_coarse_mid,
             'middle_to_fine_difference_over_mie_pct':delta_mid_fine,
             'three_mesh_error_decreases_monotonically':improved,
             'mesh_step_change_shrinks':reduced_increment,
             'cutoffs':{'max_fine_mie_error_pct':MAX_FINE_MIE_ERR_PCT,
                        'max_middle_to_fine_difference_over_mie_pct':MAX_FINE_MIDDLE_DELTA_MIE_PCT,
                        'max_lossless_absorption_ratio':MAX_LOSSLESS_RESIDUAL},
             'screen_status':'SPHERE_3MESH_TREND_SUPPORTIVE' if passed else 'SPHERE_3MESH_NEEDS_REVIEW',
             'sphere_mesh_convergence_proved':False,
             'au_ag_science_gate':'BLOCKED','au_ag_materials_qualified':False,
             'au_ag_mesh_convergence_demonstrated':False,
             'scientific_novelty_certified':False}
    return {'schema_version':VERSION,'metrics':metrics,'source_hashes':inputs,
            'directories':{'r6b':str(r6b_dir),'middle':str(mid_dir),'fine':str(fine_dir)},
            'provenance_note':'Calculated from hash-verified source artifacts. No SCUFF/Gmsh/LLM invoked.'}


def report(snapshot:dict)->str:
    m=snapshot['metrics']; c=m['coarse']; i=m['middle']; f=m['fine']
    return (f'# M15-R7B — SCUFF sphere 12 / 8 / 6 nm mesh screen\n\n'
            f'| Mesh target | Triangles | SCUFF sigma (um^2) | Relative error vs Mie |\n'
            f'|---|---:|---:|---:|\n'
            f'| {c["edge_nm"]:.0f} nm | {c["triangles"]} | {c["sigma_um2"]:.12g} | {c["mie_error_pct"]:.5g}% |\n'
            f'| {i["edge_nm"]:.0f} nm | {i["triangles"]} | {i["sigma_um2"]:.12g} | {i["mie_error_pct"]:.5g}% |\n'
            f'| {f["edge_nm"]:.0f} nm | {f["triangles"]} | {f["sigma_um2"]:.12g} | {f["mie_error_pct"]:.5g}% |\n\n'
            f'- Analytical Mie reference: {m["reference_mie_um2"]:.12g} um^2\n'
            f'- 12→8 nm change relative to Mie: {m["coarse_to_middle_difference_over_mie_pct"]:.5g}%\n'
            f'- 8→6 nm change relative to Mie: {m["middle_to_fine_difference_over_mie_pct"]:.5g}%\n'
            f'- Monotonic error decrease: {m["three_mesh_error_decreases_monotonically"]}\n'
            f'- Step difference shrinks: {m["mesh_step_change_shrinks"]}\n'
            f'- Screening outcome: **{m["screen_status"]}**\n\n'
            '**A supportive result is a bounded single-material/single-wavelength verification screen, NOT numerical convergence proof.**\n'
            '**Au/Ag nanorod science gate remains BLOCKED.** Actual Au/Ag material dispersion, '+
            'surface nesting, metal solver output, and target-observable convergence need independent tests.\n')


def audit(r6b_dir: Path, mid_dir: Path, fine_dir: Path, out_dir: Path) -> dict:
    out_dir=out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError('R7B_OUT_DIR_EXISTS_NO_OVERWRITE')
    snapshot=load_snapshot(r6b_dir,mid_dir,fine_dir)
    out_dir.mkdir(parents=True)
    rep=report(snapshot)
    (out_dir/'M15R7B_REPORT.md').write_text(rep,encoding='utf-8')
    snapshot['report_sha256']=sha(out_dir/'M15R7B_REPORT.md')
    with (out_dir/'M15R7B_AUDIT.json').open('x',encoding='utf-8') as fp:
        json.dump(snapshot,fp,indent=2,sort_keys=True,ensure_ascii=False)
        fp.write('\n')
    m=snapshot['metrics']
    return {'status':'M15R7B_AUDITED','screen_status':m['screen_status'],
            'fine_error_pct':m['fine']['mie_error_pct'],
            'middle_to_fine_delta_over_mie_pct':m['middle_to_fine_difference_over_mie_pct'],
            'nanorod_science_gate':'BLOCKED','new_scuff_runs':0,'new_paid_calls':0}


def check(out_dir: Path) -> dict:
    out_dir=out_dir.resolve()
    saved=json.loads((out_dir/'M15R7B_AUDIT.json').read_text())
    if saved['schema_version']!=VERSION or saved['metrics']['au_ag_science_gate']!='BLOCKED' or saved['metrics']['sphere_mesh_convergence_proved'] is not False:
        raise ValueError('R7B_AUTHORITY_INVALID')
    dirs=saved['directories']
    new=load_snapshot(Path(dirs['r6b']),Path(dirs['middle']),Path(dirs['fine']))
    if new!= {k:v for k,v in saved.items() if k!='report_sha256'}:
        raise ValueError('R7B_SOURCES_OR_COMPUTATION_CHANGED')
    if saved['report_sha256']!=sha(out_dir/'M15R7B_REPORT.md') or report(new)!=(out_dir/'M15R7B_REPORT.md').read_text():
        raise ValueError('R7B_REPORT_CHANGED')
    m=new['metrics']
    return {'status':'M15R7B_CHECK_OK','screen_status':m['screen_status'],
            'three_mesh_error_decreases_monotonically':m['three_mesh_error_decreases_monotonically'],
            'au_ag_science_gate':'BLOCKED','new_scuff_runs':0}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('audit');p.add_argument('--r6b-dir',type=Path,required=True)
    p.add_argument('--middle-dir',type=Path,required=True)
    p.add_argument('--fine-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p=sub.add_parser('check');p.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=(audit(args.r6b_dir,args.middle_dir,args.fine_dir,args.out_dir)
                if args.command=='audit' else check(args.out_dir))
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,KeyError,FileNotFoundError,FileExistsError,OSError,ImportError,TypeError,ZeroDivisionError) as exc:
        parser.exit(2,f'M15R7B_ERROR: {exc}\n')

if __name__=='__main__': main()
