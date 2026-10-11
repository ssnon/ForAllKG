"""M15-R5: execute Gmsh meshing and audit closed surfaces for scuff-em.

No SCUFF execution, metal dielectric inference, convergence or science authority.
Requires prior M15-R4 plan and hashes; outputs immutable run receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

VERSION = 'm15r5-mesh-verification-v1'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_r4(r4: Path) -> dict:
    plan = json.loads((r4/'M15R4_BEM_CANDIDATE_PLAN.json').read_text())
    if plan.get('schema_version') != 'm15r4-bem-candidate-v0':
        raise ValueError('R4_SCHEMA_MISMATCH')
    source = Path(plan['source_request'])
    if not source.is_file() or sha(source) != plan['source_request_sha256']:
        raise ValueError('SOURCE_REQUEST_SHA_MISMATCH')
    for filename,digest in plan['generated_files'].items():
        if filename not in ['inner_core.geo','outer_shell.geo','candidate.scuffgeo.template','scuff-scatter.args.template']:
            raise ValueError('UNEXPECTED_R4_SOURCE')
        if sha(r4/filename) != digest:
            raise ValueError('R4_GENERATED_SOURCE_HASH_MISMATCH_'+filename)
    return plan


def _section(lines: list[str], name: str) -> list[str]:
    try:
        b=lines.index('$'+name); e=lines.index('$End'+name,b+1)
    except ValueError as exc:
        raise ValueError('INVALID_MSH_SECTION_'+name) from exc
    return lines[b+1:e]


def read_msh2(path: Path) -> tuple[dict[int,tuple[float,float,float]],list[tuple[int,int,int]]]:
    lines = path.read_text(encoding='utf-8').splitlines()
    version = _section(lines, 'MeshFormat')[0].split()
    if version[:2] != ['2.2','0']:
        raise ValueError('REQUIRES_ASCII_GMSH_MSH_2_2')
    nblock=_section(lines,'Nodes')
    n=int(nblock[0])
    if len(nblock)!=n+1:raise ValueError('BAD_NODE_COUNT')
    nodes={}
    for row in nblock[1:]:
        parts=row.split();
        if len(parts)!=4:raise ValueError('BAD_NODE_ROW')
        i=int(parts[0]); xyz=tuple(map(float,parts[1:]))
        if i in nodes or not all(map(math.isfinite,xyz)):raise ValueError('BAD_NODE_COORDS')
        nodes[i]=xyz
    eblock=_section(lines,'Elements'); n=int(eblock[0])
    if len(eblock)!=n+1:raise ValueError('BAD_ELEMENT_COUNT')
    tris=[]; invalid_types=set()
    for row in eblock[1:]:
        p=row.split(); typ=int(p[1]); ntags=int(p[2]); values=p[3+ntags:]
        if len(p)<3+ntags:raise ValueError('BAD_ELEMENT_TAGS')
        if typ==2:
            if len(values)!=3:raise ValueError('BAD_TRIANGLE')
            tri=tuple(map(int,values));
            if len(set(tri))!=3 or any(x not in nodes for x in tri):raise ValueError('BAD_TRIANGLE_NODES')
            tris.append(tri)
        elif typ not in (1,15):
            invalid_types.add(typ)
    if invalid_types:raise ValueError('UNEXPECTED_ELEMENT_TYPES_'+str(sorted(invalid_types)))
    if not tris:raise ValueError('NO_TRIANGLES')
    return nodes,tris


def audit(nodes:dict[int,tuple[float,float,float]], tris:list[tuple[int,int,int]], length:float,diam:float,mesh_edge:float)->dict:
    edges=Counter(); oriented=Counter(); faces=set(); used=set()
    area=volume=0.0
    for tri in tris:
        key=frozenset(tri)
        if key in faces: raise ValueError('DUPLICATE_FACE')
        faces.add(key); used.update(tri)
        for u,v in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            edges[tuple(sorted((u,v)))]+=1
            oriented[(u,v)]+=1
        a,b,c=[nodes[t] for t in tri]
        ab=[b[i]-a[i] for i in range(3)];ac=[c[i]-a[i] for i in range(3)]
        cross=(ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0])
        norm=math.sqrt(sum(x*x for x in cross))
        if norm<1e-20:raise ValueError('DEGENERATE_FACE')
        area+=0.5*norm
        volume+=(a[0]*cross[0]+a[1]*cross[1]+a[2]*cross[2])/6.0
    boundary=sum(x==1 for x in edges.values())
    nonmanifold=sum(x>2 for x in edges.values())
    inconsistent=sum(oriented[(u,v)]!=1 or oriented[(v,u)]!=1 for u,v in edges)
    xyz=[nodes[t] for t in used]
    bounds=[[min(pt[i] for pt in xyz),max(pt[i] for pt in xyz)] for i in range(3)]
    radius=diam/2.; barrel=length-diam
    expected=math.pi*radius**2*barrel+(4./3.)*math.pi*radius**3
    verror=abs(abs(volume)-expected)/expected
    extent=[hi-lo for lo,hi in bounds]
    # Coarse triangular approximations may differ, but should not be wildly incompatible with CAD dimensions.
    extent_ok=all(abs(extent[i]-target)<=max(2*mesh_edge,0.10*target) for i,target in enumerate((length,diam,diam)))
    status='MESH_TOPOLOGY_PASS_PRELIMINARY' if (
       boundary==0 and nonmanifold==0 and inconsistent==0 and volume>0 and verror<=.20 and extent_ok
    ) else 'MESH_GEOMETRY_REVIEW_REQUIRED'
    return {'status':status,'nodes':len(used),'faces':len(tris),'edge_count':len(edges),
      'boundary_edges':boundary,'nonmanifold_edges':nonmanifold,'orientation_inconsistent_edges':inconsistent,
      'signed_volume_um3':volume,'analytic_volume_um3':expected,'relative_volume_deviation':verror,
      'surface_area_um2':area,'bbox_um':bounds,'bbox_extents_um':extent,'extent_plausible':extent_ok,
      'note':'Topology/geometry screening only; no mesh convergence, no optical science authority.'}


def evaluate(r4:Path,out:Path,*,gmsh:str='gmsh',timeout:int=180)->dict:
    if not 1<=timeout<=1800:raise ValueError('INVALID_TIMEOUT')
    r4=r4.resolve();out=out.resolve()
    plan=load_r4(r4)
    exe=shutil.which(gmsh) if '/' not in gmsh else str(Path(gmsh).resolve())
    if not exe or not Path(exe).is_file():raise ValueError('GMSH_EXECUTABLE_NOT_FOUND')
    if out.exists():raise FileExistsError('R5_OUTPUT_ALREADY_EXISTS_DO_NOT_OVERWRITE')
    out.mkdir(parents=True)
    outputs={}; results={}
    for part,prefix in (('inner_core','inner'),('outer_shell','outer')):
        target=out/(part+'.msh')
        cmd=[exe,'-2','-format','msh2','-v','2','-o',str(target),str(r4/(part+'.geo'))]
        run=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,check=False,cwd=out)
        (out/(part+'.stdout.log')).write_text(run.stdout)
        (out/(part+'.stderr.log')).write_text(run.stderr)
        if run.returncode!=0 or not target.is_file():
            raise ValueError('GMSH_MESH_FAILED_'+part+'_RC_'+str(run.returncode)+'_SEE_LOGS')
        nodes,tris=read_msh2(target)
        geom=plan['geometry_um']
        results[prefix]=audit(nodes,tris,geom[f'{prefix}_length'],geom[f'{prefix}_diameter'],plan['mesh_edge_nm']/1000.)
        outputs[part+'.msh']=sha(target)
    # Core must lie strictly inside shell in the analytic CAD; discrete intersection not certified here.
    g=plan['geometry_um']
    shell_gap=(g['outer_diameter']-g['inner_diameter'])/2.
    if shell_gap<=0 or abs((g['outer_length']-g['inner_length'])/2.-shell_gap)>1e-10:
        raise ValueError('ANALYTIC_SHELL_CONTAINMENT_INVALID')
    overall='MESH_TOPOLOGY_PASS_PRELIMINARY' if all(x['status']=='MESH_TOPOLOGY_PASS_PRELIMINARY' for x in results.values()) else 'MESH_GEOMETRY_REVIEW_REQUIRED'
    receipt={'schema_version':VERSION,'status':overall,'r4_plan_sha256':sha(r4/'M15R4_BEM_CANDIDATE_PLAN.json'),
     'r4_dir':str(r4),'source_request_sha256':plan['source_request_sha256'],
     'gmsh_executable':exe,'gmsh_version':subprocess.run([exe,'--version'],capture_output=True,text=True,timeout=20).stdout.strip(),
     'files':outputs,'meshes':results,'analytic_shell_thickness_um':shell_gap,
     'no_solver_run':True,'science_backend_qualified':False,'nanorod_science_gate':'BLOCKED',
     'next_gate':'MATERIAL_PROVENANCE_AND_SOLVER_BENCHMARK',
     'warnings':['SCUFF-EM not run','Au/Ag dispersion not defined','Mesh convergence not established',
       'Mesh-to-mesh contact/collision not independently checked','Absolute scattering normalization not validated']}
    (out/'M15R5_MESH_GATE.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return {'status':overall,'inner_triangles':results['inner']['faces'],'outer_triangles':results['outer']['faces'],
      'gate':str(out/'M15R5_MESH_GATE.json'),'science_gate':'BLOCKED'}


def check(out:Path)->dict:
    out=out.resolve();p=out/'M15R5_MESH_GATE.json';receipt=json.loads(p.read_text())
    if receipt.get('schema_version')!=VERSION or receipt.get('science_backend_qualified') is not False:
        raise ValueError('INVALID_R5_GATE')
    r4=Path(receipt['r4_dir']);plan=load_r4(r4)
    if sha(r4/'M15R4_BEM_CANDIDATE_PLAN.json')!=receipt['r4_plan_sha256']:raise ValueError('R4_PLAN_CHANGED')
    for part in ('inner_core','outer_shell'):
        name=part+'.msh'
        if sha(out/name)!=receipt['files'][name]:raise ValueError('MESH_CHANGED_'+part)
        nodes,tris=read_msh2(out/name); key='inner' if part=='inner_core' else 'outer'
        fresh=audit(nodes,tris,plan['geometry_um'][key+'_length'],plan['geometry_um'][key+'_diameter'],plan['mesh_edge_nm']/1000.)
        if fresh!=receipt['meshes'][key]:raise ValueError('AUDIT_CHANGED_'+part)
    return {'status':'M15R5_GATE_CHECK_OK','mesh_topology':receipt['status'],
        'nanorod_science_gate':'BLOCKED','new_scuff_runs':0}


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    run=sub.add_parser('mesh');run.add_argument('--r4-dir',type=Path,required=True)
    run.add_argument('--out-dir',type=Path,required=True)
    run.add_argument('--gmsh',default='gmsh');run.add_argument('--timeout-seconds',type=int,default=180)
    verify=sub.add_parser('check');verify.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args()
    try:
        outcome=evaluate(args.r4_dir,args.out_dir,gmsh=args.gmsh,timeout=args.timeout_seconds) if args.cmd=='mesh' else check(args.out_dir)
        print(json.dumps(outcome,indent=2,ensure_ascii=False))
        if outcome.get('status')=='MESH_GEOMETRY_REVIEW_REQUIRED':sys.exit(3)
    except (ValueError,FileExistsError,KeyError,OSError,subprocess.TimeoutExpired,AssertionError) as exc:
        ap.exit(2,'M15R5_ERROR: '+str(exc)+'\n')

if __name__=='__main__':main()
