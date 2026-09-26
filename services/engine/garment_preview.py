"""Shape-guided, source-metric cloth preview. Not calibrated drape or fit.

Keeps source vertices/triangles and rest lengths; solves elastic edge and seam
constraints against explicit synthetic posing guides. The guides are forces,
not replacement fabric meshes. All residuals remain in the output report.
"""
import hashlib, json, math, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parent
from assembly import compile_inventory, compile_assembly
from meshing import mesh_panel
from simulation_validation import validate_rest_mesh

PROFILE='sew-guided-cloth-preview/2'
MATERIAL={'name':'Assumed soft woven cotton, visual approximation','calibrated':False,
 'edgeWeight':1.0,'bendingWeight':0.025,'seamWeight':18.0,'guideWeight':0.075,
 'iterations':110,'contact':'not solved','gravity':'represented by posing guides, not dynamic simulation'}
def digest(b): return hashlib.sha256(b).hexdigest()
def enc(v): return json.dumps(v,separators=(',',':'),sort_keys=True,allow_nan=False).encode()
def skirt_guide(panel, xy, panels, pattern, role):
    from skirt import QUARTERS
    name = panel['id']; x, y = np.asarray(xy).T
    band = name.startswith('waistband_')
    quarter = name.removeprefix('waistband_' if band else 'skirt_')
    index = QUARTERS.index(quarter)
    source = panels['skirt_' + quarter]
    waistband = panels['waistband_' + quarter]
    length, depth, width = source['heightMm'], waistband['heightMm'], waistband['widthMm']
    elastic = next(row['valueMm'] for row in pattern['drafting']['measurements'] if row['name'] == 'Assumed relaxed elastic circumference')
    # Elastic is an explicit, uncalibrated posing assumption. Rest dimensions stay unchanged.
    hem = source['widthMm']; inset = (hem - width) / 2
    t = np.zeros_like(y) if band else np.clip(y / length, 0, 1)
    local_width = width + (hem - width) * t
    u = x / width if band else (x - inset * (1 - t)) / local_width
    angle = (index + u) * math.pi / 2
    r0 = elastic / (2 * math.pi)
    # Smoothly release waist gathering into a hanging skirt, then flare toward the hem.
    release = 1 - np.exp(-t * 5)
    circumference = elastic * (1 - release) + 4 * local_width * release
    radius = circumference / (2 * math.pi)
    fold = np.sin(angle * 24) * (2 if band else 4 + 9 * np.sin(t * math.pi / 2))
    radius += fold - (.7 if role == 'facing' else 0)
    yy = -y if band else -depth - y + 3 * np.cos(angle * 24) * t
    return np.c_[radius * np.sin(angle), yy, radius * np.cos(angle)]

def guide(panel,xy,panels,role='shell'):
    name=panel['id']; x,y=np.asarray(xy).T
    sign=-1 if (name.startswith('opening_binding_right_') or (not name.startswith('opening_binding_') and name.endswith('_right'))) else 1
    layer=.6 if role=='facing' else 0
    back=panels['back_left']; front_panel=panels['front_left']
    edge=lambda p,n: next(e for e in p['draft']['edges'] if e['name']==n)
    w=back['points'][edge(back,'shoulder')['start']][0]
    front_w=front_panel['points'][edge(front_panel,'shoulder')['start']][0]
    opening_width=w-front_w
    neckline=front_panel['points'][edge(front_panel,'center')['start']][1]
    body_length=front_panel['points'][edge(front_panel,'hem')['start']][1]
    arm=edge(front_panel,'armhole')['lengthMm']
    r=arm/math.pi; theta=math.radians(42); along=np.array([sign*math.cos(theta),-math.sin(theta),0]); across=np.array([sign*math.sin(theta),math.cos(theta),0])
    shoulder=np.array([sign*(w-arm*.107),-arm*.17,0.]); sleeve_center=shoulder-across*r
    if name.startswith(('front_','back_')):
        front=name.startswith('front_'); side=1 if front else -1
        opening=opening_width if front else 0
        hem_width=back['points'][edge(back,'hem')['end']][0]
        local_width=w+(hem_width-w)*np.clip((y-arm)/(body_length-arm),0,1)
        u=(x+opening)/local_width; a=u*math.pi/2
        blend=np.clip(y/arm,0,1)
        radius=local_width/(math.pi/2)*1.02
        X=(1-blend)*(x+opening)*.94+blend*radius*np.sin(a)
        Y=-y-arm*.17*u**4*(1-blend)+arm*.364*np.exp(-((x+opening-local_width)/(w*.4))**2)*blend*np.clip((body_length-y)/(body_length-arm),0,1)
        depth=radius*.65*np.cos(a)*np.sqrt(np.clip(y/arm,0,1))
        depth+=r*np.sin(np.clip(y/arm,0,1)*math.pi)*np.exp(-((x+opening-w)/60)**2)
        # Small, deterministic posing forces create relaxed vertical fabric folds.
        wave=(3.5*np.sin(x/24+y/145)+2*np.sin(x/12-y/170))*np.clip(y/350,0,1)*np.sin(a)
        return np.c_[sign*X,Y,side*(depth+wave)]
    if name.startswith('sleeve_'):
        length=panel['heightMm']; t=y/length
        width=panel['widthMm']+(next(e['lengthMm'] for e in panel['draft']['edges'] if e['name'] in ('wrist','hem'))-panel['widthMm'])*t
        left=(panel['widthMm']-width)/2
        angle=(x-left)/width*2*math.pi
        rad=width/(2*math.pi)
        if 'cuff_'+('right' if sign<0 else 'left') in panels:
            cuff=panels['cuff_'+('right' if sign<0 else 'left')]
            wrist=edge(panel,'wrist')['lengthMm']; target=edge(cuff,'attachment')['lengthMm']
            rad=rad*(1-(1-target/wrist)*np.maximum(0,(t-.62)/.38))
        rad+=2.3*np.sin(angle*7+y/34)*np.sin(math.pi*t)**2
        return sleeve_center+np.outer(y,along)+np.outer(-np.cos(angle)*rad,across)+np.c_[0*x,0*x,np.sin(angle)*rad]
    if name.startswith('cuff_'):
        sleeve=panels['sleeve_'+('right' if sign<0 else 'left')]; length=sleeve['heightMm']
        circ=next(e['lengthMm'] for e in panel['draft']['edges'] if e['name']=='attachment')
        angle=x/circ*2*math.pi; rad=circ/(2*math.pi)+layer
        return sleeve_center+np.outer(length+y,along)+np.outer(-np.cos(angle)*rad,across)+np.c_[0*x,0*x,np.sin(angle)*rad]
    if name.startswith('opening_binding_'):
        sleeve=panels['sleeve_'+('right' if sign<0 else 'left')]; length=sleeve['heightMm']
        is_right=name.endswith('_right'); opening=edge(sleeve,'opening_right' if is_right else 'opening_left')
        ends=np.array([sleeve['points'][opening['start']],sleeve['points'][opening['end']]])
        ends=ends[np.argsort(ends[:,1])]
        fraction=y/panel['heightMm']
        sx=ends[0,0]+(ends[1,0]-ends[0,0])*fraction+(x-panel['widthMm'])*(1 if is_right else -1)
        sy=ends[0,1]+(ends[1,1]-ends[0,1])*fraction
        return guide(sleeve,np.c_[sx,sy],panels)+np.array([0,0,layer])
    if name.startswith(('placket_','frill_')):
        frill=name.startswith('frill_'); fullness=panel['heightMm']/panels['placket_left']['heightMm'] if frill else 1
        yy=neckline+y/fullness
        base=guide(panels['front_'+('right' if sign<0 else 'left')],np.c_[0*x,yy],panels)
        base[:,0]=sign*(opening_width-x if not frill else opening_width+panel['widthMm']-x)
        base[:,2]+=layer+1.3
        if frill:
            free=1-x/panel['widthMm']
            base[:,2]+=4+free*(12+14*np.sin(y/18))
            base[:,1]+=free*6*np.cos(y/18)
        return base
    if name.startswith('collar_'):
        # Follow the actual six-edge neckline chain for a continuous attachment guide.
        chain=[('placket_left','top'),('front_left','neck'),('back_left','neck'),('back_right','neck'),('front_right','neck'),('placket_right','top')]
        paths=[]; previous=None
        for pn,en in chain:
            p=panels[pn]; e=next(e for e in p['draft']['edges'] if e['name']==en)
            q=np.array(p['points'][e['start']:e['end']+1]); world=guide(p,q,panels)
            if previous is not None and np.linalg.norm(world[-1]-previous)<np.linalg.norm(world[0]-previous): q=q[::-1];world=world[::-1]
            elif previous is None and world[0,0]>world[-1,0]: q=q[::-1];world=world[::-1]
            arcs=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
            paths.append((arcs,world)); previous=world[-1]
        arc=[]; world=[]; total=0
        for aa,ww in paths:
            arc.extend(aa+total);world.extend(ww);total+=aa[-1]
        stand=panels['collar_stand']; fall_edge=edge(stand,'fall')
        fall_start=min(stand['points'][fall_edge['start']][0],stand['points'][fall_edge['end']][0])
        xx=x+(fall_start if name=='collar_fall' else 0)
        arr=np.array(world); base=np.column_stack([np.interp(xx,arc,arr[:,i]) for i in range(3)])
        radial=base[:,[0,2]]; radial/=np.maximum(np.linalg.norm(radial,axis=1)[:,None],1)
        if name=='collar_stand': base[:,1]+=y;base[:,[0,2]]+=radial*layer
        else:
            # Top source edge is attached to the stand; lower edge folds outward.
            base[:,1]+=stand['heightMm']-y*.65
            base[:,[0,2]]+=radial*(y*.76+layer)[:,None]
        return base
    raise ValueError(name)

def build_preview(source,construction):
    started=time.monotonic(); pattern,inventory=compile_inventory(source,construction)
    panels={p['id']:p for p in pattern['panels']}; assembly=compile_assembly(pattern,inventory)
    templates={}
    for p in pattern['panels']:
        m=mesh_panel(p,18 if not p['id'].startswith('frill_') else 9,quality_refinement=True);validate_rest_mesh(p,m);templates[p['id']]=m
    pieces=[];rest=[];guides=[];offsets={}; start=0
    for inst in inventory['instances']:
        name=inst['templateId'];m=templates[name];uv=np.array(m['restPositions']); n=len(uv)
        offsets[inst['id']]=start
        rest.extend(np.c_[uv,np.zeros(n)]); guides.extend(skirt_guide(panels[name],uv,panels,pattern,inst['role']) if pattern['family']=='skirt' else guide(panels[name],uv,panels,inst['role']))
        pieces.append({**inst,'offset':start,'count':n,'mesh':m});start+=n
        if start>60000: raise ValueError('Preview vertex budget exceeded')
    rest=np.array(rest);targets=np.array(guides); n=len(rest)
    # Projective elastic springs: every rest edge remains immutable, including gathers.
    er=[];ec=[];ev=[];lengths=[];weights=[];edges=[]; row=0
    for p in pieces:
        mesh=p['mesh'];off=p['offset'];seen={};opposites={}
        for tri in mesh['triangles']:
            for k in range(3):
                a,b=sorted((tri[k],tri[(k+1)%3]));seen[(a,b)]=1
                opposites.setdefault((a,b),[]).append(tri[(k+2)%3])
        for (a,b),weight in [(e,1.) for e in seen]+[(tuple(v),MATERIAL['bendingWeight']) for v in opposites.values() if len(v)==2]:
            a+=off;b+=off;edges.append((a,b));er.extend((row,row));ec.extend((a,b));ev.extend((-1,1))
            lengths.append(np.linalg.norm(rest[b]-rest[a]));weights.append(weight);row+=1
    E=sparse.coo_matrix((ev,(er,ec)),shape=(row,n)).tocsr(); weights=np.array(weights); lengths=np.array(lengths);edges=np.array(edges)
    # Actual source-edge arc registrations, with their explicit guide-selected directions.
    rows=[];cols=[];vals=[];seams=[];r=0
    def sample(member,f):
        p=next(p for p in pieces if p['id']==member['instanceId']);edge=next(e for e in p['mesh']['boundaries'] if e['name']==member['edgeName'])
        arcs=np.array([s['arcMm'] for s in edge['samples']]); ids=np.array([s['vertex']+p['offset'] for s in edge['samples']]);t=f*edge['lengthMm']
        k=min(len(arcs)-2,max(0,np.searchsorted(arcs,t)-1));u=(t-arcs[k])/(arcs[k+1]-arcs[k])
        return {int(ids[k]):1-u,int(ids[k+1]):u}
    def position(m,f):return sum(targets[k]*v for k,v in sample(m,f).items())
    for op in assembly['operations']:
        first=op['participants'][0]
        for other in op['participants'][1:]:
            direct=sum(np.linalg.norm(position(first,f)-position(other,f)) for f in (0,1))
            reverse=sum(np.linalg.norm(position(first,f)-position(other,1-f)) for f in (0,1))
            flip=reverse<direct
            count=max(4,int(max(first['intervalMm'][1],other['intervalMm'][1])/12)+1)
            firstrow=r
            for f in np.linspace(0,1,count):
                for m,frac,s in ((first,f,1),(other,1-f if flip else f,-1)):
                    for k,v in sample(m,frac).items():rows.append(r);cols.append(k);vals.append(s*v)
                r+=1
            seams.append({'operation':op['id'],'a':first,'b':other,'reverse':bool(flip),'rows':[firstrow,r]})
    S=sparse.coo_matrix((vals,(rows,cols)),shape=(r,n)).tocsr()
    gw=MATERIAL['guideWeight'];sw=MATERIAL['seamWeight'];A=E.T@sparse.diags(weights)@E+sw*S.T@S+sparse.eye(n)*gw
    solve=factorized(A.tocsc())
    # Start in the original rigid flat planes centered at their pose guide centroids.
    pos=rest.copy()
    for p in pieces:
        sl=slice(p['offset'],p['offset']+p['count']); pos[sl]=rest[sl]-rest[sl].mean(axis=0)+targets[sl].mean(axis=0)
    history=[]
    for iteration in range(MATERIAL['iterations']):
        delta=E@pos;norm=np.maximum(np.linalg.norm(delta,axis=1),1e-10)
        projected=delta*(lengths/norm)[:,None]
        rhs=E.T@(projected*weights[:,None])+gw*targets
        nextpos=np.column_stack([solve(rhs[:,axis]) for axis in range(3)])
        if not np.isfinite(nextpos).all():raise ValueError('Nonfinite preview solve')
        pos=nextpos
        if iteration%20==0 or iteration==MATERIAL['iterations']-1:
            strain=np.abs(np.linalg.norm(E@pos,axis=1)/lengths-1);gap=np.linalg.norm(S@pos,axis=1)
            history.append({'iteration':iteration+1,'edgeStrainP95':float(np.percentile(strain[weights==1],95)),'edgeStrainMax':float(strain[weights==1].max()),'seamGapMaxMm':float(gap.max())})
    output=[];buttons=[]
    for p in pieces:
        sl=slice(p['offset'],p['offset']+p['count']);m=p['mesh'];xy=np.array(m['restPositions']);pp=pos[sl]
        # Winding is resolved from source side and guide-facing surface normals.
        faces=np.array(m['triangles']); sign=-1 if p['mirrorX'] else 1
        if p['templateId'].startswith('front_') and sign>0 or p['templateId'].startswith('back_') and sign<0:faces=faces[:,::-1]
        uv=(xy/1000).tolist()
        output.append({'instanceId':p['id'],'templateId':p['templateId'],'role':p['role'],'mirrorX':p['mirrorX'],
            'restXY':uv,'positions':(pp/1000).round(7).tolist(),'triangles':faces.tolist(),'restTriangles':m['triangles'],'sourceWeights':m['sourceWeights'],
            'boundaryLoops':[[s['vertex'] for s in b['samples']] for b in m['boundaries']]})
        if p['role']=='shell':
            tree=cKDTree(xy)
            for mark in panels[p['templateId']]['draft']['marks']:
                if mark['kind']=='button':
                    _,idx=tree.query(mark['point'],k=3);a=pp[idx];center=a.mean(axis=0); normal=np.cross(a[1]-a[0],a[2]-a[0]);norm=np.linalg.norm(normal)
                    if norm<1e-8:normal=np.array([0,0,1.])
                    else:normal/=norm
                    if normal[2]<0:normal=-normal
                    buttons.append({'position':((center+normal*2)/1000).tolist(),'normal':normal.tolist(),'templateId':p['templateId']})
    report={'profile':PROFILE,'classification':'guided-shape-approximation','acceptedSimulation':False,'patternDigest':digest(source),'units':'m',
        'material':MATERIAL,'assembly':seams,'iterations':history,'collisionCheck':'not performed','convergence':'fixed iteration budget; not certified',
        'omitted':['seam allowances','interfacing','validated turning and binding wraps','body and self contact','calibrated material response'],
        'wallSeconds':time.monotonic()-started,'sourceVertices':n,'pieces':output,'buttons':buttons,
        'generatorSha256':digest(Path(__file__).read_bytes()),
        'constructionDigest':inventory['constructionDigest'],'assemblyDigest':digest(enc(seams)),
        'materialDigest':digest(enc(MATERIAL)),'pose':{'recipe':f'source-derived-{pattern["family"]}-guides/1','calibrated':False,'source':'pattern dimensions and explicit guide equations in the pinned generator'},
        'sourceHashes':{name:digest((ROOT/name).read_bytes()) for name in ('assembly.py','skirt.py','skirt_assembly.py','meshing.py','quality_meshing.py','simulation_validation.py')}}
    if len(enc(report))>16*1024*1024: raise ValueError('Preview artifact budget exceeded')
    return report
