"""Independent, standard-library audit of saved synthetic preview source metrics."""
import bisect, hashlib, json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ASSETS=ROOT/'apps/web/public/demo-fixtures'

def area(points):return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1]))/2)
def percentile(v,q):
    v=sorted(v);k=(len(v)-1)*q;i=int(k);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(k-i)
def check(identity):
    source=(ASSETS/(identity+'.json')).read_bytes();pattern=json.loads(source);panels={p['id']:p for p in pattern['panels']}
    raw=(ASSETS/(identity+'-shape.json')).read_bytes();data=json.loads(raw)
    assert data['patternDigest']==hashlib.sha256(source).hexdigest()
    assert data['generatorSha256']==hashlib.sha256((ROOT/'scripts/build-garment-preview.py').read_bytes()).hexdigest()
    for name,sha in data['sourceHashes'].items():assert hashlib.sha256((ROOT/'services/engine'/name).read_bytes()).hexdigest()==sha
    expected={p['id']+':'+role for p in pattern['panels'] for role in (['shell','facing'] if p['cutQuantity']==2 else ['shell'])}
    assert {p['instanceId'] for p in data['pieces']}==expected
    assert len(data['pieces'])==len(expected) and data['acceptedSimulation'] is False
    strains=[];byid={}
    for p in data['pieces']:
        byid[p['instanceId']]=p;uv=p['restXY'];positions=p['positions'];panel=panels[p['templateId']]
        assert len(uv)==len(positions)==len(p['sourceWeights'])
        for point,weights in zip(uv,p['sourceWeights']):
            rebuilt=[sum(panel['points'][w['point']][axis]*w['weight']/1000 for w in weights) for axis in (0,1)]
            assert math.dist(point,rebuilt)<8e-8
            assert abs(sum(w['weight'] for w in weights)-1)<1e-8
        for point in positions:assert len(point)==3 and all(math.isfinite(x) and abs(x)<5 for x in point)
        sourcearea=area(panel['points'])/1e6
        mesharea=sum(area([uv[i] for i in triangle]) for triangle in p['triangles'])
        assert abs(mesharea-sourcearea)<2e-7,(identity,p['templateId'],sourcearea,mesharea)
        edges={tuple(sorted((t[i],t[(i+1)%3]))) for t in p['triangles'] for i in range(3)}
        for a,b in edges:
            length=math.dist(uv[a],uv[b]);assert length>0
            strains.append(abs(math.dist(positions[a],positions[b])/length-1))
    def sample(member,f):
        p=byid[member['instanceId']];panel=panels[p['templateId']]
        k=next(i for i,e in enumerate(panel['draft']['edges']) if e['name']==member['edgeName'])
        ids=p['boundaryLoops'][k];arcs=[0.]
        for a,b in zip(ids,ids[1:]):arcs.append(arcs[-1]+math.dist(p['restXY'][a],p['restXY'][b])*1000)
        t=arcs[-1]*f;k=max(0,min(len(arcs)-2,bisect.bisect_left(arcs,t)-1));u=(t-arcs[k])/(arcs[k+1]-arcs[k])
        return [p['positions'][ids[k]][j]*(1-u)+p['positions'][ids[k+1]][j]*u for j in range(3)]
    gaps=[]
    for seam in data['assembly']:
        n=seam['rows'][1]-seam['rows'][0]
        for i in range(n):
            f=i/(n-1);gaps.append(math.dist(sample(seam['a'],f),sample(seam['b'],1-f if seam['reverse'] else f))*1000)
    final=data['iterations'][-1]
    assert abs(percentile(strains,.95)-final['edgeStrainP95'])<.0005
    assert abs(max(strains)-final['edgeStrainMax'])<.002
    assert abs(max(gaps)-final['seamGapMaxMm'])<.001
    print(json.dumps({'variant':identity,'instances':len(expected),'sourceAreaAndCorrespondence':'pass','edgeStrainP95':percentile(strains,.95),'edgeStrainMax':max(strains),'seamGapMaxMm':max(gaps)}))
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
if __name__=='__main__':
    catalog=json.loads((ASSETS/'catalog.json').read_bytes())
    files={v['id']+'-shape.json':check(v['id']) for v in catalog['variants']}
    manifest=json.loads((ASSETS/'shape-catalog.json').read_bytes())
    assert manifest['files']==files
    print('All four source-preserving preview audits pass. These checks do not validate physical drape or contact.')
