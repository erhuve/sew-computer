"""Parametric bodice + skirt/tier construction, independent of the shirt recipe.

Original seam-line geometry. Center back is deliberately open for closure
finishing; no zip, lining, anatomical fit or calibrated drape is claimed.
"""
import hashlib
import json
import math
from shapely.geometry import Polygon, LineString


def compile_dress(inputs, commit):
    d, body = inputs['design'], inputs['bodyMm']
    if inputs['family'] != 'dress' or d['block'] != 'panel-dress':
        raise ValueError('Dress construction mismatch')
    chest = (max(body['bust'], body['hip']) + inputs['easeMm']) / 4
    arm = body['bust'] / 10 + 110
    waist, length = chest * d['waistRatio'], inputs['lengthMm'] - d['bodiceLengthMm']
    tiers = 2 if d['skirtStyle'] == 'tiered' else 1
    if d['skirtStyle'] not in ('flared', 'gathered', 'tiered') or d['neckline'] not in ('round', 'v', 'square') or d['sleeves'] not in ('none', 'short', 'long'):
        raise ValueError('Unsupported dress choice')
    if chest * 2 < body['shoulder'] + 20 or d['bodiceLengthMm'] < arm + 80 or length < (350 if tiers == 2 else 250) or d['neckWidthMm'] > chest - 35 or d['neckDepthMm'] > arm - 15:
        raise ValueError('DESIGN_INPUT: Dress neckline, shoulder, waist or skirt proportions leave insufficient fabric. Review construction dimensions; body measurements are preserved.')
    max_hem=waist*(1 if d['skirtStyle']=='flared' else d['skirtFullness'])*inputs['flare']*(d['tierFullness']*inputs['flare'] if tiers==2 else 1)
    if max_hem > 1800: raise ValueError('DESIGN_INPUT: Reduce skirt gathering or flare; each panel must stay within 1800 mm.')
    panels, seams = [], []
    def panel(name, names, paths):
        points, edges = [paths[0][0]], []
        for label, path in zip(names, paths):
            if math.dist(points[-1], path[0]) > 1e-8:
                raise ValueError('Disconnected dress boundary')
            start = len(points) - 1
            points.extend(path[1:])
            edges.append({'name': label, 'start': start, 'end': len(points)-1, 'lengthMm': sum(math.dist(a,b) for a,b in zip(path,path[1:])), 'finish': 'single-turn-overlocked'})
        shape = Polygon(points)
        cut = shape.buffer(d['seamAllowanceMm'], join_style='mitre', mitre_limit=3)
        if not shape.is_valid or shape.area < 100 or cut.geom_type != 'Polygon' or not cut.is_valid or cut.interiors or not cut.covers(shape):
            raise ValueError('DESIGN_INPUT: Dress outline or allowance is invalid for these proportions.')
        middle = shape.representative_point(); half = min(30,shape.bounds[3]/4)
        while not shape.covers(LineString([(middle.x,middle.y-half),(middle.x,middle.y+half)])):
            half /= 2
            if half < .5: raise ValueError('Dress grainline cannot fit')
        panels.append({'id':name,'name':name.replace('_',' '),'points':points,'widthMm':shape.bounds[2],'heightMm':shape.bounds[3],'cutQuantity':1,
          'draft':{'component':'sleeves' if name.startswith('sleeve_') else 'body','material':'shell','cutQuantity':1,'edges':edges,'cutLine':[list(p) for p in cut.exterior.coords],'grainline':[[middle.x,middle.y-half],[middle.x,middle.y+half]],'marks':[]}})
    def join(name,a,ae,b,be,treatment='plain'):
        sides, lengths = [], []
        for pn,en in ((a,ae),(b,be)):
            p=next(p for p in panels if p['id']==pn);i=next(i for i,e in enumerate(p['draft']['edges']) if e['name']==en);e=p['draft']['edges'][i]
            if e['finish']=='assembly': raise ValueError('Duplicate dress attachment')
            e['finish']='assembly';lengths.append(e['lengthMm']);sides.append({'panel':pn,'edge':i})
            # Selected seams are straight; curved neckline edges remain free.
            first,last=p['points'][e['start']],p['points'][e['end']]
            p['draft']['marks'].append({'kind':'notch','point':[(first[0]+last[0])/2,(first[1]+last[1])/2],'label':name})
        ratio=lengths[0]/lengths[1]
        if (treatment=='plain' and abs(lengths[0]-lengths[1])>.001) or (treatment=='gather' and not 1<=ratio<=3):
            raise ValueError('Dress seam metric mismatch')
        seams.append({'id':name,'sides':sides,'treatment':treatment,'ratio':ratio,'instruction':'Gather the first edge evenly to the second, match midpoint marks, sew and finish.' if treatment=='gather' else 'Match registration marks, sew the named edges, finish and press. Verify assembly orientation in a toile.'})
    for face in ('front','back'):
        for side in ('left','right'):
            depth=d['neckDepthMm'] if face=='front' else 25;w=d['neckWidthMm'];l=d['bodiceLengthMm']
            if face=='front' and d['neckline']=='v': neck=[[w,0],[0,depth]]
            elif face=='front' and d['neckline']=='square': neck=[[w,0],[w,depth],[0,depth]]
            else: neck=[[w*math.cos(math.pi*i/48),depth*math.sin(math.pi*i/48)] for i in range(25)]
            neck[0],neck[-1]=[w,0],[0,depth]
            panel(f'bodice_{face}_{side}',['center','waist','side','armhole','shoulder','neck'],[[[0,depth],[0,l]],[[0,l],[waist,l]],[[waist,l],[chest,arm]],[[chest,arm],[chest,0]],[[chest,0],[w,0]],neck])
    join('center_front','bodice_front_left','center','bodice_front_right','center')
    for side in ('left','right'):
        for edge in ('side','shoulder'): join(f'{edge}_{side}',f'bodice_front_{side}',edge,f'bodice_back_{side}',edge)
        if d['sleeves']!='none':
            cap=arm*2;wrist=cap*.72;inset=(cap-wrist)/2;l=d['sleeveLengthMm']
            panel(f'sleeve_{side}',['cap_front','cap_back','underarm_right','wrist','underarm_left'],[[[0,0],[arm,0]],[[arm,0],[cap,0]],[[cap,0],[cap-inset,l]],[[cap-inset,l],[inset,l]],[[inset,l],[0,0]]])
            join(f'sleeve_front_{side}',f'sleeve_{side}','cap_front',f'bodice_front_{side}','armhole');join(f'sleeve_back_{side}',f'sleeve_{side}','cap_back',f'bodice_back_{side}','armhole');join(f'underarm_{side}',f'sleeve_{side}','underarm_right',f'sleeve_{side}','underarm_left')
    previous=waist
    for tier in range(tiers):
        top=waist*(1 if d['skirtStyle']=='flared' else d['skirtFullness']) if tier==0 else previous*d['tierFullness']
        hem=top*inputs['flare'];h=length/tiers;inset=(hem-top)/2
        for face in ('front','back'):
            for side in ('left','right'):
                name=f'skirt{tier}_{face}_{side}'
                panel(name,['waist','outer','hem','center'],[[[inset,0],[inset+top,0]],[[inset+top,0],[hem,h]],[[hem,h],[0,h]],[[0,h],[inset,0]]])
                join(f'waist{tier}_{face}_{side}',name,'waist',f'bodice_{face}_{side}' if tier==0 else f'skirt{tier-1}_{face}_{side}','waist' if tier==0 else 'hem','gather' if top>previous+.001 else 'plain')
        for face in ('front','back'):join(f'center{tier}_{face}',f'skirt{tier}_{face}_left','center',f'skirt{tier}_{face}_right','center')
        for side in ('left','right'):join(f'side{tier}_{side}',f'skirt{tier}_front_{side}','outer',f'skirt{tier}_back_{side}','outer')
        previous=hem
    measurements={'Chest circumference':chest*4,'Bodice waist circumference':waist*4,'Waist seam from shoulder':d['bodiceLengthMm'],'Skirt length':length,'Hem circumference':previous*4,'Dress length':inputs['lengthMm']}
    warnings=['Bodice and skirt are separate source pieces, joined at an explicit waist seam. This silhouette draft is not fit or sewing certification.',
      'Center-back bodice edges remain open for closure development. Zip/ties, lining, neckline facings and finishing construction are not drafted. Resolve them and verify head/hip passage in a toile.',
      'Waist ratio and waist placement are design assumptions, not anatomical measurements. Neck and sleeve shapes use explicit construction dimensions; dropped-shoulder sleeves are not fitted set-in sleeves.',
      'The display uses uncalibrated posing guides. Source dimensions remain unchanged; gathers, contact and fabric behavior are not physically verified.']
    return {'schemaVersion':1,'units':'mm','inputDigest':inputs['inputDigest'],'engineVersion':f'Sew panel dress compiler 1 / runtime pin {commit}','family':'dress','panels':panels,
      'stitches':[{'panelA':s['sides'][0]['panel'],'edgeA':s['sides'][0]['edge'],'panelB':s['sides'][1]['panel'],'edgeB':s['sides'][1]['edge']} for s in seams],
      'warnings':warnings+inputs.get('provenance',[]),'classification':'printable-reference','assumptions':warnings,
      'drafting':{'compiler':'sew-panel-dress/1','seamAllowanceMm':d['seamAllowanceMm'],'components':['body','waist-seam','neckline']+(['sleeves'] if d['sleeves']!='none' else []),'assembly':seams,
        'measurements':[{'name':k,'valueMm':v,'method':'Derived from source seam-line dimensions; not a measured finished garment.'} for k,v in measurements.items()],
        'materials':['Shell fabric for separate bodice and skirt pieces. Material and color follow the reviewed design; yield is not calculated.','Center-back closure and neckline finishing need development; no closure hardware, lining or facing pattern is supplied.'],
        'operations':['Cut each explicitly named piece once, mirroring left/right pairs; transfer grain and registration marks.','Assemble bodice shoulders, front center and sides. Leave back center open for closure development.','Attach optional sleeves and close their underarm seams.','Assemble skirt panels by named center and outer seams. Join tiers and attach to bodice using the specified gather ratios.','Develop a suitable center-back closure and neckline finish, verify wearer passage and mobility in a toile, then finish hems. This is a silhouette prototype.']}}


def dress_inventory(source,construction):
    pattern=json.loads(source);panels={p['id']:p for p in pattern['panels']};m={r['name']:r['valueMm'] for r in pattern['drafting']['measurements']}
    arm=next(e['lengthMm'] for e in panels['bodice_front_left']['draft']['edges'] if e['name']=='armhole')
    skirt=panels['skirt0_front_left'];edge=lambda name:next(e['lengthMm'] for e in skirt['draft']['edges'] if e['name']==name)
    expected=compile_dress({'family':'dress','design':construction,'bodyMm':{'bust':(arm-110)*10,'hip':m['Chest circumference'],'shoulder':m['Chest circumference']/2-21},'easeMm':0,'lengthMm':m['Dress length'],'flare':edge('hem')/edge('waist'),'inputDigest':pattern['inputDigest']},'validation')
    def equal(a,b):
        if type(a) in (int,float) and type(b) in (int,float):return math.isfinite(a) and math.isfinite(b) and abs(a-b)<1e-5
        if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    if pattern.get('schemaVersion')!=1 or pattern.get('units')!='mm' or pattern.get('family')!='dress' or any(not equal(pattern[k],expected[k]) for k in ('panels','stitches','drafting')):
        raise ValueError('Dress source differs from captured construction')
    return pattern,{'schemaVersion':1,'compiler':'sew-panel-dress-inventory/1','patternDigest':hashlib.sha256(source).hexdigest(),'constructionDigest':hashlib.sha256(json.dumps(construction,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),'units':'mm','classification':'placement-inspection',
      'instances':[{'id':p['id']+':shell','templateId':p['id'],'role':'shell','mirrorX':p['id'].endswith('_right'),'sourceGrainline':p['draft']['grainline']} for p in pattern['panels']],
      'unresolvedPhysicalRoles':[],'capabilityGaps':['Center-back closure and neckline finishes are not drafted.','Posed silhouette only; material, contact, fit and assembly behavior are unverified.']}


def dress_assembly(pattern,inventory):
    # Same general two-edge assembly semantics; source was verified above.
    from custom_pattern import custom_assembly
    result=custom_assembly(pattern,inventory)
    result['compiler']='sew-panel-dress-assembly/1'
    result['limitations']=['Gathers use equal-fraction source registrations and uncalibrated posing guides.','Center-back closure, neckline finishing, allowances and contact remain unresolved.']
    return result


def dress_guide(panel,xy,panels,construction):
    import numpy as np
    from garment_preview import guide
    name=panel['id']
    if not name.startswith('skirt'):
        mapped={key.removeprefix('bodice_'):dict(value,id=key.removeprefix('bodice_')) for key,value in panels.items() if not key.startswith('skirt')}
        # Reuse the upper-body posing primitive, not its pattern compiler.
        for p in mapped.values():
            if p['id'].startswith(('front_','back_')):p['draft']={**p['draft'],'edges':[{**e,'name':'hem' if e['name']=='waist' else e['name']} for e in p['draft']['edges']]}
        return guide(mapped[name.removeprefix('bodice_')],xy,mapped)
    x,y=np.asarray(xy).T;tier=int(name[5]);front='_front_' in name;sign=-1 if name.endswith('_right') else 1
    top=next(e['lengthMm'] for e in panel['draft']['edges'] if e['name']=='waist');hem=panel['widthMm'];height=panel['heightMm'];t=np.clip(y/height,0,1)
    inset=(hem-top)/2;local=top+(hem-top)*t;u=(x-inset*(1-t))/local;angle=u*math.pi/2
    bodice=panels['bodice_back_left'];waist=next(e['lengthMm'] for e in bodice['draft']['edges'] if e['name']=='waist')
    previous=waist if tier==0 else panels['skirt0_back_left']['widthMm']
    release=(1-np.exp(-6*t))/(1-math.exp(-6));width=previous*(1-release)+local*release
    radius=width/(math.pi/2)*1.02
    fold=(3+min(16,(top/previous-1)*22))*np.sin(u*math.pi*12)*np.sin(np.pi*t/2)
    xx=radius*np.sin(angle);zz=radius*.65*np.cos(angle)
    # Fold guides release gathering without altering the source/rest outline.
    xx+=fold*np.sin(angle);zz+=fold*np.cos(angle)
    return np.c_[sign*xx,-construction['bodiceLengthMm']-tier*height-y,(1 if front else -1)*zz]
