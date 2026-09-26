"""Data-only polygon authoring with explicit seams and non-physical posing guides."""
import hashlib
import json
import math
from shapely.geometry import Polygon, LineString


def compile_custom(inputs, commit):
    design = inputs['design']
    if inputs['family'] != 'custom' or design.get('block') != 'custom-pattern':
        raise ValueError('Custom construction family mismatch')
    source = design['pieces']; seams = design['seams']; allowance = design['seamAllowanceMm']
    if not 1 <= len(source) <= 16 or len(seams) > 100 or not 6 <= allowance <= 20:
        raise ValueError('Custom construction budget exceeded')
    panels = []; ids = set()
    for piece in source:
        points = piece['points']
        if piece['id'] in ids or not 3 <= len(points) <= 30 or any(len(p) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 2000 for v in p) for p in points):
            raise ValueError('Invalid custom outline')
        ids.add(piece['id']); closed = points + [points[0]]
        shape = Polygon(closed)
        if not shape.is_valid or shape.area < 100 or any(math.dist(a, b) < 1 for a, b in zip(closed, closed[1:])):
            raise ValueError('DESIGN_INPUT: Custom outlines must be simple, closed shapes without crossing edges or repeated points.')
        cut = shape.buffer(allowance, join_style='mitre', mitre_limit=3)
        if cut.geom_type != 'Polygon' or not cut.is_valid or cut.interiors or not cut.covers(shape):
            raise ValueError('DESIGN_INPUT: This outline cannot produce a single seam-allowance contour.')
        x0, y0, x1, y1 = shape.bounds; center = shape.representative_point(); half = min(30, (y1-y0)/4)
        while half >= .5 and not shape.covers(LineString([(center.x,center.y-half),(center.x,center.y+half)])):
            half /= 2
        if half < .5:
            raise ValueError('DESIGN_INPUT: This piece is too narrow for a grainline.')
        edges = [{'name':f'edge_{i}','start':i,'end':i+1,'lengthMm':math.dist(a,b),'finish':'single-turn-overlocked'} for i,(a,b) in enumerate(zip(closed,closed[1:]))]
        panels.append({'id':piece['id'],'name':piece['name'],'points':closed,'widthMm':x1-x0,'heightMm':y1-y0,'cutQuantity':1,'draft':{'component':'body','material':'shell','cutQuantity':1,'edges':edges,'cutLine':[list(p) for p in cut.exterior.coords],'grainline':[[center.x,center.y-half],[center.x,center.y+half]],'marks':[]}})
    by_id = {p['id']:p for p in panels}; assembly = []; occupied = set(); seam_ids = set()
    for seam in seams:
        if seam['id'] in seam_ids:
            raise ValueError('Duplicate custom seam identity')
        seam_ids.add(seam['id']); sides = []; lengths = []
        for name,index in ((seam['a'],seam['edgeA']),(seam['b'],seam['edgeB'])):
            if name not in by_id or type(index) is not int or not 0 <= index < len(by_id[name]['draft']['edges']) or (name,index) in occupied:
                raise ValueError('DESIGN_INPUT: A seam refers to a missing or already attached edge.')
            occupied.add((name,index)); p=by_id[name]; edge=p['draft']['edges'][index]; edge['finish']='assembly'; lengths.append(edge['lengthMm'])
            a,b=p['points'][index:index+2]
            p['draft']['marks'].append({'kind':'notch','point':[(a[0]+b[0])/2,(a[1]+b[1])/2],'label':seam['id']})
            sides.append({'panel':name,'edge':index})
        ratio=lengths[0]/lengths[1]
        if seam['treatment'] not in ('plain','gather') or (seam['treatment']=='plain' and abs(lengths[0]-lengths[1])>.1) or (seam['treatment']=='gather' and not 1 <= ratio <= 3):
            raise ValueError('DESIGN_INPUT: Seam lengths differ. Edit the outline or explicitly choose a 1–3× gather; source pieces will not be resized.')
        assembly.append({'id':seam['id'],'sides':sides,'treatment':seam['treatment'],'ratio':ratio,'instruction':'Owner-defined edge connection; verify orientation and construction with a toile.'})
    return {'schemaVersion':1,'units':'mm','family':'custom','inputDigest':inputs['inputDigest'],'engineVersion':f'Sew custom pattern compiler 1 / runtime pin {commit}','classification':'printable-reference',
      'panels':panels,'stitches':[{'panelA':s['sides'][0]['panel'],'edgeA':s['sides'][0]['edge'],'panelB':s['sides'][1]['panel'],'edgeB':s['sides'][1]['edge']} for s in assembly],
      'warnings':['Custom outlines and seams are owner-authored. Disconnected pieces are allowed. Placement is an approximation; physical fit, seam orientation and sewability are not verified.'],
      'assumptions':inputs.get('provenance',[])+['One shell instance per named piece. Duplicate a piece to cut additional copies. No holes, darts, internal cuts or automatic grading.'],
      'drafting':{'compiler':'sew-custom-pattern/1','seamAllowanceMm':allowance,'components':['body'],'assembly':assembly,'measurements':[],'materials':['Owner-selected shell fabric; material response is uncalibrated.'],'operations':['Cut one of each explicitly named piece using the offset allowance contour; review grain direction and all edge connections before sewing.']}}


def custom_inventory(pattern_bytes, construction):
    pattern=json.loads(pattern_bytes)
    expected=compile_custom({'family':'custom','design':construction,'inputDigest':pattern['inputDigest']},'validation')
    if pattern.get('schemaVersion') != 1 or pattern.get('units') != 'mm' or pattern.get('family') != 'custom' or any(pattern.get(key)!=expected[key] for key in ('panels','stitches','drafting')):
        raise ValueError('Custom source does not match captured construction')
    return pattern,{'schemaVersion':1,'compiler':'sew-custom-inventory/1','patternDigest':hashlib.sha256(pattern_bytes).hexdigest(),'constructionDigest':hashlib.sha256(json.dumps(construction,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),'units':'mm','classification':'placement-inspection',
      'instances':[{'id':p['id']+':shell','templateId':p['id'],'role':'shell','mirrorX':False,'sourceGrainline':p['draft']['grainline']} for p in pattern['panels']],
      'unresolvedPhysicalRoles':[],'capabilityGaps':['Custom placement guides and seams are owner-defined; no automatic fit, contact or drape certification.']}


def custom_assembly(pattern,inventory):
    panels={p['id']:p for p in pattern['panels']}; operations=[]; occupied=set()
    def participant(name,index):
        edge=panels[name]['draft']['edges'][index]
        return {'instanceId':name+':shell','edgeName':edge['name'],'intervalMm':[0,edge['lengthMm']],'sourcePointInterval':[edge['start'],edge['end']]}
    for seam in pattern['drafting']['assembly']:
        members=[]
        for side in seam['sides']:
            key=(side['panel'],side['edge'])
            if key in occupied: raise ValueError('Duplicate custom attachment')
            occupied.add(key); members.append(participant(*key))
        operations.append({'id':seam['id'],'kind':'gather' if seam['treatment']=='gather' else 'seam','participants':members,'dependsOn':[],'registrationFractions':[0,.5,1],'distribution':'uniform' if seam['treatment']=='gather' else 'equal-arc','orientationStatus':'guide-selected','executionStatus':'approximation-only'})
    free=[{**participant(name,i),'finish':edge['finish'],'classification':'free-boundary'} for name,p in panels.items() for i,edge in enumerate(p['draft']['edges']) if (name,i) not in occupied]
    return {'schemaVersion':1,'compiler':'sew-custom-assembly/1','units':'mm','operations':operations,'closures':[],'freeBoundaries':free,'sourceOperationCount':len(operations),'physicalInstanceCount':len(inventory['instances']),'semanticCoverage':'explicit-attachments-and-unresolved-execution','solverReady':False,'limitations':['Owner-defined posing only. Disconnected pieces remain separate. Contact and physical construction are unchecked.']}


def custom_guide(panel,xy,construction):
    import numpy as np
    piece=next(p for p in construction['pieces'] if p['id']==panel['id']); pose=piece['placement']; x,y=np.asarray(xy).T
    middle=(min(p[0] for p in panel['points'])+max(p[0] for p in panel['points']))/2; x=x-middle
    angle=math.radians(pose['bendDeg']); width=panel['widthMm']
    if abs(angle)<1e-6: result=np.c_[x,-y,0*x]
    else:
        # Follow the drawn horizontal extent. This changes posing forces only;
        # the solver retains the exact source metric and reports deformation.
        low=np.full(len(y),np.inf); high=np.full(len(y),-np.inf)
        for a,b in zip(panel['points'],panel['points'][1:]):
            if abs(a[1]-b[1])<1e-9:
                mask=np.abs(y-a[1])<1e-7
                low=np.where(mask,np.minimum(low,min(a[0],b[0])),low);high=np.where(mask,np.maximum(high,max(a[0],b[0])),high)
            else:
                t=(y-a[1])/(b[1]-a[1]); mask=(t>=-1e-8)&(t<=1+1e-8); hit=a[0]+t*(b[0]-a[0])
                low=np.where(mask,np.minimum(low,hit),low);high=np.where(mask,np.maximum(high,hit),high)
        low=np.where(np.isfinite(low),low,middle-width/2);high=np.where(np.isfinite(high),high,middle+width/2)
        local_width=np.maximum(high-low,1); local_x=x+middle-(low+high)/2
        radius=local_width/angle;theta=local_x/radius
        result=np.c_[radius*np.sin(theta),-y,radius*(np.cos(theta)-math.cos(angle/2))]
        # Cylindrical sectors share a common axis, allowing explicitly rotated
        # quarter pieces to meet. Small bends stay centered to avoid huge offsets.
        if abs(angle)>=math.pi/2-1e-8: result[:,2]+=radius*math.cos(angle/2)
    for axis,degrees in enumerate(pose['rotationDeg']):
        a=math.radians(degrees);c,s=math.cos(a),math.sin(a); matrix=np.eye(3); first,second=((1,2),(2,0),(0,1))[axis];matrix[first,first]=matrix[second,second]=c;matrix[first,second]=-s;matrix[second,first]=s;result=result@matrix.T
    return result+np.array(pose['positionMm'])
