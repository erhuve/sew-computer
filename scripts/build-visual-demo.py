"""Build synthetic, source-derived assets for the local visual demo.

Uses Sew's component compiler and inspection mesher, without simulation or a
model call. Run in the pinned local engine environment; output is a new folder.
"""
import argparse, hashlib, json, math, platform, sys
from pathlib import Path
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'services/engine'))
from shirt import compile_shirt
from assembly import build_inspection
from inspection_gltf import inspection_glb

def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(data):return hashlib.sha256(data).hexdigest()
def pattern_svg(pattern):
    spacing=40;columns=4;cell_width=max(p['widthMm'] for p in pattern['panels'])+spacing*2
    cell_height=max(p['heightMm'] for p in pattern['panels'])+spacing*2
    width=columns*cell_width;height=math.ceil(len(pattern['panels'])/columns)*cell_height
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}">',
           '<title>Draft pattern pieces — synthetic demo; not fit validated</title>',
           '<rect width="100%" height="100%" fill="white"/>']
    for i,p in enumerate(pattern['panels']):
        points=lambda values:' '.join(f'{x},{y}' for x,y in values)
        parts += [f'<g transform="translate({(i%columns)*cell_width+spacing},{(i//columns)*cell_height+spacing})">',
          f'<text y="-18" font-family="sans-serif" font-size="12">{escape(p["name"])} · cut {p["cutQuantity"]}</text>',
          f'<polyline points="{points(p["draft"]["cutLine"])}" fill="none" stroke="#171717" stroke-width=".5"/>',
          f'<polyline points="{points(p["points"])}" fill="none" stroke="#527263" stroke-width=".4" stroke-dasharray="3 2"/>',
          f'<polyline points="{points(p["draft"]["grainline"])}" fill="none" stroke="#527263" stroke-width=".4"/>','</g>']
    parts.append('</svg>');return '\n'.join(parts).encode()

def build(output):
    output.mkdir(parents=True,exist_ok=False)
    design=dict(block='relaxed-drop-shoulder',sleeves='long',sleeveLengthMm=550,cuff='button',
        cuffCircumferenceMm=220,cuffDepthMm=55,collar='stand-and-fall',collarStandMm=30,collarFallMm=60,
        opening='buttons',placketWidthMm=30,buttonSpacingMm=80,hem='curved-back-tail',tailExtensionMm=100,
        frill='front-opening',frillWidthMm=35,frillFullness=1.8,seamAllowanceMm=10,
        rationale='A relaxed woven shirt with an extended curved back hem. Synthetic demo measurements; edit in the studio for your own design.')
    source_paths=['scripts/build-visual-demo.py',*[f'services/engine/{name}' for name in
        ('shirt.py','assembly.py','meshing.py','simulation_validation.py','inspection_gltf.py','guard.py')]]
    source_hashes={p:digest((ROOT/p).read_bytes()) for p in source_paths}
    catalog={'profile':'sew-visual-demo/1','synthetic':True,'classification':'placement-inspection',
        'scope':'Original component patterns and rigid display arrangement; no assembly simulation or fit validation.',
        'bodyMm':{'height':1700,'bust':960,'waist':760,'hip':1000,'shoulder':400},'variants':[]}
    for sleeves in ('long','short'):
        for frill in ('none','front-opening'):
            selected={**design,'sleeves':sleeves,'sleeveLengthMm':550 if sleeves=='long' else 220,
                'cuff':'button' if sleeves=='long' else 'none','frill':frill}
            inputs={'design':selected,'bodyMm':catalog['bodyMm'],'lengthMm':650,'easeMm':100,'flare':1,
                'provenance':['Synthetic example measurements for the visual demo; no wearer measurements or reference images.']}
            inputs['inputDigest']=digest(encoded(inputs))
            identity=f'{sleeves}-'+('frill' if frill!='none' else 'clean')
            pattern=compile_shirt(inputs,'original-component-demo')
            pattern_bytes=encoded(pattern)
            construction={k:selected[k] for k in ('sleeves','cuff','collar','opening','hem','frill')}
            inspection=build_inspection(pattern_bytes,construction)
            glb=inspection_glb(inspection)
            files={identity+'.json':pattern_bytes,identity+'.glb':glb,identity+'.svg':pattern_svg(pattern)}
            for name,data in files.items():(output/name).write_bytes(data)
            catalog['variants'].append({'id':identity,'design':selected,'patternDigest':digest(pattern_bytes),
                'pieces':len(inspection['instances']),'templates':len(pattern['panels']),
                'files':{name:{'sha256':digest(data),'bytes':len(data)} for name,data in files.items()}})
    catalog['sources']=source_hashes
    catalog['runtime']={'python':platform.python_version()}
    for p,h in source_hashes.items():assert digest((ROOT/p).read_bytes())==h
    (output/'catalog.json').write_bytes(encoded(catalog)+b'\n')
    return catalog
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();catalog=build(args.output)
    print(json.dumps({'synthetic':True,'variants':len(catalog['variants']),'output':str(args.output)}))
