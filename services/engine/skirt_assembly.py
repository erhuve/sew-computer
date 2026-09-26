"""Physical skirt inventory connections, derived from the trusted source recipe."""
import math
from skirt import seam_recipe


def compile_skirt_assembly(pattern, inventory):
    panels = {p['id']: p for p in pattern['panels']}
    source = pattern['drafting']['assembly']
    recipe = seam_recipe()
    if len(source) != len(recipe) or len({s['id'] for s in source}) != len(recipe):
        raise ValueError('Skirt assembly inventory mismatch')
    by_id = {s['id']: s for s in source}
    operations, occupied = [], set()

    def participant(name, label, role='shell'):
        boundary = next(e for e in panels[name]['draft']['edges'] if e['name'] == label)
        return {'instanceId': f'{name}:{role}', 'edgeName': label, 'intervalMm': [0, boundary['lengthMm']], 'sourcePointInterval': [boundary['start'], boundary['end']]}

    def add(identity, members, kind='seam'):
        for member in members:
            key = (member['instanceId'], member['edgeName'])
            if key in occupied:
                raise ValueError('Duplicate skirt attachment')
            occupied.add(key)
        operations.append({'id': identity, 'kind': kind, 'participants': members, 'dependsOn': [], 'registrationFractions': [0, .5, 1], 'distribution': 'equal-arc', 'orientationStatus': 'guide-selected', 'executionStatus': 'approximation-only'})

    for identity, a, ae, b, be in recipe:
        seam = by_id.get(identity)
        sides = [{'panel': name, 'edge': next(i for i, e in enumerate(panels[name]['draft']['edges']) if e['name'] == label)} for name, label in ((a, ae), (b, be))]
        if not seam or seam['sides'] != sides or seam['treatment'] != 'plain' or seam['ratio'] != 1:
            raise ValueError('Skirt source attachment differs from recipe')
        lengths = [panels[s['panel']]['draft']['edges'][s['edge']]['lengthMm'] for s in sides]
        if not all(math.isfinite(v) and v > 0 for v in lengths) or abs(lengths[0] - lengths[1]) > .001:
            raise ValueError('Skirt seam metric mismatch')
        if identity.startswith('band_side_'):
            for role in ('shell', 'facing'):
                add(f'{identity}:{role}', [participant(a, ae, role), participant(b, be, role)])
        else:
            members = [participant(a, ae), participant(b, be)]
            if identity.startswith('waist_'):
                members.append(participant(b, be, 'facing'))
            add(identity, members)
    stitches = [{'panelA': s['sides'][0]['panel'], 'edgeA': s['sides'][0]['edge'], 'panelB': s['sides'][1]['panel'], 'edgeB': s['sides'][1]['edge']} for s in source]
    if pattern['stitches'] != stitches:
        raise ValueError('Skirt stitch graph differs from assembly')
    for name in panels:
        if name.startswith('waistband_'):
            add(f'casing_top:{name}', [participant(name, 'top'), participant(name, 'top', 'facing')], 'layer-perimeter')
    free = []
    for instance in inventory['instances']:
        for boundary in panels[instance['templateId']]['draft']['edges']:
            if (instance['id'], boundary['name']) not in occupied:
                if boundary['name'] != 'hem' or instance['role'] != 'shell':
                    raise ValueError('Unexplained skirt boundary')
                free.append({**participant(instance['templateId'], boundary['name'], instance['role']), 'finish': boundary['finish'], 'classification': 'free-boundary'})
    return {'schemaVersion': 1, 'compiler': 'sew-skirt-assembly/1', 'units': 'mm', 'operations': operations, 'closures': [], 'freeBoundaries': free,
        'sourceOperationCount': len(source), 'physicalInstanceCount': len(inventory['instances']), 'semanticCoverage': 'explicit-attachments-and-unresolved-execution', 'solverReady': False,
        'limitations': ['Elastic gathering uses declared shape guides, not calibrated elastic response.', 'Seam allowances, casing insertion, turning and fabric contact are not simulated.']}
