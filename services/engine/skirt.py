"""Original four-panel woven skirt with a separate two-layer elastic casing."""
import math
from shapely.geometry import Polygon

QUARTERS = ('front_left', 'back_left', 'back_right', 'front_right')


def dimensions(inputs):
    design, body = inputs['design'], inputs['bodyMm']
    waist = max(body['hip'] + inputs['easeMm'], body['waist'] + 80) * design['fullness']
    return waist / 4, inputs['lengthMm'] - design['waistbandDepthMm'], body['waist'] + design['elasticEaseMm']


def seam_recipe():
    result = []
    for i, quarter in enumerate(QUARTERS):
        following = QUARTERS[(i + 1) % 4]
        result.extend([
            (f'side_{quarter}', f'skirt_{quarter}', 'right', f'skirt_{following}', 'left'),
            (f'band_side_{quarter}', f'waistband_{quarter}', 'right', f'waistband_{following}', 'left'),
            (f'waist_{quarter}', f'skirt_{quarter}', 'waist', f'waistband_{quarter}', 'bottom'),
        ])
    return result


def compile_skirt(inputs, commit):
    design = inputs['design']
    width, length, elastic = dimensions(inputs)
    depth, allowance = design['waistbandDepthMm'], design['seamAllowanceMm']
    flare = inputs['flare']
    if not (1 <= flare <= 1.8 and 1 <= design['fullness'] <= 1.8 and 25 <= depth <= 60 and length >= 200 and 6 <= allowance <= 20 and 390 <= elastic <= 1540):
        raise ValueError('DESIGN_INPUT: Unsupported elastic skirt proportions.')
    hem = width * flare
    inset = (hem - width) / 2
    panels, assembly = [], []

    def panel(name, component, points, names, quantity):
        shape = Polygon(points)
        cut = shape.buffer(allowance, join_style='mitre', mitre_limit=3)
        if not shape.is_valid or not cut.is_valid or cut.geom_type != 'Polygon' or cut.interiors or not cut.covers(shape):
            raise ValueError('Invalid skirt cut contour')
        grain = shape.representative_point()
        half = min(30, (shape.bounds[3] - shape.bounds[1]) / 4)
        panels.append({'id': name, 'name': name.replace('_', ' '), 'points': points, 'widthMm': shape.bounds[2], 'heightMm': shape.bounds[3], 'cutQuantity': quantity,
            'draft': {'component': component, 'material': 'shell', 'cutQuantity': quantity,
                'edges': [{'name': label, 'start': i, 'end': i + 1, 'lengthMm': math.dist(points[i], points[i + 1]), 'finish': 'bagged-facing' if quantity == 2 else 'single-turn-overlocked'} for i, label in enumerate(names)],
                'cutLine': [list(point) for point in cut.exterior.coords], 'grainline': [[grain.x, grain.y - half], [grain.x, grain.y + half]], 'marks': []}})

    for quarter in QUARTERS:
        panel(f'skirt_{quarter}', 'body', [[inset, 0], [inset + width, 0], [hem, length], [0, length], [inset, 0]], ['waist', 'right', 'hem', 'left'], 1)
        panel(f'waistband_{quarter}', 'waistband', [[0, 0], [width, 0], [width, depth], [0, depth], [0, 0]], ['top', 'right', 'bottom', 'left'], 2)
    by_id = {item['id']: item for item in panels}
    for identity, first, first_edge, second, second_edge in seam_recipe():
        sides = []
        for name, label in ((first, first_edge), (second, second_edge)):
            piece = by_id[name]
            index = next(i for i, edge in enumerate(piece['draft']['edges']) if edge['name'] == label)
            edge = piece['draft']['edges'][index]
            edge['finish'] = 'assembly'
            a, b = piece['points'][edge['start']], piece['points'][edge['end']]
            piece['draft']['marks'].append({'kind': 'notch', 'point': [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], 'label': identity})
            sides.append({'panel': name, 'edge': index})
        instruction = 'Join matching panel edges, then finish and press the seam.'
        if identity.startswith('band_side_'):
            instruction = 'Join shell band quarters into one ring and facing quarters into another. Keep the elastic insertion opening in the final waist attachment, not this side seam.'
        elif identity.startswith('waist_'):
            instruction = 'Attach shell band to skirt waist, matching quarter marks. Turn facing allowance inside and secure over the attachment; leave a temporary insertion gap, insert tested elastic, then close the gap.'
        assembly.append({'id': identity, 'sides': sides, 'treatment': 'plain', 'ratio': 1, 'instruction': instruction})
    measurements = [
        {'name': 'Fabric waist circumference', 'valueMm': width * 4, 'method': 'Sum of four source waist edges; ungathered fabric must pass over the hips.'},
        {'name': 'Hem circumference', 'valueMm': hem * 4, 'method': 'Sum of four source hem edges before single-turn finishing.'},
        {'name': 'Skirt length including waistband', 'valueMm': inputs['lengthMm'], 'method': 'Panel vertical length plus finished band depth; not a simulated worn length.'},
        {'name': 'Finished waistband depth', 'valueMm': depth, 'method': 'Width between upper and lower band seam lines.'},
        {'name': 'Assumed relaxed elastic circumference', 'valueMm': elastic, 'method': 'Entered or assumed waist plus selected elastic ease. Test on wearer and add the chosen joining overlap when cutting elastic; stretch is not calibrated.'},
    ]
    warnings = ['Four-panel woven elastic-waist skirt. Original pattern geometry, not fit or cutting certification; make a toile.',
        'Separate waistband shell and facing are cut once per named quarter. Elastic is a trim, not a fabric mesh. Band fabric is not shortened to the elastic length.',
        'Hip passage uses max(hip + ease, waist + 80 mm) times fullness; these are disclosed drafting assumptions, not measured stretch or fit.',
        f'Uniform {allowance} mm allowances include a single-turn overlocked hem. Check elastic stretch, casing width, hem mobility and printer calibration before sewing.']
    operations = ['Cut each named skirt quarter once and each band quarter twice for shell and facing. Transfer grainlines and seam registrations.',
        'Sew the skirt seams and the two waistband rings, matching the named quarter marks.',
        'Join shell and facing rings along the upper band edges, turn, understitch and press. Elastic casing remains hollow; do not fuse its layers.',
        'Attach the lower shell band to the skirt. Secure facing over the seam while leaving a temporary insertion gap.',
        f'Test elastic width below {depth - 4:g} mm and its stretch over the hips. Start from {elastic:g} mm relaxed finished circumference, adjust on the wearer and add the joining overlap before cutting; no stretch factor is assumed.',
        'Thread and join elastic without twisting. Distribute gathers, close the insertion gap, and finish the hem with the stated single-turn allowance. Make a toile before final fabric.']
    return {'schemaVersion': 1, 'units': 'mm', 'inputDigest': inputs['inputDigest'], 'engineVersion': f'Sew elastic skirt compiler 1 / runtime pin {commit}', 'family': 'skirt', 'panels': panels,
        'stitches': [{'panelA': seam['sides'][0]['panel'], 'edgeA': seam['sides'][0]['edge'], 'panelB': seam['sides'][1]['panel'], 'edgeB': seam['sides'][1]['edge']} for seam in assembly],
        'warnings': warnings + inputs['provenance'], 'classification': 'printable-reference', 'assumptions': warnings,
        'drafting': {'compiler': 'sew-elastic-skirt/1', 'seamAllowanceMm': allowance, 'components': ['body', 'waistband'], 'assembly': assembly, 'measurements': measurements,
            'materials': ['Woven shell: skirt and two-layer casing; yardage requires a fabric-width-specific layout.', f'Elastic trim: assumed {elastic:g} mm finished relaxed circumference plus joining overlap; select and test stretch and width. No interfacing is specified.'], 'operations': operations}}
