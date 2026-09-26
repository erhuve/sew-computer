"""Allowlisted original component compilers; no model-provided executable code."""
from shirt import compile_shirt
from skirt import compile_skirt


def compile_garment(inputs, commit):
    block = inputs['design']['block']
    expected = {'relaxed-drop-shoulder': 'shirt', 'relaxed-dress': 'dress', 'elastic-waist-skirt': 'skirt'}
    if expected.get(block) != inputs['family']:
        raise ValueError('DESIGN_INPUT: Construction and garment family differ.')
    if block == 'elastic-waist-skirt':
        return compile_skirt(inputs, commit)
    result = compile_shirt(inputs, commit)
    if block == 'relaxed-dress':
        result['family'] = 'dress'
        result['drafting']['compiler'] = 'sew-relaxed-dress/1'
        result['engineVersion'] = f'Sew relaxed dress compiler 1 / runtime pin {commit}'
        result['warnings'].insert(0, 'Relaxed straight or flared dress with dropped-shoulder construction. No fitted waist, darts, waist seam or separate skirt is implied; check hem width and walking ease in a toile.')
    return result
