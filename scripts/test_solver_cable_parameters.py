"""Independent fixed-position cable parameter work and recipe tests.

Exact one-dimensional positive-part antiderivatives and an independent
Decimal radial antiderivative provide expectations. No ignored research
oracle, numerical producer private helper, solver or source binder is used.
"""
import copy
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
import unittest

import numpy as np

from solver_continuous_cable_sewing import ContinuousCableSewing
from solver_cable_parameters import CableParameterRecipe


FIELDS = ('targetWorkJoules', 'activationWorkJoules', 'totalWorkJoules',
          'activationIncreaseWorkJoules', 'releaseEnergyRemovedJoules')
BUDGETS = dict(max_boundary_depth=80, max_boundary_panels=256,
               moment_max_terms=128, moment_max_panels=256, moment_max_depth=64)


def rat(value):
    value = F(value)
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def rational(value):
    assert type(value) is dict and set(value) == {'numerator', 'denominator'}
    assert type(value['numerator']) is str and type(value['denominator']) is str
    result = F(int(value['numerator']), int(value['denominator']))
    assert rat(result) == value
    return result


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def anchor(terms):
    return [{'vertex': vertex, 'weight': rat(weight)} for vertex, weight in sorted(terms.items())]


def cell(name='span', **changes):
    value = {'id': name, 'positiveStart': anchor({0: 1}), 'positiveEnd': anchor({1: 1}),
             'negativeStart': anchor({2: 1}), 'negativeEnd': anchor({3: 1}),
             'targetsMeters': [1., 1.], 'activation': 1.,
             'stiffnessDensityNPerM2': 1., 'referenceLengthMeters': rat(1)}
    value.update(changes)
    return value


def recipe(cells=None, vertex_count=4):
    return CableParameterRecipe(ContinuousCableSewing(vertex_count, [cell()] if cells is None else cells))


def positions(radii=(2., 2.)):
    return np.array([[radii[0], 0., 0.], [radii[1], 0., 0.], [0., 0., 0.], [0., 0., 0.]])


def exact_energy(radius, targets, activation, measure=F(1)):
    """Integrate (abs(linear signed radius)-linear target)_+^2 exactly.

    Split at the radius zero and then at each affine residual zero. This
    independently handles a collapsed interior and up to four combined
    old/new taut boundaries without asking the producer for its partitions.
    """
    r0, r1 = map(F, radius); d0, d1 = map(F, targets)
    dr, dd = r1-r0, d1-d0
    cuts = {F(), F(1)}
    if dr and 0 < -r0/dr < 1: cuts.add(-r0/dr)
    result = F()
    for lo, hi in zip(sorted(cuts), sorted(cuts)[1:]):
        sign = 1 if r0+dr*(lo+hi)/2 >= 0 else -1
        a, b = sign*r0-d0, sign*dr-dd
        subcuts = [lo, hi]
        if b and lo < -a/b < hi: subcuts.insert(1, -a/b)
        for left, right in zip(subcuts, subcuts[1:]):
            if a+b*(left+right)/2 <= 0: continue
            result += a*a*(right-left)+a*b*(right**2-left**2)+b*b*(right**3-left**3)/3
    return F(activation)*F(measure)*result/2


def exact_work(radius, before, a0, after, a1, measure=F(1)):
    old = exact_energy(radius, before, a0, measure)
    middle = exact_energy(radius, after, a0, measure)
    new = exact_energy(radius, after, a1, measure)
    activation = new-middle
    return dict(zip(FIELDS, (middle-old, activation, new-old,
                            max(activation, F()), max(-activation, F()))))


def work(control, q, before=(1., 1.), a0=1., after=(1., 1.), a1=1., *, tolerance=1e-11, **budgets):
    return control.parameter_energy_change(q, [list(before)], [a0], [list(after)], [a1],
                                          absolute_tolerance_joules=tolerance, **budgets)


class CableParameterTests(unittest.TestCase):
    def assert_enclosed(self, result, expected, tolerance, component_tolerances=None):
        self.assertTrue(set(FIELDS) <= set(result))
        certificate = result['certificate']
        self.assertIs(certificate['verified'], True)
        self.assertIs(certificate['accepted'], False)
        self.assertEqual(set(certificate['errorsJoules']), set(FIELDS))
        self.assertEqual(rational(certificate['requestedToleranceJoules']), F(tolerance))
        requested = {field:tolerance for field in FIELDS}
        if component_tolerances is not None: requested.update(component_tolerances)
        self.assertEqual(certificate['requestedTolerancesJoules'], {key:rat(value) for key,value in requested.items()})
        for field in FIELDS:
            self.assertIs(type(result[field]), float, field)
            self.assertTrue(math.isfinite(result[field]), field)
            bound = rational(certificate['errorsJoules'][field])
            self.assertGreaterEqual(bound, 0, field)
            self.assertLessEqual(bound, F(requested[field]), field)
            self.assertLessEqual(abs(F(result[field])-expected[field]), bound, field)
        for field in ('activationIncreaseWorkJoules', 'releaseEnergyRemovedJoules'):
            self.assertGreaterEqual(result[field], 0.)
        # The independently enclosed direct total need not equal the sum of
        # separately rounded components, but their uncertainty must explain it.
        component_error = sum((rational(certificate['errorsJoules'][key]) for key in FIELDS[:3]), F())
        self.assertLessEqual(abs(F(result['totalWorkJoules'])-F(result['targetWorkJoules'])-
                                 F(result['activationWorkJoules'])), component_error)

    def test_exact_root_partition_grid_covers_engage_release_and_retarget(self):
        control = recipe(); tol = 1e-10
        radii = ((1., 3.), (3., 1.), (2., 2.))
        limits = ((.5, .5), (2., 2.), (1., 3.))
        cases = 0
        for radius, before, after, a0, a1 in itertools.product(radii, limits, limits, (0., .5, 1.), (0., .5, 1.)):
            with self.subTest(radius=radius, before=before, after=after, a0=a0, a1=a1):
                result = work(control, positions(radius), before, a0, after, a1, tolerance=tol)
                self.assert_enclosed(result, exact_work(radius, before, a0, after, a1), tol)
                cases += 1
        self.assertEqual(cases, 243)

    def test_direct_total_survives_retarget_release_component_cancellation(self):
        control = recipe(); q = positions((3., 3.)); a1 = math.nextafter(.25, math.inf)
        expected = exact_work((3., 3.), (2., 2.), 1., (1., 1.), a1)
        result = work(control, q, (2., 2.), 1., (1., 1.), a1, tolerance=1e-12)
        self.assert_enclosed(result, expected, 1e-12)
        self.assertEqual(expected['totalWorkJoules'], F(1, 2**53))
        self.assertEqual(result['totalWorkJoules'], 2.**-53)
        self.assertEqual(math.fsum((result['targetWorkJoules'], result['activationWorkJoules'])), 0.)
        self.assertEqual(rational(result['certificate']['errorsJoules']['totalWorkJoules']), 0)
        # A tiny total limit cannot silently excuse unrepresentable large
        # component outputs. Separate explicit limits may permit that case.
        with self.assertRaises(ValueError):
            work(control,q,(2.,2.),1.,(1.,1.),a1,tolerance=1e-30)
        separate = {field:1e-12 for field in FIELDS if field!='totalWorkJoules'}
        strict_total = work(control,q,(2.,2.),1.,(1.,1.),a1,tolerance=1e-30,
                            component_tolerances_joules=separate)
        self.assert_enclosed(strict_total,expected,1e-30,separate)
        self.assertEqual(strict_total['totalWorkJoules'],2.**-53)

    def test_beta_uses_original_exact_product_not_rounded_fold_convention(self):
        control = recipe([cell(stiffnessDensityNPerM2=1e12)])
        old, new = .25, math.nextafter(.25, math.inf)
        expected = exact_work((2., 2.), (1., 1.), old, (1., 1.), new, F(1e12))
        rounded_beta_work = (F(float(F(1e12)*F(new)))-F(float(F(1e12)*F(old))))/2
        self.assertNotEqual(expected['totalWorkJoules'], rounded_beta_work)
        result = work(control, positions(), a0=old, a1=new, tolerance=1e-12)
        self.assert_enclosed(result, expected, 1e-12)
        self.assertGreater(abs(F(result['totalWorkJoules'])-rounded_beta_work), F(1e-7))

    def test_exact_rational_measure_and_affine_targets(self):
        control = recipe([cell(referenceLengthMeters=rat(F(1, 7)), stiffnessDensityNPerM2=.1)])
        radius, before, after = (1.25, 3.5), (.75, 3.75), (2., .5)
        result = work(control, positions(radius), before, .3, after, .7, tolerance=1e-12)
        self.assert_enclosed(result, exact_work(radius, before, .3, after, .7, F(.1)/7), 1e-12)

    def test_collapsed_interior_four_rational_taut_boundaries(self):
        # D(u)=(2u-1,0,0); old ±1/4 and new ±1/2 boundaries
        # yield four distinct roots while the center remains safely slack.
        control = recipe(); radius = (-1., 1.)
        expected = exact_work(radius, (.25, .25), .75, (.5, .5), .25)
        result = work(control, positions(radius), (.25, .25), .75, (.5, .5), .25, tolerance=1e-12)
        self.assert_enclosed(result, expected, 1e-12)
        self.assertLess(result['totalWorkJoules'], 0.)

    def test_non_square_four_irrational_roots_against_decimal_antiderivative(self):
        control = recipe(); q = np.array([[-.5, 1., 0.], [.5, 1., 0.], [0., 0., 0.], [0., 0., 0.]])
        d0, d1, a0, a1 = 65/64, 33/32, .75, .25
        with localcontext() as context:
            context.prec = 150
            def energy(target):
                d = Decimal.from_float(target); lower = (d*d-1).sqrt()
                def primitive(t):
                    r = (1+t*t).sqrt()
                    return (1+d*d)*t+t*t*t/3-d*(t*r+(t+r).ln())
                return primitive(Decimal('.5'))-primitive(lower)
            old, new = energy(d0), energy(d1)
            target = Decimal.from_float(a0)*(new-old)
            activation = (Decimal.from_float(a1)-Decimal.from_float(a0))*new
            expected = dict(zip(FIELDS, map(F, (target, activation, target+activation, Decimal(0), -activation))))
        result = work(control, q, (d0, d0), a0, (d1, d1), a1, tolerance=1e-11)
        self.assert_enclosed(result, expected, 1e-11)

    def test_mixed_cell_engagement_and_release_remain_separate_nonnegative_sums(self):
        cells = [cell('release'), cell('engage', stiffnessDensityNPerM2=2.)]
        control = recipe(cells); q = positions((3., 3.))
        before, after = [[2., 2.], [1., 1.]], [[1., 1.], [2., 2.]]
        a0, a1 = [1., 0.], [0., 1.]
        expected = {key:F() for key in FIELDS}
        for index in range(2):
            for key,value in exact_work((3., 3.), before[index], a0[index], after[index], a1[index], F(index+1)).items():
                expected[key] += value
        result = control.parameter_energy_change(q, before, a0, after, a1, absolute_tolerance_joules=1e-12)
        self.assert_enclosed(result, expected, 1e-12)
        self.assertEqual(expected['activationWorkJoules'], -1)
        self.assertEqual(result['activationIncreaseWorkJoules'], 1.)
        self.assertEqual(result['releaseEnergyRemovedJoules'], 2.)

    def test_tiny_target_work_survives_equal_rounded_endpoint_energies(self):
        control = recipe(); q = positions((100., 100.))
        d0, d1 = .001, math.nextafter(.001, math.inf)
        expected = exact_work((100., 100.), (d0, d0), 1., (d1, d1), 1.)
        self.assertEqual(float(exact_energy((100., 100.), (d0, d0), 1.)),
                         float(exact_energy((100., 100.), (d1, d1), 1.)))
        result = work(control, q, (d0, d0), 1., (d1, d1), 1., tolerance=1e-28)
        self.assert_enclosed(result, expected, 1e-28)
        self.assertLess(result['totalWorkJoules'], 0.)

    def test_opposite_stiff_cells_do_not_erase_small_third_cell_work(self):
        cells=[cell('first',stiffnessDensityNPerM2=1e12,referenceLengthMeters=rat(100)),
               cell('opposite',stiffnessDensityNPerM2=1e12,referenceLengthMeters=rat(100)),
               cell('small')]
        control=recipe(cells);q=positions((100.,100.));tiny_target=math.nextafter(1.,math.inf)
        before=[[1.,1.],[2.,2.],[1.,1.]]
        after=[[2.,2.],[1.,1.],[tiny_target,tiny_target]]
        expected={key:F() for key in FIELDS}
        for index,measure in enumerate((F(1e12)*100,F(1e12)*100,F(1))):
            for key,value in exact_work((100.,100.),before[index],1.,after[index],1.,measure).items():
                expected[key]+=value
        result=control.parameter_energy_change(q,before,[1.,1.,1.],after,[1.,1.,1.],
                                               absolute_tolerance_joules=1e-25)
        self.assert_enclosed(result,expected,1e-25)
        self.assertLess(result['totalWorkJoules'],0.)
        self.assertEqual(result['totalWorkJoules'],float(expected['totalWorkJoules']))

    def test_original_converted_anchor_coefficients_are_not_renormalized(self):
        terms=anchor({0:F(1,3),1:F(2,3)})
        control=recipe([cell('converted',positiveStart=terms,positiveEnd=terms,stiffnessDensityNPerM2=1e12),
                        cell('opposite',positiveStart=anchor({4:1}),positiveEnd=anchor({4:1}),
                             stiffnessDensityNPerM2=1e12)],vertex_count=5)
        q=np.array([[3.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[1.,0.,0.]])
        radius=3*F(float(F(1,3)))
        expected=exact_work((radius,radius),(.25,.25),1.,(.5,.5),1.,F(1e12))
        normalized_radius=radius/(F(float(F(1,3)))+F(float(F(2,3))))
        normalized=exact_work((normalized_radius,normalized_radius),(.25,.25),1.,(.5,.5),1.,F(1e12))
        opposite=exact_work((1.,1.),(.5,.5),1.,(.25,.25),1.,F(1e12))
        for key in FIELDS:
            expected[key]+=opposite[key]
            normalized[key]+=opposite[key]
        self.assertEqual(normalized['totalWorkJoules'],0)
        self.assertNotEqual(expected['totalWorkJoules'],normalized['totalWorkJoules'])
        result=control.parameter_energy_change(q,[[.25,.25],[.5,.5]],[1.,1.],
            [[.5,.5],[.25,.25]],[1.,1.],absolute_tolerance_joules=1e-18)
        self.assert_enclosed(result,expected,1e-18)
        # Opposite work removes the common large term; normalization would
        # now erase a positive signal far above the certified rounding error.
        bound=rational(result['certificate']['errorsJoules']['totalWorkJoules'])
        self.assertGreater(abs(F(result['totalWorkJoules'])-normalized['totalWorkJoules']),bound)

    def test_identical_crossing_targets_have_exact_zero_target_work_bound(self):
        control=recipe();q=positions((1.,3.))
        result=work(control,q,(2.,2.),.25,(2.,2.),.75,tolerance=1e-11)
        self.assert_enclosed(result,exact_work((1.,3.),(2.,2.),.25,(2.,2.),.75),1e-11)
        self.assertEqual(result['targetWorkJoules'],0.)
        self.assertEqual(rational(result['certificate']['errorsJoules']['targetWorkJoules']),0)

    def test_inactive_old_target_roots_do_not_consume_boundary_depth(self):
        control=recipe();q=positions((1.,3.))
        # Old target crosses at u=1/2 but contributes no work because its
        # activation is zero. New targets are wholly taut or wholly slack.
        for target in ((.5,.5),(4.,4.)):
            with self.subTest(target=target):
                result=work(control,q,(2.,2.),0.,target,.5,tolerance=1e-11,
                            max_boundary_depth=0)
                self.assert_enclosed(result,exact_work((1.,3.),(2.,2.),0.,target,.5),1e-11)
                self.assertEqual(result['targetWorkJoules'],0.)
                self.assertEqual(rational(result['certificate']['errorsJoules']['targetWorkJoules']),0)

    def test_positive_subnormal_work_retains_nonzero_enclosure_and_aggregation(self):
        tiny = math.ulp(0.)
        for count in (1, 3):
            control = recipe([cell('span-'+str(i)) for i in range(count)])
            result = control.parameter_energy_change(positions(), [[1.,1.]]*count, [0.]*count,
                [[1.,1.]]*count, [tiny]*count, absolute_tolerance_joules=tiny)
            exact = F(tiny)*count/2
            expected = dict(zip(FIELDS, (F(), exact, exact, exact, F())))
            self.assert_enclosed(result, expected, tiny)
            self.assertGreater(rational(result['certificate']['errorsJoules']['totalWorkJoules']), 0)
            self.assertEqual(result['totalWorkJoules'], float(exact))
        # A coefficient that rounds to zero remains inadmissible under the
        # base primitive's existing positive-effective-stiffness rule.
        control = recipe([cell(stiffnessDensityNPerM2=.5)])
        with self.assertRaises(ValueError): control.potential([[1.,1.]], [tiny])

    def test_requested_error_below_output_rounding_floor_rejects(self):
        control = recipe([cell(referenceLengthMeters=rat(F(1,3)))])
        exact = F(1,6); requested = 1e-30
        self.assertGreater(abs(F(float(exact))-exact), F(requested))
        with self.assertRaises(ValueError): work(control, positions(), a0=0., a1=1., tolerance=requested)

    def test_slack_coincidence_and_all_pending_controls_have_zero_work(self):
        control = recipe(); q = np.zeros((4,3))
        for a0,a1 in ((0.,0.),(0.,1.),(1.,0.),(1.,1.)):
            result = work(control, q, (100.,.001), a0, (.5,100.), a1, tolerance=1e-20)
            self.assert_enclosed(result, {key:F() for key in FIELDS}, 1e-20)
            for field in FIELDS:
                self.assertEqual(result[field], 0.)
                self.assertEqual(rational(result['certificate']['errorsJoules'][field]), 0)

    def test_fixed_position_parameter_cycles_telescope_and_reverse_attribution(self):
        control = recipe(); q = positions((1.,3.)); state=((.5,.5),0.); actual=[]; exact=[]
        for target,activation in (((2.,2.),.5),((.75,1.5),1.),((.5,.5),0.)):
            result = work(control, q, state[0], state[1], target, activation, tolerance=1e-11)
            expected = exact_work((1.,3.), state[0], state[1], target, activation)
            self.assert_enclosed(result, expected, 1e-11)
            actual.append(result);exact.append(expected);state=(target,activation)
        self.assertEqual(sum((x['totalWorkJoules'] for x in exact),F()),0)
        total=sum((F(x['totalWorkJoules']) for x in actual),F())
        error=sum((rational(x['certificate']['errorsJoules']['totalWorkJoules']) for x in actual),F())
        self.assertLessEqual(abs(total),error)
        forward=work(control,q,(.5,.5),.25,(2.,2.),.75)
        reverse=work(control,q,(2.,2.),.75,(.5,.5),.25)
        self.assertLessEqual(abs(F(forward['totalWorkJoules'])+F(reverse['totalWorkJoules'])),
            rational(forward['certificate']['errorsJoules']['totalWorkJoules'])+rational(reverse['certificate']['errorsJoules']['totalWorkJoules']))
        self.assertNotEqual(forward['targetWorkJoules'],-reverse['targetWorkJoules'])

    def test_recipe_geometry_identity_excludes_only_varying_parameters(self):
        original = recipe(); changed = recipe([cell(targetsMeters=[.5,2.],activation=0.)])
        self.assertEqual(original.geometry_sha256, changed.geometry_sha256)
        self.assertEqual(original.vertex_count,4)
        self.assertEqual(tuple(original.cell_ids),('span',))
        self.assertRegex(original.geometry_sha256,r'^[0-9a-f]{64}$')
        geometry_cell=cell();geometry_cell.pop('targetsMeters');geometry_cell.pop('activation')
        self.assertEqual(original.geometry_sha256,
            hashlib.sha256(encoded({'vertexCount':4,'cells':[geometry_cell]})).hexdigest())
        self.assertEqual(original.stiffness_measures,(F(1),))
        for modified in (cell(name='other'),cell(stiffnessDensityNPerM2=2.),
                         cell(referenceLengthMeters=rat(F(1,2))),
                         cell(positiveStart=anchor({0:F(1,3),1:F(2,3)}))):
            self.assertNotEqual(original.geometry_sha256,recipe([modified]).geometry_sha256)
        self.assertNotEqual(recipe([cell('a'),cell('b')]).geometry_sha256,
                            recipe([cell('b'),cell('a')]).geometry_sha256)

    def test_fresh_parameter_arrays_potentials_and_immutable_recipe(self):
        base = ContinuousCableSewing(4,[cell()]); control = CableParameterRecipe(base)
        first_targets,first_activation=control.initial_parameters
        self.assertEqual(first_targets.shape,(1,2));self.assertEqual(first_activation.shape,(1,))
        first_targets[:]=.25;first_activation[:]=0.
        np.testing.assert_array_equal(control.initial_parameters[0],[[1.,1.]])
        np.testing.assert_array_equal(control.initial_parameters[1],[1.])
        targets=np.array([[.5,.75]]);activation=np.array([.25])
        sampled=control.parameters(targets,activation)
        tuple_parameters=control.parameters(((.5,.75),),(.25,))
        np.testing.assert_array_equal(tuple_parameters[0],targets)
        effective=control.potential(targets,activation)
        self.assertIs(type(effective),ContinuousCableSewing)
        targets[:]=3.;activation[:]=0.;sampled[0][:]=4.;sampled[1][:]=1.
        self.assertEqual(effective.cells[0]['targetsMeters'],[.5,.75])
        self.assertEqual(effective.cells[0]['activation'],.25)
        self.assertEqual(base.cells[0]['targetsMeters'],[1.,1.])
        snapshot=control.description();snapshot.clear();self.assertTrue(control.description())
        exposed=effective.cells;exposed[0]['positiveStart'][0]['weight']=rat(0)
        self.assertEqual(effective.cells[0]['positiveStart'][0]['weight'],rat(1))
        for name in ('vertex_count','cell_ids','geometry_sha256'):
            with self.assertRaises(AttributeError):setattr(control,name,None)
            with self.assertRaises(AttributeError):delattr(control,name)

    def test_certificate_state_parameter_potential_identity_and_budget_binding(self):
        control=recipe();q=positions((1.,3.));old=[[.5,.75]];new=[[1.25,1.5]];a0=[.25];a1=[.75]
        result=control.parameter_energy_change(q,old,a0,new,a1,absolute_tolerance_joules=1e-11,**BUDGETS)
        certificate=result['certificate']
        self.assertEqual(certificate['profile'],'continuous-cable-parameter-work-enclosure-v1')
        self.assertEqual(certificate['parameterOrder'],'target-first-at-old-activation-then-activation-at-new-target')
        self.assertEqual(certificate['geometrySha256'],control.geometry_sha256)
        self.assertEqual(certificate['positionsSha256'],hashlib.sha256(encoded(q.tolist())).hexdigest())
        self.assertEqual(certificate['beforePotentialSha256'],control.potential(old,a0).description()['inputSha256'])
        self.assertEqual(certificate['afterPotentialSha256'],control.potential(new,a1).description()['inputSha256'])
        self.assertEqual(certificate['budgets'],BUDGETS)
        for field in ('beforeParameterSha256','afterParameterSha256'):
            self.assertRegex(certificate[field],r'^[0-9a-f]{64}$')
        self.assertEqual(certificate['beforeParameterSha256'],hashlib.sha256(encoded({'targetsMeters':old,'activation':a0})).hexdigest())
        self.assertEqual(certificate['afterParameterSha256'],hashlib.sha256(encoded({'targetsMeters':new,'activation':a1})).hexdigest())
        self.assertNotEqual(certificate['beforeParameterSha256'],certificate['afterParameterSha256'])
        again=control.parameter_energy_change(q,old,a0,new,a1,absolute_tolerance_joules=1e-11,**BUDGETS)
        self.assertEqual(result,again)
        certificate['errorsJoules']['totalWorkJoules']=rat(999)
        self.assertNotEqual(certificate,again['certificate'])
        json.dumps(again,allow_nan=False)

    def test_raw_inputs_shapes_narrowing_and_budget_failures_reject(self):
        control=recipe();q=positions()
        for targets,activation in (([[True,1.]],[1.]),([[1.,1.]],[False]),([[0.,1.]],[1.]),
            ([[1.,math.inf]],[1.]),([[101.,1.]],[1.]),([[1.,1.]],[-.1]),([[1.,1.]],[1.01]),
            ([1.,1.],[1.]),([[1.,1.],[1.,1.]],[1.]),([[1.,1.]],[[1.]])):
            with self.subTest(targets=targets,activation=activation),self.assertRaises(ValueError):
                control.parameters(targets,activation)
        if np.dtype(np.longdouble).itemsize>8:
            with self.assertRaises(ValueError):control.parameters(np.ones((1,2),dtype=np.longdouble),[1.])
        for bad in (None,object(),{'vertex_count':4}):
            with self.assertRaises(ValueError):CableParameterRecipe(bad)
        for bad in ([[True,0.,0.]]+q[1:].tolist(),np.full_like(q,math.nan),q[:2]):
            with self.assertRaises(ValueError):work(control,bad)
        for bad in (0.,-1.,True,math.inf):
            with self.assertRaises(ValueError):work(control,q,tolerance=bad)
        for name,value in (('max_boundary_depth',True),('max_boundary_panels',0),('moment_max_terms',0),
                           ('moment_max_panels',1.5),('moment_max_depth',-1)):
            with self.subTest(budget=name),self.assertRaises(ValueError):work(control,q,**{name:value})
        # Genuine old/new active-set boundary with insufficient fixed depth.
        with self.assertRaises(ValueError):
            work(control,positions((.5,2.)),(.75,.75),1.,(1.25,1.25),.5,
                 tolerance=1e-12,max_boundary_depth=0)

    def test_optional_component_tolerances_are_explicit_complete_and_non_boolean(self):
        control=recipe();q=positions();valid={key:1e-9 for key in FIELDS if key!='totalWorkJoules'}
        admitted=work(control,q,a0=0.,a1=1.,tolerance=1e-12,component_tolerances_joules=valid)
        self.assert_enclosed(admitted,exact_work((2.,2.),(1.,1.),0.,(1.,1.),1.),1e-12,valid)
        attacks=[{},dict(valid,totalWorkJoules=1e-9),list(valid.items()),
                 {key:value for key,value in valid.items() if key!='targetWorkJoules'}]
        for bad in (True,0.,-1.,math.inf,np.float64(1e-9)):
            attacked=valid.copy();attacked['targetWorkJoules']=bad;attacks.append(attacked)
        for attack in attacks:
            with self.subTest(component_tolerances=attack),self.assertRaises(ValueError):
                work(control,q,component_tolerances_joules=attack)

    def test_calls_preserve_original_inputs_and_declarations(self):
        control=recipe();q=positions((1.,3.));before=np.array([[.5,.75]]);after=np.array([[1.25,1.5]])
        a0=np.array([.25]);a1=np.array([.75]);arrays=[q,before,a0,after,a1]
        snapshots=[x.copy() for x in arrays];definition=copy.deepcopy(control.description())
        control.parameter_energy_change(*arrays,absolute_tolerance_joules=1e-11)
        for actual,expected in zip(arrays,snapshots):np.testing.assert_array_equal(actual,expected)
        self.assertEqual(control.description(),definition)


if __name__=='__main__':unittest.main()
