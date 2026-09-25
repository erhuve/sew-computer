"""Independent contraction oracles with lazy locked-runtime integration tests.

Only CableContractionMathTests run in the host stdlib review. Response tests
import numerical modules inside setUp, for the declared locked runtime.
"""
from fractions import Fraction as F
import importlib.util
import io
import itertools
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

S=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent_tracked_contraction',S/'solver_cable_contraction.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
P=F(-1,2);T=F(-3,2)
PROFILE='independent-cable-contraction-stdlib-v1'

def oracle(local,moments,width,*,corners=False):
    """Independent Minkowski endpoint formula, with optional Cartesian extrema."""
    result={}
    for key,polynomial,half,three in local:
        offset=sum((c/F(i+1) for i,c in enumerate(polynomial)),F())
        intervals=[]
        for power,coefficients in ((P,half),(T,three)):
            for i,c in enumerate(coefficients):
                if c:
                    a,b=moments['moments'][power][i]
                    intervals.append((c*a,c*b))
        if corners:
            values=[width*(offset+sum(choice,F())) for choice in itertools.product(*intervals)]
            result[key]=min(values),max(values)
        else:
            a=width*(offset+sum((min(v) for v in intervals),F()))
            b=width*(offset+sum((max(v) for v in intervals),F()))
            result[key]=min(a,b),max(a,b)
    return result

def rows(poly=(F(3,7),),half=(F(2,3),),three=(F(-4,5),)):
    return [(key,poly,half,three) for key in (('e',),('g',1),('h',1,1),('h',1,2))]

def moments():
    return {'moments':{P:tuple((F(i+1,11),F(i+3,7)) for i in range(6)),
                       T:tuple((F(i+2,13),F(i+5,9)) for i in range(6))}}

class CableContractionMathTests(unittest.TestCase):
    def equal(self,local,table,width,*,corners=False):
        expected=oracle(local,table,width,corners=corners);actual=M.contract(local,table,width)
        self.assertIsNotNone(actual);self.assertEqual(actual,expected)
        for pair in actual.values():
            self.assertEqual(len(pair),2)
            for value in pair:self.assertIs(type(value),F)
        return actual
    def test_signed_linear_forms_match_cartesian_extrema(self):
        cases=[((F(0),),(F(1),),(F(0),)),((F(-5,7),F(3,2)),(F(-2,3),F(4,9)),(F(7,11),)),
               ((F(2),F(0),F(-3)),(F(0),F(1,7)),(F(-4,5),F(2,9))),
               ((F(0),),(F(1),F(-1)),(F(1),F(-1)))]
        for poly,half,three in cases:
            for width in (F(1),F(2,7),F(13,5)):
                with self.subTest(width=str(width),terms=len(half)+len(three)):
                    self.equal(rows(poly,half,three),moments(),width,corners=True)
    def test_cancellation_retains_uncertainty_width(self):
        local=rows((F(0),),(F(1),F(-1)),(F(0),))
        table={'moments':{P:((F(2),F(5)),(F(2),F(5)))}}
        result=self.equal(local,table,F(2,3),corners=True)
        self.assertEqual(result[('e',)],(F(-2),F(2)))
        self.assertNotEqual(result[('e',)],(F(0),F(0)))
    def test_full_degree_two_powers_and_deterministic_rational_cases(self):
        rng=random.Random(170129)
        for _ in range(18):
            make=lambda:tuple(F(rng.randrange(-11,12),rng.choice((1,2,3,5,7,11))) for _ in range(6))
            local=rows(make(),make(),make())
            self.equal(local,moments(),F(rng.randrange(1,9),rng.randrange(1,9)))
    def test_point_reversed_signed_and_shared_endpoint_intervals(self):
        table={'moments':{P:((F(2),F(2)),(F(5),F(-3)),(F(-8,9),F(-1,3))),
                          T:((F(-2),F(4)),)}}
        self.equal(rows((F(-2),),(F(1),F(-3),F(5,7)),(F(-2),)),table,F(5,13),corners=True)
    def test_subnormal_large_rational_and_nonidentity_panel_widths(self):
        tiny=F(float.fromhex('0x0.0000000000001p-1022'))
        huge=F(2**2048+1,2**701-1)
        for width in (tiny,F(2**301-1,2**599+1),F(7,9)-F(2,11)):
            self.equal(rows((tiny,-huge,F(7,13)),(huge,tiny),(F(-11,17),)),moments(),width)
    def test_zero_terms_do_not_lookup_missing_or_malformed_moments(self):
        class Poison:
            def __getattribute__(self,name):raise AssertionError('unused moments accessed')
        local=rows((F(2),F(-7,3),F(0)),(F(0),),(F(0),F(0)))
        self.equal(local,Poison(),F(2,5));self.equal(local,None,F(1))
        # The used first entry is valid; trailing absent/bad entries are irrelevant.
        local=rows(half=(F(2,3),F(0),F(0)),three=(F(0),))
        table={'moments':{P:((F(1,5),F(2,3)),object()),T:object()},'ignored':object()}
        self.equal(local,table,F(1))
    def test_structural_fallback_never_invokes_user_protocols(self):
        seen=[]
        def bad(*args,**kwargs):seen.append('hook');raise AssertionError('user hook')
        class EvilF(F):
            __bool__=__mul__=__eq__=bad
            numerator=property(bad)
        class EvilTuple(tuple):__len__=__iter__=__getitem__=bad
        class EvilList(list):__len__=__iter__=__getitem__=bad
        class EvilDict(dict):__len__=__iter__=__getitem__=get=bad
        class EvilStr(str):__eq__=bad;__hash__=str.__hash__
        class Object:
            __bool__=__mul__=__rmul__=__iter__=bad
            numerator=property(bad);denominator=property(bad)
        local=rows();table=moments()
        probes=[(EvilList(local),table,F(1)),([EvilTuple(local[0])]+local[1:],table,F(1)),
                (rows(poly=EvilTuple((F(1),))),table,F(1)),(local,EvilDict(table),F(1)),
                (rows(poly=(EvilF(1),)),table,F(1)),(rows(half=(Object(),)),table,F(1)),
                (local,table,EvilF(1)),(local,table,Object()),
                ([(EvilTuple(('e',)),*local[0][1:])]+local[1:],table,F(1)),
                ([((EvilStr('e'),),*local[0][1:])]+local[1:],table,F(1)),
                (local,{'moments':EvilDict(table['moments'])},F(1)),
                (local,{'moments':{P:EvilTuple(table['moments'][P]),T:table['moments'][T]}},F(1))]
        for a,b,c in probes:self.assertIsNone(M.contract(a,b,c))
        self.assertEqual(seen,[])
    def test_numeric_noncanonical_and_width_fallback(self):
        for value in (True,1,0.0,float('inf'),float('nan')):
            self.assertIsNone(M.contract(rows(half=(value,)),moments(),F(1)))
        for numerator,denominator in ((2,4),(0,2),(1,0),(1,-3),(True,1),(1.0,1)):
            value=F(1);object.__setattr__(value,'_numerator',numerator);object.__setattr__(value,'_denominator',denominator)
            self.assertIsNone(M.contract(rows(half=(value,)),moments(),F(1)))
        for width in (F(0),F(-1,7),1,True,1.0):self.assertIsNone(M.contract(rows(),moments(),width))
    def test_shape_degree_row_and_operand_caps_fall_back(self):
        self.assertEqual((M.MAX_BITS,M.MIN_ROWS,M.MAX_ROWS,M.MAX_DEGREE),(65536,4,1024,5))
        for n in range(4):self.assertIsNone(M.contract(rows()[:n],moments(),F(1)))
        self.assertIsNone(M.contract([rows()[0]]*(M.MAX_ROWS+1),moments(),F(1)))
        self.equal([rows()[0]]*M.MAX_ROWS,moments(),F(1))
        self.assertIsNone(M.contract(rows(poly=(F(1),)*7),moments(),F(1)))
        self.assertIsNone(M.contract(rows(poly=()),moments(),F(1)))
        oversized=F(1<<M.MAX_BITS)
        self.assertIsNone(M.contract(rows(half=(oversized,)),moments(),F(1)))
        for table in (None,{}, {'moments':{}},{'moments':{P:()}}, {'moments':{P:((F(1),),)}}):
            self.assertIsNone(M.contract(rows(three=(F(0),)),table,F(1)))
    def test_per_operation_growth_checks_precede_products_and_additions(self):
        seen=[]
        class Probe(int):
            def __mul__(self,other):seen.append('mul');return int(self)*other
            def __add__(self,other):seen.append('add');return int(self)+other
        with patch.object(M,'MAX_BITS',12):
            self.assertIsNone(M._product(Probe(64),64));self.assertEqual(seen,[])
            self.assertIsNone(M._sum(Probe(2048),-2048));self.assertEqual(seen,[])
            self.assertEqual(M._product(Probe(3),5),15);self.assertEqual(seen,['mul'])
            self.assertEqual(M._sum(Probe(3),5),8);self.assertEqual(seen,['mul','add'])
            # Each operand fits, but coprime endpoint denominators do not.
            table={'moments':{P:((F(1,61),F(1,67)),)}}
            self.assertIsNone(M.contract(rows(three=(F(0),)),table,F(1)))
            # Shared moment basis fits; final positive width product does not.
            self.assertIsNone(M.contract(rows((F(2048),),(F(0),),(F(0),)),None,F(2)))
            # Coefficient LCM and offset denominator overflow use the same fallback.
            self.assertIsNone(M.contract(rows((F(1,61),),(F(1,67),),(F(0),)),{'moments':{P:((F(1),F(1)),)}},F(1)))
    def test_fallback_dispatch_preserves_exception_identity_once(self):
        local=rows(half=(True,));table=moments();calls=[]
        for failure in (ValueError('original'),KeyboardInterrupt('stop')):
            def original():calls.append(failure);raise failure
            def dispatched():
                value=M.contract(local,table,F(1))
                return original() if value is None else value
            with self.assertRaises(type(failure)) as caught:dispatched()
            self.assertIs(caught.exception,failure)
        self.assertEqual(len(calls),2)
    def test_results_are_fresh_and_inputs_do_not_retain_output_aliases(self):
        local=rows();table=moments();before=repr((local,table));a=self.equal(local,table,F(1));b=self.equal(local,table,F(1))
        keys=list(a);self.assertIsNot(a[keys[0]][0],a[keys[1]][0]);self.assertIsNot(a[keys[0]][0],b[keys[0]][0])
        expected=b[keys[1]][0];object.__setattr__(a[keys[0]][0],'_numerator',123456789)
        self.assertEqual(a[keys[1]][0],expected);self.assertEqual(b[keys[0]][0],expected)
        self.assertEqual(repr((local,table)),before)

DECLARED_CONTROLS=tuple(sorted('CableContractionMathTests.'+n for n in vars(CableContractionMathTests) if n.startswith('test_')))
def run_checks():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CableContractionMathTests)
    names=sorted(type(t).__name__+'.'+t._testMethodName for t in suite);out=io.StringIO()
    result=unittest.TextTestRunner(stream=out,verbosity=2).run(suite)
    native=any(n.split('.')[0] in ('numpy','scipy','warp','ipctk','newton') for n in sys.modules)
    def records(items):return [{'test':type(t).__name__+'.'+t._testMethodName,'detail':detail} for t,detail in items]
    return {'profile':PROFILE,'verified':result.wasSuccessful() and not result.skipped and not native,'successful':result.wasSuccessful() and not result.skipped and not native,
        'testsRun':result.testsRun,'testNames':names,'errors':records(result.errors),'failures':records(result.failures),'skipped':records(result.skipped),
        'nativeModulesImported':native,'newMotionExecuted':False,'output':out.getvalue()}


class CableContractionResponseTests(unittest.TestCase):
    """Fresh locked-runtime responses against the unchanged forced fallback."""
    def setUp(self):
        import solver_continuous_cable_sewing as cable
        import solver_cable_parameters as parameters
        import test_solver_cable_composition as composition
        self.cable, self.parameters, self.composition = cable, parameters, composition
        self.potential = cable.ContinuousCableSewing(4, [composition.response_cell()])
        self.precision = {'energy_tolerance_joules':1e-8, 'gradient_tolerance_newtons':1e-8,
                          'hessian_tolerance_newtons_per_meter':1e-8}
        self.tolerances = {'e':F(1,10**8), 'g':F(1,10**8), 'h':F(1,10**8)}
        self.budgets = {'moment_max_panels':256, 'moment_max_terms':128, 'moment_max_depth':64}

    def assert_response_identical(self, first, second):
        self.composition.CableCompositionResponseTests.assert_response_identical(self, first, second)

    def test_complete_responses_and_fresh_moment_requests_match_original(self):
        fixtures = [[[0.,0.,0.], [0.,0.,0.], [1.,.2,.1], [2.,.4,.2]],
                    [[0.,0.,0.], [0.,0.,0.], [.1,0.,0.], [.2,0.,0.]],
                    [[0.,0.,0.], [0.,0.,0.], [.25,0.,0.], [1.25,0.,0.]]]
        radial, contract = self.cable.radial_moment_bounds, self.cable._contract_intervals
        fast = []
        for positions in fixtures:
            records = []
            def observed_radial(*args, **kwargs):
                result = radial(*args, **kwargs)
                records.append((args, kwargs, result))
                return result
            def observed_contract(*args):
                result = contract(*args)
                fast.append(result is not None)
                return result
            with patch.object(self.cable, 'radial_moment_bounds', observed_radial), \
                 patch.object(self.cable, '_contract_intervals', observed_contract):
                actual = self.potential.evaluate(positions, **self.precision)
            first_calls = list(records);records.clear()
            with patch.object(self.cable, 'radial_moment_bounds', observed_radial), \
                 patch.object(self.cable, '_contract_intervals', return_value=None):
                expected = self.potential.evaluate(positions, **self.precision)
            self.assert_response_identical(actual, expected)
            self.assertEqual(first_calls, records)
            for first, second in zip(first_calls, records):
                self.assertIsNot(first[2], second[2])
        self.assertIn(True, fast)

    def test_fixed_and_varying_parameter_work_and_effective_responses_match_original(self):
        first = [[0.,0.,0.], [0.,0.,0.], [.25,0.,0.], [1.25,0.,0.]]
        last = [[0.,0.,0.], [0.,0.,0.], [.5,0.,0.], [1.5,0.,0.]]
        actual = self.potential.energy_change(first, last, absolute_tolerance_joules=1e-8)
        with patch.object(self.cable, '_contract_intervals', return_value=None):
            expected = self.potential.energy_change(first, last, absolute_tolerance_joules=1e-8)
        self.assertNotEqual(actual['changeJoules'], 0.)
        self.assertEqual(json.dumps(actual, sort_keys=True, allow_nan=False),
                         json.dumps(expected, sort_keys=True, allow_nan=False))
        recipe = self.parameters.CableParameterRecipe(self.potential)
        # Parameter work has its own unchanged contraction, not an imported
        # _contract_intervals alias. Its fresh effective potentials use the new
        # cable hook and are compared separately; no alias is invented here.
        for positions in (first, [[0.,0.,0.], [0.,0.,0.], [1.,0.,0.], [2.,0.,0.]]):
            arguments = (positions, [[.5,.5]], [1.], [[.75,.75]], [.5])
            actual = recipe.parameter_energy_change(*arguments, absolute_tolerance_joules=1e-8)
            with patch.object(self.cable, '_contract_intervals', return_value=None):
                expected = recipe.parameter_energy_change(*arguments, absolute_tolerance_joules=1e-8)
            self.assertNotEqual(actual['totalWorkJoules'], 0.)
            self.assertGreater(actual['releaseEnergyRemovedJoules'], 0.)
            self.assertEqual(json.dumps(actual, sort_keys=True, allow_nan=False),
                             json.dumps(expected, sort_keys=True, allow_nan=False))
            effective = recipe.potential([[.75,.75]], [.5])
            actual_response = effective.evaluate(positions, **self.precision)
            with patch.object(self.cable, '_contract_intervals', return_value=None):
                expected_response = effective.evaluate(positions, **self.precision)
            self.assert_response_identical(actual_response, expected_response)

    def test_zero_radial_and_energy_only_paths_preserve_laziness_and_stats(self):
        q = (F(4),F(),F())
        zero_radial = rows((F(2),F(-3,7)), (F(),), (F(),))
        with patch.object(self.cable, 'radial_moment_bounds', side_effect=AssertionError('unused kernel')):
            a_stats = {'momentPanels':7, 'maxMomentTerms':11};b_stats = dict(a_stats)
            actual = self.potential._smooth(q, zero_radial, F(1,7), F(6,7), self.tolerances, self.budgets, a_stats)
            with patch.object(self.cable, '_contract_intervals', return_value=None):
                expected = self.potential._smooth(q, zero_radial, F(1,7), F(6,7), self.tolerances, self.budgets, b_stats)
        self.assertEqual(actual, expected);self.assertEqual(a_stats, {'momentPanels':7, 'maxMomentTerms':11})
        self.assertEqual(a_stats, b_stats)
        energy = [(('e',), (F(2),), (F(-3,7),), (F(),))]
        radial, contract = self.cable.radial_moment_bounds, self.cable._contract_intervals
        admissions = []
        def observed(*args):
            result = contract(*args);admissions.append(result);return result
        with patch.object(self.cable, 'radial_moment_bounds', wraps=radial) as calls:
            a_stats = {'momentPanels':0, 'maxMomentTerms':0};b_stats = dict(a_stats)
            with patch.object(self.cable, '_contract_intervals', observed):
                actual = self.potential._smooth(q, energy, F(), F(1), self.tolerances, self.budgets, a_stats)
            with patch.object(self.cable, '_contract_intervals', return_value=None):
                expected = self.potential._smooth(q, energy, F(), F(1), self.tolerances, self.budgets, b_stats)
            self.assertEqual(calls.call_count, 2)
            self.assertEqual(calls.call_args_list[0], calls.call_args_list[1])
        self.assertEqual(admissions, [None]);self.assertEqual(actual, expected);self.assertEqual(a_stats, b_stats)
        self.assertGreater(a_stats['momentPanels'], 0)

    def test_kernel_failures_and_unverified_results_precede_contraction(self):
        q = (F(4),F(),F());functionals = rows()
        for failure in (ValueError('original kernel failure'), KeyboardInterrupt('original stop')):
            for forced_fallback in (False, True):
                stats = {'momentPanels':0, 'maxMomentTerms':0}
                with patch.object(self.cable, 'radial_moment_bounds', side_effect=failure) as radial, \
                     patch.object(self.cable, '_contract_intervals',
                                  side_effect=(lambda *args: None) if forced_fallback else self.cable._contract_intervals) as contraction:
                    with self.assertRaises(type(failure)) as caught:
                        self.potential._smooth(q, functionals, F(), F(1), self.tolerances, self.budgets, stats)
                    self.assertIs(caught.exception, failure);self.assertEqual(radial.call_count, 1)
                    contraction.assert_not_called()
                self.assertEqual(stats, {'momentPanels':0, 'maxMomentTerms':0})
        stats = {'momentPanels':0, 'maxMomentTerms':0}
        with patch.object(self.cable, 'radial_moment_bounds', return_value={'verified':False}), \
             patch.object(self.cable, '_contract_intervals') as contraction:
            with self.assertRaisesRegex(ValueError, 'Verified radial moment enclosures required'):
                self.potential._smooth(q, functionals, F(), F(1), self.tolerances, self.budgets, stats)
            contraction.assert_not_called()
        self.assertEqual(stats, {'momentPanels':0, 'maxMomentTerms':0})


if __name__ == '__main__':
    unittest.main()
