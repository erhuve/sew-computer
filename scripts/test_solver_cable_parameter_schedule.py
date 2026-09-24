"""Independent supplied-cell interpolation and finite-grid admission checks."""
import ast
import copy
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import struct
import unittest
from unittest import mock

import numpy as np

from solver_continuous_cable_sewing import ContinuousCableSewing
from solver_cable_parameters import CableParameterRecipe
from solver_cable_parameter_schedule import CableParameterSchedule
import solver_cable_parameter_schedule as module


def rat(value):
    value=F(value)
    return {'numerator':str(value.numerator),'denominator':str(value.denominator)}


def read(value):
    result=F(int(value['numerator']),int(value['denominator']))
    if value!=rat(result): raise AssertionError('Canonical rational witness required')
    return result


def bits(value):
    return struct.pack('>d',value)


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def recipe(count=3, *, density=2., length=F(1,2), activation=1.):
    cells=[]
    for index in range(count):
        cells.append({'id':'cell:'+str(index),
            'positiveStart':[{'vertex':2,'weight':rat(1)}],
            'positiveEnd':[{'vertex':3,'weight':rat(1)}],
            'negativeStart':[{'vertex':0,'weight':rat(1)}],
            'negativeEnd':[{'vertex':1,'weight':rat(1)}],
            'targetsMeters':[.25,.5], 'activation':activation,
            'stiffnessDensityNPerM2':density,'referenceLengthMeters':rat(length)})
    return CableParameterRecipe(ContinuousCableSewing(4,cells))


def raw_schedule(bound, *, fractions=(0.,.25,.5,.75,1.)):
    knots=[]
    sequence=((0.,1.,.25),(1.,1.,.5),(1.,.5,.25),(0.,0.,.75),(.5,1.,0.))
    for index,fraction in enumerate(fractions):
        knots.append({'fraction':fraction,
            'targetsMeters':[[.125+(index+j)/16.,.5+(j-index)/32.] for j in range(len(bound.cell_ids))],
            'activation':[sequence[index%5][j%3] for j in range(len(bound.cell_ids))]})
    return {'profile':'cable-target-activation-v1','geometrySha256':bound.geometry_sha256,
            'cellIds':list(bound.cell_ids),'knots':knots}


def two_knots(bound, a0, a1, targets0=None, targets1=None):
    count=len(bound.cell_ids)
    return {'profile':'cable-target-activation-v1','geometrySha256':bound.geometry_sha256,
        'cellIds':list(bound.cell_ids),'knots':[
            {'fraction':0.,'targetsMeters':[[.25,.5] for _ in range(count)] if targets0 is None else targets0,
             'activation':list(a0)},
            {'fraction':1.,'targetsMeters':[[.5,.25] for _ in range(count)] if targets1 is None else targets1,
             'activation':list(a1)}]}


def independent_parameters(raw,fraction):
    """Direct Fraction endpoint interpolation; no schedule or recipe calls."""
    for knot in raw['knots']:
        if F(knot['fraction'])==fraction:
            return np.asarray(knot['targetsMeters'],dtype=float),np.asarray(knot['activation'],dtype=float)
    for before,after in zip(raw['knots'],raw['knots'][1:]):
        left,right=F(before['fraction']),F(after['fraction'])
        if left<fraction<right:
            amount=(fraction-left)/(right-left)
            def interpolate(a,b): return float((1-amount)*F(a)+amount*F(b))
            targets=[[interpolate(a,b) for a,b in zip(first,last)]
                     for first,last in zip(before['targetsMeters'],after['targetsMeters'])]
            activation=[interpolate(a,b) for a,b in zip(before['activation'],after['activation'])]
            return np.asarray(targets),np.asarray(activation)
    raise AssertionError('Oracle domain error')


class CableParameterScheduleTests(unittest.TestCase):
    def test_explicit_ramps_hold_release_reengage_and_geometry_identity(self):
        bound=recipe(); raw=raw_schedule(bound)
        schedule=CableParameterSchedule(raw,4,cable_recipe=bound)
        for fraction in (F(1,8),F(3,8),F(5,8),F(7,8)):
            targets,activation=schedule.parameters(fraction)
            expected=independent_parameters(raw,fraction)
            np.testing.assert_array_equal(targets,expected[0]);np.testing.assert_array_equal(activation,expected[1])
        np.testing.assert_array_equal(schedule.parameters(F(3,8))[1],[1.,.75,.375])
        self.assertEqual(schedule.cell_ids,bound.cell_ids)
        self.assertEqual(schedule.geometry_sha256,bound.geometry_sha256)
        self.assertEqual(schedule.fractions,tuple(F(k['fraction']) for k in raw['knots']))
        self.assertEqual(schedule.subdivisions,4)
        description=schedule.description()
        self.assertEqual(description['scheduleSha256'],hashlib.sha256(encoded(raw)).hexdigest())
        self.assertEqual(description['capturedSchedule'],raw)
        self.assertIs(description['accepted'],False)
        self.assertIs(description['sourceAdmissionGranted'],False)
        self.assertIs(description['controlsInstalled'],False)
        # Standalone declaration permits explicit different initial parameters;
        # it neither executes nor accounts an initial jump from the base recipe.
        self.assertNotEqual(schedule.parameters(0)[0].tolist(),bound.initial_parameters[0].tolist())

    def test_original_fraction_interpolation_matches_independent_decimal(self):
        bound=recipe(2)
        raw=two_knots(bound,[.1,.73],[.83,.12],[[.13,.71],[.03,.999]],[[.67,.42],[.89,1.125]])
        raw['knots'][1]['fraction']=.75
        raw['knots'].append({'fraction':1.,'targetsMeters':[[.67,.42],[.89,1.125]],'activation':[.83,.12]})
        schedule=CableParameterSchedule(raw,4,cable_recipe=bound)
        queries=[F(1,2**40),F(2**40-1,2**40)]+[F(i,128) for i in range(129)]
        rounded_amount_disagrees=False
        with localcontext() as context:
            context.prec=220
            for fraction in queries:
                output=schedule.parameters(fraction)
                expected=independent_parameters(raw,fraction)
                self.assertEqual(tuple(a.tobytes() for a in output),tuple(a.tobytes() for a in expected))
                if fraction in schedule.fractions:continue
                lower=next(i for i in range(len(raw['knots'])-1)
                           if F(raw['knots'][i]['fraction'])<fraction<F(raw['knots'][i+1]['fraction']))
                first,last=raw['knots'][lower:lower+2]
                t=(fraction-F(first['fraction']))/(F(last['fraction'])-F(first['fraction']))
                decimal_t=Decimal(t.numerator)/Decimal(t.denominator)
                pairs=[(output[0].ravel(),sum(first['targetsMeters'],[]),sum(last['targetsMeters'],[])),
                       (output[1],first['activation'],last['activation'])]
                for values,starts,ends in pairs:
                    for result,a,b in zip(values,starts,ends):
                        oracle=(1-decimal_t)*Decimal.from_float(a)+decimal_t*Decimal.from_float(b)
                        self.assertEqual(bits(result),bits(float(oracle)))
                        wrong=float((1-F(float(t)))*F(a)+F(float(t))*F(b))
                        rounded_amount_disagrees|=bits(result)!=bits(wrong)
        self.assertTrue(rounded_amount_disagrees,'The fixture must distinguish early amount rounding')

    def test_nearest_even_ties_signed_zero_and_minimum_positive_target(self):
        bound=recipe(2); upper=math.nextafter(1.,math.inf); next_upper=math.nextafter(upper,math.inf)
        tiny=math.ulp(0.)
        raw=two_knots(bound,[-0.,.5],[0.,.5],[[1.,upper],[tiny,100.]],[[upper,next_upper],[tiny,tiny]])
        schedule=CableParameterSchedule(raw,1,cable_recipe=bound)
        self.assertEqual(bits(schedule.parameters(0)[1][0]),bits(-0.))
        self.assertEqual(bits(schedule.parameters(F(1,2))[1][0]),bits(0.))
        self.assertEqual(schedule.parameters(F(1,2))[0][0].tolist(),[1.,next_upper])
        for fraction in (F(1,2),F(1,2**40),F(2**40-1,2**40),F(1)):
            values=schedule.parameters(fraction)[0]
            self.assertEqual(values[1,0],tiny)
            self.assertGreater(values[1,1],0.)

    def test_query_order_retry_and_output_copies_do_not_change_controls(self):
        bound=recipe();raw=raw_schedule(bound);schedule=CableParameterSchedule(raw,4,cable_recipe=bound)
        queries=[F(17,64),F(9,16),F(1,4),F(7,8),F(3,32),F(1),F()]
        recorded={f:tuple(a.tobytes() for a in schedule.parameters(f)) for f in queries}
        for fraction in queries[::-1]+queries[::2]+[queries[0]]*3:
            result=schedule.parameters(fraction)
            self.assertEqual(tuple(a.tobytes() for a in result),recorded[fraction])
            for array in result:
                self.assertEqual(array.dtype,np.float64);self.assertTrue(array.flags.writeable)
                array[:]=99
        replay=CableParameterSchedule(copy.deepcopy(raw),4,cable_recipe=bound)
        for fraction in queries:
            self.assertEqual(tuple(a.tobytes() for a in replay.parameters(fraction)),recorded[fraction])
        with self.assertRaises(ValueError):schedule.parameters(F(1,3))
        self.assertEqual(tuple(a.tobytes() for a in schedule.parameters(queries[0])),recorded[queries[0]])

    def test_captured_raw_recipe_description_and_backing_arrays_are_immutable(self):
        bound=recipe();raw=raw_schedule(bound);schedule=CableParameterSchedule(raw,4,cable_recipe=bound)
        identity=encoded(schedule.description());sample=tuple(a.tobytes() for a in schedule.parameters(F(3,8)))
        raw['cellIds'].reverse();raw['knots'][1]['targetsMeters'][0][0]=99.;raw['knots'].clear()
        exposed=schedule.description();exposed['capturedSchedule']['knots'].clear();exposed['cellIds'].clear()
        initial=bound.initial_parameters;initial[0][:]=9.;initial[1][:]=0.
        self.assertEqual(encoded(schedule.description()),identity)
        self.assertEqual(tuple(a.tobytes() for a in schedule.parameters(F(3,8))),sample)
        for name in schedule.__slots__:
            with self.subTest(name=name),self.assertRaises(AttributeError):setattr(schedule,name,None)
            with self.subTest(delete=name),self.assertRaises(AttributeError):delattr(schedule,name)
        replacement=raw_schedule(bound);replacement['knots'][0]['activation'][0]=.75
        with self.assertRaises(AttributeError):schedule.__init__(replacement,4,cable_recipe=bound)
        self.assertEqual(encoded(schedule.description()),identity)
        self.assertEqual(tuple(a.tobytes() for a in schedule.parameters(F(3,8))),sample)
        for array in schedule._arrays():
            with self.assertRaises(ValueError):array.setflags(write=True)
            with self.assertRaises(ValueError):array.base.setflags(write=True)
            with self.assertRaises(ValueError):array.flat[0]=1.
        preflight=schedule.preflight(2);preflight['cellWitnesses'].clear()
        self.assertEqual(len(schedule.preflight(2)['cellWitnesses']),3)

    def test_no_retained_ndarray_metadata_can_change_sample_or_identity(self):
        bound=recipe();schedule=CableParameterSchedule(raw_schedule(bound),4,cable_recipe=bound)
        queries=(F(),F(1,8),F(3,8),F(7,8),F(1))
        def state():
            return (encoded(schedule.description()),encoded(schedule.preflight(3)),
                    tuple(tuple(a.tobytes() for a in schedule.parameters(f)) for f in queries))
        original=state()
        for name in schedule.__slots__:
            self.assertNotIsInstance(getattr(schedule,name),np.ndarray)
        self.assertIs(type(schedule._target_bytes),bytes)
        self.assertIs(type(schedule._activation_bytes),bytes)
        self.assertIs(type(schedule._target_shape),tuple)
        self.assertIs(type(schedule._activation_shape),tuple)
        for channel in (0,1):
            for attribute in ('shape','strides','dtype'):
                for use_base in (False,True):
                    view=schedule._arrays()[channel]
                    target=view.base if use_base else view
                    if attribute=='shape':target.shape=(target.size,)
                    elif attribute=='strides':target.strides=(0,)*target.ndim
                    else:target.dtype=np.uint8
                    with self.subTest(channel=channel,attribute=attribute,base=use_base):
                        self.assertEqual(state(),original)
                # Public writable returned arrays are also disconnected from
                # future sampled bytes, even when only their metadata changes.
                target=schedule.parameters(1)[channel]
                if attribute=='shape':target.shape=(target.size,)
                elif attribute=='strides':target.strides=(0,)*target.ndim
                else:target.dtype=np.uint8
                self.assertEqual(state(),original)

    def test_raw_json_representation_identity_retains_numbers_and_signed_zero(self):
        bound=recipe(1)
        first=two_knots(bound,[0],[1],[[1,2]],[[2,1]])
        second=copy.deepcopy(first);second['knots'][0]['targetsMeters'][0][0]=1.
        third=copy.deepcopy(first);third['knots'][0]['activation'][0]=-0.
        schedules=[CableParameterSchedule(raw,1,cable_recipe=bound) for raw in (first,second,third)]
        self.assertEqual(len({s.description()['scheduleSha256'] for s in schedules}),3)
        for fraction in (F(1,4),F(1,2),F(3,4)):
            values=[tuple(a.tobytes() for a in s.parameters(fraction)) for s in schedules]
            self.assertEqual(values[0],values[1]);self.assertEqual(values[1],values[2])

    def test_recipe_type_geometry_and_complete_order_are_bound(self):
        bound=recipe();raw=raw_schedule(bound)
        for invalid in (None,object(),mock.Mock(spec=CableParameterRecipe)):
            with self.subTest(type=type(invalid)),self.assertRaises(ValueError):
                CableParameterSchedule(raw,4,cable_recipe=invalid)
        class Subclass(CableParameterRecipe):pass
        subclass=object.__new__(Subclass)
        with self.assertRaises(ValueError):CableParameterSchedule(raw,4,cable_recipe=subclass)
        changed=recipe(density=3.)
        with self.assertRaises(ValueError):CableParameterSchedule(raw,4,cable_recipe=changed)
        for attack in (lambda r:r.__setitem__('geometrySha256','0'*64),
                       lambda r:r['cellIds'].reverse(),lambda r:r['cellIds'].pop(),
                       lambda r:r['cellIds'].__setitem__(1,r['cellIds'][0])):
            value=copy.deepcopy(raw);attack(value)
            with self.assertRaises(ValueError):CableParameterSchedule(value,4,cable_recipe=bound)

    def test_raw_shape_scalar_profile_and_budget_rejections(self):
        bound=recipe();raw=raw_schedule(bound)
        attacks=[lambda r:r.__setitem__('profile','legacy'),lambda r:r.__setitem__('extra',0),
            lambda r:r['knots'][0].__setitem__('extra',0),
            lambda r:r.__setitem__('cellIds',tuple(r['cellIds'])),
            lambda r:r['knots'][0].__setitem__('targetsMeters',np.ones((3,2))),
            lambda r:r['knots'][0].__setitem__('activation',(0.,0.,0.)),
            lambda r:r['knots'][0]['targetsMeters'].pop(),
            lambda r:r['knots'][0]['targetsMeters'][0].append(.2),
            lambda r:r['knots'][0].__setitem__('activation',[0.,0.]),
            lambda r:r['knots'].__setitem__(0,[])]
        for field,values in (('target',[True,False,0.,-0.,-1.,101.,2**53+1,float('nan'),float('inf'),
                                            np.float64(.5),np.longdouble(.5),F(1,2),'0.5',10**1000]),
                             ('activation',[True,False,-math.ulp(0.),math.nextafter(1.,math.inf),2**53+1,
                                            float('nan'),float('inf'),np.float64(.5),F(1,2)])):
            for value in values:
                def attack(r,value=value,field=field):
                    if field=='target':r['knots'][0]['targetsMeters'][0][0]=value
                    else:r['knots'][0]['activation'][0]=value
                attacks.append(attack)
        for index,attack in enumerate(attacks):
            value=copy.deepcopy(raw);attack(value)
            with self.subTest(attack=index),self.assertRaises(ValueError):CableParameterSchedule(value,4,cable_recipe=bound)
        with mock.patch.object(module,'MAX_RAW_BYTES',100):
            with self.assertRaisesRegex(ValueError,'byte budget'):CableParameterSchedule(raw,4,cable_recipe=bound)

    def test_knot_fraction_subdivision_and_public_fraction_admission(self):
        bound=recipe();raw=raw_schedule(bound)
        for divisions in (True,False,0,-1,3,8192,4.,np.int64(4)):
            with self.subTest(divisions=divisions),self.assertRaises(ValueError):CableParameterSchedule(raw,divisions,cable_recipe=bound)
        for bad in (True,float('nan'),float('inf'),F(1,4),np.float64(.25),.1):
            value=copy.deepcopy(raw);value['knots'][1]['fraction']=bad
            with self.subTest(knot=bad),self.assertRaises(ValueError):CableParameterSchedule(value,4,cable_recipe=bound)
        for field,value in ((0,.25),(-1,.75),(2,.25),(2,.125)):
            source=copy.deepcopy(raw);source['knots'][field]['fraction']=value
            with self.assertRaises(ValueError):CableParameterSchedule(source,4,cable_recipe=bound)
        schedule=CableParameterSchedule(raw,4,cable_recipe=bound)
        for fraction in (True,False,np.float64(.5),np.int64(1),-.1,1.1,float('nan'),float('inf'),
                         F(1,3),F(1,2**41),F(1,2**10000),'0.5',10**1000):
            with self.subTest(fraction=str(fraction)[:80]),self.assertRaises(ValueError):schedule.parameters(fraction)
        for count in (0,1,66):
            source=copy.deepcopy(raw);source['knots']=[copy.deepcopy(raw['knots'][0]) for _ in range(count)]
            with self.assertRaises(ValueError):CableParameterSchedule(source,4,cable_recipe=bound)
        source=raw_schedule(recipe(1),fractions=tuple(i/64 for i in range(65)))
        for knot in source['knots']:
            knot['targetsMeters']=[[.25,.5]]
        self.assertEqual(len(CableParameterSchedule(source,64,cable_recipe=recipe(1)).fractions),65)

    def test_positive_activation_and_beta_underflow_are_distinct(self):
        tiny=math.ulp(0.)
        for first,last in ((0.,tiny),(tiny,0.)):
            bound=recipe(1);raw=two_knots(bound,[first],[last])
            schedule=CableParameterSchedule(raw,1,cable_recipe=bound)
            self.assertTrue(schedule.preflight(0)['verified'])
            with self.assertRaisesRegex(ValueError,'activation underflows'):schedule.parameters(F(1,2))
            with self.assertRaisesRegex(ValueError,'activation underflows'):schedule.preflight(1)
        bound=recipe(1,density=2*tiny,length=F(1))
        raw=two_knots(bound,[0.],[1.]);schedule=CableParameterSchedule(raw,1,cable_recipe=bound)
        self.assertEqual(schedule.parameters(F(1,2))[1][0],.5)
        self.assertTrue(schedule.preflight(1)['verified'])
        with self.assertRaises(ValueError):schedule.parameters(F(1,4))
        with self.assertRaisesRegex(ValueError,'beta underflows'):schedule.preflight(2)
        # Endpoints themselves must be mechanically admitted before storage.
        bad=two_knots(bound,[.25],[1.])
        with self.assertRaises(ValueError):CableParameterSchedule(bad,1,cable_recipe=bound)
        above=math.nextafter(.25,1.)
        admitted=CableParameterSchedule(two_knots(bound,[above],[above]),1,cable_recipe=bound)
        witness=admitted.preflight(30)['cellWitnesses'][0]['minimumPositiveActivation']
        self.assertGreater(read(witness['exactEffectiveStiffnessNPerM']),F(tiny)/2)
        self.assertLess(read(witness['exactEffectiveStiffnessNPerM']),F(tiny))
        self.assertEqual(read(witness['exactEffectiveStiffnessNPerM']),F(above)*F(2*tiny))
        self.assertEqual(witness['binary64AdmissionValueNPerM'],tiny)
        self.assertGreater(read(witness['admissionRoundingResidualNPerM']),0)

    def test_representable_endpoint_beta_does_not_require_representable_beta_delta(self):
        tiny=math.ulp(0.);bound=recipe(1,density=2*tiny,length=F(1))
        last=math.nextafter(.5,1.)
        schedule=CableParameterSchedule(two_knots(bound,[.5],[last]),1,cable_recipe=bound)
        evidence=schedule.preflight(20)['cellWitnesses'][0]['minimumPositiveActivation']
        self.assertEqual(read(evidence['exactEffectiveStiffnessNPerM']),F(tiny))
        self.assertEqual(F(last)*F(2*tiny)-F(tiny),F(1,2**1126))
        self.assertEqual(schedule.parameters(1)[1][0],last)
        self.assertEqual(float(F(last)*F(2*tiny)),tiny)

    def test_finite_grid_minima_match_exhaustive_independent_sampling(self):
        bound=recipe(2,density=.75,length=F(1,7))
        count=0
        for pattern in product((0.,.5,1.),repeat=3):
            raw={'profile':module.PROFILE,'geometrySha256':bound.geometry_sha256,'cellIds':list(bound.cell_ids),
                 'knots':[{'fraction':i/2,'targetsMeters':[[.25,.5],[.75,.25]],
                           'activation':[pattern[i],pattern[2-i]]} for i in range(3)]}
            schedule=CableParameterSchedule(raw,2,cable_recipe=bound)
            for depth in range(4):
                report=schedule.preflight(depth);denominator=2*2**depth
                self.assertEqual(report['gridPointCount'],denominator+1)
                samples=[independent_parameters(raw,F(i,denominator))[1] for i in range(denominator+1)]
                for column,witness in enumerate(report['cellWitnesses']):
                    positive=[float(values[column]) for values in samples if values[column]>0]
                    self.assertEqual(witness['activeOnGrid'],bool(positive))
                    if not positive:
                        self.assertIsNone(witness['minimumPositiveActivation']);continue
                    minimum=witness['minimumPositiveActivation'];fraction=read(minimum['fraction'])
                    self.assertEqual(minimum['numericalActivation'],min(positive))
                    self.assertEqual((fraction*denominator).denominator,1)
                    self.assertEqual(minimum['numericalActivation'],independent_parameters(raw,fraction)[1][column])
                    self.assertEqual(read(minimum['exactEffectiveStiffnessNPerM']),F(min(positive))*F(.75)*F(1,7))
                    self.assertEqual(read(minimum['admissionRoundingResidualNPerM']),
                                     F(minimum['binary64AdmissionValueNPerM'])-read(minimum['exactEffectiveStiffnessNPerM']))
                count+=1
        self.assertEqual(count,108)

    def test_huge_grid_preflight_uses_bounded_candidates_not_enumeration(self):
        bound=recipe();schedule=CableParameterSchedule(raw_schedule(bound),4096,cable_recipe=bound)
        original=CableParameterRecipe.parameters;calls=[]
        def validate(instance,targets,activation):
            calls.append(tuple(activation));self.assertLessEqual(len(calls),1)
            return original(instance,targets,activation)
        with mock.patch.object(CableParameterRecipe,'parameters',validate), \
             mock.patch.object(CableParameterSchedule,'parameters',side_effect=AssertionError('Grid enumeration')):
            report=schedule.preflight(28)
        self.assertEqual(len(calls),1)
        self.assertEqual(report['gridDenominator'],2**40)
        self.assertEqual(read(report['minimumFractionStep']),F(1,2**40))
        self.assertLessEqual(sum(w['positiveCandidateCount'] for w in report['cellWitnesses']),48)
        # The final re-engagement climbs only to .5 over the final quarter,
        # giving slope2 and a smaller witness than the first slope4 ramp.
        self.assertEqual(report['cellWitnesses'][0]['minimumPositiveActivation']['numericalActivation'],2**-39)
        self.assertEqual(read(report['cellWitnesses'][0]['minimumPositiveActivation']['exactEffectiveStiffnessNPerM']),F(1,2**39))
        for bad in (-1,29,30,31,True,1.,np.int64(1)):
            with self.subTest(depth=bad),self.assertRaises(ValueError):schedule.preflight(bad)
        self.assertTrue(CableParameterSchedule(raw_schedule(bound),4,cable_recipe=bound).preflight(30)['verified'])

    def test_rounding_plateau_witness_need_not_be_first_full_grid_minimizer(self):
        bound=recipe(1)
        first=math.nextafter(.5,1.);last=.5
        schedule=CableParameterSchedule(two_knots(bound,[first],[last]),1,cable_recipe=bound)
        report=schedule.preflight(3);witness=report['cellWitnesses'][0]['minimumPositiveActivation']
        # Endpoint minimum is a valid witness even though the midpoint ties to
        # .5 and therefore reaches that rounded minimum earlier on this grid.
        self.assertEqual(read(witness['fraction']),1)
        self.assertEqual(schedule.parameters(F(1,2))[1][0],last)
        self.assertIn('not necessarily the earliest',report['scope'])

    def test_maximum_cell_count_and_standalone_import_boundary(self):
        bound=recipe(4096);raw=two_knots(bound,[0.]*4096,[1.]*4096)
        schedule=CableParameterSchedule(raw,1,cable_recipe=bound)
        self.assertEqual(schedule.parameters(F(1,2))[0].shape,(4096,2))
        excessive=copy.deepcopy(raw);excessive['cellIds'].append('extra')
        with self.assertRaises(ValueError):CableParameterSchedule(excessive,1,cable_recipe=bound)
        tree=ast.parse(Path(module.__file__).read_text())
        imports={node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
        self.assertEqual({name for name in imports if name and name.startswith('solver_')},{'solver_cable_parameters'})
        self.assertNotIn('solver_global_sewing',imports)


if __name__=='__main__':
    unittest.main()
