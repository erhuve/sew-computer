"""Portable mathematical controls for independent per-hinge time sampling."""

import copy
from decimal import Decimal, localcontext
from fractions import Fraction as F
import math
import struct
import unittest

import numpy as np

from solver_fold_control_schedule import FoldControlSchedule, ANGLE_LIMIT


HINGES = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


def recipe(hinges=None):
    return {"profile": "fold-angle-activation-v1", "hinges": copy.deepcopy(HINGES if hinges is None else hinges),
        "knots": [
            {"fraction": 0., "targetsRadians": [0., -.0, -.25], "activation": [0., 1., .25]},
            {"fraction": .25, "targetsRadians": [1., .5, -.5], "activation": [1., 1., .5]},
            {"fraction": .5, "targetsRadians": [1., -.5, 0.], "activation": [1., .5, .25]},
            {"fraction": .75, "targetsRadians": [-1., -.5, .5], "activation": [0., 0., .75]},
            {"fraction": 1., "targetsRadians": [0., .5, 0.], "activation": [.5, 1., 0.]}]}


def two_knots(start_angles, end_angles, start_activation=None, end_activation=None):
    count = len(start_angles)
    hinges = [[4*i+j for j in range(4)] for i in range(count)]
    return {"profile": "fold-angle-activation-v1", "hinges": hinges,
        "knots": [{"fraction": 0., "targetsRadians": start_angles,
                   "activation": [1.]*count if start_activation is None else start_activation},
                  {"fraction": 1., "targetsRadians": end_angles,
                   "activation": [1.]*count if end_activation is None else end_activation}]}


def bits(value):
    return struct.pack(">d", value)


class FoldControlScheduleTests(unittest.TestCase):
    def test_independent_hinge_ramps_hold_release_and_reengagement(self):
        source = recipe()
        schedule = FoldControlSchedule(source, 4, hinges=HINGES)
        angles, activation = schedule.parameters(F(3, 8))
        np.testing.assert_array_equal(angles, [1., 0., -.25])
        np.testing.assert_array_equal(activation, [1., .75, .375])
        angles, activation = schedule.parameters(F(5, 8))
        np.testing.assert_array_equal(angles, [0., -.5, .25])
        np.testing.assert_array_equal(activation, [.5, .25, .5])
        angles, activation = schedule.parameters(F(7, 8))
        np.testing.assert_array_equal(angles, [-.5, 0., .25])
        np.testing.assert_array_equal(activation, [.25, .5, .375])
        self.assertEqual(schedule.hinges, tuple(map(tuple, HINGES)))
        self.assertEqual(schedule.fractions, (F(), F(1,4), F(1,2), F(3,4), F(1)))
        self.assertEqual(schedule.subdivisions, 4)

    def test_exact_original_binary_interpolation_against_decimal(self):
        source = two_knots([.13, -.71, math.nextafter(1., math.inf)],
                           [-.67, .42, -.3], [.1, .73, 0.], [.83, .12, 1.])
        schedule = FoldControlSchedule(source, 1, hinges=source["hinges"])
        fractions = [F(1,2), F(1,2**40), F(2**40-1,2**40)] + [F(i,128) for i in range(1,128)]
        with localcontext() as context:
            context.prec = 200
            for fraction in fractions:
                amount = Decimal(fraction.numerator)/Decimal(fraction.denominator)
                outputs = schedule.parameters(fraction)
                for values, field in zip(outputs, ("targetsRadians", "activation")):
                    for value, a, b in zip(values, source["knots"][0][field], source["knots"][1][field]):
                        expected = (1-amount)*Decimal.from_float(a)+amount*Decimal.from_float(b)
                        self.assertEqual(bits(value), bits(float(expected)))

    def test_exact_midpoints_and_cancellation_do_not_round_intermediate_difference(self):
        upper = math.nextafter(1., math.inf)
        source = two_knots([1., upper, 1., -.0], [upper, math.nextafter(upper,math.inf), -upper, 0.],
                           [.75, 0., .5, -0.], [math.nextafter(.75,math.inf), 1., .5, 0.])
        schedule = FoldControlSchedule(source, 1, hinges=source["hinges"])
        angles, activation = schedule.parameters(F(1,2))
        self.assertEqual(angles[0], 1.)
        self.assertEqual(angles[1], math.nextafter(upper, math.inf))
        self.assertEqual(angles[2], -2**-53)
        self.assertEqual(1. + .5*(-upper-1.), 0., "The control must expose rounded-difference cancellation")
        self.assertEqual(bits(angles[3]), bits(0.))
        self.assertEqual(activation[0], .75)
        self.assertEqual(bits(activation[3]), bits(0.))
        # Dyadic original fractions need not produce a dyadic local amount:
        # the interval [0,3/4] sampled at 1/4 has exact amount 1/3. Rounding
        # that amount before interpolation changes this result by one ulp.
        uneven = two_knots([1.], [-upper])
        uneven["knots"][1]["fraction"] = .75
        uneven["knots"].append({"fraction": 1., "targetsRadians": [-upper], "activation": [1.]})
        uneven_schedule = FoldControlSchedule(uneven, 4, hinges=uneven["hinges"])
        expected = float(F(2,3)-F(upper)*F(1,3))
        rounded_amount = F(float(F(1,3)))
        self.assertEqual(uneven_schedule.parameters(F(1,4))[0][0], expected)
        self.assertNotEqual(expected, float(1-rounded_amount-F(upper)*rounded_amount))

    def test_original_fraction_retry_and_query_order_are_stateless(self):
        source = recipe()
        schedule = FoldControlSchedule(source, 4, hinges=HINGES)
        queries = [F(17,64), F(9,16), F(1,4), F(7,8), F(3,32), F(1), F()]
        recorded = {fraction: tuple(array.tobytes() for array in schedule.parameters(fraction)) for fraction in queries}
        for fraction in queries[::-1] + queries[::2] + [F(17,64)]*3:
            self.assertEqual(tuple(array.tobytes() for array in schedule.parameters(fraction)), recorded[fraction])
        # Recreating a captured recipe for a retry samples the original global
        # fraction identically; no accepted-step state or local ramp is stored.
        retried = FoldControlSchedule(copy.deepcopy(source),4,hinges=np.asarray(HINGES,dtype=np.int32))
        for fraction in queries:
            self.assertEqual(tuple(array.tobytes() for array in retried.parameters(fraction)), recorded[fraction])
        with self.assertRaises(ValueError):
            schedule.parameters(F(1,3))
        self.assertEqual(tuple(array.tobytes() for array in schedule.parameters(F(17,64))), recorded[F(17,64)])

    def test_knot_binary64_bytes_and_all_return_arrays_are_isolated(self):
        source = recipe()
        source["knots"][0]["activation"][0] = -0.
        schedule = FoldControlSchedule(source,4,hinges=HINGES)
        for knot in source["knots"]:
            result = schedule.parameters(F(knot["fraction"]))
            for array, field in zip(result,("targetsRadians","activation")):
                expected = np.asarray(knot[field],dtype=np.float64)
                self.assertEqual(array.tobytes(),expected.tobytes())
                self.assertEqual(array.dtype,np.float64)
                self.assertEqual(array.shape,(3,))
                self.assertTrue(array.flags.writeable)
                array[:] = 99
                self.assertEqual(schedule.parameters(F(knot["fraction"]))[("targetsRadians","activation").index(field)].tobytes(), expected.tobytes())
        a, b = schedule.parameters(F(3,8))
        a[:] = 2.
        b[:] = 0.
        np.testing.assert_array_equal(schedule.parameters(F(3,8))[0],[1.,0.,-.25])
        np.testing.assert_array_equal(schedule.parameters(F(3,8))[1],[1.,.75,.375])

    def test_captured_recipe_native_arrays_and_backing_storage_cannot_mutate_schedule(self):
        source = recipe()
        native = np.asarray(HINGES,dtype=np.int64)
        schedule = FoldControlSchedule(source,4,hinges=native)
        expected = tuple(array.tobytes() for array in schedule.parameters(F(3,8)))
        source["hinges"][0][0] = 99
        source["knots"][1]["targetsRadians"][0] = 2.
        source["knots"].clear()
        native[:] = 99
        self.assertEqual(tuple(array.tobytes() for array in schedule.parameters(F(3,8))),expected)
        for name, value in (("_hinges",()),("_fractions",()),("_subdivisions",1),("_targets",None),("_activation",None)):
            with self.subTest(name=name),self.assertRaises(AttributeError):
                setattr(schedule,name,value)
            with self.subTest(delete=name),self.assertRaises(AttributeError):
                delattr(schedule,name)
        for array in (schedule._targets,schedule._activation):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
            with self.assertRaises(ValueError):
                array.base.setflags(write=True)
            with self.assertRaises(ValueError):
                array[0,0] = 1.
        with self.assertRaises(TypeError):
            schedule.hinges[0][0] = 99

    def test_positive_interior_activation_underflow_rejects_both_engage_and_release(self):
        tiny = math.ulp(0.)
        for before, after in ((0.,tiny),(tiny,0.)):
            source = two_knots([0.],[1.],[before],[after])
            schedule = FoldControlSchedule(source,1,hinges=source["hinges"])
            for fraction in (F(1,2), F(1,2**40) if before==0 else F(2**40-1,2**40)):
                with self.subTest(before=before, fraction=fraction),self.assertRaisesRegex(ValueError,"underflows"):
                    schedule.parameters(fraction)
            self.assertEqual(schedule.parameters(0)[1][0],before)
            self.assertEqual(schedule.parameters(1)[1][0],after)
        source = two_knots([0.],[tiny],[tiny],[3*tiny])
        schedule = FoldControlSchedule(source,1,hinges=source["hinges"])
        self.assertEqual(schedule.parameters(F(1,2))[1][0],2*tiny)
        # An angle is a signed parameter, not active/inactive state. Correct
        # nearest rounding of its positive subnormal midpoint to zero is kept.
        self.assertEqual(schedule.parameters(F(1,2))[0][0],0.)
        source = two_knots([0.],[0.],[tiny],[tiny])
        schedule = FoldControlSchedule(source,1,hinges=source["hinges"])
        self.assertEqual(schedule.parameters(F(1,2**40))[1][0],tiny)

    def test_raw_recipe_shape_number_profile_and_boolean_attacks(self):
        attacks = [
            lambda r:r.__setitem__("profile","legacy"), lambda r:r.__setitem__("extra",0),
            lambda r:r["knots"][1].__setitem__("extra",0),
            lambda r:r["knots"][0]["targetsRadians"].__setitem__(0,True),
            lambda r:r["knots"][0]["activation"].__setitem__(0,False),
            lambda r:r["knots"][0]["targetsRadians"].__setitem__(0,float("nan")),
            lambda r:r["knots"][0]["targetsRadians"].__setitem__(0,float("inf")),
            lambda r:r["knots"][0]["targetsRadians"].__setitem__(0,10**1000),
            lambda r:r["knots"][0]["targetsRadians"].__setitem__(0,ANGLE_LIMIT),
            lambda r:r["knots"][0]["targetsRadians"].__setitem__(0,-ANGLE_LIMIT),
            lambda r:r["knots"][0]["activation"].__setitem__(0,math.nextafter(1.,math.inf)),
            lambda r:r["knots"][0]["activation"].__setitem__(0,-math.ulp(0.)),
            lambda r:r["knots"][0]["activation"].__setitem__(0,10**1000),
            lambda r:r["knots"][0]["activation"].__setitem__(0,np.float64(.5)),
            lambda r:r["knots"][0]["activation"].__setitem__(0,np.longdouble(.5)),
            lambda r:r["knots"][0].__setitem__("targetsRadians",[0.,0.]),
            lambda r:r["knots"][0].__setitem__("targetsRadians",[[0.],[0.],[0.]]),
            lambda r:r["knots"][0].__setitem__("activation",(0.,0.,0.)),
            lambda r:r["knots"][0].__setitem__("activation",np.zeros(3)),
        ]
        for index,mutate in enumerate(attacks):
            source = recipe()
            mutate(source)
            with self.subTest(attack=index),self.assertRaises(ValueError):
                FoldControlSchedule(source,4,hinges=HINGES)
        for value in (None,[],(),{"profile":"fold-angle-activation-v1"}):
            with self.assertRaises(ValueError):
                FoldControlSchedule(value,4,hinges=HINGES)

    def test_strict_complete_ordered_native_hinge_identity(self):
        source = recipe()
        for native in ([],[[0,1,2,3]],HINGES[::-1],[[1,0,2,3],*HINGES[1:]],
                       [[True,1,2,3],*HINGES[1:]],[[0,0,2,3],*HINGES[1:]],
                       [[-1,1,2,3],*HINGES[1:]],[[2**63,1,2,3],*HINGES[1:]],
                       [HINGES[0],HINGES[0],HINGES[2]],np.asarray(HINGES,dtype=float),
                       np.asarray(HINGES,dtype=object),np.zeros((3,4),dtype=bool),np.zeros((3,3),dtype=int)):
            with self.subTest(native=native),self.assertRaises(ValueError):
                FoldControlSchedule(source,4,hinges=native)
        for raw in (HINGES[::-1], [[1,0,2,3],*HINGES[1:]], [[0,1,2,False],*HINGES[1:]],
                    [tuple(row) for row in HINGES], np.asarray(HINGES,dtype=int)):
            changed = copy.deepcopy(source)
            changed["hinges"] = raw
            with self.assertRaises(ValueError):
                FoldControlSchedule(changed,4,hinges=HINGES)
        self.assertEqual(FoldControlSchedule(source,4,hinges=tuple(map(tuple,HINGES))).hinges,tuple(map(tuple,HINGES)))

    def test_fraction_order_grid_and_initial_subdivision_admission(self):
        for subdivisions in (True,0,-1,3,4097,8192,4.,np.int64(4)):
            with self.subTest(subdivisions=subdivisions),self.assertRaises(ValueError):
                FoldControlSchedule(recipe(),subdivisions,hinges=HINGES)
        for index,value in ((0,.25),(4,.75),(1,.5),(1,.3),(1,1/8),(1,True),(1,F(1,4)),(1,np.float64(.25))):
            changed=recipe()
            changed["knots"][index]["fraction"]=value
            with self.subTest(index=index,value=value),self.assertRaises(ValueError):
                FoldControlSchedule(changed,4,hinges=HINGES)
        for knots in ([],recipe()["knots"][:1],recipe()["knots"][::-1],recipe()["knots"]*14):
            changed=recipe()
            changed["knots"]=knots
            with self.assertRaises(ValueError):
                FoldControlSchedule(changed,4,hinges=HINGES)
        schedule=FoldControlSchedule(recipe(),4,hinges=HINGES)
        for fraction in (True,np.bool_(False),np.float64(.25),np.int64(0),"0.25",None,float("nan"),float("inf"),
                         F(1,3),F(1,2**41),F(2**41-1,2**41),-.1,1.1,10**1000,F(1,2**10000)):
            with self.subTest(fraction=str(fraction)[:80]),self.assertRaises(ValueError):
                schedule.parameters(fraction)

    def test_declared_maximum_shapes_are_bounded_before_conversion(self):
        hinges = [[4*i+j for j in range(4)] for i in range(4096)]
        source = {"profile":"fold-angle-activation-v1","hinges":hinges,
                  "knots":[{"fraction":i/64,"targetsRadians":[0.]*4096,"activation":[float(i%2)]*4096} for i in range(65)]}
        schedule=FoldControlSchedule(source,4096,hinges=np.asarray(hinges,dtype=np.int32))
        target, activation=schedule.parameters(F(1,128))
        self.assertEqual(target.shape,(4096,))
        np.testing.assert_array_equal(target,np.zeros(4096))
        np.testing.assert_array_equal(activation,np.full(4096,.5))
        self.assertEqual(schedule._targets.nbytes+schedule._activation.nbytes,2*65*4096*8)
        source["hinges"].append([20000,20001,20002,20003])
        with self.assertRaises(ValueError):
            FoldControlSchedule(source,4096,hinges=source["hinges"])


if __name__=="__main__":
    unittest.main()
