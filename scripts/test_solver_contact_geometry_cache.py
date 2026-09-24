"""Exact equivalence and admission/resource boundaries of geometry memoization."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as F
import math
import random
import unittest

from solver_contact_work import (WorkBudget, WorkPolicy, _distance_choices,
    _distance_choices_uncached, _distance_choice_integers, closest_feature,
    distance_squared)


class GeometryCacheTests(unittest.TestCase):
    def setUp(self):
        _distance_choice_integers.cache_clear()

    def tearDown(self):
        _distance_choice_integers.cache_clear()

    def test_seeded_all_stencil_results_equal_uncached_arithmetic(self):
        rng=random.Random(230926)
        for kind,size in (('vv',2),('ev',3),('fv',4),('ee',4)):
            for _ in range(200):
                points=tuple(tuple(rng.randrange(-12,13)/8 for _ in range(3)) for _ in range(size))
                try:
                    expected=_distance_choices_uncached(kind,points)
                except ValueError as error:
                    with self.assertRaisesRegex(ValueError,str(error)):
                        _distance_choices(kind,points)
                else:
                    self.assertEqual(_distance_choices(kind,points),expected)
                    self.assertEqual(_distance_choices(kind,points),expected)

    def test_repeated_geometry_reuses_only_integer_payload(self):
        points=((0.,0.,1.),(0.,0.,0.),(2.,0.,0.))
        first=_distance_choices('ev',points)
        second=_distance_choices('ev',points)
        self.assertEqual(first,second)
        self.assertIsNot(first,second)
        self.assertEqual(_distance_choice_integers.cache_info().misses,1)
        self.assertEqual(_distance_choice_integers.cache_info().hits,1)
        payload=_distance_choice_integers('ev',points)
        self.assertTrue(all(type(name) is str and type(n) is int and type(d) is int for name,n,d in payload))

    def test_returned_mapping_and_fraction_cannot_poison_later_calls(self):
        points=((0.,0.,0.),(1.,2.,2.))
        first=_distance_choices('vv',points)
        first['P_P']._numerator=123
        first.clear()
        self.assertEqual(_distance_choices('vv',points),{'P_P':F(9)})
        value=distance_squared('vv','P_P',points,WorkBudget())
        value._denominator=17
        self.assertEqual(distance_squared('vv','P_P',points,WorkBudget()),9)

    def test_equal_hash_invalid_coordinate_types_reject_after_warming(self):
        points=((0.,0.,0.),(1.,2.,2.))
        _distance_choices('vv',points)
        for value in (0,False,F(),None,'0'):
            with self.assertRaisesRegex(ValueError,'Finite binary64 input'):
                _distance_choices('vv',((value,0.,0.),points[1]))
        for value in (math.nan,math.inf,-math.inf):
            with self.assertRaisesRegex(ValueError,'Finite binary64 input'):
                _distance_choices('vv',((value,0.,0.),points[1]))
        self.assertEqual(_distance_choice_integers.cache_info().currsize,1)

    def test_shape_admission_precedes_hashing(self):
        points=((0.,0.,0.),(1.,2.,2.))
        _distance_choices('vv',points)
        for value in ([*points],(list(points[0]),points[1]),(points[0],),(points[0]+(0.,),points[1])):
            with self.assertRaises(ValueError):
                _distance_choices('vv',value)

    def test_tighter_policy_still_rejects_warm_geometry(self):
        points=((1e308,0.,0.),(math.ulp(0.),0.,0.))
        generous=WorkPolicy(max_fraction_bits=131072)
        _,value,_=closest_feature('vv',points,WorkBudget(generous))
        self.assertGreater(value.numerator.bit_length(),4096)
        for operation in (lambda:closest_feature('vv',points,WorkBudget(WorkPolicy(max_fraction_bits=4096))),
                          lambda:distance_squared('vv','P_P',points,WorkBudget(WorkPolicy(max_fraction_bits=4096)))):
            with self.assertRaisesRegex(ValueError,'rational-size budget'):
                operation()

    def test_feature_admission_and_exact_ties_still_apply_on_hits(self):
        points=((1.,0.,1.),(0.,0.,0.),(2.,0.,0.))
        self.assertEqual(closest_feature('ev',points,WorkBudget()),('P_E',F(1),('P_E',)))
        with self.assertRaisesRegex(ValueError,'Native closest feature'):
            distance_squared('ev','P_E0',points,WorkBudget())
        tied=((0.,0.,1.),points[1],points[2])
        expected=('P_E0',F(1),('P_E0','P_E'))
        self.assertEqual(closest_feature('ev',tied,WorkBudget()),expected)
        self.assertEqual(closest_feature('ev',tied,WorkBudget()),expected)

    def test_adjacent_binary64_positions_have_distinct_values(self):
        a=((0.,0.,0.),(0.,0.,1.))
        b=(a[0],(0.,0.,math.nextafter(1.,math.inf)))
        x=_distance_choices('vv',a)['P_P']
        y=_distance_choices('vv',b)['P_P']
        self.assertLess(x,y)
        self.assertEqual(y,F(b[1][2])**2)
        self.assertEqual(_distance_choice_integers.cache_info().currsize,2)

    def test_capacity_and_eviction_do_not_change_values(self):
        first=((0.,0.,0.),(1.,0.,0.))
        expected=_distance_choices('vv',first)
        for i in range(300):
            _distance_choices('vv',(first[0],(2.+i,0.,0.)))
        self.assertLessEqual(_distance_choice_integers.cache_info().currsize,256)
        misses=_distance_choice_integers.cache_info().misses
        self.assertEqual(_distance_choices('vv',first),expected)
        self.assertEqual(_distance_choice_integers.cache_info().misses,misses+1)

    def test_degenerate_geometry_does_not_install_a_cached_value(self):
        points=((0.,0.,1.),(0.,0.,0.),(0.,0.,0.))
        for _ in range(2):
            with self.assertRaisesRegex(ValueError,'Degenerate contact edge'):
                _distance_choices('ev',points)
        self.assertEqual(_distance_choice_integers.cache_info().currsize,0)

    def test_nonstandard_kind_retains_uncached_behavior(self):
        class Kind(str):
            pass
        points=((0.,0.,0.),(1.,2.,2.))
        self.assertEqual(_distance_choices(Kind('vv'),points),_distance_choices_uncached(Kind('vv'),points))
        self.assertEqual(_distance_choice_integers.cache_info().currsize,0)

    def test_concurrent_callers_receive_detached_values(self):
        points=((0.,0.,0.),(1.,2.,2.))
        def call(_):
            values=_distance_choices('vv',points)
            result=values['P_P'].numerator,values['P_P'].denominator
            values['P_P']._numerator=123
            values.clear()
            return result
        with ThreadPoolExecutor(max_workers=4) as workers:
            self.assertEqual(list(workers.map(call,range(64))),[(9,1)]*64)
        self.assertEqual(_distance_choices('vv',points),{'P_P':F(9)})


if __name__=='__main__':
    unittest.main()
