"""Arithmetic and admission tests; high-precision formulas use Decimal.ln."""
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import math
import random
import unittest

from solver_contact_work import (BarrierParameters, ContactTerm, Interval, WorkBudget,
    WorkPolicy, barrier_change, barrier_value, contact_work, distance_squared, log_interval)


PARAMETERS = BarrierParameters.capture(.002,.0001,10000.)


def decimal(fraction):
    return D(fraction.numerator)/D(fraction.denominator)


def oracle_barrier(s,h):
    if s >= h:
        return D(0)
    s,h = decimal(s),decimal(h)
    return -(s-h)**2*(s/h).ln()


def vv(z, *, weight=1., bucket=0, ids=(0,1), parameters=PARAMETERS):
    return ContactTerm(bucket,"vv",ids,(("vertex0_id",ids[0]),("vertex1_id",ids[1])),"P_P",
        ((0.,0.,0.),(0.,0.,z)),weight,None,parameters)


class LogarithmTests(unittest.TestCase):
    def test_independent_decimal_enclosure(self):
        rng=random.Random(809)
        values=[F(1),F(2),F(1,2),F(1,2**1074),F(2**1024-2**971),
            F(1)+F(1,2**120),F(1)-F(1,2**120)]
        values += [F(rng.randrange(1,10**20),rng.randrange(1,10**20))*F(2)**rng.randrange(-200,200) for _ in range(80)]
        with localcontext() as context:
            context.prec=220
            for x in values:
                bound=log_interval(x,WorkBudget())
                reference=decimal(x).ln()
                self.assertLessEqual(decimal(bound.lower),reference)
                self.assertLessEqual(reference,decimal(bound.upper))

    def test_tiny_argument_change_is_bounded_not_advertised_exact(self):
        result=log_interval(F(1)+F(1,2**1074),WorkBudget())
        self.assertLessEqual(result.lower,0)
        self.assertGreater(result.upper,0)

    def test_zero_and_inverse(self):
        self.assertEqual(log_interval(F(1),WorkBudget()),Interval(F(),F()))
        for x in (F(3,2),F(1,10),F(1,2**1074)):
            a,b=log_interval(x,WorkBudget()),log_interval(1/x,WorkBudget())
            self.assertEqual((a.lower,a.upper),(-b.upper,-b.lower))

    def test_log_work_budgets_and_domain(self):
        for x in (F(),F(-1),1.):
            with self.assertRaises(ValueError):log_interval(x,WorkBudget())
        for policy in (WorkPolicy(max_log_terms=1),WorkPolicy(max_total_log_terms=1)):
            with self.assertRaisesRegex(ValueError,"budget"):log_interval(F(2),WorkBudget(policy))
        with self.assertRaisesRegex(ValueError,"rational-size"):
            log_interval(F(1,2**4097),WorkBudget(WorkPolicy(max_fraction_bits=4096)))
        for fields in ({'bits':True},{'bits':63},{'max_log_terms':0},{'max_endpoint_terms':1e5}):
            with self.assertRaises(ValueError):WorkPolicy(**fields)


class BarrierTests(unittest.TestCase):
    def test_tiny_changes_clamp_and_independent_endpoint_formula(self):
        h=F(PARAMETERS.h)
        for s0,s1 in ((h/2,h/2+F(1,2**90)),(h-F(1,2**80),h),(h,h-F(1,2**80)),
                (h/2,2*h),(2*h,3*h),(F(1,2**1074),h/2)):
            result=barrier_change(s0,s1,h,WorkBudget())
            with localcontext() as context:
                context.prec=400
                reference=oracle_barrier(s1,h)-oracle_barrier(s0,h)
                self.assertLessEqual(decimal(result.lower),reference)
                self.assertLessEqual(reference,decimal(result.upper))

    def test_domain_and_exact_equal_distance(self):
        self.assertEqual(barrier_change(F(1),F(1),F(2),WorkBudget()),Interval(F(),F()))
        for s in (F(),F(-1)):
            with self.assertRaises(ValueError):barrier_value(s,F(1),WorkBudget())
            with self.assertRaises(ValueError):barrier_change(s,s,F(1),WorkBudget())

    def test_factorization_and_distinct_activation_boundaries(self):
        difference=F(PARAMETERS.candidate_outer_squared)-F(PARAMETERS.minimum_squared)-F(PARAMETERS.h)
        self.assertEqual(difference,F(-285,302231454903657293676544))
        with self.assertRaises(ValueError):replace(PARAMETERS,h=math.nextafter(PARAMETERS.h,math.inf))
        for values in ((True,.001,1.),(.001,.001,math.inf),(1e-300,1e-300,1.),(.001,0.,1.)):
            with self.assertRaises(ValueError):BarrierParameters.capture(*values)


class FeatureTests(unittest.TestCase):
    def distance(self,kind,feature,*positions):
        return distance_squared(kind,feature,tuple(tuple(float(v) for v in p) for p in positions),WorkBudget())

    def test_point_point_and_every_point_edge_feature(self):
        self.assertEqual(self.distance('vv','P_P',(0,0,0),(1,2,2)),9)
        for feature,p in (('P_E0',(-1,0,1)),('P_E1',(3,0,1)),('P_E',(1,0,1))):
            self.assertEqual(self.distance('ev',feature,p,(0,0,0),(2,0,0)),1 if feature=='P_E' else 2)

    def test_every_point_triangle_feature(self):
        points={'P_T0':(-1,-1,1),'P_T1':(3,-1,1),'P_T2':(-1,3,1),
                'P_E0':(1,-1,1),'P_E1':(1.5,1.5,1),'P_E2':(-1,1,1),'P_T':(.5,.5,1)}
        expected={'P_T0':3,'P_T1':3,'P_T2':3,'P_E0':2,'P_E1':1.5,'P_E2':2,'P_T':1}
        for feature,p in points.items():
            self.assertEqual(self.distance('fv',feature,p,(0,0,0),(2,0,0),(0,2,0)),F(expected[feature]))

    def test_every_edge_edge_feature(self):
        # Horizontal A and vertical B; move B across each Voronoi cell.
        for x,astate in ((-1.,'EA0'),(1.,'EA'),(3.,'EA1')):
            for y,bstate in ((1.,'EB0'),(-1.,'EB'),(-3.,'EB1')):
                expected=1+(1 if astate!='EA' else 0)+(1 if bstate!='EB' else 0)
                self.assertEqual(self.distance('ee',astate+'_'+bstate,
                    (0,0,0),(2,0,0),(x,y,1),(x,y+2,1)),expected)

    def test_tie_equivalence_and_no_feature_reselection(self):
        args=((0.,0.,1.),(0.,0.,0.),(2.,0.,0.))
        self.assertEqual(self.distance('ev','P_E0',*args),self.distance('ev','P_E',*args))
        with self.assertRaisesRegex(ValueError,'Native closest feature'):
            self.distance('ev','P_E0',(1e-30,0,1),(0,0,0),(2,0,0))
        with self.assertRaisesRegex(ValueError,'Native closest feature'):
            self.distance('ee','EA_EB',(0,0,0),(2,0,0),(0,0,1),(2,0,1))

    def test_degenerate_and_invalid_inputs(self):
        for kind,feature,args in (('ev','P_E',((0,0,1),(0,0,0),(0,0,0))),
                ('fv','P_T',((0,0,1),(0,0,0),(1,0,0),(2,0,0)))):
            with self.assertRaisesRegex(ValueError,'Degenerate'):self.distance(kind,feature,*args)
        with self.assertRaises(ValueError):self.distance('vv','AUTO',(0,0,0),(1,0,0))
        with self.assertRaises(ValueError):self.distance('vv','P_P',(0,0,0),(math.nan,0,0))


class InventoryTests(unittest.TestCase):
    def assert_encloses(self,result,reference):
        with localcontext() as context:
            context.prec=200
            self.assertLessEqual(decimal(result.lower),reference)
            self.assertLessEqual(reference,decimal(result.upper))
            self.assertLessEqual(abs(D(result.value)-reference),decimal(result.absolute_error))

    def oracle(self,term):
        s=F(term.positions[1][2])**2-F(term.parameters.minimum_squared)
        return oracle_barrier(s,F(term.parameters.h))*D(term.weight)*D(term.parameters.pressure)*D(term.parameters.normalization)

    def test_tiny_native_scale_work_and_output_rounding(self):
        start,end=vv(.00208),vv(.00208+1e-14)
        result=contact_work((start,),(end,))
        with localcontext() as context:
            context.prec=200
            self.assert_encloses(result,self.oracle(end)-self.oracle(start))
        self.assertLess(result.absolute_error,F(1,10**25))

    def test_zero_reversal_and_telescope(self):
        a,b,c=vv(.0015),vv(.0016),vv(.0017)
        zero=contact_work((a,),(a,))
        self.assertEqual((zero.value,zero.absolute_error),(0.,F()))
        ab,ba,bc,ac=(contact_work((x,),(y,)) for x,y in ((a,b),(b,a),(b,c),(a,c)))
        self.assertLessEqual(abs(F(ab.value)+F(ba.value)),ab.absolute_error+ba.absolute_error)
        self.assertLessEqual(abs(F(ab.value)+F(bc.value)-F(ac.value)),ab.absolute_error+bc.absolute_error+ac.absolute_error)

    def test_missing_duplicate_weight_change_buckets_and_signed_cancellation(self):
        a=vv(.0015,weight=1e100)
        b=replace(a,weight=-1e100)
        c=vv(.0016,weight=1.,bucket=1,ids=(2,3))
        end=vv(.0017,weight=-2.,bucket=1,ids=(2,3))
        result=contact_work((a,a,b,b,c),(end,))
        self.assertEqual((result.endpoint_terms,result.union_terms),(6,5))
        with localcontext() as context:
            context.prec=250
            self.assert_encloses(result,self.oracle(end)-self.oracle(c))
        self.assertGreater(result.absolute_error,0) # cancellation does not cancel radii

    def test_changed_feature_is_retained_and_mollifier_crosses_threshold(self):
        params=BarrierParameters.capture(2.,.1,1.)
        def edge(offset,height):
            return ContactTerm(0,'ee',(0,1,2,3),(('edge0_id',0),('edge1_id',1)),
                'EA0_EB0' if offset<0 else 'EA_EB0',
                ((0.,0.,0.),(1.,0.,0.),(offset,0.,.5),(offset, height,.5)),1.,1.,params)
        a,b=edge(-.25,.5),edge(.25,1.5)
        result=contact_work((a,),(b,))
        self.assertEqual(result.union_terms,1)
        s0,m0=a.scalar(WorkBudget());s1,m1=b.scalar(WorkBudget())
        with localcontext() as context:
            context.prec=200
            self.assert_encloses(result,decimal(m1)*oracle_barrier(s1,F(params.h))-decimal(m0)*oracle_barrier(s0,F(params.h)))

    def test_parallel_mollifier_zero_but_domain_still_validated(self):
        params=BarrierParameters.capture(2.,.1,1.)
        t=ContactTerm(0,'ee',(0,1,2,3),(),'EA0_EB0',
            ((0.,0.,0.),(1.,0.,0.),(0.,0.,.5),(1.,0.,.5)),1.,1.,params)
        result=contact_work((),(t,))
        self.assertEqual((result.value,result.absolute_error),(0.,F()))
        with self.assertRaises(ValueError):contact_work((),(replace(t,positions=t.positions[:2]+t.positions[:2]),))

    def test_subnormal_work_never_loses_its_error_radius(self):
        parameters=BarrierParameters.capture(.002,.0001,math.ulp(0.))
        a=vv(.0015,weight=1e-20,parameters=parameters)
        result=contact_work((),(a,))
        self.assertEqual(result.value,0.)
        self.assertGreater(result.absolute_error,0)
        self.assertEqual(result.binary64_error_bound(),math.ulp(0.))
        with localcontext() as context:
            context.prec=220
            self.assert_encloses(result,self.oracle(a))

    def test_overflow_and_count_budget_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'Unrepresentable'):
            contact_work((),(vv(.001,weight=1e308,parameters=BarrierParameters.capture(.002,.0001,1e308)),))
        with self.assertRaisesRegex(ValueError,'endpoint-term budget'):
            contact_work((vv(.001),),(vv(.0015),),policy=WorkPolicy(max_endpoint_terms=1))

    def test_outward_error_certificate_includes_rounding(self):
        result=contact_work((vv(.0015),),(vv(.0016),))
        self.assertGreaterEqual(F(result.binary64_error_bound()),result.absolute_error)
        self.assertEqual(contact_work((),()).binary64_error_bound(),0.)
        with self.assertRaisesRegex(ValueError,'error certificate'):
            replace(result,absolute_error=F(2)**1024).binary64_error_bound()

    def test_invalid_capture_identity_geometry_and_mutability(self):
        a=vv(.0015)
        with self.assertRaisesRegex(ValueError,'vertex coordinates'):
            contact_work((a,vv(.0016)),())
        with self.assertRaisesRegex(ValueError,'bucket parameters'):
            contact_work((a,vv(.0016,ids=(2,3),parameters=BarrierParameters.capture(.003,.0001,10000.))),())
        for obj in (a,PARAMETERS,WorkPolicy()):
            self.assertFalse(hasattr(obj,'__dict__'))
            with self.assertRaises((FrozenInstanceError,TypeError)):obj.unknown=0
        policy=WorkPolicy(max_endpoint_terms=1)
        for update in ({'max_endpoint_terms':100000},{'bits':1}):
            with self.assertRaisesRegex(AttributeError,'already initialized'):policy.__init__(**update)
            self.assertEqual((policy.bits,policy.max_endpoint_terms),(160,1))
        previous=PARAMETERS.pressure
        with self.assertRaisesRegex(AttributeError,'already initialized'):
            PARAMETERS.__init__(.002,.0001,2.,1e-8,4.4e-6,1.936e-11,103305785.12396693,4.409999999999999e-6)
        self.assertEqual(PARAMETERS.pressure,previous)
        with self.assertRaises(ValueError):replace(a,positions=[[0.,0.,0.],[0.,0.,.0015]])
        with self.assertRaises(ValueError):replace(a,weight=True)
        with self.assertRaises(ValueError):replace(a,eps_x=1.)
        with self.assertRaises(ValueError):contact_work([a],())
        # This profile subtracts captured RN(dmin*dmin). Exact dmin² is
        # slightly larger here; native state admission remains a separate gate.
        self.assertGreater(F(.0001)**2-F(PARAMETERS.minimum_squared),0)
        with self.assertRaises(ValueError):contact_work((),(vv(math.nextafter(.0001,0.)),))


if __name__=='__main__':unittest.main()
