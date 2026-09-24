"""Independent continuum cable mechanics oracles; no garment admission."""
import copy
from decimal import Decimal, localcontext
from fractions import Fraction as F
from functools import lru_cache
import json
import math
import sys
import unittest
from unittest import mock

import numpy as np

from solver_continuous_cable_sewing import ContinuousCableSewing
import solver_continuous_cable_sewing as cable


def rat(value):
    value=F(value)
    return {'numerator':str(value.numerator),'denominator':str(value.denominator)}


def read(value):
    if type(value) is not dict or set(value)!={'numerator','denominator'}:
        raise AssertionError('Canonical rational error bound required')
    result=F(int(value['numerator']),int(value['denominator']))
    if rat(result)!=value:raise AssertionError('Unreduced rational witness')
    return result


def dec(value):
    value=F(value)
    return Decimal(value.numerator)/Decimal(value.denominator)


def anchor(values):
    return [{'vertex':v,'weight':rat(w)} for v,w in sorted(values.items()) if w]


def cell(name='curve',**changes):
    result={'id':name,'negativeStart':anchor({0:F(1)}),'negativeEnd':anchor({1:F(1)}),
        'positiveStart':anchor({2:F(1)}),'positiveEnd':anchor({3:F(1)}),
        'targetsMeters':[.25,.5],'referenceLengthMeters':rat(F(1,2)),
        'stiffnessDensityNPerM2':16.,'activation':1.}
    result.update(changes)
    return result


def constant_cell(name='point',negative=0,positive=1,**changes):
    options=dict(negativeStart=anchor({negative:F(1)}),negativeEnd=anchor({negative:F(1)}),
                 positiveStart=anchor({positive:F(1)}),positiveEnd=anchor({positive:F(1)}),
                 targetsMeters=[.5,.5],referenceLengthMeters=rat(F(1)),stiffnessDensityNPerM2=1.)
    options.update(changes)
    return cell(name,**options)


TOL={'energy_tolerance_joules':1e-9,'gradient_tolerance_newtons':1e-9,
     'hessian_tolerance_newtons_per_meter':1e-9}
BOUND_KEYS=('energyErrorBoundJoules','gradientMaxAbsoluteErrorBoundNewtons',
            'hessianMaxEntryErrorBoundNewtonsPerMeter')


def numerical_rows(c,vertex_count):
    result=[]
    for ending in ('Start','End'):
        row=[F() for _ in range(vertex_count)]
        for sign,name in ((1,'positive'),(-1,'negative')):
            for term in c[name+ending]:
                row[term['vertex']]+=sign*F(float(read(term['weight'])))
        result.append(row)
    return result


@lru_cache(None)
def gauss_legendre(order):
    """Independent 80-digit root iteration; no SciPy integration or kernel."""
    with localcontext() as context:
        context.prec=80
        result=[]
        for k in range(1,order//2+1):
            z=Decimal(math.cos(math.pi*(k-.25)/(order+.5)))
            for _ in range(80):
                first,second=Decimal(1),z
                for degree in range(2,order+1):
                    first,second=second,((2*degree-1)*z*second-(degree-1)*first)/degree
                derivative=order*(z*second-first)/(z*z-1)
                update=second/derivative
                z-=update
                if abs(update)<Decimal('1e-74'):break
            else:raise AssertionError('Independent quadrature roots did not converge')
            weight=Decimal(2)/((1-z*z)*derivative*derivative)
            result.extend([(-z,weight),(z,weight)])
        return tuple(sorted(result))


def independent_decimal_integral(cells,positions,order=40):
    """Integrate analytic E/g/H with independent Decimal active-root splits.

    The tests compare two quadrature orders and retain their discrepancy as
    reference uncertainty; this is an independent numerical oracle, not a new
    certified integration implementation.
    """
    with localcontext() as context:
        context.prec=80
        n=len(positions);size=3*n;zero=Decimal(0)
        q=[[dec(F(float(v))) for v in p] for p in positions]
        total=[zero]*(1+size+size*size)
        for c in cells:
            if not c['activation']:continue
            rows=[[dec(x) for x in row] for row in numerical_rows(c,n)]
            vectors=[[sum((row[v]*q[v][a] for v in range(n)),zero) for a in range(3)] for row in rows]
            a=vectors[0];b=[y-x for x,y in zip(*vectors)]
            targets=[dec(F(x)) for x in c['targetsMeters']];d0=targets[0];dd=targets[1]-d0
            beta=dec(F(c['activation'])*F(c['stiffnessDensityNPerM2'])*read(c['referenceLengthMeters']))
            quadratic=sum((x*x for x in b),zero)-dd*dd
            linear=2*(sum((x*y for x,y in zip(a,b)),zero)-d0*dd)
            constant=sum((x*x for x in a),zero)-d0*d0
            if quadratic==linear==constant==0:continue # declared slack-sided equality
            breaks=[zero,Decimal(1)]
            if quadratic:
                discr=linear*linear-4*quadratic*constant
                if discr>=0:
                    square=discr.sqrt()
                    breaks.extend(x for x in ((-linear-square)/(2*quadratic),(-linear+square)/(2*quadratic)) if 0<x<1)
            elif linear:
                x=-constant/linear
                if 0<x<1:breaks.append(x)
            breaks=sorted(set(breaks))
            for left,right in zip(breaks,breaks[1:]):
                midpoint=(left+right)/2
                if constant+linear*midpoint+quadratic*midpoint*midpoint<=0:continue
                half=(right-left)/2
                for node,weight in gauss_legendre(order):
                    u=midpoint+half*node;factor=beta*half*weight
                    D=[x+u*y for x,y in zip(a,b)];d=d0+u*dd
                    r=sum((x*x for x in D),zero).sqrt()
                    assert r>d>0
                    A=[(1-u)*x+u*y for x,y in zip(*rows)]
                    tangent=1-d/r
                    total[0]+=factor*(r-d)*(r-d)/2
                    for i in range(n):
                        for axis in range(3):
                            total[1+3*i+axis]+=factor*A[i]*tangent*D[axis]
                            for j in range(n):
                                for other in range(3):
                                    radial=tangent*int(axis==other)+d*D[axis]*D[other]/(r*r*r)
                                    index=1+size+(3*i+axis)*size+3*j+other
                                    total[index]+=factor*A[i]*A[j]*radial
        return total


class ContinuousCableSewingTests(unittest.TestCase):
    def setUp(self):
        self.q=np.array([[0.,0.,0.],[0.,1.,0.],[.5,0.,1.5],[1.25,1.,2.]])
        self.cells=[cell()]

    def evaluate(self,potential,q,**changes):
        result=potential.evaluate(q,**(TOL|changes))
        self.assertEqual(set(result),{'energy','gradient','hessian','certificate'})
        self.assertIs(type(result['energy']),float)
        self.assertEqual(result['gradient'].shape,(3*len(q),))
        self.assertEqual(result['hessian'].shape,(3*len(q),3*len(q)))
        self.assertEqual(result['hessian'].format,'csr')
        self.assertTrue(np.isfinite(result['gradient']).all())
        self.assertTrue(np.isfinite(result['hessian'].data).all())
        self.assertTrue(math.isfinite(result['energy']))
        certificate=result['certificate']
        self.assertIs(certificate['verified'],True);self.assertIs(certificate['accepted'],False)
        self.assertEqual(certificate['law'],'continuous-tension-only-separation-v1')
        self.assertEqual(certificate['hessianPolicy'],'slack-sided-generalized-curvature')
        self.assertEqual(certificate['inputSha256'],potential.description()['inputSha256'])
        self.assertEqual(len(certificate['positionsSha256']),64)
        for key,tolerance in zip(BOUND_KEYS,(TOL|changes).values()):
            self.assertGreaterEqual(read(certificate[key]),0)
        for key,tolerance_key in zip(BOUND_KEYS,TOL):
            self.assertLessEqual(read(certificate[key]),F((TOL|changes)[tolerance_key]))
        json.dumps(certificate,allow_nan=False)
        return result

    def flatten(self,result):
        return [result['energy'],*result['gradient'].tolist(),*result['hessian'].toarray().ravel().tolist()]

    def assert_reference(self,result,expected,uncertainty=None):
        size=len(result['gradient']);bounds=[read(result['certificate'][key]) for key in BOUND_KEYS]
        uncertainties=uncertainty or [Decimal(0)]*len(expected)
        with localcontext() as context:
            context.prec=80
            for i,(actual,want,extra) in enumerate(zip(self.flatten(result),expected,uncertainties)):
                category=0 if i==0 else 1 if i<=size else 2
                difference=abs(dec(F(actual))-(want if isinstance(want,Decimal) else dec(want)))
                self.assertLessEqual(difference,dec(bounds[category])+extra+Decimal('1e-65'),(i,difference,bounds[category]))

    def decimal_reference(self,cells,q):
        first=independent_decimal_integral(cells,q,32)
        second=independent_decimal_integral(cells,q,48)
        uncertainty=[abs(a-b)*2+Decimal('1e-65') for a,b in zip(first,second)]
        self.assertLess(max(uncertainty),Decimal('1e-24'))
        return second,uncertainty

    def test_exact_slack_coincident_and_fully_taut_generalized_zero(self):
        c=constant_cell();potential=ContinuousCableSewing(2,[c])
        self.assertFalse(hasattr(potential,'exact_hessian'))
        for gap in (0.,.25,.5):
            with self.subTest(gap=gap):
                result=self.evaluate(potential,[[0.,0.,0.],[gap,0.,0.]],max_boundary_depth=0,max_boundary_panels=1,
                                     moment_max_terms=1,moment_max_panels=1,moment_max_depth=0)
                self.assertEqual(self.flatten(result),[0.]*43)
                self.assertEqual([read(result['certificate'][key]) for key in BOUND_KEYS],[0,0,0])

    def test_exact_constant_extended_curve_rational_energy_gradient_hessian(self):
        c=constant_cell(targetsMeters=[2.,2.],stiffnessDensityNPerM2=8.)
        result=self.evaluate(ContinuousCableSewing(2,[c]),[[0.,0.,0.],[3.,4.,0.]])
        D=[F(3),F(4),F()];beta=F(8);r=F(5);d=F(2)
        gradient=[sign*beta*(1-d/r)*x for sign in (-1,1) for x in D]
        H=[]
        for i in range(6):
            for j in range(6):
                sign=(-1 if i<3 else 1)*(-1 if j<3 else 1);a,b=i%3,j%3
                H.append(sign*beta*((1-d/r)*int(a==b)+d*D[a]*D[b]/r**3))
        self.assert_reference(result,[beta*(r-d)**2/2,*gradient,*H])

    def test_exact_collinear_active_polynomial_and_beta_not_rounded(self):
        c=cell(targetsMeters=[.5,1.],referenceLengthMeters=rat(F(1,3)),stiffnessDensityNPerM2=1.)
        q=[[0.,0.,0.],[0.,0.,0.],[2.,0.,0.],[4.,0.,0.]]
        result=self.evaluate(ContinuousCableSewing(4,[c]),q)
        beta=F(1,3);gradient=[]
        for value in (-beta,-5*beta/4,beta,5*beta/4):gradient.extend([value,F(),F()])
        H=[];signs=[-1,-1,1,1]
        for i in range(12):
            for j in range(12):
                a,b=i//3,j//3
                gram=F(1,3) if a%2==b%2 else F(1,6)
                H.append(beta*signs[a]*signs[b]*gram*(F(1) if i%3==0 else F(3,4)) if i%3==j%3 else F())
        self.assert_reference(result,[21*beta/8,*gradient,*H])

    def test_full_active_independent_decimal_energy_gradient_and_hessian(self):
        result=self.evaluate(ContinuousCableSewing(4,self.cells),self.q)
        self.assert_reference(result,*self.decimal_reference(self.cells,self.q))

    def test_derivatives_match_centered_differences_away_from_thresholds(self):
        potential=ContinuousCableSewing(4,self.cells)
        precision={'energy_tolerance_joules':1e-13,'gradient_tolerance_newtons':1e-13,
                   'hessian_tolerance_newtons_per_meter':1e-12}
        center=self.evaluate(potential,self.q,**precision);step=2e-5
        for dof,direction in enumerate(np.eye(12).reshape(12,4,3)):
            before=self.evaluate(potential,self.q-step*direction,**precision)
            after=self.evaluate(potential,self.q+step*direction,**precision)
            derivative=(after['energy']-before['energy'])/(2*step)
            self.assertAlmostEqual(derivative,center['gradient'][dof],delta=3e-8)
            derivative=(after['gradient']-before['gradient'])/(2*step)
            np.testing.assert_allclose(derivative,center['hessian'].toarray()[:,dof],rtol=0,atol=3e-8)

    def test_overlapping_signed_support_cancels_and_pending_cells_keep_definitions(self):
        c=constant_cell(negativeStart=anchor({0:F(1,2),1:F(1,2)}),
            negativeEnd=anchor({0:F(1,2),1:F(1,2)}),
            positiveStart=anchor({0:F(1,2),2:F(1,2)}),positiveEnd=anchor({0:F(1,2),2:F(1,2)}))
        potential=ContinuousCableSewing(3,[c])
        for irrelevant in (-1e6,1e6):
            q=[[irrelevant,irrelevant,irrelevant],[0.,0.,0.],[2.,0.,0.]]
            result=self.evaluate(potential,q)
            self.assertEqual(result['energy'],.125)
            self.assertEqual(result['gradient'].tolist(),[0.,0.,0.,-.25,0.,0.,.25,0.,0.])
            self.assertFalse(np.any(result['hessian'].toarray()[:3]))
        pending=copy.deepcopy(c);pending['activation']=0.
        inactive=ContinuousCableSewing(3,[pending])
        result=self.evaluate(inactive,[[1e6,1e6,1e6],[-1e6,0.,0.],[1e6,0.,0.]],
            max_boundary_depth=0,max_boundary_panels=1,moment_max_terms=1,moment_max_panels=1,moment_max_depth=0)
        self.assertEqual(inactive.cells,[pending])
        self.assertEqual(result['energy'],0.);self.assertFalse(np.any(result['gradient']))
        self.assertEqual(result['hessian'].nnz,0)

    def test_two_irrational_active_crossings_are_integrated_with_bounds(self):
        c=cell(targetsMeters=[1.,1.],referenceLengthMeters=rat(F(1)),stiffnessDensityNPerM2=1.)
        q=[[0.,0.,0.],[0.,0.,0.],[-2.,0.,.5],[2.,0.,.5]]
        result=self.evaluate(ContinuousCableSewing(4,[c]),q,energy_tolerance_joules=1e-8,
                             gradient_tolerance_newtons=1e-7,hessian_tolerance_newtons_per_meter=1e-7)
        self.assert_reference(result,*self.decimal_reference([c],q))
        self.assertGreater(result['energy'],0.)

    def test_isolated_double_root_keeps_active_measure(self):
        c=cell(targetsMeters=[1.,1.],referenceLengthMeters=rat(F(1)),stiffnessDensityNPerM2=1.)
        q=[[0.,0.,0.],[0.,0.,0.],[-.5,0.,1.],[.5,0.,1.]]
        result=self.evaluate(ContinuousCableSewing(4,[c]),q)
        self.assert_reference(result,*self.decimal_reference([c],q))
        self.assertGreater(result['energy'],0.)

    def test_exact_crease_roll_wrong_side_and_shear_fixtures_remain_taut_zero(self):
        L,w,d=1/8,1/16,5/1024
        q=np.array([[0,0,0],[0,L,0],[w,0,0],[w,L,0],[-w,0,0],[-w,L,0],
                    [0,0,d],[0,L,d],[-w,0,d],[-w,L,d]],dtype=float)
        c=cell(positiveStart=anchor({6:F(1)}),positiveEnd=anchor({7:F(1)}),targetsMeters=[d,d])
        poses=[q.copy()]
        for sign in (-1,1):
            pose=q.copy()
            for v in (2,3):pose[v]=[0.,pose[v,1],sign*w]
            poses.append(pose)
        pose=q.copy();pose[6:,0]*=-1;pose[6:,2]*=-1;poses.append(pose)
        pose=q.copy();pose[6:,1]+=3/1024;pose[6:,2]-=1/1024;poses.append(pose)
        potential=ContinuousCableSewing(10,[c])
        for pose in poses:
            result=self.evaluate(potential,pose)
            self.assertEqual(result['energy'],0.);self.assertFalse(np.any(result['gradient']))
            self.assertEqual(result['hessian'].nnz,0)

    def test_curved_offset_chord_and_collapsed_interval_are_exactly_slack(self):
        d=5/1024;c=cell(targetsMeters=[d,d],stiffnessDensityNPerM2=1000.)
        for q in ([[0,0,0],[0,0,0],[0,0,d],[3/1024,0,4/1024]],np.zeros((4,3))):
            result=self.evaluate(ContinuousCableSewing(4,[c]),q)
            self.assertEqual(self.flatten(result),[0.]*(1+12+144))
            self.assertEqual([read(result['certificate'][key]) for key in BOUND_KEYS],[0,0,0])

    def test_measure_preserving_subdivision_and_proper_rigid_covariance(self):
        original=self.cells[0];first,last=copy.deepcopy(original),copy.deepcopy(original)
        first['id'],last['id']='first','last';t=F(1,4)
        for sign in ('positive','negative'):
            middle={}
            for scale,end in ((1-t,'Start'),(t,'End')):
                for term in original[sign+end]:middle[term['vertex']]=middle.get(term['vertex'],F())+scale*read(term['weight'])
            first[sign+'End']=anchor(middle);last[sign+'Start']=anchor(middle)
        target=float((1-t)*F(original['targetsMeters'][0])+t*F(original['targetsMeters'][1]))
        first['targetsMeters'][1]=last['targetsMeters'][0]=target
        first['referenceLengthMeters']=rat(read(original['referenceLengthMeters'])*t)
        last['referenceLengthMeters']=rat(read(original['referenceLengthMeters'])*(1-t))
        base=self.evaluate(ContinuousCableSewing(4,[original]),self.q)
        split=self.evaluate(ContinuousCableSewing(4,[first,last]),self.q)
        for a,b in zip(self.flatten(base),self.flatten(split)):self.assertAlmostEqual(a,b,delta=3e-9)
        R=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        moved=self.q@R.T+[.75,.5,-.125]
        result=self.evaluate(ContinuousCableSewing(4,[original]),moved)
        self.assertAlmostEqual(result['energy'],base['energy'],delta=2e-9)
        np.testing.assert_allclose(result['gradient'].reshape(-1,3),base['gradient'].reshape(-1,3)@R.T,atol=2e-9,rtol=0)
        transform=np.kron(np.eye(4),R)
        np.testing.assert_allclose(result['hessian'].toarray(),transform@base['hessian'].toarray()@transform.T,atol=2e-9,rtol=0)
        gradient=base['gradient'].reshape(-1,3)
        np.testing.assert_allclose(gradient.sum(axis=0),0.,atol=4e-9)
        np.testing.assert_allclose(np.cross(self.q,gradient).sum(axis=0),0.,atol=2e-8)

    def test_global_assembly_retains_small_force_between_cancelled_large_terms(self):
        cells=[constant_cell('left',negative=0,positive=3,targetsMeters=[.25,.25],stiffnessDensityNPerM2=1e12),
               constant_cell('small',negative=2,positive=3,targetsMeters=[.25,.25],stiffnessDensityNPerM2=2**-28),
               constant_cell('right',negative=1,positive=3,targetsMeters=[.25,.25],stiffnessDensityNPerM2=1e12)]
        q=[[-1.,0.,0.],[1.,0.,0.],[-.5,0.,0.],[0.,0.,0.]]
        result=self.evaluate(ContinuousCableSewing(4,cells),q,energy_tolerance_joules=1e-6,
                             gradient_tolerance_newtons=1e-12,hessian_tolerance_newtons_per_meter=1e-7)
        want=F(1,2**30);error=read(result['certificate'][BOUND_KEYS[1]])
        self.assertLessEqual(abs(F(float(result['gradient'][9]))-want),error)
        self.assertNotEqual(result['gradient'][9],0.)

    def test_declared_anchor_conversion_defect_is_not_silently_normalized(self):
        exact={2:F(1,10),3:F(1,5),4:F(7,10)}
        c=constant_cell(positiveStart=anchor(exact),positiveEnd=anchor(exact))
        potential=ContinuousCableSewing(5,[c]);description=potential.description()
        conversion=description['conversions'][0]
        weights={v:F(float(w)) for v,w in exact.items()};defect=sum(weights.values(),F())-1
        self.assertEqual(defect,-F(1,2**55))
        self.assertEqual([read(x) for x in conversion['numericalSignedCoefficientSums']],[defect,defect])
        for term in conversion['anchorConversions']['positiveStart']:
            self.assertEqual(read(term['exactWeight']),exact[term['vertex']])
            self.assertEqual(F(term['numericalWeight']),weights[term['vertex']])
            self.assertEqual(read(term['numericalMinusExact']),weights[term['vertex']]-exact[term['vertex']])
        A=[-F(1),F(),weights[2],weights[3],weights[4]];energies=[]
        for translation in (0,1024):
            q=[[float(translation),0.,0.],[float(translation),0.,0.]]+[[float(translation+1),0.,0.] for _ in range(3)]
            result=self.evaluate(potential,q,energy_tolerance_joules=1e-16,
                gradient_tolerance_newtons=1e-16,hessian_tolerance_newtons_per_meter=1e-15)
            radius=sum(weights.values(),F())+translation*defect;target=F(1,2)
            gradient=[value for coefficient in A for value in (coefficient*(radius-target),F(),F())]
            H=[A[i//3]*A[j//3]*(1 if i%3==0 else 1-target/radius) if i%3==j%3 else F()
               for i in range(15) for j in range(15)]
            self.assert_reference(result,[(radius-target)**2/2,*gradient,*H]);energies.append(result['energy'])
        self.assertLess(energies[1],energies[0]) # Declared map defect, not a rigid-covariance claim.

    def test_tiny_fixed_parameter_work_survives_large_unchanged_energy(self):
        cells=[constant_cell('large',negative=0,positive=1,targetsMeters=[1.,1.],stiffnessDensityNPerM2=1e12),
               constant_cell('small',negative=2,positive=3)]
        q=np.array([[0.,0.,0.],[1000.,0.,0.],[0.,0.,0.],[1.,0.,0.]])
        end=q.copy();end[3,0]=math.nextafter(1.,math.inf)
        potential=ContinuousCableSewing(4,cells)
        result=potential.energy_change(q,end,absolute_tolerance_joules=1e-20)
        self.assertIs(result['certificate']['verified'],True);self.assertIs(result['certificate']['accepted'],False)
        delta=F(float(end[3,0]))-1;expected=delta/2+delta*delta/2
        bound=read(result['certificate']['changeErrorBoundJoules'])
        self.assertLessEqual(bound,F(1e-20));self.assertLessEqual(abs(F(result['changeJoules'])-expected),bound)
        self.assertGreater(result['changeJoules'],0.)
        same=q.copy();same[:,0],same[:,2]=q[:,2],-q[:,0]
        result=potential.energy_change(q,same,absolute_tolerance_joules=1e-30,moment_max_terms=1,moment_max_panels=1,moment_max_depth=0)
        self.assertEqual(result['changeJoules'],0.);self.assertEqual(read(result['certificate']['changeErrorBoundJoules']),0)

    def test_requested_precision_below_output_rounding_fails_closed(self):
        c=constant_cell(referenceLengthMeters=rat(F(1,3)))
        with self.assertRaises(ValueError):
            self.evaluate(ContinuousCableSewing(2,[c]),[[0.,0.,0.],[1.,0.,0.]],energy_tolerance_joules=1e-30)

    def test_nonzero_underflow_output_carries_a_nonzero_valid_bound(self):
        tiny=math.ulp(0.)
        c=constant_cell(stiffnessDensityNPerM2=tiny)
        result=self.evaluate(ContinuousCableSewing(2,[c]),[[0.,0.,0.],[1.,0.,0.]],
            energy_tolerance_joules=tiny,gradient_tolerance_newtons=tiny,hessian_tolerance_newtons_per_meter=tiny)
        self.assertEqual(result['energy'],0.)
        self.assertGreater(read(result['certificate'][BOUND_KEYS[0]]),0)
        self.assertLessEqual(F(tiny)/8,read(result['certificate'][BOUND_KEYS[0]]))

    def test_error_bound_ceiling_handles_huge_rationals_and_exact_tolerance(self):
        original_limit=sys.get_int_max_str_digits()
        epsilon=F(1,2**20000);base=F(1,2**60);tiny=F(math.ulp(0.))
        huge=base+epsilon
        self.assertGreater(huge.numerator.bit_length(),19000)
        self.assertGreater(huge.denominator.bit_length(),19000)
        if 0<original_limit<6000:
            with self.assertRaises(ValueError):str(huge.denominator)
        cases=[(F(),tiny,F()),(epsilon,tiny,tiny),
               (huge,F(math.nextafter(float(base),math.inf)),F(math.nextafter(float(base),math.inf))),
               (base-epsilon,base,base),(base,base,base),
               (F(1)+F(1,2**53),F(math.nextafter(1.,math.inf)),F(math.nextafter(1.,math.inf)))]
        for exact_error,tolerance,expected in cases:
            encoded_bound=cable._error_bound(exact_error,tolerance)
            bound=read(encoded_bound)
            self.assertTrue(exact_error<=bound<=tolerance)
            self.assertEqual(bound,expected)
            self.assertLess(len(json.dumps(encoded_bound,allow_nan=False)),800)
        with self.assertRaises(ValueError):cable._error_bound(base+epsilon,base)
        with self.assertRaises(ValueError):cable._error_bound(-epsilon,tiny)
        self.assertEqual(sys.get_int_max_str_digits(),original_limit)

    def test_publication_serializes_large_exact_uncertainty_for_all_responses_and_work(self):
        # Publication-only mock: every enclosed true response is exactly zero
        # for the admitted slack states. This isolates the previously failing
        # conversion without pretending mocked intervals are a mechanics test.
        epsilon=F(1,2**20000);tiny=math.ulp(0.);potential=ContinuousCableSewing(2,[constant_cell()])
        intervals={('e',):(F(),epsilon),('g',0):(-epsilon,epsilon),
                   ('h',0,0):(F(),epsilon),('h',0,1):(-epsilon,epsilon)}
        stats={'testScope':'Publication-only valid zero-response enclosures'}
        q=[[0.,0.,0.],[.25,0.,0.]];end=[[0.,0.,0.],[.375,0.,0.]]
        original_limit=sys.get_int_max_str_digits()
        with mock.patch.object(ContinuousCableSewing,'_intervals',return_value=(intervals,stats)):
            result=self.evaluate(potential,q,energy_tolerance_joules=tiny,
                gradient_tolerance_newtons=tiny,hessian_tolerance_newtons_per_meter=tiny)
            self.assertEqual(result['energy'],0.);self.assertFalse(np.any(result['gradient']))
            self.assertEqual(result['hessian'].nnz,0)
            for key in BOUND_KEYS:
                bound=read(result['certificate'][key])
                self.assertTrue(epsilon<=bound<=F(tiny))
                self.assertEqual(bound,F(tiny))
            self.assertLess(len(json.dumps(result['certificate'],allow_nan=False)),10000)
            work=potential.energy_change(q,end,absolute_tolerance_joules=tiny)
            self.assertEqual(work['changeJoules'],0.)
            self.assertTrue(epsilon<=read(work['certificate']['changeErrorBoundJoules'])<=F(tiny))
            self.assertLess(len(json.dumps(work['certificate'],allow_nan=False)),10000)
        self.assertEqual(sys.get_int_max_str_digits(),original_limit)

    def test_immutable_capture_and_copied_public_views(self):
        cells=copy.deepcopy(self.cells);potential=ContinuousCableSewing(4,cells)
        baseline=potential.description();result=self.evaluate(potential,self.q)
        cells[0]['activation']=0.;cells[0]['positiveStart'][0]['weight']=rat(F(1,2))
        exposed=potential.cells;exposed[0]['targetsMeters'][0]=99.
        exposed=potential.description();exposed['inputSha256']='changed'
        potential.activation[:]=0.
        self.assertEqual(potential.description(),baseline)
        current=self.evaluate(potential,self.q)
        self.assertEqual(self.flatten(current),self.flatten(result))
        with self.assertRaises((AttributeError,TypeError)):potential._vertex_count=1
        with self.assertRaises((AttributeError,TypeError)):del potential._vertex_count

    def test_raw_constructor_fields_unit_sums_and_positive_underflow_reject(self):
        bad=[]
        for key,value in [('activation',True),('stiffnessDensityNPerM2',math.inf),('targetsMeters',[False,.5]),
                          ('frameVertices',[0,1,2]),('side',1),('referenceLengthMeters',rat(F()))]:
            changed=copy.deepcopy(self.cells[0]);changed[key]=value;bad.append(changed)
        changed=copy.deepcopy(self.cells[0]);changed['positiveStart']=anchor({2:F(1,3),3:F(1,3)});bad.append(changed)
        changed=copy.deepcopy(self.cells[0]);changed['positiveStart'][0]['vertex']=True;bad.append(changed)
        changed=copy.deepcopy(self.cells[0]);changed['positiveStart'][0]['weight']={'numerator':'2','denominator':'2'};bad.append(changed)
        changed=copy.deepcopy(self.cells[0]);changed['activation']=math.ulp(0.);changed['stiffnessDensityNPerM2']=.25;bad.append(changed)
        changed=copy.deepcopy(self.cells[0]);changed['positiveStart']=anchor({2:F(1)-F(1,2**1100),3:F(1,2**1100)});bad.append(changed)
        for c in bad:
            with self.subTest(cell=c):
                with self.assertRaises(ValueError):ContinuousCableSewing(4,[c])
        for count,cells in ((True,self.cells),(0,self.cells),(4,[]),(4,self.cells*4097),(4,self.cells*2)):
            with self.assertRaises(ValueError):ContinuousCableSewing(count,cells)

    def test_positions_tolerances_and_budget_raw_admission(self):
        potential=ContinuousCableSewing(4,self.cells)
        for q in (self.q.tolist()[:3],np.full((4,3),math.nan),np.full((4,3),1e6+1)):
            with self.assertRaises(ValueError):self.evaluate(potential,q)
        boolean=self.q.tolist();boolean[0][0]=False
        with self.assertRaises(ValueError):self.evaluate(potential,boolean)
        if np.dtype(np.longdouble).itemsize>8:
            with self.assertRaises(ValueError):self.evaluate(potential,self.q.astype(np.longdouble))
        for key in TOL:
            for value in (True,0.,-1.,math.inf,math.nan,1e6+1):
                with self.subTest(key=key,value=value):
                    with self.assertRaises(ValueError):self.evaluate(potential,self.q,**{key:value})
        for key,values in {'max_boundary_depth':(True,-1,129),'max_boundary_panels':(True,0,4097),
                           'moment_max_terms':(True,0,257),'moment_max_panels':(True,0,4097),
                           'moment_max_depth':(True,-1,129)}.items():
            for value in values:
                with self.assertRaises(ValueError):self.evaluate(potential,self.q,**{key:value})

    def test_mixed_boundary_budget_exhaustion_is_not_a_false_zero_certificate(self):
        c=cell(targetsMeters=[1.,1.],stiffnessDensityNPerM2=1.)
        q=[[0.,0.,0.],[0.,0.,0.],[-2.,0.,.5],[2.,0.,.5]]
        with self.assertRaises(ValueError):
            self.evaluate(ContinuousCableSewing(4,[c]),q,max_boundary_depth=0,max_boundary_panels=1)


if __name__=='__main__':unittest.main()
