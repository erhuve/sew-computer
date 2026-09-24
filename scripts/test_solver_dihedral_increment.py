"""Independent original-input Decimal angle oracles; no solver trajectories."""

from decimal import Decimal, localcontext
import math
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from solver_bending import ElasticDihedralBending
from solver_dihedral_increment import dihedral_angle_increment


PI = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521")


def _atan(value):
    sign = -1 if value < 0 else 1
    value = abs(value)
    factor = 1
    while value > Decimal(".1"):
        value = value / (1 + (1 + value*value).sqrt())
        factor *= 2
    total = term = value
    squared = value*value
    for index in range(1, 1000):
        term *= -squared
        candidate = total + term / (2*index+1)
        if candidate == total:
            return sign * factor * total
        total = candidate
    raise AssertionError("Decimal atan did not converge")


def _atan2(y, x):
    if not x:
        return PI/2 if y > 0 else -PI/2
    angle = _atan(y/x)
    return angle if x > 0 else angle + (PI if y >= 0 else -PI)


def _sub(first, second):
    return [a-b for a,b in zip(first,second)]


def _cross(a,b):
    return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]


def _dot(a,b):
    return sum((x*y for x,y in zip(a,b)),Decimal())


def decimal_angle(points):
    q=[[Decimal.from_float(float(v)) for v in row] for row in points]
    first=_cross(_sub(q[2],q[0]),_sub(q[3],q[0]))
    second=_cross(_sub(q[3],q[1]),_sub(q[2],q[1]))
    edge=_sub(q[3],q[2])
    return _atan2(_dot(_cross(first,second),edge)/_dot(edge,edge).sqrt(),_dot(first,second))


def exact_increment(start,end):
    with localcontext() as context:
        context.prec=160
        return float(decimal_angle(end)-decimal_angle(start))


def geometry(count=1):
    indices=np.arange(4*count).reshape(count,4)
    return ElasticDihedralBending(4*count,indices,np.zeros(count),np.ones(count),np.ones(count))


def points(angle):
    return np.array([[.25,1.,0.],[.75,-math.cos(angle),-math.sin(angle)],[0.,0.,0.],[1.,0.,0.]])


class DihedralIncrementTests(unittest.TestCase):
    def assertOracle(self,start,end,*,relative=4e-14,absolute=1e-30):
        g=geometry()
        angle,delta=dihedral_angle_increment(g,start,end)
        np.testing.assert_array_equal(angle,g.angles(start))
        self.assertEqual(angle.dtype,np.dtype(np.float64))
        self.assertEqual(delta.dtype,np.dtype(np.float64))
        expected=exact_increment(start,end)
        self.assertAlmostEqual(delta[0],expected,delta=max(absolute,abs(expected)*relative))
        return delta[0]

    def test_one_ulp_motion_retains_change_lost_by_endpoint_atan_subtraction(self):
        start=np.array([[.2,.8,.3],[.6,-.7,-.1],[0.,0.,0.],[1.,0.,0.]])
        end=start.copy();end[0,1]=np.nextafter(.8,math.inf)
        g=ElasticDihedralBending(4,[[0,1,2,3]],[.1],[1.],[3.])
        self.assertEqual(g.angles(start)[0],g.angles(end)[0])
        if np.finfo(np.longdouble).nmant == 52:
            self.assertEqual(g.energy_change(start,end),0.)
        delta=self.assertOracle(start,end)
        self.assertGreater(delta,0.)
        self.assertAlmostEqual(delta,-self.assertOracle(end,start),delta=1e-30)
        # Exercise the native-binary64 arithmetic route on hosts where the
        # normal longdouble type also provides extra precision.
        with patch('solver_dihedral_increment.np.longdouble',np.float64):
            self.assertOracle(start,end)

    def test_tiny_general_hinge_displacements_match_original_input_decimal(self):
        rng=np.random.default_rng(81263)
        for case in range(12):
            start=points(.6)+rng.normal(0,.03,(4,3))
            end=start+rng.normal(0,1e-12,(4,3))
            with self.subTest(case=case):
                self.assertOracle(start,end,relative=1e-12)

    def test_signed_zero_crossing_is_local_and_not_suppressed(self):
        for angle in (1e-12,1e-5,.02):
            for sign in (-1,1):
                start,end=points(-sign*angle),points(sign*angle)
                with self.subTest(angle=angle,sign=sign):
                    change=self.assertOracle(start,end)
                    self.assertEqual(math.copysign(1,change),sign)

    def test_large_updates_and_shrink_use_validated_endpoint_fallback(self):
        for start,end in ((points(-1.2),points(.8)),(points(.8)*1e20,points(-.4)*1e-20),
                          (points(.8),points(.8)*1e-12),(points(.8)*1e-12,points(.8)),
                          (points(.8)*1e-78,points(-.4)*1e77)):
            with self.subTest(startScale=float(np.max(np.abs(start))),endScale=float(np.max(np.abs(end)))):
                with warnings.catch_warnings():
                    warnings.simplefilter('error',RuntimeWarning)
                    g=geometry();_,actual=dihedral_angle_increment(g,start,end)
                self.assertEqual(actual[0],g.angles(end)[0]-g.angles(start)[0])
                self.assertAlmostEqual(actual[0],exact_increment(start,end),delta=1e-15)

    def test_uniform_scale_tiny_steps_retain_angular_change(self):
        for exponent in (-180,-80,0,80,180):
            start=np.ldexp(points(.7),exponent)
            end=np.ldexp(points(.7+1e-12),exponent)
            with self.subTest(exponent=exponent):
                self.assertOracle(start,end,relative=2e-13)

    def test_exact_uniform_scale_has_zero_increment(self):
        cases = [np.array([[-2.,-1.,.5],[1.75,-2.,1.75],[1.5,1.5,1.75],[-1.25,1.5,.25]]),
                 np.array([[0.,1.25,.75],[-1.5,1.,1.25],[2.,-1.,1.5],[.25,1.25,.75]])]
        for index,start in enumerate(cases):
            end=start*(33/32)
            with self.subTest(index=index):
                self.assertEqual(exact_increment(start,end),0.)
                self.assertEqual(self.assertOracle(start,end),0.)
                with patch('solver_dihedral_increment.np.longdouble',np.float64):
                    self.assertEqual(self.assertOracle(start,end),0.)

    def test_one_ulp_angular_signal_superimposed_on_scaling_keeps_both_signs(self):
        base=np.array([[-2.,-1.,.5],[1.75,-2.,1.75],[1.5,1.5,1.75],[-1.25,1.5,.25]])
        for exponent in (-120,0,120):
            for direction in (-math.inf,math.inf):
                start=np.ldexp(base,exponent);end=start*(33/32)
                end[0,2]=np.nextafter(end[0,2],direction)
                with self.subTest(exponent=exponent,direction=direction):
                    expected=exact_increment(start,end)
                    self.assertNotEqual(expected,0.)
                    self.assertOracle(start,end,relative=2e-14,absolute=1e-32)
                    with patch('solver_dihedral_increment.np.longdouble',np.float64):
                        self.assertOracle(start,end,relative=2e-14,absolute=1e-32)

    def test_small_angle_signal_on_nonsimilarity_change_matches_decimal(self):
        # Move the two opposite vertices within their own triangle planes.
        # This changes triangle shape, not merely a common scale. A subsequent
        # one-ulp out-of-plane change must retain its tiny signed angle.
        start=np.array([[.25,1.,.5],[.75,-1.,.5],[0.,0.,0.],[1.,0.,0.]])
        end=start.copy();end[0]*=33/32;end[1]*=31/32
        for direction in (-math.inf,math.inf):
            candidate=end.copy();candidate[0,2]=np.nextafter(candidate[0,2],direction)
            with self.subTest(direction=direction):
                self.assertOracle(start,candidate,relative=2e-14,absolute=1e-32)
                with patch('solver_dihedral_increment.np.longdouble',np.float64):
                    self.assertOracle(start,candidate,relative=2e-14,absolute=1e-32)

    def test_small_edge_motion_with_large_normal_change_uses_fallback(self):
        start=np.array([[.25,1e-9,0.],[.75,-1.,0.],[0.,0.,0.],[1.,0.,0.]])
        end=start.copy();end[0,1]=1e-10;end[0,2]=5e-10
        g=geometry();_,delta=dihedral_angle_increment(g,start,end)
        self.assertEqual(delta[0],g.angles(end)[0]-g.angles(start)[0])
        self.assertGreater(abs(delta[0]),1.)
        self.assertAlmostEqual(delta[0],exact_increment(start,end),delta=3e-16)

    def test_mixed_local_large_and_zero_crossing_batch_keeps_row_identity(self):
        pairs=[(points(.7),points(.7+1e-12)),(points(-1.),points(.8)),
               (points(-1e-12),points(1e-12)),(points(.7)*1e10,points(-.2)*1e-10)]
        start=np.vstack([a for a,b in pairs]);end=np.vstack([b for a,b in pairs])
        g=geometry(len(pairs));angles,deltas=dihedral_angle_increment(g,start,end)
        np.testing.assert_array_equal(angles,g.angles(start))
        for index,(a,b) in enumerate(pairs):
            expected=self.assertOracle(a,b,relative=2e-13,absolute=1e-15 if index in (1,3) else 1e-30)
            self.assertEqual(deltas[index],expected)

    def test_branch_crossing_and_degenerate_endpoints_fail_closed(self):
        g=geometry()
        for start,end in ((points(3.1),points(-3.1)),(points(-3.1),points(3.1)),
                          (points(0),points(math.pi)),(points(math.pi),points(0))):
            with self.subTest(start=start.tolist()),self.assertRaisesRegex(ValueError,'branch'):
                dihedral_angle_increment(g,start,end)
        collapsed=points(.4);collapsed[0]=collapsed[2]
        for start,end in ((collapsed,points(.4)),(points(.4),collapsed)):
            with self.assertRaisesRegex(ValueError,'Degenerate'):
                dihedral_angle_increment(g,start,end)

    def test_rigid_covariance_and_reversal_with_rounded_input_oracle(self):
        start=points(.43);end=points(.43000001)
        rotation=np.array([[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]])
        for translation in (np.zeros(3),np.array([.125,-.25,.0625])):
            first,last=start@rotation.T+translation,end@rotation.T+translation
            change=self.assertOracle(first,last,relative=3e-13)
            self.assertAlmostEqual(change,-self.assertOracle(last,first,relative=3e-13),delta=1e-22)
        pure=self.assertOracle(start,end,relative=3e-13)
        rotated=self.assertOracle(start@rotation.T,end@rotation.T,relative=3e-13)
        self.assertAlmostEqual(pure,rotated,delta=1e-22)

    def test_empty_subset_still_validates_complete_positions_and_fresh_arrays(self):
        g=ElasticDihedralBending(4,np.empty((0,4),dtype=int),[],[],[])
        first,last=dihedral_angle_increment(g,points(0),points(.2))
        self.assertEqual(first.shape,(0,));self.assertEqual(last.shape,(0,))
        self.assertIsNot(first,last)
        for bad in (np.zeros((3,3)),np.full((4,3),np.nan),np.full((4,3),np.inf)):
            for endpoints in ((bad,points(0)),(points(0),bad)):
                with self.assertRaises(ValueError):dihedral_angle_increment(g,*endpoints)
        with self.assertRaises(ValueError):dihedral_angle_increment(None,points(0),points(0))

    def test_identical_positions_are_exact_zero_without_mutating_inputs(self):
        start=points(.6);saved=start.copy();g=geometry()
        angle,delta=dihedral_angle_increment(g,start,start)
        np.testing.assert_array_equal(delta,[0.])
        angle[0]=1.;delta[0]=1.
        np.testing.assert_array_equal(start,saved)
        self.assertNotEqual(g.angles(start)[0],1.)


if __name__=='__main__':unittest.main()
