from fractions import Fraction as F
import unittest
from unittest.mock import patch

import ipctk
import numpy as np

from solver_ipc_contact import IpcSurfaceContact
from solver_swept_separation import certified_candidates
from solver_temporal_separation import certify_linear_path, temporal_support_bounds, _primitive_ids


def verify_leaf(start, end, leaf, minimum):
    normal = list(map(F, leaf['normal']))
    norm2 = sum(n * n for n in normal)
    values = []
    for time in (leaf['t0'], leaf['t1']):
        t = F(time)
        for i in leaf['first']:
            for j in leaf['second']:
                values.append(sum(normal[k] * ((1-t) * (F(float(start[i,k])) - F(float(start[j,k])))
                    + t * (F(float(end[i,k])) - F(float(end[j,k])))) for k in range(3)))
    gap = max(min(values), -max(values))
    assert gap > 0 and gap * gap > F(minimum) ** 2 * norm2
    assert F(leaf['lowerBoundM']) ** 2 * norm2 <= gap * gap


class TemporalSupportTests(unittest.TestCase):
    def test_exact_rational_support_at_arbitrary_time_endpoints(self):
        rng = np.random.default_rng(1931)
        checked = 0
        for count_a, count_b in ((1,1), (1,2), (2,2), (1,3)):
            a0 = rng.normal(size=(150,count_a,3)) * .001
            a1 = a0 + rng.normal(size=a0.shape) * .0001
            normal = rng.normal(size=(150,3))
            shift = normal / np.linalg.norm(normal, axis=1)[:,None] * .012
            b0 = rng.normal(size=(150,count_b,3)) * .001 + shift[:,None]
            b1 = b0 + rng.normal(size=b0.shape) * .0001
            offset = rng.uniform(-100,100,(150,1,3))
            a0, b0, a1, b1 = (x + offset for x in (a0,b0,a1,b1))
            for lo, hi in ((0.,1.), (.1,.73), (2**-25, .5)):
                safe, bounds = temporal_support_bounds(a0,b0,a1,b1,normal,lo,hi,.0001)
                for i in np.flatnonzero(safe):
                    start, end = np.concatenate((a0[i],b0[i])), np.concatenate((a1[i],b1[i]))
                    leaf = dict(first=list(range(count_a)), second=list(range(count_a,count_a+count_b)),
                                normal=normal[i].tolist(), t0=lo, t1=hi, lowerBoundM=float(bounds[i]))
                    verify_leaf(start,end,leaf,.0001)
                    checked += 1
        self.assertGreater(checked,1000)

    def test_boundary_and_crossing_are_not_certificates(self):
        a = np.zeros((1,1,3))
        for distance in (.0001, np.nextafter(.0001,0.)):
            b = np.array([[[0.,0.,distance]]])
            self.assertFalse(temporal_support_bounds(a,b,a,b,[[0,0,1]],0.,1.,.0001)[0][0])
        b = np.array([[[0.,0.,.001]]])
        self.assertFalse(temporal_support_bounds(a,b,a,-b,[[0,0,1]],0.,1.,.0001)[0][0])

    def test_bad_directions_fail_closed(self):
        a,b = np.zeros((1,1,3)),np.ones((1,1,3))
        for normal in ([0,0,0],[np.nan,0,0],[np.inf,0,0],[1e300,0,0],[1e-300,0,0]):
            self.assertFalse(temporal_support_bounds(a,b,a,b,[normal],0.,1.,.1)[0][0])

    def test_higher_precision_interval_cannot_collapse_to_an_endpoint(self):
        lo = np.nextafter(np.longdouble(1), np.longdouble(0))
        if lo == float(lo):
            self.skipTest("Platform has no extended-precision long double")
        a0 = np.array([[[-1e300, 0., 0.]]])
        a1 = np.array([[[1., 0., 0.]]])
        b = np.zeros((1, 1, 3))
        self.assertEqual(float(lo), 1.)
        with self.assertRaises(ValueError):
            temporal_support_bounds(a0, b, a1, b, [[1., 0., 0.]], lo, 1., .1)
        with self.assertRaises(ValueError):
            temporal_support_bounds(b, a1, b, a1, [[1., 0., 0.]], 0., 1., lo)

    def test_bad_inputs_reject(self):
        a,b = np.zeros((1,1,3)),np.ones((1,1,3))
        for minimum in (0.,-1.,True,np.nan,np.inf):
            with self.assertRaises(ValueError):
                temporal_support_bounds(a,b,a,b,[[0,0,1]],0.,1.,minimum)
        for lo,hi in ((True,1.),(0.,0.),(-1.,1.),(0.,np.nan)):
            with self.assertRaises(ValueError):
                temporal_support_bounds(a,b,a,b,[[0,0,1]],lo,hi,.1)
        with self.assertRaises(ValueError):
            temporal_support_bounds(a,b,a,b,[[0,1]],0.,1.,.1)


class TemporalPathTests(unittest.TestCase):
    def setUp(self):
        ipctk.set_num_threads(1)

    def fixture(self):
        start = np.array([[-1.,0.,0.],[1.,0.,0.],[0.,.1,0.]])
        end = np.array([[0.,-1.,0.],[0.,1.,0.],[-.1,0.,0.]])
        return ipctk.CollisionMesh(start,np.array([[0,1]])),start,end

    def verify(self,mesh,start,end,minimum,report):
        from solver_temporal_separation import _GROUPS, _primitive_ids

        def identity(group, first, second):
            first, second = tuple(sorted(first)), tuple(sorted(second))
            if group in ("vv_candidates", "ee_candidates") and first > second:
                first, second = second, first
            return group, first, second

        self.assertTrue(report['safe'],report)
        intervals = {}
        for leaf in report['certificateLeaves']:
            verify_leaf(start,end,leaf,minimum)
            key = identity(leaf['group'], leaf['first'], leaf['second'])
            intervals.setdefault(key,[]).append((leaf['t0'],leaf['t1']))
        candidates = ipctk.Candidates()
        candidates.build(mesh,start,end,inflation_radius=np.nextafter(minimum/2,np.inf),
                         broad_phase=ipctk.BruteForce())
        expected = {identity(group, *_primitive_ids(group, candidate, np.asarray(mesh.edges), np.asarray(mesh.faces)))
                    for group in _GROUPS for candidate in getattr(candidates, group)}
        self.assertEqual(set(intervals),expected)
        self.assertEqual(len(expected), len(candidates))
        self.assertEqual(len(expected), report['candidateCount'])
        for spans in intervals.values():
            cursor=0.
            for lo,hi in sorted(spans):
                self.assertEqual(lo,cursor)
                self.assertGreater(hi,lo)
                cursor=hi
            self.assertEqual(cursor,1.)

    def test_rotation_requires_subdivision_and_complete_exact_cover(self):
        mesh,a,b=self.fixture()
        report=certify_linear_path(mesh,a,b,.06,keep_leaves=True)
        self.verify(mesh,a,b,.06,report)
        self.assertGreater(report['deepest'],0)
        self.assertGreater(report['leafCount'],report['candidateCount'])
        self.assertFalse(certify_linear_path(mesh,a,b,.06,max_depth=0)['safe'])

    def test_node_budget_rejects_partial_certificate(self):
        mesh,a,b=self.fixture()
        report=certify_linear_path(mesh,a,b,.06,max_nodes=1,keep_leaves=True)
        self.assertFalse(report['safe'])
        self.assertEqual(report['reason'],'node-budget-exhausted')
        self.assertLessEqual(report['nodeCount'],1)

    def test_invalid_candidate_thickness_assignments_reject(self):
        mesh, start, end = self.fixture()
        for minimum in (0., -1., True, np.nan, np.inf, .061, "small", None):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                certify_linear_path(mesh, start, end, .06,
                    candidate_minimum_distance=lambda name, candidate: minimum)
        with self.assertRaises(ValueError):
            certify_linear_path(mesh, start, end, .06, candidate_minimum_distance=.01)

    def test_assigned_thickness_preserves_exact_subdivision_proof(self):
        mesh, start, end = self.fixture()
        report = certify_linear_path(mesh, start, end, .08, keep_leaves=True,
            candidate_minimum_distance=lambda name, candidate: .06)
        self.assertTrue(report["safe"], report)
        self.assertGreater(report["deepest"], 0)
        for leaf in report["certificateLeaves"]:
            self.assertEqual(leaf["minimumDistanceM"], .06)
            verify_leaf(start, end, leaf, .06)
        self.assertFalse(certify_linear_path(mesh, start, end, .08)["safe"])

    def test_interior_clearance_violation_despite_safe_endpoints(self):
        mesh,a,b=self.fixture()
        self.assertFalse(certify_linear_path(mesh,a,b,.08)['safe'])
        self.assertFalse(certify_linear_path(mesh,a,b,.1)['safe'])
        crossing=a.copy();crossing[2,1]=-.1
        self.assertFalse(certify_linear_path(mesh,a,crossing,.001)['safe'])

    def test_common_motion_separates_even_when_static_hulls_overlap(self):
        a=np.array([[0.,0.,0.],[0.,0.,.01],[.00008,.00008,0.],[.00008,.00008,.01]])
        b=a+[.02,.02,.03]
        mesh=ipctk.CollisionMesh(a,np.array([[0,1],[2,3]]))
        self.assertGreater(certified_candidates(mesh,a,b,.0001)[1]['remainingCount'],0)
        self.verify(mesh,a,b,.0001,certify_linear_path(mesh,a,b,.0001,keep_leaves=True))

    def test_rigid_transform_controls(self):
        _,a,b=self.fixture()
        rotation=np.linalg.qr(np.random.default_rng(114).normal(size=(3,3)))[0]
        for r in (np.eye(3),rotation):
            for offset in (np.zeros(3),np.array([100.,-192.,213.])):
                aa,bb=a@r+offset,b@r+offset
                mesh=ipctk.CollisionMesh(aa,np.array([[0,1]]))
                self.verify(mesh,aa,bb,.06,certify_linear_path(mesh,aa,bb,.06,keep_leaves=True))

    def test_invalid_shapes_and_budgets_reject_before_native_code(self):
        mesh,a,b=self.fixture()
        for bad in (a[:2],a.ravel(),np.vstack((a,a[:1])),a*np.nan):
            for function in (certify_linear_path,certified_candidates):
                with self.assertRaises(ValueError):
                    function(mesh,bad,bad,.01)
        for options in ({'max_depth':True},{'max_depth':31},{'max_nodes':0},{'keep_leaves':1}):
            with self.assertRaises(ValueError):
                certify_linear_path(mesh,a,b,.01,**options)


class TemporalContactTests(unittest.TestCase):
    def setUp(self):
        ipctk.set_num_threads(1)
        n=np.array([1.,2.,3.])/np.sqrt(14.)
        t=np.cross(n,[1.,0.,0.]);t/=np.linalg.norm(t)
        triangle=np.array([np.zeros(3),.01*t,.01*np.cross(n,t)])
        self.a=np.concatenate((triangle,triangle+.00011*n))
        self.faces=np.array([[0,1,2],[3,4,5]])
        self.contact=self.make('temporal-separation-tight-inclusion')

    def make(self,profile):
        return IpcSurfaceContact(self.a,self.faces,activation_distance_m=.00002,
            minimum_distance_m=.0001,stiffness=10000.,energy_profile='area-improved-max',ccd_profile=profile)

    def test_predicate_is_not_step_proposal(self):
        b=self.a+[.001,.001,.001]
        with patch.object(self.contact,'step_limit',side_effect=AssertionError('Not a predicate')):
            self.assertTrue(self.contact.path_safe(self.a,b))
        report=certify_linear_path(self.contact.mesh,self.a,b,.0001,keep_leaves=True)
        TemporalPathTests().verify(self.contact.mesh,self.a,b,.0001,report)
        self.assertTrue(any(leaf['group']=='fv_candidates' for leaf in report['certificateLeaves']))

    def test_unchanged_energy_force_metric_and_step_proposal(self):
        other=self.make('swept-plane-tight-inclusion')
        self.assertGreater(self.contact.energy(self.a),0.)
        self.assertEqual(self.contact.energy(self.a),other.energy(self.a))
        np.testing.assert_array_equal(self.contact.gradient(self.a),other.gradient(self.a))
        self.assertEqual((self.contact.hessian(self.a)-other.hessian(self.a)).nnz,0)
        self.assertEqual(self.contact.step_limit(self.a,self.a+[.00001,0.,0.]),
                         other.step_limit(self.a,self.a+[.00001,0.,0.]))
        self.assertEqual(self.contact.profile()['ccd']['conservativeRescaling'],.8)

    def test_unresolved_never_falls_back_to_samples_or_step_bound(self):
        with patch('solver_temporal_separation.certify_linear_path',return_value={'safe':False}):
            with patch.object(self.contact,'step_limit',return_value=1.):
                self.assertFalse(self.contact.path_safe(self.a,self.a))


if __name__ == '__main__':
    unittest.main()
