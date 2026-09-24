import unittest
from dataclasses import replace
from fractions import Fraction as F

import ipctk
import numpy as np

from solver_contact_work import WorkPolicy, contact_work
from solver_contact_work_native import capture_endpoint, bounded_native_work
from solver_rest_filtered_contact import RestFilteredSurfaceContact
from test_solver_contact_range_adversarial import square_grid


def fixture(gap=.0019, *, rotated=True):
    panel=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],[.01,.01,0.]])
    face=np.array([[0,1,2],[1,3,2]],dtype=int)
    rest=np.vstack((panel,panel))
    faces=np.vstack((face,face+4))
    contact=RestFilteredSurfaceContact(rest,faces,activation_distance_m=.002,
        minimum_distance_m=.0001,stiffness=10000.)
    q=rest.copy()
    if rotated:
        angle=.037
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        q[4:,:2]=q[4:,:2]@rotation.T
    q[4:]+=[.000321,.000271,gap]
    return contact,q


class NativeCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):ipctk.set_num_threads(1)

    def test_complete_native_inventory_and_owned_immutable_coordinates(self):
        contact,q=fixture()
        endpoint=capture_endpoint(contact,q)
        self.assertEqual(len(endpoint.terms),sum(len(bucket) for _,bucket in contact._buckets(q)[1]))
        self.assertGreater(len(endpoint.terms),0)
        self.assertEqual(len(endpoint.terms),len(endpoint.observations))
        self.assertEqual(endpoint.native_energy,contact.energy(q))
        self.assertTrue(all(o.exact_distance_squared>0 for o in endpoint.observations))
        first=endpoint.terms[0].positions
        q[:]=0
        self.assertEqual(first,endpoint.terms[0].positions)
        self.assertFalse(hasattr(endpoint,'__dict__'))

    def test_native_energy_agreement_and_tiny_difference(self):
        contact,q=fixture()
        first=capture_endpoint(contact,q)
        scalar=contact_work((),first.terms)
        self.assertAlmostEqual(scalar.value,first.native_energy,delta=abs(first.native_energy)*1e-11)
        end=q.copy();end[4:,2]+=1e-14
        result,_,_=bounded_native_work(contact,q,end)
        self.assertLess(result.value,0)
        self.assertLess(result.absolute_error,F(1,10**25))

    def test_missing_endpoint_terms_activation_entry_exit_and_reversal(self):
        contact,q=fixture()
        outside=q.copy();outside[4:,2]=.003
        forward,a,b=bounded_native_work(contact,outside,q)
        backward,_,_=bounded_native_work(contact,q,outside)
        self.assertEqual(a.terms,())
        self.assertGreater(len(b.terms),0)
        self.assertGreater(forward.value,0)
        self.assertLessEqual(abs(F(forward.value)+F(backward.value)),forward.absolute_error+backward.absolute_error)

    def test_native_gradient_consistency_at_three_displacements(self):
        contact,q=fixture()
        direction=np.zeros_like(q);direction[4:,2]=1
        expected=float(np.sum(contact.gradient(q)*direction))
        for epsilon in (1e-7,1e-9,1e-12):
            a,b=q-epsilon*direction,q+epsilon*direction
            result,_,_=bounded_native_work(contact,a,b)
            # Actual binary64 displacement is used, not a presumed 2*epsilon.
            displacement=b[4,2]-a[4,2]
            observed=result.value/displacement
            self.assertAlmostEqual(observed,expected,delta=abs(expected)*2e-5)

    def test_multiple_filtered_parameter_buckets(self):
        rest,faces=square_grid(2)
        contact=RestFilteredSurfaceContact(rest,faces,activation_distance_m=.01,
            minimum_distance_m=.0001,stiffness=10000.)
        q=rest*.4
        q+=np.random.default_rng(83).normal(size=q.shape)*1e-6
        endpoint=capture_endpoint(contact,q)
        self.assertGreater(len({t.bucket for t in endpoint.terms}),1)
        result=contact_work((),endpoint.terms)
        self.assertAlmostEqual(result.value,endpoint.native_energy,delta=endpoint.native_energy*1e-11)

    def test_admission_and_budget_failures(self):
        contact,q=fixture()
        with self.assertRaises(ValueError):capture_endpoint(object(),q)
        for raw in (q.tolist(),q.astype(int),q.astype(bool),q.astype(np.longdouble)):
            with self.assertRaisesRegex(ValueError,'Raw native binary64'):capture_endpoint(contact,raw)
        with self.assertRaisesRegex(ValueError,'budget'):
            capture_endpoint(contact,q,policy=WorkPolicy(max_endpoint_terms=1))
        near=q.copy();near[4:]=near[:4]+[0.,0.,.0001]
        with self.assertRaises(ValueError):capture_endpoint(contact,near)
        contact._potentials[0].dhat*=2
        with self.assertRaisesRegex(ValueError,'Native contact parameters'):
            capture_endpoint(contact,q)

    def test_native_profile_and_weight_mutations_reject(self):
        contact,q=fixture()
        _,buckets=contact._buckets(q)
        bucket=buckets[0][1]
        bucket.use_area_weighting=False
        with self.assertRaisesRegex(ValueError,'collision set'):capture_endpoint(contact,q)
        bucket.use_area_weighting=True
        bucket.collision_set_type=ipctk.NormalCollisions.IMPROVED_MAX_APPROX
        with self.assertRaisesRegex(ValueError,'collision set'):capture_endpoint(contact,q)
        bucket.collision_set_type=ipctk.NormalCollisions.IPC
        bucket[0].weight=-1.
        with self.assertRaisesRegex(ValueError,'weight/minimum'):capture_endpoint(contact,q)

    def test_existing_energy_change_and_profile_remain_native(self):
        contact,q=fixture()
        end=q.copy();end[4:,2]+=1e-14
        before=contact.profile()
        bounded_native_work(contact,q,end)
        self.assertEqual(contact.profile(),before)
        self.assertEqual(contact.energy_change(q,end),contact.energy(end)-contact.energy(q))

    def test_nearly_parallel_native_feature_mismatch_is_explicitly_unsupported(self):
        # Native roundoff selects a different minimum on one almost-parallel
        # diagonal. Its tiny nonzero mollifier is retained; never drop it or
        # silently replace the native feature to make capture pass.
        contact,q=fixture(rotated=False)
        self.assertGreater(contact.energy(q),0)
        with self.assertRaisesRegex(ValueError,'Native closest feature'):
            capture_endpoint(contact,q)


if __name__=='__main__':unittest.main()
