"""Independent generic cell-mechanics tests; no garment or trajectory authority."""

import copy
from decimal import Decimal, localcontext
from fractions import Fraction as F
import json
import math
import unittest

import numpy as np

from solver_continuous_normal_sewing import ContinuousNormalOffsetSewing


def rat(value):
    value = F(value)
    return {"numerator":str(value.numerator),"denominator":str(value.denominator)}


def rational(value):
    if set(value) != {"numerator","denominator"}:
        raise AssertionError("Exact conversion witnesses require only canonical rational strings")
    result = F(int(value["numerator"]),int(value["denominator"]))
    if rat(result) != value:
        raise AssertionError("Unreduced or noncanonical rational witness")
    return result


def anchor(values):
    return [{"vertex":v,"weight":rat(w)} for v,w in sorted(values.items()) if w]


def cell(identifier="one", **changes):
    result = {"id":identifier,
        "positiveStart":anchor({3:F(3,4),4:F(1,4)}),
        "positiveEnd":anchor({4:F(1,2),5:F(1,2)}),
        "negativeStart":anchor({0:F(1,4),1:F(1,4),2:F(1,2)}),
        "negativeEnd":anchor({0:F(1,2),1:F(3,8),2:F(1,8)}),
        "frameVertices":[0,1,2],"side":1,"targetsMeters":[.125,.375],
        "referenceLengthMeters":rat(F(3,8)),"stiffnessDensityNPerM2":16.,"activation":.75}
    result.update(changes)
    return result


def mixed_anchor(first,last,fraction):
    result = {}
    for scale,raw in ((1-fraction,first),(fraction,last)):
        for term in raw:
            v = term["vertex"]
            result[v] = result.get(v,F()) + scale*rational(term["weight"])
    return anchor(result)


def numeric_anchor(raw,q):
    # Explicitly convert each coefficient once; never normalize its rounded sum.
    result = np.zeros(3)
    for term in raw:
        result += float(rational(term["weight"]))*q[term["vertex"]]
    return result


def direct_residual(cells,q):
    q = np.asarray(q,dtype=float)
    result = []
    for c in cells:
        if not c["activation"]:
            result.extend([0.]*6)
            continue
        a,b,d = q[c["frameVertices"]]
        area = np.cross(b-a,d-a)
        normal = area/math.hypot(*area)
        endpoint = [numeric_anchor(c["positive"+end],q)-numeric_anchor(c["negative"+end],q)
            - c["side"]*target*normal for end,target in zip(("Start","End"),c["targetsMeters"])]
        gamma = F(c["activation"])*F(c["stiffnessDensityNPerM2"])*rational(c["referenceLengthMeters"])
        result.extend(math.sqrt(float(gamma))*(endpoint[0]+endpoint[1])/2)
        result.extend(math.sqrt(float(gamma/12))*(endpoint[1]-endpoint[0]))
    return np.asarray(result)


def decimal_integral_value(cells,q):
    """Integrate R(u)^2 coefficients at 100 digits, independent of residual factors."""
    with localcontext() as context:
        context.prec = 100
        positions = [[Decimal(float(x)) for x in point] for point in q]
        total = Decimal(0)
        for c in cells:
            if not c["activation"]:
                continue
            a,b,d = [positions[v] for v in c["frameVertices"]]
            u,v = [[last-first for first,last in zip(a,p)] for p in (b,d)]
            cross = [u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
            length = sum((x*x for x in cross),Decimal(0)).sqrt()
            normal = [x/length for x in cross]
            residual = []
            for ending,target in zip(("Start","End"),c["targetsMeters"]):
                r = [-Decimal(c["side"])*Decimal(target)*n for n in normal]
                for sign in ("positive","negative"):
                    for term in c[sign+ending]:
                        # The numerical law uses RN(exact coefficient), whose
                        # error from source RAT is checked separately below.
                        weight = Decimal(float(rational(term["weight"])))
                        for axis in range(3):r[axis] += (1 if sign=="positive" else -1)*weight*positions[term["vertex"]][axis]
                residual.append(r)
            raw_measure = rational(c["referenceLengthMeters"])
            measure = Decimal(raw_measure.numerator)/Decimal(raw_measure.denominator)
            gamma = Decimal(c["activation"])*Decimal(c["stiffnessDensityNPerM2"])*measure
            for first,last in zip(*residual):
                change = last-first
                total += gamma*(first*first+first*change+change*change/Decimal(3))/Decimal(2)
        return total


def decimal_integral(cells,q):
    return float(decimal_integral_value(cells,q))


def decimal_constant_point_reference(q, beta=Decimal(10)**14):
    """160-digit geometric-normal reference for a constant pos3-minus-neg0 cell.

    Differentiate the original area cross product directly in Decimal. This
    does not sample a binary64 normal or call any producer geometry routine.
    Decimal-valued perturbations are retained for independent derivative checks.
    """
    with localcontext() as context:
        context.prec=160
        points=[[x if isinstance(x,Decimal) else Decimal(float(x)) for x in p] for p in q]
        zero=Decimal(0)
        def cross(a,b):
            return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
        def dot(a,b):return sum((x*y for x,y in zip(a,b)),zero)
        u=[points[1][j]-points[0][j] for j in range(3)]
        v=[points[2][j]-points[0][j] for j in range(3)]
        area=cross(u,v);length=dot(area,area).sqrt()
        normal=[a/length for a in area]
        residual=[points[3][j]-points[0][j]-normal[j] for j in range(3)]
        gradient=[]
        for vertex in range(4):
            for axis in range(3):
                du=[Decimal((int(vertex==1)-int(vertex==0))*int(j==axis)) for j in range(3)]
                dv=[Decimal((int(vertex==2)-int(vertex==0))*int(j==axis)) for j in range(3)]
                da=[a+b for a,b in zip(cross(du,v),cross(u,dv))]
                projection=dot(normal,da)
                dn=[(a-n*projection)/length for a,n in zip(da,normal)]
                direct=(int(vertex==3)-int(vertex==0))*residual[axis]
                gradient.append(beta*(direct-dot(residual,dn)))
        return residual,beta*dot(residual,residual)/2,gradient


class ContinuousNormalSewingTests(unittest.TestCase):
    def setUp(self):
        self.q = np.array([[0.,0.,0.],[2.,0.,.25],[0.,1.5,-.125],
            [.375,.25,.5],[.625,.75,.125],[1.125,.375,.75]])
        self.cells = [cell()]
        self.potential = ContinuousNormalOffsetSewing(len(self.q),self.cells)

    def test_affine_cell_integral_matches_independent_high_precision_polynomial(self):
        expected = decimal_integral(self.cells,self.q)
        self.assertAlmostEqual(self.potential.energy(self.q)/expected,1.,delta=3e-15)
        expected_residual = direct_residual(self.cells,self.q)
        np.testing.assert_allclose(self.potential.residual(self.q),expected_residual,rtol=3e-15,atol=3e-16)
        residual = self.potential.residual(self.q)
        self.assertAlmostEqual(self.potential.energy(self.q),.5*float(residual@residual),delta=3e-15*expected)
        # Endpoint-only or independent endpoint point penalties omit the cross
        # term. The chosen residuals make that error visible.
        mean,slope = expected_residual[:3],expected_residual[3:]
        self.assertGreater(np.linalg.norm(slope),.1)
        self.assertGreater(np.linalg.norm(mean),.1)

    def test_jacobian_gradient_and_full_curvature_against_independent_differences(self):
        epsilon = 2e-6
        jacobian = self.potential.jacobian(self.q).toarray()
        gradient = self.potential.gradient(self.q)
        exact = self.potential.exact_hessian(self.q).toarray()
        self.assertEqual(jacobian.shape,(6,18));self.assertEqual(gradient.shape,(18,))
        for index,direction in enumerate(np.eye(18).reshape(18,6,3)):
            plus,minus = self.q+epsilon*direction,self.q-epsilon*direction
            independent_j = (direct_residual(self.cells,plus)-direct_residual(self.cells,minus))/(2*epsilon)
            np.testing.assert_allclose(jacobian[:,index],independent_j,rtol=3e-7,atol=2e-9)
            difference = (decimal_integral(self.cells,plus)-decimal_integral(self.cells,minus))/(2*epsilon)
            self.assertAlmostEqual(gradient[index],difference,delta=2e-9)
            difference_gradient = (self.potential.gradient(plus)-self.potential.gradient(minus))/(2*epsilon)
            np.testing.assert_allclose(exact[:,index],difference_gradient,rtol=3e-6,atol=2e-8)
        np.testing.assert_allclose(exact,exact.T,rtol=0,atol=1e-13)
        np.testing.assert_allclose(gradient,jacobian.T@self.potential.residual(self.q),rtol=2e-14,atol=1e-14)
        gn = self.potential.hessian(self.q).toarray()
        np.testing.assert_allclose(gn,jacobian.T@jacobian,rtol=2e-14,atol=2e-14)
        np.testing.assert_allclose(self.potential.hessian(self.q,project_psd=True).toarray(),gn,rtol=0,atol=1e-13)
        self.assertGreater(np.max(np.abs(exact-gn)),.01)
        self.assertGreater(np.linalg.eigvalsh(gn).min(),-1e-12)

    def test_proper_rotation_force_torque_and_required_frame_reactions(self):
        axis = np.array([1.,2.,3.]);axis /= np.linalg.norm(axis)
        angle = .73;c,s = math.cos(angle),math.sin(angle)
        skew = np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
        rotation = c*np.eye(3)+(1-c)*np.outer(axis,axis)+s*skew
        self.assertAlmostEqual(np.linalg.det(rotation),1.,delta=1e-15)
        moved = self.q@rotation.T+[.5,-.75,.125]
        gradient = self.potential.gradient(self.q).reshape(-1,3)
        np.testing.assert_allclose(self.potential.gradient(moved).reshape(-1,3),gradient@rotation.T,rtol=2e-13,atol=1e-14)
        np.testing.assert_allclose(self.potential.residual(moved).reshape(-1,3),self.potential.residual(self.q).reshape(-1,3)@rotation.T,rtol=2e-13,atol=1e-14)
        np.testing.assert_allclose(gradient.sum(axis=0),0.,atol=1e-14)
        np.testing.assert_allclose(np.cross(self.q,gradient).sum(axis=0),0.,atol=1e-14)
        transform = np.kron(np.eye(6),rotation)
        np.testing.assert_allclose(self.potential.exact_hessian(moved).toarray(),
            transform@self.potential.exact_hessian(self.q).toarray()@transform.T,rtol=2e-12,atol=1e-13)
        # An independent frozen-director force has an uncancelled torque. The
        # full frame derivatives must supply the missing response.
        c = self.cells[0];gamma = float(F(c['activation'])*F(c['stiffnessDensityNPerM2'])*rational(c['referenceLengthMeters']))
        a,b,d = self.q[c['frameVertices']];normal = np.cross(b-a,d-a);normal /= np.linalg.norm(normal)
        endpoint = [numeric_anchor(c['positive'+e],self.q)-numeric_anchor(c['negative'+e],self.q)-t*normal
                    for e,t in zip(('Start','End'),c['targetsMeters'])]
        frozen = np.zeros_like(self.q)
        for ending,r in zip(('Start','End'),((2*endpoint[0]+endpoint[1])/6,(endpoint[0]+2*endpoint[1])/6)):
            for sign,name in ((1,'positive'),(-1,'negative')):
                for term in c[name+ending]:frozen[term['vertex']] += sign*float(rational(term['weight']))*gamma*r
        self.assertGreater(np.linalg.norm(np.cross(self.q,frozen).sum(axis=0)),.01)
        self.assertGreater(np.max(np.abs(gradient-frozen)),.01)
        np.testing.assert_allclose((gradient-frozen)[3:],0.,atol=1e-14)

    def test_endpoint_reversal_and_exact_measure_preserving_subdivision(self):
        original = self.cells[0]
        reversed_cell = copy.deepcopy(original);reversed_cell['id']='reversed'
        for prefix in ('positive','negative'):
            reversed_cell[prefix+'Start'],reversed_cell[prefix+'End'] = original[prefix+'End'],original[prefix+'Start']
        reversed_cell['targetsMeters'] = original['targetsMeters'][::-1]
        reverse = ContinuousNormalOffsetSewing(6,[reversed_cell])
        np.testing.assert_allclose(reverse.residual(self.q)[:3],self.potential.residual(self.q)[:3],atol=1e-14)
        np.testing.assert_allclose(reverse.residual(self.q)[3:],-self.potential.residual(self.q)[3:],atol=1e-14)
        fraction = F(3,8)
        first,last = copy.deepcopy(original),copy.deepcopy(original)
        first['id'],last['id']='first','last'
        for prefix in ('positive','negative'):
            middle = mixed_anchor(original[prefix+'Start'],original[prefix+'End'],fraction)
            first[prefix+'End'],last[prefix+'Start'] = middle,copy.deepcopy(middle)
        target = float((1-fraction)*F(original['targetsMeters'][0])+fraction*F(original['targetsMeters'][1]))
        first['targetsMeters'][1],last['targetsMeters'][0]=target,target
        length = rational(original['referenceLengthMeters'])
        first['referenceLengthMeters'],last['referenceLengthMeters']=rat(length*fraction),rat(length*(1-fraction))
        subdivided = ContinuousNormalOffsetSewing(6,[first,last])
        opposite = copy.deepcopy(original)
        opposite['frameVertices']=[0,2,1];opposite['side']=-1
        oriented = ContinuousNormalOffsetSewing(6,[opposite])
        for candidate in (reverse,subdivided,oriented):
            self.assertAlmostEqual(candidate.energy(self.q),self.potential.energy(self.q),delta=2e-15)
            np.testing.assert_allclose(candidate.gradient(self.q),self.potential.gradient(self.q),rtol=2e-14,atol=2e-14)
            np.testing.assert_allclose(candidate.hessian(self.q).toarray(),self.potential.hessian(self.q).toarray(),rtol=3e-14,atol=3e-14)
            np.testing.assert_allclose(candidate.exact_hessian(self.q).toarray(),self.potential.exact_hessian(self.q).toarray(),rtol=3e-13,atol=3e-14)
        # This intentionally preserves the cell measure. Reusing the complete
        # length for both children doubles the material penalty instead.
        first['referenceLengthMeters']=rat(length);last['referenceLengthMeters']=rat(length)
        wrong = ContinuousNormalOffsetSewing(6,[first,last])
        self.assertGreater(abs(wrong.energy(self.q)-self.potential.energy(self.q)),.01)

    def test_shared_and_disjoint_frames_accumulate_without_cross_cell_curvature(self):
        second = cell('two',side=-1,targetsMeters=[.25,.5],activation=.5)
        combined = ContinuousNormalOffsetSewing(6,[self.cells[0],second])
        single = ContinuousNormalOffsetSewing(6,[second])
        self.assertAlmostEqual(combined.energy(self.q),self.potential.energy(self.q)+single.energy(self.q),delta=2e-15)
        np.testing.assert_allclose(combined.gradient(self.q),self.potential.gradient(self.q)+single.gradient(self.q),atol=2e-14)
        np.testing.assert_allclose(combined.exact_hessian(self.q).toarray(),self.potential.exact_hessian(self.q).toarray()+single.exact_hessian(self.q).toarray(),rtol=2e-13,atol=3e-14)
        duplicate = copy.deepcopy(self.cells[0]);duplicate['id']='duplicate-physical-cell'
        doubled = ContinuousNormalOffsetSewing(6,[self.cells[0],duplicate])
        self.assertAlmostEqual(doubled.energy(self.q),2*self.potential.energy(self.q),delta=2e-15)
        disjoint = copy.deepcopy(second)
        disjoint['frameVertices']=[v+6 for v in disjoint['frameVertices']]
        for name in ('positiveStart','positiveEnd','negativeStart','negativeEnd'):
            for term in disjoint[name]:term['vertex']+=6
        separate = ContinuousNormalOffsetSewing(12,[self.cells[0],disjoint])
        q = np.vstack((self.q,self.q+[3.,-2.,1.]))
        matrix = separate.exact_hessian(q).toarray()
        np.testing.assert_array_equal(matrix[:18,18:],0.)
        np.testing.assert_array_equal(matrix[18:,:18],0.)
        np.testing.assert_allclose(matrix[:18,:18],self.potential.exact_hessian(self.q).toarray(),atol=2e-14)
        np.testing.assert_allclose(matrix[18:,18:],single.exact_hessian(self.q).toarray(),atol=2e-14)

    def test_interior_bump_invisible_to_two_samples_has_positive_integrated_response(self):
        q = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,.25],[.5,0.,.75],[1.,0.,.25]])
        common = {'targetsMeters':[.25,.25],'referenceLengthMeters':rat(1),'stiffnessDensityNPerM2':3.,'activation':1.}
        first = cell('left',positiveStart=anchor({3:1}),positiveEnd=anchor({4:1}),
            negativeStart=anchor({0:1}),negativeEnd=anchor({0:F(1,2),1:F(1,2)}),**common)
        last = cell('right',positiveStart=anchor({4:1}),positiveEnd=anchor({5:1}),
            negativeStart=anchor({0:F(1,2),1:F(1,2)}),negativeEnd=anchor({1:1}),**common)
        continuous = ContinuousNormalOffsetSewing(6,[first,last])
        outer = cell('coarse-only',positiveStart=anchor({3:1}),positiveEnd=anchor({5:1}),
            negativeStart=anchor({0:1}),negativeEnd=anchor({1:1}),**(common|{'referenceLengthMeters':rat(2)}))
        sparse_model = ContinuousNormalOffsetSewing(6,[outer])
        self.assertEqual(sparse_model.energy(q),0.)
        self.assertAlmostEqual(continuous.energy(q),.25,delta=1e-15)
        self.assertAlmostEqual(continuous.gradient(q).reshape(-1,3)[4,2],1.,delta=2e-15)
        flat = q.copy();flat[4,2]=.25
        self.assertEqual(continuous.energy(flat),0.)
        np.testing.assert_allclose(continuous.exact_hessian(flat).toarray(),continuous.hessian(flat).toarray(),atol=2e-14)

    def test_adjacent_one_sided_directors_can_make_a_continuous_offset_incompatible(self):
        # Two triangles meet along edge 0--1 with orthogonal normals +z,+y.
        # Their shared negative material point 0 has one shared positive point
        # 3. Positive distance d cannot make that single point equal d*n on both.
        d = .25
        q = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,d/2,d/2],
            [0.,1.,d],[0.,d,1.],[0.,0.,1.]])
        settings = {'targetsMeters':[d,d],'referenceLengthMeters':rat(1),'stiffnessDensityNPerM2':1.,'activation':1.}
        left = cell('z-director',positiveStart=anchor({4:1}),positiveEnd=anchor({3:1}),
            negativeStart=anchor({2:1}),negativeEnd=anchor({0:1}),frameVertices=[0,1,2],**settings)
        right = cell('y-director',positiveStart=anchor({3:1}),positiveEnd=anchor({5:1}),
            negativeStart=anchor({0:1}),negativeEnd=anchor({6:1}),frameVertices=[1,0,6],**settings)
        potential = ContinuousNormalOffsetSewing(7,[left,right])
        expected = d*d/6
        self.assertAlmostEqual(potential.energy(q),expected,delta=1e-17)
        # The optimum of the shared point still has strictly positive energy.
        # This is a limitation witness, not a garment equilibrium/force solve.
        for y,z in ((0.,d),(d,0.),(0.,0.),(d,d)):
            moved=q.copy();moved[3]=[0.,y,z]
            self.assertGreater(potential.energy(moved),potential.energy(q))
        np.testing.assert_allclose(potential.gradient(q).reshape(-1,3)[3],0.,atol=1e-16)
        self.assertGreater(potential.energy(q),0.)
        self.assertAlmostEqual(potential.energy(q),decimal_integral([left,right],q),delta=1e-17)

    def test_stable_directional_work_near_equilibrium_and_large_energy_cancellation(self):
        definition = cell(positiveStart=anchor({3:1}),positiveEnd=anchor({3:1}),
            negativeStart=anchor({0:1}),negativeEnd=anchor({0:1}),targetsMeters=[1.,1.],
            referenceLengthMeters=rat(1),stiffnessDensityNPerM2=4.,activation=1.)
        potential = ContinuousNormalOffsetSewing(4,[definition])
        gamma_effective = F(math.sqrt(4.))**2
        for base in (0.,2**-30,1024.):
            for sign in (-1,1):
                q0 = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[base,0.,1.]])
                q1 = q0.copy()
                q1[3,0] = sign*2**-60 if base==0 else math.nextafter(base,math.inf if sign>0 else -math.inf)
                delta = F(float(q1[3,0]))-F(float(q0[3,0]))
                expected = float(gamma_effective*(F(base)+delta/2)*delta)
                actual = potential.energy_change(q0,q1)
                self.assertAlmostEqual(actual/expected,1.,delta=3e-14)
                self.assertAlmostEqual(potential.energy_change(q1,q0)/(-expected),1.,delta=3e-14)
        self.assertEqual(potential.energy_change(self.q[:4],self.q[:4]),0.)

    def test_one_ulp_frame_tilt_work_preserves_high_precision_direction_and_magnitude(self):
        definition = cell(positiveStart=anchor({3:1}),positiveEnd=anchor({3:1}),
            negativeStart=anchor({0:1}),negativeEnd=anchor({0:1}),targetsMeters=[1.,1.],
            referenceLengthMeters=rat(1),stiffnessDensityNPerM2=1.,activation=1.)
        potential = ContinuousNormalOffsetSewing(4,[definition])
        for tilt in (.1,.3,.7,1.,1.5,3.,10.):
            q0=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,tilt],[.2,.4,.6]])
            for direction in (-math.inf,math.inf):
                q1=q0.copy();q1[2,2]=math.nextafter(tilt,direction)
                with localcontext() as context:
                    context.prec=100
                    expected=float(decimal_integral_value([definition],q1)-decimal_integral_value([definition],q0))
                actual=potential.energy_change(q0,q1)
                with self.subTest(tilt=tilt,direction=direction):
                    self.assertNotEqual(expected,0.)
                    self.assertAlmostEqual(actual/expected,1.,delta=3e-13)
                    self.assertAlmostEqual(potential.energy_change(q1,q0)/(-expected),1.,delta=3e-13)

    def test_large_positive_frame_scaling_has_zero_director_work_and_retains_anchor_work(self):
        definition=cell(positiveStart=anchor({3:1}),positiveEnd=anchor({3:1}),
            negativeStart=anchor({0:1}),negativeEnd=anchor({0:1}),targetsMeters=[1.,1.],
            referenceLengthMeters=rat(1),stiffnessDensityNPerM2=1.,activation=1.)
        potential=ContinuousNormalOffsetSewing(4,[definition])
        start=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,.3],[.2,.4,.6]])
        for exponent in (-36,-20,16,18):
            scale=math.ldexp(1.,exponent)
            end=start.copy();end[1:3]*=scale
            # Exact binary powers preserve the original-input positive area
            # direction; the anchor and target stay fixed. No path is certified.
            with self.subTest(exponent=exponent):
                self.assertEqual(potential.energy_change(start,end),0.)
                self.assertEqual(potential.energy_change(end,start),0.)
                moved=end.copy();moved[3,0]=math.nextafter(float(start[3,0]),math.inf)
                delta=F(float(moved[3,0]))-F(float(start[3,0]))
                expected=float((F(float(start[3,0]))+delta/2)*delta)
                self.assertAlmostEqual(potential.energy_change(start,moved)/expected,1.,delta=3e-13)
                # A resolved large normal change also exercises the endpoint
                # fallback. This does not claim small angular signal accuracy
                # under a simultaneous ill-conditioned large coordinate change.
                moved=end.copy();moved[2,2]=-.7*scale
                with localcontext() as context:
                    context.prec=100
                    expected=float(decimal_integral_value([definition],moved)-decimal_integral_value([definition],start))
                self.assertAlmostEqual(potential.energy_change(start,moved)/expected,1.,delta=3e-13)

    def test_opposed_large_cells_retain_small_net_force_and_signed_work(self):
        q=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,0.]])
        definitions=[cell(str(index),positiveStart=anchor({3:1}),positiveEnd=anchor({3:1}),
            negativeStart=anchor({0:1}),negativeEnd=anchor({0:1}),targetsMeters=[distance,distance],side=side,
            referenceLengthMeters=rat(100),stiffnessDensityNPerM2=1e12,activation=1.)
            for index,(side,distance) in enumerate(((-1,100.),(-1,1e-14),(1,100.)))]
        potential=ContinuousNormalOffsetSewing(4,definitions)
        expected=float(F(10**14)*F(1e-14))
        gradient=potential.gradient(q).reshape(-1,3)
        np.testing.assert_array_equal(gradient[:,:2],0.)
        np.testing.assert_allclose(gradient[:,2],[-expected,0.,0.,expected],rtol=3e-15,atol=0.)
        # At coincident anchors, expand E directly as .5*beta*|Cq|^2
        # -side*beta*d*(Cq).n +constant. This removes the identically
        # cancelling d^2 normal-curvature terms analytically. The O(1) cross
        # entries beside O(1e14) entries must survive as well as the force.
        c=np.zeros((3,12));c[:,:3]=-np.eye(3);c[:,9:]=np.eye(3)
        dn=np.zeros((3,12));dn[0,2]=dn[1,2]=1.;dn[0,5]=dn[1,8]=-1.
        hessian=3e14*(c.T@c)+expected*(c.T@dn+dn.T@c)
        np.testing.assert_allclose(potential.exact_hessian(q).toarray(),hessian,rtol=3e-15,atol=1e-13)
        # These entries receive no residual-curvature correction; GN must also
        # retain the small cross-cell signal independently of full curvature.
        gn=potential.hessian(q).toarray()
        self.assertAlmostEqual(gn[9,5],-expected,delta=3e-15)
        self.assertAlmostEqual(gn[10,8],-expected,delta=3e-15)
        for increment in (-1e-16,1e-16,-2**-48):
            end=q.copy();end[3,2]=increment
            with localcontext() as context:
                context.prec=100
                work=float(decimal_integral_value(definitions,end)-decimal_integral_value(definitions,q))
            self.assertNotEqual(work,0.)
            self.assertAlmostEqual(potential.energy_change(q,end)/work,1.,delta=3e-14)
        # Input order may not erase the middle small cell between cancelling
        # large forces. This is a force identity, not an optimizer convergence test.
        for ordered in (definitions[::-1],[definitions[0],definitions[2],definitions[1]]):
            np.testing.assert_allclose(ContinuousNormalOffsetSewing(4,ordered).gradient(q),gradient.ravel(),rtol=3e-15,atol=0.)

    def test_converted_thirds_retain_nonzero_anchor_residual_before_force_reduction(self):
        q=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,3.],[0.,0.,0.]])
        definition=cell(positiveStart=anchor({3:F(1,3),4:F(2,3)}),positiveEnd=anchor({3:F(1,3),4:F(2,3)}),
            negativeStart=anchor({0:1}),negativeEnd=anchor({0:1}),targetsMeters=[1.,1.],
            referenceLengthMeters=rat(1),stiffnessDensityNPerM2=1.,activation=1.)
        potential=ContinuousNormalOffsetSewing(5,[definition])
        a,b=F(float(F(1,3))),F(float(F(2,3)))
        error=3*a-1
        self.assertEqual(error,-F(1,2**54))
        np.testing.assert_array_equal(potential.residual(q),[0.,0.,float(error),0.,0.,0.])
        self.assertEqual(potential.energy(q),float(error*error/2))
        expected=np.zeros((5,3));expected[0,2]=float(-error);expected[3,2]=float(a*error);expected[4,2]=float(b*error)
        np.testing.assert_allclose(potential.gradient(q).reshape(-1,3),expected,rtol=3e-15,atol=0.)
        for direction in (-math.inf,math.inf):
            end=q.copy();end[3,2]=math.nextafter(3.,direction)
            delta=a*(F(float(end[3,2]))-F(3))
            expected_work=float((error+delta/2)*delta)
            self.assertNotEqual(expected_work,0.)
            self.assertAlmostEqual(potential.energy_change(q,end)/expected_work,1.,delta=3e-14)

    def test_rounded_normal_anchor_retains_true_geometric_residual_and_high_stiffness_force(self):
        definition=cell(positiveStart=anchor({3:1}),positiveEnd=anchor({3:1}),
            negativeStart=anchor({0:1}),negativeEnd=anchor({0:1}),targetsMeters=[1.,1.],
            referenceLengthMeters=rat(100),stiffnessDensityNPerM2=1e12,activation=1.)
        potential=ContinuousNormalOffsetSewing(4,[definition])
        for tilt in (.3,.7,1.3):
            # Matching a rounded normal does not give an exact geometric
            # equilibrium. At this stiffness the sub-ULP mismatch creates
            # measurable force, including the frame reaction; do not project
            # the normal derivative merely to make this state force-free.
            q=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,tilt],[0.,0.,0.]])
            area=np.cross(q[1]-q[0],q[2]-q[0])
            q[3]=area/np.linalg.norm(area)
            with self.subTest(tilt=tilt):
                residual,energy,gradient=decimal_constant_point_reference(q)
                self.assertGreater(float(energy),0.)
                expected_residual=[float(x*Decimal(10**7)) for x in residual]+[0.]*3
                np.testing.assert_allclose(potential.residual(q),expected_residual,rtol=3e-14,atol=1e-48)
                self.assertAlmostEqual(potential.energy(q)/float(energy),1.,delta=3e-14)
                np.testing.assert_allclose(potential.gradient(q),[float(x) for x in gradient],rtol=3e-14,atol=1e-48)
                # Decimal finite differences retain perturbations far below
                # binary64 spacing. They independently check both curvature
                # and GN without subtracting O(1e14) binary64 matrix entries.
                with localcontext() as context:
                    context.prec=160
                    base=[[Decimal(float(x)) for x in point] for point in q]
                    epsilon=Decimal('1e-45')
                    jacobian=[[Decimal(0) for _ in range(12)] for _ in range(3)]
                    hessian=np.zeros((12,12))
                    for column in range(12):
                        vertex,axis=divmod(column,3)
                        plus,minus=copy.deepcopy(base),copy.deepcopy(base)
                        plus[vertex][axis]+=epsilon;minus[vertex][axis]-=epsilon
                        rp,_,gp=decimal_constant_point_reference(plus)
                        rm,_,gm=decimal_constant_point_reference(minus)
                        for row in range(3):jacobian[row][column]=(rp[row]-rm[row])/(2*epsilon)
                        hessian[:,column]=[float((a-b)/(2*epsilon)) for a,b in zip(gp,gm)]
                    gn=np.array([[float(Decimal(10**14)*sum((jacobian[k][i]*jacobian[k][j] for k in range(3)),Decimal(0)))
                        for j in range(12)] for i in range(12)])
                np.testing.assert_allclose(potential.exact_hessian(q).toarray(),hessian,rtol=3e-14,atol=1e-48)
                np.testing.assert_allclose(potential.hessian(q).toarray(),gn,rtol=3e-14,atol=1e-48)
        # The exactly representable axial-normal state really does have zero
        # geometric residual and therefore zero force and residual curvature.
        exact=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        np.testing.assert_array_equal(potential.residual(exact),np.zeros(6))
        self.assertEqual(potential.energy(exact),0.)
        np.testing.assert_array_equal(potential.gradient(exact),np.zeros(12))
        np.testing.assert_array_equal(potential.exact_hessian(exact).toarray(),potential.hessian(exact).toarray())

    def test_directed_normal_enclosures_bind_original_binary_area_without_global_error_claim(self):
        self.assertEqual(self.potential.description()['normalSquareRootFractionBits'],256)
        axial=self.q.copy();axial[:3]=[[0.,0.,0.],[1.,0.,0.],[0.,3.,4.]]
        for q in (self.q,self.q*2**-60,self.q*2**12,axial):
            with self.subTest(scale=float(np.max(np.abs(q)))):
                report=self.potential.normal_diagnostics(q)
                self.assertIs(report['accepted'],False)
                self.assertIs(report['globalForceErrorBoundVerified'],False)
                self.assertEqual(len(report['frames']),1)
                frame=report['frames'][0]
                self.assertEqual(frame['cellId'],'one')
                self.assertEqual(frame['frameVertices'],[0,1,2])
                points=[[F(float(x)) for x in point] for point in q[:3]]
                u,v=([p-a for p,a in zip(point,points[0])] for point in points[1:])
                area=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
                squared=sum((x*x for x in area),F())
                self.assertEqual(rational(frame['doubleAreaSquaredMeters4']),squared)
                lower,upper=map(rational,frame['doubleAreaMagnitudeBoundsMeters2'])
                self.assertGreater(lower,0)
                self.assertLessEqual(lower*lower,squared)
                self.assertGreaterEqual(upper*upper,squared)
                self.assertLessEqual(upper-lower,max(map(abs,area))/2**256)
                midpoint=(lower+upper)/2
                defect=sum(((x/midpoint)**2 for x in area),F())-1
                self.assertEqual(rational(frame['normalMagnitudeSquaredMinusOne']),defect)
                self.assertIs(frame['enclosureVerified'],True)
                # Returned evidence is isolated and cannot alter geometry.
                frame['frameVertices'][0]=5
                self.assertEqual(self.potential.normal_diagnostics(q)['frames'][0]['frameVertices'],[0,1,2])
        pending=ContinuousNormalOffsetSewing(6,[cell('pending',activation=0.)])
        self.assertEqual(pending.normal_diagnostics(np.zeros((6,3)))['frames'],[])

    def test_pending_cell_skips_degenerate_frames_but_preserves_identity_and_shape(self):
        pending = cell('pending',activation=0.)
        p = ContinuousNormalOffsetSewing(6,[pending])
        collapsed = np.zeros((6,3));collapsed[3:]=1e5
        self.assertEqual(p.energy(collapsed),0.)
        self.assertEqual(p.energy_change(collapsed,-collapsed),0.)
        np.testing.assert_array_equal(p.residual(collapsed),np.zeros(6))
        np.testing.assert_array_equal(p.gradient(collapsed),np.zeros(18))
        for matrix in (p.jacobian(collapsed),p.hessian(collapsed),p.exact_hessian(collapsed)):
            self.assertEqual(matrix.nnz,0)
        self.assertEqual(p.cells,[pending])
        for method in ('energy','residual','jacobian','gradient','hessian','exact_hessian'):
            with self.subTest(method=method),self.assertRaises(ValueError):getattr(self.potential,method)(collapsed)
        with self.assertRaises(ValueError):self.potential.energy_change(self.q,collapsed)
        independent=cell('independent-pending',activation=0.,frameVertices=[6,7,8],
            positiveStart=anchor({9:1}),positiveEnd=anchor({9:1}),
            negativeStart=anchor({6:1}),negativeEnd=anchor({6:1}))
        mixed=ContinuousNormalOffsetSewing(10,[self.cells[0],independent])
        q=np.vstack((self.q,np.zeros((3,3)),[1e5,-1e5,1e5]))
        self.assertEqual(mixed.energy(q),self.potential.energy(self.q))
        np.testing.assert_array_equal(mixed.residual(q)[6:],0.)
        np.testing.assert_array_equal(mixed.gradient(q)[18:],0.)
        np.testing.assert_array_equal(mixed.exact_hessian(q).toarray()[18:,:],0.)

    def test_strict_raw_cell_identity_coefficients_and_frame_support(self):
        self.assertGreater(self.potential.energy(self.q),0.)
        attacks = [
            lambda c:c.update(id=''),lambda c:c.update(extra=True),lambda c:c.update(side=True),
            lambda c:c.update(side=0),lambda c:c.update(frameVertices=[0,0,2]),
            lambda c:c.update(frameVertices=[False,1,2]),lambda c:c.update(frameVertices=[0,1,6]),
            lambda c:c.update(positiveStart=anchor({0:1})),lambda c:c.update(negativeStart=anchor({3:1})),
            lambda c:c['positiveStart'][0].update(vertex=True),
            lambda c:c['positiveStart'][0]['weight'].update(numerator='03',denominator='4'),
            lambda c:c['positiveStart'][0]['weight'].update(numerator='6',denominator='8'),
            lambda c:c['positiveStart'][0]['weight'].update(numerator='0'),
            lambda c:c['positiveStart'][0]['weight'].update(numerator='-3'),
            lambda c:c['positiveStart'].reverse(),lambda c:c['positiveStart'].append(copy.deepcopy(c['positiveStart'][0])),
            lambda c:c.update(positiveStart=anchor({3:F(1,2)})),
            lambda c:c.update(targetsMeters=[True,.25]),lambda c:c.update(targetsMeters=[0.,.25]),
            lambda c:c.update(targetsMeters=[.25]),lambda c:c.update(activation=-.1),
            lambda c:c.update(activation=1.1),lambda c:c.update(activation=True),
            lambda c:c.update(stiffnessDensityNPerM2=float('inf')),
            lambda c:c.update(stiffnessDensityNPerM2=0.),lambda c:c.update(stiffnessDensityNPerM2=1e12+1.),
            lambda c:c.update(referenceLengthMeters=rat(0)),
            lambda c:c.update(referenceLengthMeters=rat(101)),
            # JSON serialization alone changes tuples into lists and accepts
            # some NumPy scalar subclasses. Admission must inspect raw types
            # before immutable serialization can conceal those distinctions.
            lambda c:c.update(frameVertices=tuple(c['frameVertices'])),
            lambda c:c.update(positiveStart=tuple(c['positiveStart'])),
            lambda c:c.update(negativeEnd=tuple(c['negativeEnd'])),
            lambda c:c.update(targetsMeters=tuple(c['targetsMeters'])),
            lambda c:c.update(targetsMeters=[np.float64(.125),.375]),
            lambda c:c.update(activation=np.float64(.75)),
            lambda c:c.update(activation=np.float32(.75)),
            lambda c:c.update(activation=np.float16(.75)),
            lambda c:c.update(activation=np.longdouble(.75)),
            lambda c:c.update(activation=np.bool_(True)),
            lambda c:c.update(stiffnessDensityNPerM2=np.float64(16.)),
            lambda c:c.update(side=np.int64(1)),
            lambda c:c.update(frameVertices=[0,np.int64(1),2]),
            lambda c:c['positiveStart'][0].update(vertex=np.uint64(3)),
            lambda c:c.update(id=np.str_('one')),
            lambda c:c['positiveStart'][0]['weight'].update(numerator=np.str_('3')),
            lambda c:c.update({np.str_('activation'):c.pop('activation')}),
        ]
        for index,attack in enumerate(attacks):
            altered=copy.deepcopy(self.cells[0]);attack(altered)
            with self.subTest(index=index),self.assertRaises(ValueError):ContinuousNormalOffsetSewing(6,[altered])
        for count in (True,0,100001,6.,np.int64(6)):
            with self.subTest(count=count),self.assertRaises(ValueError):ContinuousNormalOffsetSewing(count,self.cells)
        for cells in ([],self.cells*4097,self.cells*2,tuple(self.cells),[tuple(self.cells[0].items())]):
            with self.subTest(type=type(cells).__name__,count=len(cells)),self.assertRaises(ValueError):ContinuousNormalOffsetSewing(6,cells)

    def test_positive_coefficient_and_integrated_scale_underflow_fail_closed(self):
        tiny = F(1,2**1075)
        definition = cell(positiveStart=anchor({3:tiny,4:1-tiny}))
        with self.assertRaises(ValueError):ContinuousNormalOffsetSewing(6,[definition])
        for definition in (cell(referenceLengthMeters=rat(F(1,2**1074)),stiffnessDensityNPerM2=1.,activation=1.),
                cell(stiffnessDensityNPerM2=math.ulp(0.),referenceLengthMeters=rat(1),activation=1.),
                cell(stiffnessDensityNPerM2=1.,referenceLengthMeters=rat(1),activation=math.ulp(0.))):
            with self.subTest(definition=definition),self.assertRaises(ValueError):ContinuousNormalOffsetSewing(6,[definition])

    def test_exact_to_binary_conversions_and_non_dyadic_measure_are_visible(self):
        definition = cell(positiveStart=anchor({3:F(1,10),4:F(9,10)}),
            positiveEnd=anchor({4:F(1,3),5:F(2,3)}),negativeStart=anchor({0:1}),
            referenceLengthMeters=rat(F(7,10)),stiffnessDensityNPerM2=.3,activation=.7)
        potential = ContinuousNormalOffsetSewing(6,[definition])
        description = potential.description()
        self.assertEqual(description['cells'],[definition])
        self.assertEqual(description['vertexCount'],6);self.assertEqual(description['cellCount'],1)
        conversion = description['conversions'][0]
        gamma = F(.7)*F(.3)*F(7,10)
        slope = gamma/12
        self.assertEqual(rational(conversion['exactEffectiveStiffnessNPerM']),gamma)
        self.assertEqual(conversion['numericalEffectiveStiffnessNPerM'],float(gamma))
        self.assertEqual(rational(conversion['effectiveStiffnessConversionResidualNPerM']),F(float(gamma))-gamma)
        self.assertEqual(rational(conversion['exactSlopeWeightNPerM']),slope)
        self.assertEqual(conversion['numericalSlopeWeightNPerM'],float(slope))
        self.assertEqual(rational(conversion['slopeWeightConversionResidualNPerM']),F(float(slope))-slope)
        for prefix,value in (('mean',gamma),('slope',slope)):
            scale = math.sqrt(float(value))
            self.assertEqual(conversion[prefix+'ResidualScale'],scale)
            key = 'exactSquareOf'+prefix.capitalize()+'ResidualScale'
            self.assertEqual(rational(conversion[key]),F(scale)**2)
        sums = []
        for ending in ('Start','End'):
            signed_sum = F()
            for sign,prefix in ((1,'positive'),(-1,'negative')):
                for original,actual in zip(definition[prefix+ending],conversion['anchorConversions'][prefix+ending]):
                    exact = rational(original['weight']);numeric = float(exact)
                    self.assertEqual(actual['vertex'],original['vertex'])
                    self.assertEqual(rational(actual['exactWeight']),exact)
                    self.assertEqual(actual['numericalWeight'],numeric)
                    self.assertEqual(rational(actual['numericalMinusExact']),F(numeric)-exact)
                    signed_sum += sign*F(numeric)
            sums.append(signed_sum)
        self.assertEqual(list(map(rational,conversion['numericalSignedCoefficientSums'])),sums)
        self.assertNotEqual(sums[0],0, 'Converted 1/10 and9/10 weights have a real signed-sum defect')
        self.assertAlmostEqual(potential.energy(self.q)/decimal_integral([definition],self.q),1.,delta=4e-15)
        self.assertTrue(description['limitations'])
        self.assertIs(description['accepted'],False)

    def test_raw_positions_and_active_frame_degeneracy_reject(self):
        self.assertGreater(self.potential.energy(self.q.tolist()),0.)
        attacks = [self.q[:-1],self.q[:,:2],self.q.astype(bool),self.q.tolist()]
        attacks[-1][0][0]=False
        for value in (float('nan'),float('inf'),1000001.,2**53+1,complex(1,0)):
            q=self.q.astype(object);q[0,0]=value;attacks.append(q)
        for index,q in enumerate(attacks):
            with self.subTest(index=index),self.assertRaises(ValueError):self.potential.energy(q)
        if np.dtype(np.longdouble).itemsize>8:
            with self.assertRaises(ValueError):self.potential.energy(self.q.astype(np.longdouble))
            raw=self.q.tolist();raw[0][0]=np.longdouble('0.100000000000000000001')
            with self.assertRaises(ValueError):self.potential.energy(raw)
        near=self.q.copy();near[:3]=[[0.,0.,0.],[1.,0.,0.],[1.,2**-45,0.]]
        with self.assertRaises(ValueError):self.potential.energy(near)

    def test_input_cells_positions_and_public_descriptions_are_isolated(self):
        raw=copy.deepcopy(self.cells);q=self.q.copy()
        potential=ContinuousNormalOffsetSewing(6,raw)
        before=q.tobytes();energy=potential.energy(q);gradient=potential.gradient(q).copy()
        raw[0]['targetsMeters'][0]=10.
        raw[0]['positiveStart'].clear()
        returned=potential.cells;returned[0]['frameVertices'].reverse();returned[0]['activation']=0.
        activation=potential.activation;activation[:]=0.
        description=potential.description()
        # A deep mutation of every returned mutable container must not feed back.
        def erase(value):
            if isinstance(value,dict):
                for child in list(value.values()):erase(child)
                value.clear()
            elif isinstance(value,list):
                for child in value:erase(child)
                value.clear()
        erase(description)
        self.assertEqual(potential.cells,self.cells)
        np.testing.assert_array_equal(potential.activation,[.75])
        self.assertEqual(potential.energy(q),energy)
        np.testing.assert_array_equal(potential.gradient(q),gradient)
        self.assertTrue(potential.description())
        self.assertEqual(q.tobytes(),before)
        # The previous implementation retained mutable native endpoint and CSR
        # objects: target mutation changed energy while input hashes stayed put.
        # They may be absent, immutable, or fresh disposable views after repair.
        description=potential.description()
        endpoint=getattr(potential,'_endpoint',None)
        if endpoint is not None:
            try:endpoint.targets[:]=10.
            except (ValueError,AttributeError):pass
        transform=getattr(potential,'_transform',None)
        if transform is not None:
            try:transform.data[:]=0.
            except (ValueError,AttributeError):pass
        self.assertEqual(potential.energy(q),energy)
        self.assertEqual(potential.description(),description)
        np.testing.assert_array_equal(potential.gradient(q),gradient)
        with self.assertRaises(AttributeError):potential._targets=(10.,10.)
        with self.assertRaises(AttributeError):delattr(potential,'_sealed')
        self.assertFalse(hasattr(potential,'__dict__'), 'Mutable instance dictionaries bypass the advertised recipe seal')


if __name__=='__main__':
    unittest.main()
