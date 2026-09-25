"""Fixed-control physical responses for experimental coupled time integration.

This boundary does not advance time, validate a swept path or certify a garment.
Gradients are potential gradients (the physical force has the opposite sign).
Some existing mechanical matrices are search approximations, explicitly labelled
below. No backward-Euler inertia is included or numerically subtracted.
"""
from fractions import Fraction as F
import hashlib
import json

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, issparse

from solver_cable_integration import positions_sha256, round_sum
from solver_continuous_normal_sewing import _rat, _rational
from solver_global_sewing import GlobalSewingSolver
from solver_membrane_hessian import membrane_element_derivatives
from solver_sewing_activation import validate_sewing_activation
from solver_temporal_control import problem_identity, strict_array


PROFILE = 'fixed-physical-potential-response-v1'
COMPONENTS = ('sewing', 'membrane', 'bending', 'foldActuation', 'grippers',
              'foldBarrier', 'contact', 'cable')
ERROR_SCOPE = (
    'Conditional on existing binary64 non-cable component evaluations. Known '
    'bounds include cable integration error and one final rounding of component '
    'energies/gradient entries only; component assembly, other constitutive and '
    'geometric errors and search-matrix assembly are not certified. No native '
    'stationarity equivalence, continuous path, time accuracy or garment acceptance.'
)


def _array(value, shape=None):
    result = strict_array(value)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError('Finite correctly shaped binary64 physical controls required')
    return np.frombuffer(result.tobytes(), dtype=np.float64).reshape(result.shape)


def _snapshot(value):
    return value.dtype.str, value.shape, value.strides, value.tobytes()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class PhysicalResponseFailure(ValueError):
    """Rejected response with retained attempted component calls, no new state."""
    def __init__(self, reason, component_calls):
        super().__init__(reason)
        self.component_calls = dict(component_calls)


class FixedPhysicalPotential:
    """One fixed set of controls, with fresh response evaluation on every call.

    Exposed numerical model/recipe identity is checked before and after calls.
    It is an in-process identity check with the existing contact-cache exclusions,
    not a cross-process revision digest or inspection of hidden native state.
    The future integrator owns reservations, path checks and state publication.
    """
    __slots__ = ('_solver', '_targets', '_weights', '_distance', '_actuator',
                 '_grippers', '_cable', '_cable_parameters', '_linear_sewing',
                 '_description_bytes', '_identity', '_sealed')

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError('Fixed physical controls are immutable')
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError('Fixed physical controls are immutable')

    def __init__(self, solver, targets, *, sewing_activation=None,
                 fold_targets=None, fold_activation=None,
                 gripper_targets=None, gripper_activation=None,
                 cable_targets=None, cable_activation=None):
        if getattr(self, '_sealed', False):
            raise AttributeError('Fixed physical controls cannot be reinitialized')
        if type(solver) is not GlobalSewingSolver:
            raise ValueError('An explicit GlobalSewingSolver is required')
        original_identity = problem_identity(solver)
        self._solver = solver

        def construct(function, *arguments, **keywords):
            arrays = [(value, _snapshot(value)) for value in (*arguments, *keywords.values())
                      if type(value) is np.ndarray]
            try:
                return function(*arguments, **keywords)
            finally:
                if any(_snapshot(value) != before for value, before in arrays):
                    raise ValueError('Physical control helper mutated its supplied parameters')
                if problem_identity(solver) != original_identity:
                    raise ValueError('Physical control construction mutated the solver model')

        count = solver.sewing.shape[0]
        self._targets = _array(targets, (count, 3) if solver.sewing_mode == 'vector' else (count,))
        self._weights = _array(validate_sewing_activation(sewing_activation, count), (count,))
        self._distance = (None if solver.sewing_mode == 'vector' else
                          construct(solver.sewing_potential, self._targets, activation=self._weights))
        self._linear_sewing = (solver.sewing_xyz.T @ diags(np.repeat(self._weights, 3))
                               @ solver.sewing_xyz / solver.compliance).tocsr()
        self._actuator = None
        controls = {'sewingMode': solver.sewing_mode, 'sewingTargets': self._targets.tolist(),
                    'sewingActivation': self._weights.tolist()}
        if solver.controlled_fold_actuation is not None:
            if solver.fold_actuation is not None or fold_targets is None or fold_activation is None:
                raise ValueError('Controlled fold requires explicit targets and activation without legacy actuation')
            ft, fa = _array(fold_targets), _array(fold_activation)
            self._actuator = construct(solver.controlled_fold_actuation.potential, ft, fa)
            controls.update(foldTargets=ft.tolist(), foldActivation=fa.tolist(), foldKind='controlled')
        else:
            if fold_activation is not None or (solver.fold_actuation is None) != (fold_targets is None):
                raise ValueError('Matching legacy fold recipe/targets or no fold parameters required')
            if solver.fold_actuation is not None:
                ft = _array(fold_targets)
                self._actuator = construct(solver.fold_actuation.potential, ft)
                controls.update(foldTargets=ft.tolist(), foldKind='legacy')
        self._grippers = None
        if solver.material_grippers is None:
            if gripper_targets is not None or gripper_activation is not None:
                raise ValueError('Gripper parameters require a material-gripper recipe')
        else:
            if gripper_targets is None or gripper_activation is None:
                raise ValueError('Material grippers require explicit targets and activation')
            gt, ga = _array(gripper_targets), _array(gripper_activation)
            self._grippers = construct(solver.material_grippers.potential, gt, ga)
            controls.update(gripperTargets=gt.tolist(), gripperActivation=ga.tolist())
        self._cable = solver.continuous_cable
        self._cable_parameters = None
        if solver.cable_parameter_control is not None:
            if self._cable is not None or cable_targets is None or cable_activation is None:
                raise ValueError('Varying cable requires explicit parameters without a fixed cable')
            ct, ca = construct(solver.cable_parameter_control.parameters,
                               _array(cable_targets), _array(cable_activation))
            self._cable_parameters = (_array(ct), _array(ca))
            self._cable = construct(solver.cable_parameter_control.effective, *self._cable_parameters)
            controls['cableParameters'] = construct(solver.cable_parameter_control.parameter_record,
                                                    *self._cable_parameters)
        elif cable_targets is not None or cable_activation is not None:
            raise ValueError('Cable parameters require a varying cable recipe')
        if self._cable is not None:
            from solver_cable_integration import CableControl
            if type(self._cable) is not CableControl:
                raise ValueError('An explicitly validated cable control is required')
            controls['effectiveCable'] = self._cable.description()
        if solver.contact_work_control is not None:
            solver.contact_work_control.check(solver.contact)
        kinds = {
            'sewing': 'analytic-unprojected', 'membrane': 'analytic-unprojected',
            'bending': 'gauss-newton-search-metric',
            'foldActuation': 'gauss-newton-search-metric' if self._actuator is not None else 'inactive',
            'grippers': 'analytic-quadratic' if self._grippers is not None else 'inactive',
            'foldBarrier': 'angular-pullback-search-metric' if solver.fold_barrier is not None else 'inactive',
            'contact': ('native-unprojected' if solver.contact.requires_guarded_metric else
                        'native-psd-projected-search-metric') if solver.contact is not None else 'inactive',
            'cable': 'bounded-analytic-unprojected' if self._cable is not None else 'inactive',
        }
        self._description_bytes = _encoded({
            'profile': PROFILE, 'vertexCount': len(solver.mass), 'componentOrder': list(COMPONENTS),
            'controls': controls, 'controlSha256': hashlib.sha256(_encoded(controls)).hexdigest(),
            'gradientConvention': 'Potential gradient; physical force is its negative',
            'componentSearchMatrixKinds': kinds, 'includesInertia': False,
            'configurationIdentityScope': 'Existing exposed in-process problem identity, excluding known contact caches; no hidden-native-state or cross-process revision digest claim',
            'errorScope': ERROR_SCOPE, 'accepted': False,
            'scope': 'Fixed-control response only; no swept path, timestep, power/work quadrature, source admission or garment acceptance',
        })
        if problem_identity(solver) != original_identity:
            raise ValueError('Physical control construction mutated the solver model')
        self._identity = problem_identity(solver, self._bound_controls())
        self._sealed = True

    def _bound_controls(self):
        return (self._targets, self._weights, self._distance, self._actuator,
                self._grippers, self._cable, self._cable_parameters, self._linear_sewing)

    def _check(self):
        if problem_identity(self._solver, self._bound_controls()) != self._identity:
            raise ValueError('Physical model or fixed control identity changed')
        if self._solver.contact_work_control is not None:
            self._solver.contact_work_control.check(self._solver.contact)

    def description(self):
        return json.loads(self._description_bytes)

    def evaluate(self, positions):
        """Fresh finite physical response; failure publishes no response object."""
        calls = {}
        try:
            return self._evaluate(positions, calls)
        except Exception as failure:
            raise PhysicalResponseFailure(str(failure), calls) from failure

    def _evaluate(self, positions, calls):
        self._check()
        solver = self._solver
        shape = (len(solver.mass), 3)
        if (type(positions) is not np.ndarray or positions.dtype != np.dtype(np.float64)
                or positions.shape != shape or not np.isfinite(positions).all()):
            raise ValueError('Finite raw binary64 physical positions required')
        external_snapshot = _snapshot(positions)
        q = positions.copy()
        snapshot = _snapshot(q)
        size = q.size
        energies = dict.fromkeys(COMPONENTS, 0.)
        gradients = {name: np.zeros(size) for name in COMPONENTS}
        matrices = {name: csr_matrix((size, size)) for name in COMPONENTS}
        def invoke(name, function, *arguments):
            calls[name] = calls.get(name, 0)+1
            arrays = [(value, _snapshot(value)) for value in arguments if type(value) is np.ndarray]
            try:
                return function(*arguments)
            finally:
                if (any(_snapshot(value) != before for value, before in arrays)
                        or _snapshot(q) != snapshot or _snapshot(positions) != external_snapshot):
                    raise ValueError('Physical helper mutated supplied geometry')
                self._check()

        def matrix(value):
            if (not issparse(value) or value.shape != (size, size) or value.dtype != np.dtype(np.float64)
                    or not np.isfinite(value.data).all()):
                raise ValueError('Finite correctly shaped sparse physical search matrix required')
            return value.tocsr(copy=True)

        try:
            if solver.contact is not None:
                invoke('contact.validate_state', solver.contact.validate_state, q)
            coefficients = np.concatenate((-solver.poses.sum(axis=1)[:, None, :], solver.poses), axis=1)
            deformation = np.einsum('fvc,fva->fca', coefficients, q[solver.faces])
            first, second = deformation[:, 0], deformation[:, 1]
            area_ratio = np.linalg.norm(np.cross(first, second), axis=1)
            if not np.isfinite(deformation).all() or np.any(area_ratio <= 1e-10):
                raise ValueError('Degenerate membrane physical response')
            first_norm, second_norm = np.linalg.norm(first, axis=1), np.linalg.norm(second, axis=1)
            shear_shape = ((first_norm-second_norm)**2 + 2*np.sum(first*second, axis=1)**2
                           / (first_norm*second_norm+area_ratio))
            shear = solver.materials[:, 0]
            lame = shear+solver.materials[:, 1]
            alpha = 1+shear/np.maximum(lame, 1e-6)
            strain = area_ratio-1
            energies['membrane'] = float(np.sum(solver.areas*(shear*shear_shape/2
                + lame*strain**2/2 + (shear+lame*(1-alpha))*strain)))
            # The helper's raw energy has a different additive reference. Only
            # its unprojected derivatives are used; preserve the stable energy above.
            _, element_gradient, elements = invoke('membrane.derivatives', membrane_element_derivatives,
                q[solver.faces], solver.poses, solver.areas, solver.materials[:, :3])
            np.add.at(gradients['membrane'].reshape(shape), solver.faces, element_gradient.reshape((-1, 3, 3)))
            dofs = (solver.faces[:, :, None]*3+np.arange(3)).reshape((-1, 9))
            matrices['membrane'] = coo_matrix((elements.ravel(),
                (np.broadcast_to(dofs[:, :, None], elements.shape).ravel(),
                 np.broadcast_to(dofs[:, None, :], elements.shape).ravel())), shape=(size, size)).tocsr()
            if self._distance is None:
                active = self._weights > 0
                rows = solver.sewing[active]
                # Inactive targets never participate in subtraction or squaring.
                error = rows@q-self._targets[active]
                residual = np.sqrt(self._weights[active])[:, None]*error/np.sqrt(solver.compliance)
                energies['sewing'] = float(np.sum(residual*residual)/2)
                gradients['sewing'] = np.asarray(rows.T@(self._weights[active, None]*error/solver.compliance)).ravel()
                matrices['sewing'] = self._linear_sewing.copy()
                calls['sewing.vector_response'] = 1
            else:
                residual = invoke('sewing.residual', self._distance.residual, q)
                energies['sewing'] = float(np.dot(residual, residual)/2)
                gradients['sewing'] = invoke('sewing.gradient', self._distance.gradient, q)
                matrices['sewing'] = invoke('sewing.exact_hessian', self._distance.exact_hessian, q)
            for name, potential in (('bending', solver.bending), ('foldActuation', self._actuator),
                                    ('grippers', self._grippers), ('foldBarrier', solver.fold_barrier),
                                    ('contact', solver.contact)):
                if potential is None:
                    continue
                energies[name] = invoke(name+'.energy', potential.energy, q)
                gradients[name] = invoke(name+'.gradient', potential.gradient, q)
                matrices[name] = invoke(name+'.hessian', potential.hessian, q)
            cable_response = None
            cable_energy_error = cable_gradient_error = F()
            if self._cable is not None:
                response = invoke('cable.evaluate', self._cable.evaluate, q)
                cable_response = invoke('cable.validate_response', self._cable.validate_response, q, response)
                energies['cable'] = cable_response['energy']
                gradients['cable'] = cable_response['gradient']
                matrices['cable'] = cable_response['hessian']
                cable_energy_error = _rational(cable_response['certificate']['energyErrorBoundJoules'])
                cable_gradient_error = _rational(cable_response['certificate']['gradientMaxAbsoluteErrorBoundNewtons'])
            for name in COMPONENTS:
                if type(energies[name]) not in (float, np.float64) or not np.isfinite(energies[name]):
                    raise ValueError('Finite binary64 scalar physical component energy required')
                energies[name] = float(energies[name])
                gradients[name] = np.asarray(gradients[name])
                if gradients[name].dtype != np.dtype(np.float64):
                    raise ValueError('Binary64 physical component gradient required')
                gradients[name] = gradients[name].reshape(-1)
                if gradients[name].shape != (size,) or not np.isfinite(gradients[name]).all():
                    raise ValueError('Finite correctly shaped physical component gradient required')
                matrices[name] = matrix(matrices[name])
            energy, energy_rounding = round_sum(energies.values())
            gradient = np.empty(size)
            gradient_rounding = F()
            for i in range(size):
                gradient[i], error = round_sum(gradients[name][i] for name in COMPONENTS)
                gradient_rounding = max(gradient_rounding, error)
            search_matrix = sum((matrices[name] for name in COMPONENTS), csr_matrix((size, size)))
            search_matrix = matrix(search_matrix)
            description = self.description()
            return {
                'profile': PROFILE, 'positionsSha256': positions_sha256(q),
                'controlSha256': description['controlSha256'], 'configurationIdentityChecked': True,
                'energyJoules': energy, 'gradientNewtons': gradient,
                'searchMatrixNewtonsPerMetre': search_matrix,
                'componentEnergyJoules': energies,
                'componentSearchMatrixKinds': description['componentSearchMatrixKinds'],
                'componentCalls': calls, 'cableCertificate': None if cable_response is None else cable_response['certificate'],
                'knownErrorBounds': {
                    'energyJoules': _rat(cable_energy_error+energy_rounding),
                    'gradientMaxAbsoluteNewtons': _rat(cable_gradient_error+gradient_rounding),
                    'cableEnergyJoules': _rat(cable_energy_error),
                    'cableGradientMaxAbsoluteNewtons': _rat(cable_gradient_error),
                    'energyAssemblyRoundingJoules': _rat(energy_rounding),
                    'gradientAssemblyRoundingNewtons': _rat(gradient_rounding),
                    'scope': ERROR_SCOPE,
                },
                'includesInertia': False, 'accepted': False,
            }
        finally:
            if _snapshot(q) != snapshot or _snapshot(positions) != external_snapshot:
                raise ValueError('Physical response mutated supplied geometry')
            self._check()
