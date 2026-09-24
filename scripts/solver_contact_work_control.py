"""Explicit bounded contact-work policy and fresh publication validation.

The native contact object remains trusted. This binds its captured geometry and
parameters, not garment/source ownership or native force arithmetic accuracy.
"""
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
import math

import numpy as np

from solver_contact_work import PROFILE, WorkPolicy, WorkResult, WorkBudget, ContactTerm
from solver_contact_work_native import bounded_native_work, NativeEndpoint
from solver_rest_filtered_contact import RestFilteredSurfaceContact


CONTROL_PROFILE = 'guarded-native-contact-work-control-v1'
RECORD_PROFILE = 'bounded-native-contact-work-record-v1'
SCOPE = ('Bounded declared contact scalar work on complete trusted native endpoint inventories; '
         'exact binary-input geometry and captured native-rounded constants. Native energy observations, '
         'gradient/Hessian arithmetic, other constitutive work, source authenticity and physical '
         'contact are not certified. No path, garment or temporal-accuracy acceptance.')
POLICY_KEYS = set(WorkPolicy.__dataclass_fields__)


def encoded(value):
    try:
        return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError('Finite JSON contact-work record required') from error


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _array_identity(array):
    value=np.asarray(array)
    return (value.shape,value.strides,value.dtype.str,hashlib.sha256(value.tobytes()).hexdigest())


def _native_identity(contact):
    contact._check_native_parameters()
    return encoded({'profile':contact.profile(),'parameters':contact._contact_parameters,
        'rest':_array_identity(contact.rest_positions),'faces':_array_identity(contact.faces),
        'edges':_array_identity(contact._edges),'nativeRest':_array_identity(contact.mesh.rest_positions),
        'nativeFaces':_array_identity(contact.mesh.faces),'nativeEdges':_array_identity(contact.mesh.edges)})


def _endpoint_record(endpoint):
    inventory=[]
    for term,observation in zip(endpoint.terms,endpoint.observations):
        inventory.append({'term':asdict(term),'native':{
            'knownFeature':observation.known_feature,'selectedFeature':observation.selected_feature,
            'distanceSquared':observation.distance_squared,'mollifier':observation.mollifier}})
    return {'positionsSha256':endpoint.positions_sha256,'inventorySha256':digest(inventory),
            'termCount':len(endpoint.terms),'nativeEnergyJoules':endpoint.native_energy}


def validate_record(record):
    """Structural validation only; a fresh control evaluation proves numerics."""
    keys={'profile','definition','changeJoules','changeErrorBoundJoules','start','end',
          'unionTerms','logCalls','logTerms','scope','accepted'}
    if type(record) is not dict or set(record)!=keys:
        raise ValueError('Complete bounded contact-work record required')
    if record['profile']!=RECORD_PROFILE or record['scope']!=SCOPE or record['accepted'] is not False:
        raise ValueError('Contact-work profile or scope mismatch')
    definition=record['definition']
    if (type(definition) is not dict or set(definition)!={'profile','scalarProfile','nativeIdentitySha256','vertexCount','policy','scope'}
            or definition['profile']!=CONTROL_PROFILE or definition['scalarProfile']!=PROFILE
            or definition['scope']!=SCOPE or type(definition['vertexCount']) is not int or definition['vertexCount']<3
            or type(definition['policy']) is not dict or set(definition['policy'])!=POLICY_KEYS):
        raise ValueError('Complete explicit contact-work definition required')
    policy=WorkPolicy(**definition['policy'])
    hashes=[definition['nativeIdentitySha256']]
    for name in ('start','end'):
        endpoint=record[name]
        if (type(endpoint) is not dict or set(endpoint)!={'positionsSha256','inventorySha256','termCount','nativeEnergyJoules'}
                or type(endpoint['termCount']) is not int or not 0<=endpoint['termCount']<=policy.max_endpoint_terms
                or type(endpoint['nativeEnergyJoules']) is not float or not math.isfinite(endpoint['nativeEnergyJoules'])
                or endpoint['nativeEnergyJoules']<0):
            raise ValueError('Complete native contact endpoint identity required')
        hashes.extend((endpoint['positionsSha256'],endpoint['inventorySha256']))
    if any(type(h) is not str or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
        raise ValueError('Canonical contact-work digest required')
    if any(type(record[k]) is not float or not math.isfinite(record[k]) for k in ('changeJoules','changeErrorBoundJoules')):
        raise ValueError('Finite binary64 contact work and error bound required')
    if record['changeErrorBoundJoules']<0:
        raise ValueError('Nonnegative contact-work error bound required')
    counts=(record['start']['termCount'],record['end']['termCount'])
    if sum(counts)>policy.max_endpoint_terms:
        raise ValueError('Contact endpoint inventory exceeds fixed budget')
    if (any(type(record[k]) is not int or record[k]<0 for k in ('unionTerms','logCalls','logTerms'))
            or not max(counts)<=record['unionTerms']<=sum(counts)
            or record['logTerms']>policy.max_total_log_terms
            or record['logCalls']>3*record['unionTerms']):
        raise ValueError('Invalid bounded contact-work operation counts')
    encoded(record)
    return F(record['changeErrorBoundJoules'])


def contact_error(report):
    """Match public fields to a structured work record; not fresh verification."""
    if 'boundedContactWork' not in report:
        return F()
    work=report['boundedContactWork']
    error=validate_record(work)
    for name,expected in (('contactChangeJoules',work['changeJoules']),
            ('contactBeforeJoules',work['start']['nativeEnergyJoules']),
            ('contactAfterJoules',work['end']['nativeEnergyJoules'])):
        if type(report.get(name)) is not float or report[name].hex()!=expected.hex():
            raise ValueError('Public contact energy differs from bounded contact work: '+name)
    return error


class ContactWorkControl:
    __slots__=('_contact','_native_bytes','_description_bytes','_policy','_sealed')

    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):raise AttributeError('Contact work controls are immutable')
        object.__setattr__(self,name,value)

    def __delattr__(self,name):
        raise AttributeError('Contact work controls are immutable')

    def __init__(self,contact,policy):
        if getattr(self,'_sealed',False):raise AttributeError('Contact work controls are already initialized')
        if type(contact) is not RestFilteredSurfaceContact:
            raise ValueError('Explicit rest-filtered native IPC contact required')
        if type(policy) is not dict or set(policy)!=POLICY_KEYS:
            raise ValueError('Every contact-work arithmetic budget must be explicit')
        self._policy=WorkPolicy(**policy)
        self._contact=contact
        self._native_bytes=_native_identity(contact)
        self._description_bytes=encoded({'profile':CONTROL_PROFILE,'scalarProfile':PROFILE,
            'nativeIdentitySha256':hashlib.sha256(self._native_bytes).hexdigest(),
            'vertexCount':contact.vertex_count,'policy':asdict(self._policy),'scope':SCOPE})
        self._sealed=True

    def description(self):
        return json.loads(self._description_bytes)

    def check(self,contact):
        if contact is not self._contact or _native_identity(contact)!=self._native_bytes:
            raise ValueError('Contact model identity or parameters changed during bounded work')

    def positions(self,value):
        if (type(value) is not np.ndarray or value.dtype!=np.dtype('float64')
                or value.shape!=(self._contact.vertex_count,3) or not np.isfinite(value).all()):
            raise ValueError('Finite raw binary64 contact work state required')
        return np.frombuffer(value.tobytes(),dtype=np.float64).reshape(value.shape)

    def _evaluate(self,start,end):
        self.check(self._contact)
        first,last=self.positions(start).copy(),self.positions(end).copy()
        snapshots=(_array_identity(first),_array_identity(last))
        try:
            result,a,b=bounded_native_work(self._contact,first,last,policy=self._policy)
        finally:
            if (_array_identity(first),_array_identity(last))!=snapshots:
                raise ValueError('Native contact work helper mutated isolated endpoints')
            self.check(self._contact)
        if type(result) is not WorkResult or result.profile!=PROFILE:
            raise ValueError('Exact bounded contact-work result required')
        budget=WorkBudget(self._policy)
        for value in (result.lower,result.upper,result.absolute_error):budget.check(value)
        if (type(result.value) is not float or not math.isfinite(result.value)
                or result.lower>result.upper or result.absolute_error<0):
            raise ValueError('Ordered finite contact-work interval required')
        try:
            midpoint=float((result.lower+result.upper)/2)
        except OverflowError as failure:
            raise ValueError('Unrepresentable contact-work midpoint') from failure
        if (result.value.hex()!=midpoint.hex() or result.absolute_error!=max(
                abs(F(result.value)-result.lower),abs(result.upper-F(result.value)))):
            raise ValueError('Contact-work value/radius differ from retained exact interval')
        for endpoint,positions in ((a,first),(b,last)):
            expected_hash=hashlib.sha256(np.asarray(positions,dtype='<f8').tobytes(order='C')).hexdigest()
            if (type(endpoint) is not NativeEndpoint or type(endpoint.terms) is not tuple
                    or type(endpoint.observations) is not tuple or len(endpoint.terms)!=len(endpoint.observations)
                    or endpoint.positions_sha256!=expected_hash):
                raise ValueError('Native contact capture differs from actual endpoint identity')
            for term in endpoint.terms:
                if (type(term) is not ContactTerm or any(v>=len(positions) for v in term.vertex_ids)
                        or term.positions!=tuple(tuple(float(v) for v in row) for row in positions[list(term.vertex_ids)])):
                    raise ValueError('Captured contact coordinates differ from actual endpoints')
        if result.endpoint_terms!=len(a.terms)+len(b.terms):
            raise ValueError('Incomplete bounded contact endpoint inventory')
        record={'profile':RECORD_PROFILE,'definition':self.description(),
            'changeJoules':result.value,'changeErrorBoundJoules':result.binary64_error_bound(),
            'start':_endpoint_record(a),'end':_endpoint_record(b),'unionTerms':result.union_terms,
            'logCalls':result.log_calls,'logTerms':result.log_terms,'scope':SCOPE,'accepted':False}
        validate_record(record)
        return record

    def energy_change(self,start,end):
        return self._evaluate(start,end)

    def validate_change(self,start,end,record):
        validate_record(record)
        # Recompute complete native captures and work, rather than accepting a
        # favorable but understated radius or matching only policy metadata.
        expected=ContactWorkControl._evaluate(self,start,end)
        if encoded(record)!=encoded(expected):
            raise ValueError('Bounded contact work differs from fresh endpoint evaluation')
        return json.loads(encoded(expected))


def checked_change(control,contact,start,end):
    if type(control) is not ContactWorkControl:
        raise ValueError('Explicit immutable contact-work control required')
    control.check(contact)
    first,last=control.positions(start).copy(),control.positions(end).copy()
    snapshots=(_array_identity(first),_array_identity(last))
    try:
        record=control.energy_change(first,last)
        validated=ContactWorkControl.validate_change(control,first,last,record)
    finally:
        if (_array_identity(first),_array_identity(last))!=snapshots:
            raise ValueError('Contact work validation mutated isolated endpoints')
        control.check(contact)
    return validated


def validate_contact_energy(control,contact,start,end,report):
    error=contact_error(report)
    if 'boundedContactWork' not in report:
        raise ValueError('Missing bounded contact work at publication')
    original=encoded(report)
    expected=checked_change(control,contact,start,end)
    if encoded(report['boundedContactWork'])!=encoded(expected) or encoded(report)!=original:
        raise ValueError('Contact energy publication differs from fresh complete work')
    return error
