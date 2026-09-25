"""Bounded exact cable contraction; None selects original arithmetic.

Eligibility limits bound new scratch work, never reject the original problem.
Moment intervals are fresh inputs from the unchanged kernel. No moments or
responses survive a call. Each coefficient selects its own signed endpoints;
opposite coefficients on uncertain moments must not be combined in advance.
"""
from fractions import Fraction as F
from math import gcd

MAX_BITS=65536
MAX_ROWS=1024
MIN_ROWS=4
MAX_DEGREE=5
HALF,THREE_HALVES=F(-1,2),F(-3,2)


def _fraction(value):
    if type(value) is not F:return False
    n,d=value.numerator,value.denominator
    return (type(n) is int and type(d) is int and d>0
        and max(n.bit_length(),d.bit_length())<=MAX_BITS and gcd(n,d)==1)


def _product(a,b):
    if not a or not b:return 0
    if a.bit_length()+b.bit_length()>MAX_BITS:return None
    return a*b


def _sum(a,b):
    if not a:return b
    if not b:return a
    if max(a.bit_length(),b.bit_length())+1>MAX_BITS:return None
    return a+b


def _lcm(a,b):
    return _product(a,b//gcd(a,b))


def contract(local,moments,width):
    """Return exactly the old intervals, or None before using its fallback."""
    if (type(local) is not list or not MIN_ROWS<=len(local)<=MAX_ROWS
            or not _fraction(width) or width.numerator<=0):return None
    used=[];seen=set()
    for item in local:
        if type(item) is not tuple or len(item)!=4:return None
        key,*polys=item
        if (type(key) is not tuple or not key or type(key[0]) is not str
                or key[0] not in ('e','g','h') or len(key)!={'e':1,'g':2,'h':3}[key[0]]
                or any(type(v) is not int for v in key[1:])):return None
        if any(type(p) is not tuple or not 1<=len(p)<=MAX_DEGREE+1 for p in polys):return None
        if any(not _fraction(v) for p in polys for v in p):return None
        for power,poly in ((HALF,polys[1]),(THREE_HALVES,polys[2])):
            for i,v in enumerate(poly):
                # Exact canonical Fraction truthiness only; unused entries are
                # never accessed, including when all radial terms are zero.
                if v.numerator and (power,i) not in seen:
                    used.append((power,i));seen.add((power,i))
    pairs={};moment_denominator=1
    if used:
        if type(moments) is not dict or len(moments)>16 or any(type(k) is not str for k in moments):return None
        table=moments.get('moments')
        if type(table) is not dict or len(table)>4 or any(not _fraction(k) for k in table):return None
        for power,i in used:
            values=table.get(power)
            if type(values) is not tuple or i>=len(values):return None
            pair=values[i]
            if type(pair) is not tuple or len(pair)!=2 or any(not _fraction(v) for v in pair):return None
            pairs[(power,i)]=pair
            for v in pair:
                moment_denominator=_lcm(moment_denominator,v.denominator)
                if moment_denominator is None:return None
    basis={}
    for key,pair in pairs.items():
        scaled=[]
        for v in pair:
            value=_product(v.numerator,moment_denominator//v.denominator)
            if value is None:return None
            scaled.append(value)
        # The original _times_interval sorts even reversed supplied endpoints.
        basis[key]=(min(scaled),max(scaled))
    scaled_denominator=_product(moment_denominator,width.denominator)
    if scaled_denominator is None:return None
    result={}
    for key,polynomial,half,three in local:
        coefficient_denominator=1
        polynomial_terms=[];radial_terms=[]
        for i,v in enumerate(polynomial):
            if not v.numerator:continue
            denominator=_product(v.denominator,i+1)
            if denominator is None:return None
            coefficient_denominator=_lcm(coefficient_denominator,denominator)
            if coefficient_denominator is None:return None
            polynomial_terms.append((v.numerator,denominator))
        for power,poly in ((HALF,half),(THREE_HALVES,three)):
            for i,v in enumerate(poly):
                if not v.numerator:continue
                coefficient_denominator=_lcm(coefficient_denominator,v.denominator)
                if coefficient_denominator is None:return None
                radial_terms.append((v.numerator,v.denominator,basis[(power,i)]))
        denominator=_product(coefficient_denominator,scaled_denominator)
        if denominator is None:return None
        integral=0
        for numerator,divisor in polynomial_terms:
            term=_product(numerator,coefficient_denominator//divisor)
            if term is None:return None
            integral=_sum(integral,term)
            if integral is None:return None
        lower=_product(integral,moment_denominator)
        if lower is None:return None
        upper=lower
        for numerator,divisor,(lo,hi) in radial_terms:
            coefficient=_product(numerator,coefficient_denominator//divisor)
            if coefficient is None:return None
            if coefficient<0:lo,hi=hi,lo
            left,right=_product(coefficient,lo),_product(coefficient,hi)
            if left is None or right is None:return None
            lower,upper=_sum(lower,left),_sum(upper,right)
            if lower is None or upper is None:return None
        lower,upper=_product(lower,width.numerator),_product(upper,width.numerator)
        if lower is None or upper is None:return None
        result[key]=F(lower,denominator),F(upper,denominator)
    return result
