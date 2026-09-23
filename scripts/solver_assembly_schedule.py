import math


class AssemblySchedule:
    def __init__(self, recipe, subdivisions):
        if (not isinstance(recipe, dict) or set(recipe) != {"profile", "knots"}
                or recipe["profile"] != "sewing-fold-progress-v1"
                or not isinstance(recipe["knots"], list) or not 2 <= len(recipe["knots"]) <= 65):
            raise ValueError("Bounded explicit sewing/fold schedule required")
        if type(subdivisions) is not int or not 1 <= subdivisions <= 4096 or subdivisions & (subdivisions - 1):
            raise ValueError("Dyadic schedule subdivisions required")
        self.knots = []
        for knot in recipe["knots"]:
            if (not isinstance(knot, dict) or set(knot) != {"fraction", "sewingProgress", "foldProgress"}
                    or any(type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1
                           for value in knot.values())):
                raise ValueError("Finite schedule fractions and progress in [0, 1] required")
            values = tuple(float(knot[key]) for key in ("fraction", "sewingProgress", "foldProgress"))
            if not (values[0] * subdivisions).is_integer():
                raise ValueError("Schedule knots must coincide with initial interval boundaries")
            if self.knots and (values[0] <= self.knots[-1][0]
                              or any(values[index] < self.knots[-1][index] for index in (1, 2))):
                raise ValueError("Strictly ordered time and nondecreasing operation progress required")
            self.knots.append(values)
        if self.knots[0] != (0., 0., 0.) or self.knots[-1] != (1., 1., 1.):
            raise ValueError("Schedule must span both complete captured target ramps")

    def progress(self, fraction):
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Schedule evaluation must stay in [0, 1]")
        for lower, upper in zip(self.knots[:-1], self.knots[1:]):
            if fraction <= upper[0]:
                amount = (fraction - lower[0]) / (upper[0] - lower[0])
                return tuple(lower[index] + amount * (upper[index] - lower[index]) for index in (1, 2))
        raise ValueError("Schedule does not cover requested fraction")
