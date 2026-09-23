"""Fresh source/native fold binding and static mechanical direction checks."""

import copy
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parent
os.environ.setdefault("WARP_CACHE_PATH", str(SCRIPTS.parent / ".planning/solver/warp-cache"))

import newton
import numpy as np
import warp as wp

from solver_binding_fold import (ANGLE_CONVENTION, REQUEST_PROFILE, STIFFNESS_MEASURE, _length,
    bind_binding_fold_control, build_binding_fold_control, validate_binding_fold_control)
from solver_binding_source import build_binding_source


INSTANCE = "opening_binding_left_left:shell"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def rat(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def request():
    return {"profile": REQUEST_PROFILE, "accepted": False, "angleConvention": ANGLE_CONVENTION,
        "stiffnessMeasure": STIFFNESS_MEASURE,
        "folds": [{"sourcePathName": name, "referenceRotationRegion": region,
            "targetRightHandAngleRadians": math.radians(3.), "stiffnessJoulesPerMeter": .1}
            for name, region in (("right", "body"), ("left", "allowance"))]}


def model_for(source):
    rest, faces = np.asarray(source["restMeters"]), np.asarray(source["triangles"]).reshape(-1, 3)
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    ordered = sorted(source["instanceOffsets"].items(), key=lambda item: item[1])
    for index, (_, start) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else len(rest)
        selected = faces[np.all((faces >= start) & (faces < end), axis=1)] - start
        builder.add_cloth_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1,
            vel=wp.vec3(0, 0, 0), vertices=rest[start:end].tolist(), indices=selected.ravel().tolist(),
            density=.2, tri_ke=10000, tri_ka=10000, tri_kd=0, edge_ke=.01, edge_kd=0)
    builder.set_coloring([[vertex] for vertex in range(len(rest))])
    model = builder.finalize(device="cpu")
    return model, {"vertexCount": len(model.particle_mass.numpy()), "triangles": model.tri_indices.numpy().tolist(),
                   "hinges": model.edge_indices.numpy().tolist()}


class BindingFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name)
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        for script, arguments in (("prepare-cuff-source.py", ["--output", root / "parent"]),
                ("prepare-cuff-construction.py", ["--source-canonical", root / "parent/canonical.json",
                    "--side", "left", "--attachment-policy", "inner-facing-first-outer-shell-last-v1", "--output", root / "unit"])):
            completed = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, arguments)],
                cwd=root, env=environment, capture_output=True, text=True, timeout=60)
            if completed.returncode:
                raise AssertionError(completed.stdout + completed.stderr)
        cls.source = build_binding_source(json.loads((root / "unit/unit.json").read_bytes()))
        cls.model, cls.native = model_for(cls.source)
        cls.request = request()
        cls.descriptor = build_binding_fold_control(cls.source, cls.request, cls.native)

    def test_full_cut_chains_native_order_and_source_directed_regions(self):
        descriptor = self.descriptor
        self.assertEqual([len(rail["segments"]) for rail in descriptor["rails"]], [7, 9])
        for rail in descriptor["rails"]:
            self.assertEqual(set(rail["bodyFacesLocal"]) | set(rail["allowanceFacesLocal"]), set(range(44)))
            self.assertFalse(set(rail["bodyFacesLocal"]) & set(rail["allowanceFacesLocal"]))
            self.assertEqual(set(rail["bodyVerticesOffChainLocal"]) | set(rail["allowanceVerticesOffChainLocal"])
                | set(rail["chainVerticesLocal"]), set(range(29)))
            for segment in rail["segments"]:
                self.assertEqual(segment["nativeOrderedHingeCanonical"], self.native["hinges"][segment["nativeHingeIndex"]])
                self.assertEqual(segment["nativeEdgeDirectionSign"], -1)
                self.assertEqual(segment["nativeOppositeOrderSign"], -1)
                self.assertEqual(segment["nativeAngleSign"], 1)
        self.assertEqual(descriptor["rails"][0]["sourceTangent"], [0., 1., 0.])
        self.assertEqual(descriptor["rails"][1]["sourceTangent"], [0., -1., 0.])
        for key in ("accepted", "solverReady", "executable"):
            self.assertIs(descriptor[key], False)

    def test_actual_primitive_binds_all_sixteen_ordered_native_hinges(self):
        primitive, evidence = bind_binding_fold_control(self.source, self.descriptor, self.model)
        self.assertTrue(evidence["verified"])
        self.assertEqual(evidence["hingeCount"], 16)
        np.testing.assert_array_equal(primitive.hinges, self.descriptor["foldActuation"]["hinges"])
        np.testing.assert_array_equal(primitive.potential([0.] * 16).angles(self.source["restMeters"]), np.zeros(16))
        changed = copy.deepcopy(self.native)
        changed["hinges"].reverse()
        reordered = build_binding_fold_control(self.source, self.request, changed)
        # Reordering a full valid declaration changes its native identity; it
        # cannot claim to describe this unchanged live native model.
        with self.assertRaises(ValueError):
            bind_binding_fold_control(self.source, reordered, self.model)

    def test_static_reference_region_rotations_match_both_rail_signs(self):
        primitive, _ = bind_binding_fold_control(self.source, self.descriptor, self.model)
        rest = np.asarray(self.source["restMeters"])
        offset = self.source["instanceOffsets"][INSTANCE]
        for rail_index, rail in enumerate(self.descriptor["rails"]):
            for region in ("body", "allowance"):
                for degrees in (-3., 3.):
                    origin = np.array(rail["sourceLineEndpointsMeters"][0] + [0.])
                    tangent = np.array(rail["sourceTangent"])
                    moving = np.array(rail[region + "VerticesOffChainLocal"]) + offset
                    angle = math.radians(degrees)
                    delta = rest[moving] - origin
                    pose = rest.copy()
                    pose[moving] = (origin + delta * math.cos(angle) + np.cross(tangent, delta) * math.sin(angle)
                        + (delta @ tangent)[:, None] * tangent * (1 - math.cos(angle)))
                    observed = primitive.potential([0.] * 16).angles(pose)
                    expected = np.zeros(16)
                    indices = [segment["actuatorIndex"] for segment in rail["segments"]]
                    expected[indices] = angle * (-1 if region == "body" else 1)
                    with self.subTest(rail=rail_index, region=region, degrees=degrees):
                        np.testing.assert_allclose(observed, expected, atol=2e-15, rtol=0)
                        np.testing.assert_array_equal(pose[rail["chainVerticesCanonical"]], rest[rail["chainVerticesCanonical"]])
                        np.testing.assert_array_equal(pose[:offset], rest[:offset])
                        np.testing.assert_array_equal(pose[offset + 29:], rest[offset + 29:])

    def test_line_control_energy_virtual_work_and_balanced_free_region_reactions(self):
        primitive, _ = bind_binding_fold_control(self.source, self.descriptor, self.model)
        positions = np.asarray(self.source["restMeters"])
        recipe = self.descriptor["foldActuation"]
        potential = primitive.potential(recipe["targetAnglesRadians"])
        gradient = potential.gradient(positions)
        expected_energy = .5 * math.fsum(k * target ** 2 for k, target in zip(recipe["stiffnessJoules"], recipe["targetAnglesRadians"]))
        self.assertAlmostEqual(potential.energy(positions), expected_energy, places=17)
        np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=2e-14, rtol=0)
        np.testing.assert_allclose(np.cross(positions, gradient).sum(axis=0), 0, atol=2e-14, rtol=0)
        for rail in self.descriptor["rails"]:
            for field in ("body", "allowance"):
                opposite = [segment["sourceDirectedHingeCanonical"][0 if field == "body" else 1] for segment in rail["segments"]]
                self.assertGreater(np.linalg.norm(gradient[opposite]), 0)
            tangent, origin = np.array(rail["sourceTangent"]), np.array(rail["sourceLineEndpointsMeters"][0] + [0.])
            moving = np.array(rail[rail["referenceRotationRegion"] + "VerticesOffChainLocal"]) + self.source["instanceOffsets"][INSTANCE]
            velocity = np.cross(tangent, positions[moving] - origin)
            virtual_work_derivative = math.fsum(float(value) for value in (gradient[moving] * velocity).ravel())
            angle = self.request["folds"][self.descriptor["rails"].index(rail)]["targetRightHandAngleRadians"]
            expected = -angle * math.fsum(segment["numericalStiffnessJoules"] for segment in rail["segments"])
            self.assertAlmostEqual(virtual_work_derivative, expected, places=14)

    def test_sequential_reference_folds_transport_finish_axis_without_mesh_repair(self):
        primitive, _ = bind_binding_fold_control(self.source, self.descriptor, self.model)
        rest = np.asarray(self.source["restMeters"])
        right, left = self.descriptor["rails"]
        offset = self.source["instanceOffsets"][INSTANCE]
        angle = math.radians(3.)
        def rotate(points, origin, axis, theta):
            delta = points - origin
            return (origin + delta * math.cos(theta) + np.cross(axis, delta) * math.sin(theta)
                + (delta @ axis)[..., None] * axis * (1 - math.cos(theta)))
        right_origin = np.array(right["sourceLineEndpointsMeters"][0] + [0.])
        right_axis = np.array(right["sourceTangent"])
        body = np.array(right["bodyVerticesOffChainLocal"]) + offset
        first = rest.copy()
        first[body] = rotate(rest[body], right_origin, right_axis, angle)
        np.testing.assert_array_equal(first[right["chainVerticesCanonical"]], rest[right["chainVerticesCanonical"]])
        np.testing.assert_allclose(primitive.potential([0.] * 16).angles(first), [-angle] * 7 + [0.] * 9, atol=2e-15, rtol=0)
        # The second source rail was carried with the body by the first fold.
        flat_line = np.array([point + [0.] for point in left["sourceLineEndpointsMeters"]])
        transported = rotate(flat_line, right_origin, right_axis, angle)
        finish_axis = transported[1] - transported[0]
        finish_axis /= np.linalg.norm(finish_axis)
        wing = np.array(left["allowanceVerticesOffChainLocal"]) + offset
        final = first.copy()
        final[wing] = rotate(first[wing], transported[0], finish_axis, -angle)
        np.testing.assert_array_equal(final[left["chainVerticesCanonical"]], first[left["chainVerticesCanonical"]])
        np.testing.assert_array_equal(final[:offset], rest[:offset])
        np.testing.assert_array_equal(final[offset + 29:], rest[offset + 29:])
        np.testing.assert_allclose(primitive.potential([0.] * 16).angles(final), [-angle] * 16, atol=2e-15, rtol=0)
        faces = np.array(self.source["numericalMeshes"][INSTANCE]["triangles"]) + offset
        edges = np.array(sorted({tuple(sorted((face[i], face[(i + 1) % 3]))) for face in faces for i in range(3)}))
        lengths = lambda points: np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
        np.testing.assert_allclose(lengths(final), lengths(rest), atol=1e-15, rtol=0)
        wrong = first.copy()
        wrong[wing] = rotate(first[wing], flat_line[0], np.array(left["sourceTangent"]), -angle)
        self.assertGreater(np.max(np.abs(lengths(wrong) - lengths(rest))), 1e-7)

    def test_length_midpoint_cells_and_density_product_residuals_are_exact(self):
        rest = self.source["restMeters"]
        for rail in self.descriptor["rails"]:
            ideals, numerical = [], []
            for segment in rail["segments"]:
                a, b = (rest[index] for index in segment["orientedEdgeCanonical"])
                square = sum(((Fraction(y) - Fraction(x)) ** 2 for x, y in zip(a, b)), Fraction())
                witness = segment["length"]
                self.assertEqual(rat(witness["exactSquaredLengthMetersSquared"]), square)
                self.assertLessEqual(rat(witness["lowerMidpointSquaredMetersSquared"]), square)
                self.assertLessEqual(square, rat(witness["upperMidpointSquaredMetersSquared"]))
                rounded = Fraction(witness["roundedLengthMeters"])
                self.assertEqual(rat(witness["roundedLengthSquaredMinusExactMetersSquared"]), rounded ** 2 - square)
                ideal = Fraction(rail["stiffnessJoulesPerMeter"]) * rounded
                observed = Fraction(segment["numericalStiffnessJoules"])
                self.assertEqual(rat(segment["idealStiffnessFromRoundedLengthJoules"]), ideal)
                self.assertEqual(rat(segment["stiffnessRoundingResidualJoules"]), observed - ideal)
                ideals.append(ideal)
                numerical.append(observed)
            self.assertEqual(rat(rail["totalStiffnessRoundingResidualJoules"]), sum(numerical) - sum(ideals))
            self.assertAlmostEqual(float(sum(numerical)), .0120672415, places=16)
        self.assertEqual(_length([0., 0., 0.], [3., 4., 0.])["roundedLengthMeters"], 5.)

    def test_missing_duplicate_misoriented_and_wrong_edge_native_rows_reject(self):
        index = self.descriptor["rails"][0]["segments"][0]["nativeHingeIndex"]
        mutations = [lambda n: n["hinges"].pop(),
            lambda n: n["hinges"].__setitem__(index, list(n["hinges"][0])),
            lambda n: n["hinges"][index].__setitem__(slice(0, 2), n["hinges"][index][:2][::-1]),
            lambda n: n["hinges"][index].__setitem__(slice(2, 4), n["hinges"][index][2:][::-1]),
            lambda n: n["hinges"][index].__setitem__(0, -1),
            lambda n: n["hinges"][index].__setitem__(0, True),
            lambda n: n["triangles"][0].reverse(), lambda n: n.update(vertexCount=244)]
        for mutation in mutations:
            native = copy.deepcopy(self.native)
            mutation(native)
            with self.subTest(mutation=mutations.index(mutation)), self.assertRaises(ValueError):
                build_binding_fold_control(self.source, self.request, native)

    def test_request_angles_density_units_and_positive_underflow_reject(self):
        mutations = [lambda r: r.update(accepted=0), lambda r: r.update(stiffnessMeasure="per-hinge"),
            lambda r: r.update(angleConvention="unsigned"), lambda r: r["folds"].append(copy.deepcopy(r["folds"][0])),
            lambda r: r["folds"][0].update(referenceRotationRegion="inferred"),
            lambda r: r["folds"][0].update(targetRightHandAngleRadians=True),
            lambda r: r["folds"][0].update(targetRightHandAngleRadians=math.pi),
            lambda r: r["folds"][0].update(stiffnessJoulesPerMeter=0.),
            lambda r: r["folds"][0].update(stiffnessJoulesPerMeter=1e7),
            lambda r: r["folds"][0].update(stiffnessJoulesPerMeter=math.ulp(0.))]
        for mutation in mutations:
            requested = copy.deepcopy(self.request)
            mutation(requested)
            with self.subTest(mutation=mutations.index(mutation)), self.assertRaises(ValueError):
                build_binding_fold_control(self.source, requested, self.native)

    def test_strict_rederivation_detects_repaired_control_and_frame_witnesses(self):
        mutations = [lambda d: d.update(accepted=True), lambda d: d.update(nativeTopologySha256="0" * 64),
            lambda d: d["rails"][0]["bodyFacesLocal"].pop(),
            lambda d: d["rails"][0]["segments"][0].update(nativeAngleSign=-1),
            lambda d: d["rails"][0]["segments"][0]["length"].update(roundedLengthMeters=.01),
            lambda d: d["foldActuation"]["targetAnglesRadians"].__setitem__(0, 0.),
            lambda d: d["foldActuation"]["stiffnessJoules"].__setitem__(0, 1.),
            lambda d: d["rails"][0]["storedRailOffsetsMeters"][0].update(numerator="123")]
        for mutation in mutations:
            descriptor = copy.deepcopy(self.descriptor)
            mutation(descriptor)
            self.assertNotEqual(encoded(descriptor), encoded(self.descriptor))
            with self.subTest(mutation=mutations.index(mutation)), self.assertRaises(ValueError):
                validate_binding_fold_control(self.source, descriptor)

    def test_inputs_rest_geometry_and_seam_rows_remain_unchanged_without_aliases(self):
        before = encoded([self.source, self.request, self.native])
        descriptor = validate_binding_fold_control(self.source, self.descriptor)
        descriptor["request"]["folds"].clear()
        descriptor["nativeTopology"]["hinges"].clear()
        descriptor["foldActuation"]["hinges"].clear()
        self.assertEqual(encoded([self.source, self.request, self.native]), before)
        self.assertEqual(len(self.source["embeddedConstraints"]["constraints"]), 40)
        self.assertEqual((len(self.source["restMeters"]), len(self.source["triangles"]) // 3), (245, 398))
        self.assertNotIn("foldActuation", self.source)
        for field in ("placedMeters", "foldActuation", "gripperActuation", "sewingActuation"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_binding_fold_control({**self.source, field: None}, self.request, self.native)


if __name__ == "__main__":
    unittest.main()
