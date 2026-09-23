"""Constructive source-cuff controls for explicit crease frame semantics.

These endpoint controls do not establish a dynamic path, a calibrated contact
model, turning, or garment acceptance. Source metrics and seam anchors are fixed.
"""

import copy
import unittest

import ipctk
import numpy as np
from scipy.sparse import coo_matrix

from solver_bending import ElasticDihedralBending
from solver_cuff_sequence import combine_cuff_controls
from solver_normal_sewing import NormalOffsetSewing
from solver_rest_filtered_contact import RestFilteredSurfaceContact
import test_solver_cuff_compatibility as source_fixture


class CuffFrameSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ipctk.set_num_threads(1)
        source_fixture.CuffCompatibilityTests.setUpClass()
        cls.source = source_fixture.CuffCompatibilityTests()
        cls.body = combine_cuff_controls(cls.source.sewing_source, cls.source.fold_source,
                                         crease_frame_region="body")
        cls.allowance = combine_cuff_controls(cls.source.sewing_source, cls.source.fold_source,
                                              crease_frame_region="allowance")

    @staticmethod
    def sewing_potential(control):
        offsets = control["instanceOffsets"]
        entries = [(row_id, offsets[term["instanceId"]] + term["vertex"], term["coefficient"])
                   for row_id, row in enumerate(control["embeddedConstraints"]["constraints"])
                   for term in row["terms"]]
        matrix = coo_matrix(([entry[2] for entry in entries],
                             ([entry[0] for entry in entries], [entry[1] for entry in entries])),
                            shape=(20, 66)).tocsr()
        return NormalOffsetSewing(matrix, np.full(20, .0005), 1e-8,
                                  control["sewingFrames"]["faces"], control["sewingFrames"]["sides"])

    def test_frame_choice_changes_only_explicit_frames_and_retains_material_anchors(self):
        sewing, fold = copy.deepcopy(self.source.sewing_source), copy.deepcopy(self.source.fold_source)
        sewing_before, fold_before = copy.deepcopy(sewing), copy.deepcopy(fold)
        combine_cuff_controls(sewing, fold, crease_frame_region="body")
        self.assertEqual(sewing, sewing_before)
        self.assertEqual(fold, fold_before)
        body = {key: value for key, value in self.body.items() if key != "sewingFrames"}
        allowance = {key: value for key, value in self.allowance.items() if key != "sewingFrames"}
        self.assertEqual(body, allowance)
        changed = [i for i, pair in enumerate(zip(self.body["sewingFrames"]["faces"],
                                                   self.allowance["sewingFrames"]["faces"])) if pair[0] != pair[1]]
        self.assertEqual(changed, [9, 10, 11, 12, 13, 14, 15])
        rest = np.asarray(self.body["restMeters"])
        for control, region in ((self.body, "body"), (self.allowance, "allowance")):
            for row_id in changed:
                binding = control["sewingFrames"]["bindings"][row_id]
                self.assertTrue(binding["onCrease"])
                self.assertEqual(binding["region"], region)
                face = control["sewingFrames"]["faces"][row_id]
                y = rest[face, 1]
                self.assertEqual(bool(np.all(y <= .055 + 1e-15)), region == "body")

    def test_body_frames_admit_rigid_fold_endpoint_without_changing_rest_metric(self):
        panel = self.source.folded_panel(1.2)
        positions = np.vstack((panel, panel + [0., 0., .0005]))
        rest = np.asarray(self.body["restMeters"])
        faces = np.asarray(self.body["triangles"]).reshape((-1, 3))
        edges = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2)), axis=1), axis=0)
        np.testing.assert_allclose(np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1),
                                   np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1), rtol=0, atol=1e-16)
        recipe = self.body["foldActuation"]
        # The investigated coupled run commands 1.2 radians; the extractor
        # fixture's later standalone fold target is not changed by this test.
        commanded_angles = np.full(len(recipe["hinges"]), 1.2)
        actuator = ElasticDihedralBending(len(rest), recipe["hinges"], commanded_angles,
                                          np.ones(len(recipe["hinges"])), recipe["stiffnessJoules"])
        np.testing.assert_allclose(actuator.angles(positions), commanded_angles, rtol=0, atol=2e-14)
        body_sewing = self.sewing_potential(self.body)
        allowance_sewing = self.sewing_potential(self.allowance)
        body_error = body_sewing.residual(positions).reshape((-1, 3)) * np.sqrt(body_sewing.compliance)
        allowance_error = allowance_sewing.residual(positions).reshape((-1, 3)) * np.sqrt(allowance_sewing.compliance)
        np.testing.assert_allclose(body_error, 0., rtol=0, atol=1e-15)
        self.assertAlmostEqual(float(np.max(np.linalg.norm(allowance_error, axis=1))), .001 * np.sin(.6), places=15)
        contact = RestFilteredSurfaceContact(rest, faces, activation_distance_m=.001,
            minimum_distance_m=.0001, stiffness=10000.)
        contact.validate_state(positions)
        self.assertGreater(.0005 * np.cos(1.2), .0001)
        # An admissible endpoint is still strongly resisted by this uncalibrated
        # pressure law. Frame semantics alone do not imply the targets are reached.
        self.assertGreater(contact.energy(positions), 1.)

    def test_allowance_frames_introduce_tip_rotation_work_without_tip_anchor_weights(self):
        panel = self.source.folded_panel(.6)
        positions = np.vstack((panel, panel + [0., 0., .0005]))
        potential = self.sewing_potential(self.allowance)
        tips = np.flatnonzero(self.source.panel[:, 1] > .055 + 1e-15)
        self.assertEqual(np.count_nonzero(potential.sewing[:, np.r_[tips, tips + 33]].toarray()), 0)
        gradient = potential.gradient(positions).reshape((-1, 3))
        np.testing.assert_array_equal(gradient[tips], 0.)
        self.assertGreater(float(np.linalg.norm(gradient[tips + 33])), 1.)
        body_gradient = self.sewing_potential(self.body).gradient(positions).reshape((-1, 3))
        np.testing.assert_array_equal(body_gradient[np.r_[tips, tips + 33]], 0.)


if __name__ == "__main__":
    unittest.main()
