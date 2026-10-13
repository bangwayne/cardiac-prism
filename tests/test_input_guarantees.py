"""Training recipe, trusted source identity, and finite inference boundaries."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from server import cardiac_model as cm
from server.input_profile import build_input_profile, source_sha256
import test_platform as fixtures


class InputGuaranteeTests(unittest.TestCase):
    def test_negative_extreme_and_constant_native_inputs_produce_finite_bounded_tensors(self):
        rng = np.random.default_rng(23)
        for raw in (rng.normal(-400, 1700, (2, 41, 67)),
                    rng.uniform(-1e30, 1e30, (2, 41, 67)),
                    np.zeros((2, 41, 67)), np.full((2, 41, 67), -123.)):
            original = raw.copy()
            prepared = cm.prepare_frame(raw, (1.2, .9, 7.))
            self.assertEqual(prepared.shape, (2, 256, 256))
            for plane in prepared:
                for reference in (False, True):
                    tensor = cm.image_tensor(plane, reference=reference).numpy()
                    self.assertEqual(tensor.shape, (3, 512, 512))
                    self.assertTrue(np.isfinite(tensor).all())
                    self.assertGreaterEqual(float(tensor.min()), cm.MODEL_INPUT_RANGE[0] - 1e-5)
                    self.assertLessEqual(float(tensor.max()), cm.MODEL_INPUT_RANGE[1] + 1e-5)
            np.testing.assert_array_equal(raw, original)

    def test_invalid_prepared_values_cannot_wrap_via_uint8(self):
        for bad in (np.nan, np.inf, -.1, 255.1):
            plane = np.zeros((256, 256), np.float32)
            plane[0, 0] = bad
            for reference in (False, True):
                with self.assertRaisesRegex(ValueError, 'finite.*range'):
                    cm.image_tensor(plane, reference=reference)
        with self.assertRaises(ValueError):
            cm.image_tensor(np.zeros((128, 512), np.float32))

    def test_invalid_raw_values_and_geometry_are_rejected_without_replacing_them(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                cm.prepare_frame(np.full((1, 11, 13), bad))
        for spacing in ((1, 0, 1), (1, np.nan, 1)):
            with self.assertRaisesRegex(ValueError, 'spacing'):
                cm.prepare_frame(np.zeros((1, 11, 13)), spacing)

    def test_profile_uses_actual_content_identity_not_intensity_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'same-name.nii.gz'
            source.write_bytes(b'an unfamiliar uploaded source')
            voxels = np.random.default_rng(1).uniform(0, 255, (2, 3, 256, 256)).astype(np.float32)
            with patch('server.input_profile.prepared_source_hashes', return_value=set()), \
                    patch.object(cm, 'prepare_frame', side_effect=AssertionError('must not prepare whole cine')):
                profile = build_input_profile(source, voxels)
            self.assertEqual(profile['input_state'], 'raw')
            self.assertTrue(profile['source_identity_verified'])
            self.assertLessEqual(profile['raw_stats']['sample_count'], 250_000)
            self.assertEqual(profile['source_sha256'], source_sha256(source))
            actual_hash = source_sha256(source)
            with patch('server.input_profile.prepared_source_hashes', return_value={actual_hash}):
                self.assertEqual(build_input_profile(source, voxels)['input_state'], 'trusted_prepared')
                source.write_bytes(b'different contents with the same filename')
                self.assertEqual(build_input_profile(source, voxels)['input_state'], 'raw')

    def test_profile_survives_save_and_is_reverified_on_open(self):
        from server.domain import ProjectStore
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = np.arange(3 * 27 * 43, dtype=np.float32).reshape(3, 27, 43) - 100.
            source = root / 'native.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(raw), str(source))
            store = ProjectStore(root / 'work')
            project = store.create(source, 'native')
            profile = project.input_profile
            reopened = ProjectStore(root / 'work').get(project.id)
            self.assertEqual(reopened.input_profile, profile)
            self.assertEqual(profile['input_state'], 'raw')
            np.testing.assert_array_equal(reopened.voxels[0], raw)
            self.assertEqual(reopened.prior_store.reads, 0)


class InputProfileBackupTests(unittest.TestCase):
    setUp = fixtures.PlatformTests.setUp
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post

    @patch('server.input_profile.prepared_source_hashes', return_value={'a' * 64})
    def test_backup_cannot_spoof_prepared_state_through_provenance_or_saved_profile(self, catalog_hashes):
        expected_sha = source_sha256(self.project.source_path)
        claimed_sha = 'a' * 64
        self.project.provenance = {'sha256': claimed_sha}
        self.project.input_profile.update(input_state='trusted_prepared', source_sha256=claimed_sha)
        backup = self.client.get(self.prefix + '/export?kind=study')
        self.assertEqual(backup.status_code, 200)
        restored = self.client.post('/api/projects/import-backup', files={'file': ('study.zip', backup.content)})
        self.assertEqual(restored.status_code, 200, restored.text)
        profile = restored.json()['input_profile']
        self.assertEqual(profile['input_state'], 'raw')
        self.assertEqual(profile['source_sha256'], expected_sha)
        self.assertNotEqual(profile['source_sha256'], claimed_sha)
        np.testing.assert_array_equal(self.app.state.store.get(restored.json()['id']).voxels, self.project.voxels)


if __name__ == '__main__':
    unittest.main()
