"""Check deployment tensors against original K-Prism preprocessing code."""
import ast
import importlib.util
from pathlib import Path
import unittest

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import SimpleITK as sitk
import torch

from server import cardiac_model as cm

ORIGINAL = Path(__file__).resolve().parents[2] / 'Verse_cfg'


def original_definitions(path, names, scope):
    tree = ast.parse(path.read_text())
    subset = ast.Module(body=[node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names], type_ignores=[])
    exec(compile(subset, str(path), 'exec'), scope)
    return scope


class CardiacInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_query_and_reference_match_original_test_transforms_exactly(self):
        scope = dict(vars(A), ToTensorV2=ToTensorV2)
        original_definitions(ORIGINAL / 'Verse_2_MOE/data_mapper/test_data_utils.py',
                             ['build_test_image_transform', 'build_ref_image_transform'], scope)
        prepared = np.random.default_rng(10).uniform(0, 255, (256, 256)).astype(np.float32)
        image = torch.from_numpy(prepared)[None].repeat(3, 1, 1)
        resized = torch.nn.functional.interpolate(image[None], (512, 512), mode='bilinear', align_corners=False)[0]
        expected = scope['build_test_image_transform']()(image=resized.numpy().transpose(1, 2, 0).astype('uint8'))['image']
        torch.testing.assert_close(cm.image_tensor(prepared), expected, rtol=0, atol=0)
        expected_ref = scope['build_ref_image_transform']()(image=image.numpy().transpose(1, 2, 0).astype('uint8'))['image']
        torch.testing.assert_close(cm.image_tensor(prepared, reference=True), expected_ref, rtol=0, atol=0)

    def test_raw_preparation_matches_original_slice_clip_and_physical_resize(self):
        scope = {'np': np, 'sitk': sitk}
        original_definitions(ORIGINAL / 'datasets/utils/nii_norm.py', ['normalize_data_2d'], scope)
        original_definitions(ORIGINAL / 'datasets/utils/nii_resize.py', ['resample_sitkImage_by_size'], scope)
        raw = np.random.default_rng(0).uniform(-40, 1500, (3, 103, 147)).astype(np.float64)
        raw[0].fill(30)
        normalized = scope['normalize_data_2d'](raw.copy())
        image = sitk.GetImageFromArray(normalized)
        image.SetSpacing([1.37, 2.41, 8.])
        resized = scope['resample_sitkImage_by_size'](image, [256, 256, 3])
        original = sitk.GetArrayFromImage(resized).astype(np.float32)
        original = (original-original.min())/(original.max()-original.min())*255.
        np.testing.assert_array_equal(cm.prepare_frame(raw, (1.37, 2.41, 8.)), original)
        self.assertTrue(np.isfinite(cm.prepare_frame(np.zeros((2, 55, 71)))).all())

    def test_native_rectangular_mask_and_click_align_on_model_grid(self):
        mask = np.zeros((96, 160), np.uint8)
        mask[30:41, 70:81] = 1
        prior = cm.mask_prior(mask)
        (y, x), = cm.map_points([{'x': 75, 'y': 35}], mask.shape)
        self.assertEqual(prior.shape, (1, 512, 512))
        self.assertEqual(prior[0, y, x], 1)
        edge = cm.map_points([{'x': 159.9, 'y': 95.9}], mask.shape)[0]
        self.assertTrue(all(0 <= v < 512 for v in edge))

    def test_known_prepared_example_does_not_get_clipped_twice(self):
        from server.inference import ModelService
        model = ModelService()
        frame = np.random.default_rng(2).uniform(0, 255, (2, 256, 256)).astype(np.float32)
        result = model.prepare_plane(frame, 1, ('prepared-fixture', 0), preprocessed=True)
        np.testing.assert_array_equal(result.prepared, frame[1])

    def test_output_geometry_matches_original_sigmoid_resize(self):
        logits = torch.randn(1, 1, 128, 128)
        F = torch.nn.functional
        expected = F.interpolate(F.interpolate(logits, (512, 512), mode='bilinear', align_corners=False).sigmoid(),
                                 (256, 256), mode='bilinear', align_corners=False)[0, 0].numpy()
        np.testing.assert_array_equal(cm.restore_probability(logits, (256, 256)), expected)
        result = cm.restore_probability(logits, (97, 143))
        self.assertEqual(result.shape, (97, 143))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(((result >= 0) & (result <= 1)).all())


if __name__ == '__main__':
    unittest.main()
