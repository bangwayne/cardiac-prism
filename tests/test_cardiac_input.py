"""Check deployment tensors against the fixed cardiac training recipe."""
import unittest

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import SimpleITK as sitk
import torch

from server import cardiac_model as cm

class CardiacInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_query_and_reference_match_original_test_transforms_exactly(self):
        query_transform = A.Compose([
            A.Normalize(mean=(.26, .26, .26), std=(.30, .30, .30), max_pixel_value=255.),
            ToTensorV2(transpose_mask=True),
        ])
        reference_transform = A.Compose([
            A.LongestMaxSize(max_size=512),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, value=0, mask_value=0),
            A.Normalize(mean=(.26, .26, .26), std=(.30, .30, .30), max_pixel_value=255.),
            ToTensorV2(transpose_mask=True),
        ])
        prepared = np.random.default_rng(10).uniform(0, 255, (256, 256)).astype(np.float32)
        image = torch.from_numpy(prepared)[None].repeat(3, 1, 1)
        resized = torch.nn.functional.interpolate(image[None], (512, 512), mode='bilinear', align_corners=False)[0]
        expected = query_transform(image=resized.numpy().transpose(1, 2, 0).astype('uint8'))['image']
        torch.testing.assert_close(cm.image_tensor(prepared), expected, rtol=0, atol=0)
        expected_ref = reference_transform(image=image.numpy().transpose(1, 2, 0).astype('uint8'))['image']
        torch.testing.assert_close(cm.image_tensor(prepared, reference=True), expected_ref, rtol=0, atol=0)

    def test_raw_preparation_matches_original_slice_clip_and_physical_resize(self):
        raw = np.random.default_rng(0).uniform(-40, 1500, (3, 103, 147)).astype(np.float64)
        raw[0].fill(30)
        normalized = raw.copy()
        for plane in normalized:
            lower, upper = np.min(plane), np.percentile(plane, 98)
            plane[plane > upper] = upper
            plane[:] = (plane - lower) / (upper - lower + .0001)
        image = sitk.GetImageFromArray(normalized)
        image.SetSpacing([1.37, 2.41, 8.])
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([256, 256, 3])
        resampler.SetOutputSpacing((np.array(image.GetSize()) * np.array(image.GetSpacing()) / [256, 256, 3]).tolist())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(float(normalized.min()))
        resized = resampler.Execute(image)
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
