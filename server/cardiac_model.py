"""Inference-only adapter for the versioned cardiac fine-tuning recipe.

Native studies/masks stay untouched. Only model inputs use the training grid.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / 'weight/cardiac-prism-v0.1.pth'
PROTOCOL = 'mm_slice_p0_p98_256_v1'
MODEL_INPUT_RANGE = (-.26 / .30, (1. - .26) / .30)


def validate_prepared(values, dimensions=2):
    """Reject invalid declarations before uint8 conversion can wrap values."""
    values = np.asarray(values)
    if (values.ndim != dimensions or not values.size or values.shape[-2:] != (256, 256)
            or not np.isfinite(values).all() or values.min() < 0 or values.max() > 255):
        raise ValueError('Prepared cardiac input must be finite on the 256 grid in the range 0 to 255.')
    return values


def checkpoint_path():
    return Path(os.environ.get('CARDIAC_PRISM_CHECKPOINT', DEFAULT_CHECKPOINT)).resolve()


def model_profile():
    path = checkpoint_path().with_suffix('.json')
    return json.loads(path.read_text()) if path.is_file() else {}


@dataclass
class CardiacPlane:
    prepared: np.ndarray
    shape: tuple

    def copy(self):
        return CardiacPlane(self.prepared.copy(), self.shape)


def prepare_frame(frame, spacing=(1., 1., 1.)):
    """Exactly match prepare_volume for an immutable native Z,Y,X frame."""
    values = np.array(frame, dtype=np.float64, copy=True)
    if values.ndim != 3 or not values.size or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError('Expected a finite Z,Y,X image frame.')
    if len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError('Frame spacing must contain three finite positive values.')
    low = values.min(axis=(1, 2), keepdims=True)
    high = np.percentile(values, 98, axis=(1, 2), keepdims=True)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        try:
            values = (np.clip(values, low, high) - low) / (high-low+.0001)
        except FloatingPointError as error:
            raise ValueError('Source intensity scale exceeds finite cardiac preprocessing arithmetic.') from error
    if values.shape[1:] != (256, 256):
        image = sitk.GetImageFromArray(values)
        image.SetSpacing([float(v) for v in spacing])
        depth, height, width = values.shape
        image = sitk.Resample(image, [256, 256, depth], sitk.Transform(), sitk.sitkLinear,
                              [0., 0., 0.], (np.array(image.GetSize())*np.array(spacing)/[256, 256, depth]).tolist(), image.GetDirection(),
                              float(values.min()), image.GetPixelID())
        values = sitk.GetArrayFromImage(image)
    values = values.astype(np.float32)
    lo, hi = float(values.min()), float(values.max())
    prepared = (values-lo)/(hi-lo)*255. if hi > lo else np.zeros_like(values)
    validate_prepared(prepared, dimensions=3)
    return prepared


def image_tensor(prepared, reference=False):
    import torch
    prepared = validate_prepared(prepared)
    if reference:
        # Original reference transform: uint8 -> Albumentations/OpenCV resize.
        rgb = np.repeat(prepared.astype(np.uint8)[..., None], 3, axis=2)
        rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    else:
        # Original test query: Torch float resize -> uint8 -> Normalize.
        import torch.nn.functional as F
        source = torch.from_numpy(prepared.copy())[None].repeat(3, 1, 1)
        resized = F.interpolate(source[None], size=(512, 512),
                                mode='bilinear', align_corners=False)[0]
        rgb = resized.numpy().transpose(1, 2, 0).astype(np.uint8)
    # Same arithmetic and operation order as Albumentations Normalize.
    rgb = rgb.astype(np.float32)
    rgb -= np.asarray([.26]*3, dtype=np.float32)*255.
    rgb *= np.reciprocal(np.asarray([.30]*3, dtype=np.float32)*255.)
    # Validate the actual 512 tensor after the exact original transform. This
    # adds no clipping or replacement intensities and leaves native data intact.
    low, high = MODEL_INPUT_RANGE
    if (rgb.shape != (512, 512, 3) or not np.isfinite(rgb).all()
            or float(rgb.min()) < low - 1e-5 or float(rgb.max()) > high + 1e-5):
        raise ValueError('Cardiac model tensor is outside its finite training input range.')
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


def mask_prior(mask):
    """A native mask on the trained 256->512 grid, nearest-neighbor throughout."""
    image = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8))
    height, width = mask.shape
    if mask.shape != (256, 256):
        image = sitk.Resample(image, [256, 256], sitk.Transform(), sitk.sitkNearestNeighbor,
                              [0., 0.], [width/256, height/256], image.GetDirection(), 0., image.GetPixelID())
    resized = sitk.GetArrayFromImage(image)
    return cv2.resize(resized, (512, 512), interpolation=cv2.INTER_NEAREST)[None].astype(np.float32)


def map_points(points, shape):
    """Native -> ITK 256 -> original Cardiac-Prism scale-based click coordinates."""
    height, width = shape
    return [(int(np.clip(round(p['y']*512/height), 0, 511)),
             int(np.clip(round(p['x']*512/width), 0, 511))) for p in points]


def restore_probability(logits, shape):
    """Original inference: upsample logits, sigmoid, restore probabilities."""
    import torch.nn.functional as F
    padded = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False).sigmoid()
    small = F.interpolate(padded, size=(256, 256), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
    if shape == (256, 256):
        return small.copy()
    height, width = shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32)*256/width,
                                 np.arange(height, dtype=np.float32)*256/height)
    return cv2.remap(small, grid_x, grid_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def predict(api, plane, points, label, mode, prior, iteration):
    import torch
    import torch.nn.functional as F
    from cardiac_prism.inference.inference import _pack_user_point_tokens
    from cardiac_prism.modeling.point_encoder.point_feature_map_encoder import get_batch_point_feature_map
    model = api._model
    image = image_tensor(plane.prepared).to(model.device)
    mapped = map_points(points, plane.shape)
    positive = [p for p, original in zip(mapped, points) if original['positive']]
    negative = [p for p, original in zip(mapped, points) if not original['positive']]
    pos, neg = _pack_user_point_tokens(positive, negative, model.device)
    sample = {'image': image, 'points_list': [(torch.cat((pos[0], neg[0])), torch.cat((pos[1], neg[1])))],
              'pos_point_list': [pos], 'neg_point_list': [neg], 'click_index': iteration,
              'seg_result': torch.from_numpy(prior.copy()).to(model.device) if prior is not None else None}
    click_mode = '2' if mode == 'click' else '1' if points else '0'
    with torch.inference_mode():
        features = model.backbone(image[None])
        output = model.sem_seg_head(features, get_batch_point_feature_map([sample]),
                    [sample['points_list']], [sample['pos_point_list']] if points else None,
                    [sample['neg_point_list']] if points else None, click_mode=click_mode, query_index=[label+15])
        logits = output['pred_masks']
        if not torch.isfinite(logits).all():
            raise RuntimeError('Model produced non-finite logits.')
        probability = restore_probability(logits, plane.shape)
        padded = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False).sigmoid()[0].cpu().numpy().copy()
    return {'mask': probability > .5, 'probability': probability, 'prior': padded, 'iteration': iteration+1}


def predict_reference(api, plane, references, label):
    """The original MOE in-context test path, with all selected supports.

    References are immutable prepared planes and native binary annotations. No
    previous target prediction or clicks are fed into another time frame.
    """
    import torch
    import torch.nn.functional as F
    if not 1 <= len(references) <= 3 or label not in (1, 2, 3):
        raise ValueError('Choose one to three annotated reference frames and a cardiac structure.')
    ref_images, ref_masks = [], []
    for reference, mask in references:
        if mask.shape != reference.shape or not np.isin(mask, [0, 1]).all() or not np.any(mask):
            raise ValueError('Each reference must contain a nonempty binary mask on its native image grid.')
        ref_images.append(image_tensor(reference.prepared, reference=True))
        ref_masks.append(torch.from_numpy(mask_prior(mask)))
    model = api._model
    with torch.inference_mode():
        image = image_tensor(plane.prepared).to(model.device)[None]
        sample = {'ref_img': torch.stack(ref_images), 'ref_mask': torch.stack(ref_masks)}
        summaries, features = model.process_ref_inputs([sample], image, mode='test')
        zero = torch.zeros(1, 1, 3, 512, 512, device=model.device)
        output = model.sem_seg_head(features, zero, click_mode='3', query_index=[label+15], task_query=summaries)
        logits = output['pred_masks']
        if not torch.isfinite(logits).all():
            raise RuntimeError('In-context model produced non-finite logits.')
        probability = restore_probability(logits, plane.shape)
        prior = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False).sigmoid()[0].cpu().numpy().copy()
    return {'mask': probability > .5, 'probability': probability, 'prior': prior, 'iteration': 1}
