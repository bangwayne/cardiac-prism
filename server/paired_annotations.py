"""Verified, explicitly paired external annotations used as refinement baselines."""
import hashlib
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .domain import ORGANS, array4d, require_same_geometry


def catalog(root):
    path = root / 'demo-data/derivate/refinement-masks.json'
    return json.loads(path.read_text()) if path.is_file() else {}


def read_annotation(root, image, name):
    entry = catalog(root).get(name)
    if entry is None:
        raise ValueError('No matched refinement annotation is available for this study.')
    folder = (root / 'demo-data/derivate/sax_itksnap_safe').resolve()
    filename = entry['filename']
    if Path(filename).name != filename:
        raise ValueError('Invalid paired annotation path.')
    path = (folder / filename).resolve()
    if path.parent != folder or not path.is_file():
        raise ValueError('The matched annotation file is unavailable.')
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('The matched annotation has changed. Validate and register it again.')
    annotation = sitk.ReadImage(str(path))
    require_same_geometry(image, annotation)
    labels = array4d(annotation)
    if not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError('Paired annotation labels must be 0 (background), 1 (RV), 2 (MYO), 3 (LV).')
    return entry, labels.astype(np.uint8, copy=False)


def initialize_masks(project, labels):
    """Install a labelmap with lightweight, mask-derived first-click priors."""
    project.masks = {label: (labels == label).astype(np.uint8) for label in ORGANS}
    project.contexts = {(t, z, label): {'points': [], 'prior': None, 'mask_prior': True,
                                      'iteration': 1, 'mode': None}
                        for t in range(project.voxels.shape[0])
                        for z in range(project.voxels.shape[1]) for label in ORGANS}
