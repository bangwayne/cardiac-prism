"""Small moving 4D image/labelmap with exact voxel samples for browser checks."""
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

root = Path(os.environ.get('CARDIAC_PRISM_TEST_OUTPUT', 'test-results/browser')) / 'fixtures/cine'
root.mkdir(parents=True, exist_ok=True)
t, z, y, x = np.indices((6, 6, 64, 64))
values = (t * 100 + z * 10 + y * .25 + x * .125).astype(np.float32)
labels = np.zeros(values.shape, dtype=np.uint8)
for label, cy in [(1, 12), (2, 30), (3, 47)]:
    labels[((x - 15 - t * 4) ** 2 + (y - cy) ** 2 < 25) & (z >= 1) & (z <= 4)] = label
image = sitk.GetImageFromArray(values, isVector=False)
image.SetSpacing((1.3, 1.7, 4., .035))
image.SetOrigin((12., -8., 5., 0.))
image.SetDirection((-1., 0., 0., 0., 0., -1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.))
mask = sitk.GetImageFromArray(labels, isVector=False)
mask.CopyInformation(image)
sitk.WriteImage(image, str(root / 'cine-fixture.nii.gz'))
sitk.WriteImage(mask, str(root / 'cine-fixture-labels.nii.gz'))
samples = [(3 * 64 + yy) * 64 + xx for yy in (2, 12, 30, 47) for xx in (2, 15, 19, 23, 27, 31, 35)]
(root / 'fixture.json').write_text(json.dumps({
    'shape': list(values.shape), 'samples': samples,
    'frames': [{'image': [float(values[frame].ravel()[i]) for i in samples],
                'labels': [int(labels[frame].ravel()[i]) for i in samples]} for frame in range(6)],
}))
