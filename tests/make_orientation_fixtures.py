"""Asymmetric source voxels for physical-coordinate and rendered MPR checks."""
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def main():
    output = Path(os.environ.get('KPRISM_TEST_OUTPUT', 'test-results/browser')) / 'fixtures/orientation'
    output.mkdir(parents=True, exist_ok=True)
    # A fully oblique, reflected acquisition, rather than an in-plane rotation.
    scan = np.array([[.6762558, .27805029, -.68217745],
                     [.73650106, -.23554416, .63410187],
                     [-.0156293, .93123949, .36407245]])
    u, _, v = np.linalg.svd(scan)
    scan = u @ v
    positive = scan.copy()
    positive[:, 0] *= -1
    z, y, x = np.indices((12, 48, 64))
    plane = (15 + (x // 8) * 16 + (y // 8) * 35 + (z // 3) * 65).astype(np.float32)
    records = []
    for name, direction in [('oblique-reflected', scan), ('oblique-positive', positive),
                            ('legacy-aligned', np.diag([-1., -1., 1.])),
                            ('identity-aligned', np.eye(3))]:
        image = sitk.GetImageFromArray(np.stack([plane, plane + 20]), isVector=False)
        matrix = np.eye(4)
        matrix[:3, :3] = direction
        image.SetSpacing((1.3, .9, 2.5, .05))
        image.SetOrigin((11.5, -37.2, 9.1, 0))
        image.SetDirection(matrix.ravel().tolist())
        path = output / (name + '.nii.gz')
        sitk.WriteImage(image, str(path))
        records.append({'name': name, 'path': str(path), 'dimensions': [64, 48, 12],
                        'direction': direction.ravel().tolist(),
                        'determinant': float(np.linalg.det(direction))})
    (output / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({'fixtures': len(records), 'output': str(output)}))


if __name__ == '__main__':
    main()
