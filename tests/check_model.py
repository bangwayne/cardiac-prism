"""Real checkpoint checks; run explicitly after downloading the sample assets.

Write a reference with --output, then compare another installation with
--compare. The checks exercise native-size preprocessing, organ predictions,
positive/negative clicks with prior reuse, and temporal reference inference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--atol', type=float, default=0.)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    os.environ['VERSE_DEVICE'] = args.device
    os.environ['VERSE_CHECKPOINT'] = str(source / 'weight/cardiac_achilleas_epoch10.pth')
    data = args.data_root or source / 'demo-data'
    import numpy as np
    import SimpleITK as sitk
    import torch
    from server.inference import ModelService
    torch.set_num_threads(2)
    image = sitk.ReadImage(str(data / 'images/patient001_4d.nii.gz'))
    voxels = sitk.GetArrayFromImage(image)
    labels = sitk.GetArrayFromImage(sitk.ReadImage(str(data / 'references/patient001_frame01_gt.nii.gz')))
    service = ModelService()
    z = labels.shape[0] // 2
    planes = [service.prepare_plane(voxels[t], z, ('check', t), spacing=image.GetSpacing()[:3]) for t in (0, 1)]
    outputs = {'prepared': planes[0].prepared}
    checks = []
    start = time.perf_counter()

    def record(name, result):
        for kind in ('mask', 'probability', 'prior'):
            array = np.asarray(result[kind])
            assert np.isfinite(array).all(), name
            outputs[name + '_' + kind] = array
        checks.append(name)
        print('PASS', name, flush=True)

    for organ in (1, 2, 3):
        positive_yx = np.argwhere(labels[z] == organ)
        assert len(positive_yx), 'Sample reference lacks the selected structure'
        y, x = positive_yx[len(positive_yx) // 2]
        positive = {'x': float(x), 'y': float(y), 'positive': True}
        negative = {'x': 3., 'y': 3., 'positive': False}
        initial = service.predict(planes[0], [], organ, 'organ', None, 0)
        record(f'{organ}_organ', initial)
        clicked = service.predict(planes[0], [positive], organ, 'click', None, 0)
        record(f'{organ}_click', clicked)
        repeated = service.predict(planes[0], [positive], organ, 'click', None, 0)
        for kind in ('mask', 'probability', 'prior'):
            np.testing.assert_array_equal(clicked[kind], repeated[kind])
        refined = service.predict(planes[0], [positive, negative], organ, 'click', clicked['prior'], clicked['iteration'])
        record(f'{organ}_negative', refined)
        if not args.quick:
            combined = service.predict(planes[0], [positive], organ, 'organ_click', initial['prior'], initial['iteration'])
            record(f'{organ}_organ_click', combined)
            references = [(planes[0], (labels[z] == organ).astype(np.uint8))]
            record(f'{organ}_reference', service.predict_reference(planes[1], references, organ))
            record(f'{organ}_references3', service.predict_reference(planes[1], references * 3, organ))
    comparisons = {}
    if args.compare:
        with np.load(args.compare) as expected:
            for name, actual in outputs.items():
                target = expected[name]
                difference = float(np.max(np.abs(actual.astype(np.float64) - target.astype(np.float64))))
                if name.endswith('_mask') or name == 'prepared':
                    np.testing.assert_array_equal(actual, target, err_msg=name)
                else:
                    np.testing.assert_allclose(actual, target, atol=args.atol, rtol=0., err_msg=name)
                comparisons[name] = difference
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **outputs)
    report = {'passed': True, 'device': args.device, 'torch': torch.__version__,
              'checkpoint_sha256': service.checkpoint_sha256, 'checks': checks,
              'seconds': round(time.perf_counter() - start, 3), 'max_absolute_differences': comparisons,
              'mask_hashes': {key: hashlib.sha256(value.tobytes()).hexdigest()
                              for key, value in outputs.items() if key.endswith('_mask')}}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
