"""Small, reproducible input provenance without preparing the whole cine.

Only an exact source-file identity in the maintained catalog can bypass raw
preparation. Neither a filename, a user-supplied provenance record, nor an
intensity range establishes that an upload was already prepared.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import cardiac_model

ROOT = Path(__file__).resolve().parents[1]


def source_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepared_source_hashes():
    path = ROOT / 'demo-data/provenance.json'
    entries = json.loads(path.read_text()) if path.is_file() else []
    return {entry['sha256'] for entry in entries
            if '/rescale_data/' in entry.get('geometry_source', '') and entry.get('sha256')}


def build_input_profile(source_path, voxels, geometry=None, model=None):
    """Describe a validated native T,Z,Y,X image; do not resample or alter it.

    Project construction checks all source values are finite. This helper only
    samples statistics and fingerprints the immutable copied source. Rebuilding
    on open also prevents an untrusted backup profile from declaring raw data
    prepared. Actual prepared frames and 512 tensors are checked at inference.
    """
    values = np.asarray(voxels)
    if values.ndim != 4 or not values.size or min(values.shape) < 1:
        raise ValueError('Input profile requires a nonempty scalar T,Z,Y,X study.')
    model = cardiac_model.model_profile() if model is None else model
    fingerprint = source_sha256(source_path)
    prepared = fingerprint in prepared_source_hashes()
    sampled = values.reshape(-1)[::max(1, (values.size + 249_999) // 250_000)]
    if not np.isfinite(sampled).all():
        raise ValueError('Image contains NaN or infinite values.')
    low, p1, p50, p98, p99, high = np.percentile(sampled.astype(np.float64), [0, 1, 50, 98, 99, 100])
    warnings = []
    if low == high:
        warnings.append('Sampled intensities are constant; constant slices become zero before model normalization.')
    if low < 0:
        warnings.append('Negative source values are handled by the same per-slice training normalization.')
    protocol = model.get('preprocessing', 'legacy_ui_max_divide')
    cardiac = protocol == cardiac_model.PROTOCOL
    if not cardiac:
        warnings.append('The active checkpoint uses the legacy input path; the cardiac input guarantee is unavailable.')
    if prepared and (values.shape[2:] != (256, 256) or low < 0 or high > 255):
        raise ValueError('The catalog prepared source has an incompatible shape or intensity range.')
    return {
        'schema_version': 1, 'protocol': protocol,
        'model_id': model.get('id', 'legacy'), 'model_name': model.get('name', 'K-Prism original'),
        'checkpoint_sha256': model.get('checkpoint_sha256'),
        'source_sha256': fingerprint, 'source_identity_verified': True,
        'input_state': 'trusted_prepared' if prepared and cardiac else 'raw',
        'native_shape_tzyx': list(values.shape),
        'raw_stats': {'sample_count': int(sampled.size), 'min': float(low), 'max': float(high),
                      'p1': float(p1), 'p50': float(p50), 'p98': float(p98), 'p99': float(p99),
                      'negative_fraction': float(np.mean(sampled < 0)),
                      'zero_fraction': float(np.mean(sampled == 0))},
        'model_input': {'shape': [3, 512, 512],
                        'expected_range': list(cardiac_model.MODEL_INPUT_RANGE) if cardiac else None,
                        'finite_range_checked': cardiac, 'checks_when': 'every_model_input'},
        'native_source_preserved': True, 'warnings': warnings,
    }
