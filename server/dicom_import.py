"""A deliberately small importer: one complete, classic MR series per folder.

Uploaded names and DICOM identity tags are never used as paths or provenance.
The converter is an independent process; every decoded source pixel is checked
against its physical location and phase before a study is committed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import numpy as np
import pydicom
from pydicom.uid import MRImageStorage, UID
import SimpleITK as sitk

from .domain import array4d, geometry

ROOT = Path(__file__).resolve().parents[1]
MAX_FILES = 1000
MAX_BYTES = 512 * 1024 ** 2
MAX_VOXELS = 128_000_000
DEFAULT_INTERVAL_SECONDS = .1
CONVERTER_TIMEOUT_SECONDS = 180
_IMPORT_LOCK = threading.Lock()


class ConverterUnavailable(RuntimeError):
    pass


@dataclass
class Instance:
    path: Path
    header: object
    position: np.ndarray
    projected_position: float
    layer: int = 0
    time: int = 0


def converter_path():
    path = Path(os.environ.get('KPRISM_DCM2NIIX',
                ROOT / 'runtime/tools/dcm2niix-v1.0.20260724/dcm2niix'))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ConverterUnavailable('DICOM conversion is not installed on this server. Run scripts/setup-dicom.sh first.')
    return path.resolve()


def _numbers(header, name, count):
    try:
        values = np.asarray(getattr(header, name), dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f'DICOM images require valid {name}.') from error
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError(f'DICOM images require valid {name}.')
    return values


def _number(header, name):
    try:
        value = float(getattr(header, name))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f'DICOM images contain an invalid {name}.') from error
    if not np.isfinite(value):
        raise ValueError(f'DICOM images contain an invalid {name}.')
    return value


def _groups(values, tolerance):
    """Small sorted scalar clusters, without trusting filenames or instance IDs."""
    centers = []
    assignments = [None] * len(values)
    for index in np.argsort(values, kind='stable'):
        value = float(values[index])
        if not centers or value - centers[-1] > tolerance:
            centers.append(value)
        assignments[index] = len(centers) - 1
    return np.asarray(centers), assignments


def inspect_series(paths):
    records = []
    identities, series, studies = set(), set(), set()
    orientation = spacing = image_type = complex_type = echo = None
    rows = columns = None
    for path in paths:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception as error:
            raise ValueError('The folder contains a file that cannot be read as a DICOM image.') from error
        if str(ds.get('SOPClassUID', '')) != str(MRImageStorage) or str(ds.get('Modality', '')) != 'MR':
            raise ValueError('Choose one classic MR image series. Enhanced/multiframe DICOM and other modalities are not supported yet.')
        if int(ds.get('NumberOfFrames', 1)) != 1:
            raise ValueError('Enhanced/multiframe DICOM is not supported yet. Choose a classic single-frame MR series.')
        try:
            syntax = UID(ds.file_meta.TransferSyntaxUID)
            compressed = syntax.is_compressed
        except (AttributeError, ValueError) as error:
            raise ValueError('The DICOM transfer syntax is missing or unsupported.') from error
        if compressed:
            raise ValueError('Compressed DICOM is not supported in this first version. Export an uncompressed classic MR series.')
        if int(ds.get('SamplesPerPixel', 1)) != 1 or str(ds.get('PhotometricInterpretation', '')) != 'MONOCHROME2':
            raise ValueError('Choose scalar MONOCHROME2 MR images; color and other image components are unsupported.')
        current_rows, current_columns = int(ds.get('Rows', 0)), int(ds.get('Columns', 0))
        if min(current_rows, current_columns) < 1 or current_rows * current_columns > MAX_VOXELS:
            raise ValueError('The DICOM image dimensions are invalid or exceed the 128-million-voxel study limit.')
        current_orientation = _numbers(ds, 'ImageOrientationPatient', 6)
        basis = current_orientation.reshape(2, 3)
        if not np.allclose(basis @ basis.T, np.eye(2), atol=1e-4, rtol=0):
            raise ValueError('DICOM image orientation must define orthogonal unit axes.')
        current_spacing = _numbers(ds, 'PixelSpacing', 2)
        if np.any(current_spacing <= 0):
            raise ValueError('DICOM pixel spacing must be positive.')
        current_image_type = tuple(str(value) for value in ds.get('ImageType', []))
        current_complex = str(ds.get('ComplexImageComponent', ''))
        current_echo = str(ds.get('EchoNumbers', ''))
        if orientation is None:
            orientation, spacing = current_orientation, current_spacing
            rows, columns = current_rows, current_columns
            image_type, complex_type, echo = current_image_type, current_complex, current_echo
        if ((current_rows, current_columns) != (rows, columns)
                or not np.allclose(current_orientation, orientation, atol=1e-4, rtol=0)
                or not np.allclose(current_spacing, spacing, atol=1e-4, rtol=0)):
            raise ValueError('The folder mixes image sizes, directions or pixel spacings. Choose a single consistent MR series.')
        if (current_image_type, current_complex, current_echo) != (image_type, complex_type, echo):
            raise ValueError('The folder mixes image components or echoes. Choose only one MR image component.')
        for name, collection in [('SOPInstanceUID', identities), ('SeriesInstanceUID', series), ('StudyInstanceUID', studies)]:
            value = str(ds.get(name, ''))
            if not value:
                raise ValueError(f'A DICOM image is missing {name}.')
            if name == 'SOPInstanceUID' and value in collection:
                raise ValueError('The folder contains duplicate DICOM images.')
            collection.add(value)
        position = _numbers(ds, 'ImagePositionPatient', 3)
        records.append(Instance(Path(path), ds, position, float(position @ np.cross(orientation[:3], orientation[3:]))))
    if not records:
        raise ValueError('The selected folder contains no DICOM images.')
    if len(series) != 1 or len(studies) != 1:
        raise ValueError('The folder contains multiple studies or series. Choose the folder for just one MR series.')
    centers, layers = _groups([record.projected_position for record in records], .01)
    depth = len(centers)
    for record, layer in zip(records, layers):
        record.layer = layer
    if depth > 1:
        differences = np.diff(centers)
        z_spacing = float(np.mean(differences))
        if not np.allclose(differences, z_spacing, atol=max(.01, z_spacing * .001), rtol=0):
            raise ValueError('DICOM layer spacing is irregular or layers are missing.')
    else:
        z_spacing = abs(float(records[0].header.get('SpacingBetweenSlices', records[0].header.get('SliceThickness', 1))))
        if not np.isfinite(z_spacing) or z_spacing <= 0:
            raise ValueError('DICOM slice spacing must be positive.')
    normal = np.cross(orientation[:3], orientation[3:])
    anchor = records[0].position - records[0].projected_position * normal
    for record in records:
        expected = anchor + centers[record.layer] * normal
        if np.max(np.abs(record.position - expected)) > .01:
            raise ValueError('DICOM layers shift within the image plane or have inconsistent physical positions.')
    has_tpi = ['TemporalPositionIdentifier' in record.header for record in records]
    has_trigger = ['TriggerTime' in record.header for record in records]
    if any(has_tpi) and not all(has_tpi):
        raise ValueError('DICOM time identifiers are incomplete. Choose a complete single-series folder.')
    if any(has_trigger) and not all(has_trigger):
        raise ValueError('DICOM TriggerTime values are incomplete. Choose a complete single-series folder.')
    if all(has_tpi):
        values = [_number(record.header, 'TemporalPositionIdentifier') for record in records]
        if any(value != int(value) or value < 1 for value in values):
            raise ValueError('DICOM temporal position identifiers must be positive integers.')
        phases, times = _groups(values, 0)
        if len(phases) > 1 and not np.all(np.diff(phases) == 1):
            raise ValueError('DICOM time identifiers have gaps; one or more frames may be missing.')
        grouping = 'TemporalPositionIdentifier'
    elif all(has_trigger):
        phases, times = _groups([_number(record.header, 'TriggerTime') for record in records], .1)
        grouping = 'TriggerTime'
    else:
        phases, times, grouping = np.asarray([0.]), [0] * len(records), 'single_frame'
    frames = len(phases)
    if frames == 1:
        grouping = 'single_frame'
    cells = set()
    for record, time in zip(records, times):
        record.time = time
        cell = (time, record.layer)
        if cell in cells:
            if grouping == 'single_frame':
                raise ValueError('Several images share a layer, but reliable time identifiers are missing. Frame order cannot be guessed from filenames.')
            raise ValueError('The folder contains duplicate time/layer images or mixed image components.')
        cells.add(cell)
        declared = record.header.get('NumberOfTemporalPositions')
        if declared is not None and int(declared) != frames:
            raise ValueError('The folder does not contain all declared DICOM time frames.')
    if len(cells) != frames * depth:
        raise ValueError('The DICOM time/layer grid is incomplete. Every frame must contain the same layers.')
    if frames * depth * rows * columns > MAX_VOXELS:
        raise ValueError('This workstation supports up to 128 million voxels per study.')
    warnings = []
    offsets = interval = None
    timing_source = 'not_applicable' if frames == 1 else 'unknown'
    if frames > 1 and all(has_trigger):
        trigger_phases = []
        for time in range(frames):
            values = np.asarray([_number(record.header, 'TriggerTime') for record in records if record.time == time])
            if float(np.ptp(values)) > .5:
                raise ValueError('DICOM phase timing differs between layers; a single shared time axis cannot be established.')
            trigger_phases.append(float(np.mean(values)) / 1000.)
        differences = np.diff(trigger_phases)
        if np.any(differences <= 0):
            raise ValueError('DICOM time identifiers and TriggerTime disagree about frame order.')
        offsets = (np.asarray(trigger_phases) - trigger_phases[0]).tolist()
        interval = float(np.mean(differences))
        timing_source = 'TriggerTime'
        if not np.allclose(differences, interval, atol=max(.0011, interval * .02), rtol=0):
            warnings.append('Phase intervals vary. NIfTI stores their mean; exact DICOM phase offsets are retained in the study backup.')
    elif frames > 1:
        warnings.append('Frame order is known, but acquisition frame timing is unavailable. The stored time interval is a nominal 0.1 s; adjust playback FPS as needed. MRI TR is not used.')
    metadata = {
        'schema_version': 1, 'source_format': 'DICOM', 'instance_count': len(records),
        'shape_xyzt': [columns, rows, depth, frames], 'time_grouping': grouping,
        'timing_known': interval is not None, 'timing_source': timing_source,
        'frame_interval_seconds': interval,
        'nominal_frame_interval_seconds': (interval or DEFAULT_INTERVAL_SECONDS) if frames > 1 else None,
        'phase_offsets_seconds': offsets, 'warnings': warnings,
        'spatial_sort': 'ImagePositionPatient projected on ImageOrientationPatient normal',
        'source_spacing_xyz_mm': [float(spacing[1]), float(spacing[0]), z_spacing],
    }
    return records, metadata


def verify_conversion(image, records, metadata):
    geometry(image)
    data = array4d(image)
    expected_shape = tuple(reversed(metadata['shape_xyzt']))
    if data.shape != expected_shape:
        raise ValueError('The converter produced unexpected image dimensions. This DICOM layout is not supported yet.')
    dimension = image.GetDimension()
    spatial = np.asarray(image.GetDirection()).reshape(dimension, dimension)[:3, :3]
    inverse = np.linalg.inv(spatial @ np.diag(image.GetSpacing()[:3]))
    origin = np.asarray(image.GetOrigin()[:3])
    maximum_error = maximum_grid_error = 0.
    covered = set()
    # Keep one native plane coordinate grid, rather than allocating a full cine.
    rows, columns = np.indices((expected_shape[2], expected_shape[3]), dtype=np.float64)
    for record in records:
        try:
            ds = pydicom.dcmread(record.path, force=True)
            pixels = ds.pixel_array
        except Exception as error:
            raise ValueError('A DICOM pixel image could not be decoded. Choose valid uncompressed MR images.') from error
        if pixels.shape != rows.shape:
            raise ValueError('DICOM pixel data does not match its declared dimensions.')
        iop = _numbers(ds, 'ImageOrientationPatient', 6)
        spacing = _numbers(ds, 'PixelSpacing', 2)
        start = inverse @ (record.position - origin)
        column_step = inverse @ (iop[:3] * spacing[1])
        row_step = inverse @ (iop[3:] * spacing[0])
        coordinates = start[:, None, None] + column_step[:, None, None] * columns + row_step[:, None, None] * rows
        indices = np.rint(coordinates).astype(np.int64)
        grid_error = float(np.abs(coordinates - indices).max())
        maximum_grid_error = max(maximum_grid_error, grid_error)
        if grid_error > .01:
            raise ValueError('Converted image geometry does not match the DICOM pixel positions.')
        x, y, z = indices
        if (np.any(x < 0) or np.any(x >= data.shape[3]) or np.any(y < 0) or np.any(y >= data.shape[2])
                or np.any(z < 0) or np.any(z >= data.shape[1]) or np.ptp(z) != 0):
            raise ValueError('Converted image axes do not preserve the source DICOM layers.')
        cell = (record.time, int(z.flat[0]))
        if cell in covered:
            raise ValueError('Converted DICOM layers overlap or are missing.')
        covered.add(cell)
        slope = float(ds.get('RescaleSlope', 1))
        intercept = float(ds.get('RescaleIntercept', 0))
        expected = pixels.astype(np.float64) * slope + intercept
        actual = data[record.time, z, y, x].astype(np.float64)
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError('DICOM image intensities contain non-finite values.')
        error = float(np.abs(expected - actual).max())
        maximum_error = max(maximum_error, error)
        # NIfTI stores scaling in float32; allow only its representational error.
        if not np.allclose(actual, expected, rtol=2e-6, atol=1e-5):
            raise ValueError('Converted pixel values or time ordering differ from the DICOM source. This acquisition is not supported yet.')
    if len(covered) != data.shape[0] * data.shape[1]:
        raise ValueError('Converted DICOM time/layer coverage is incomplete.')
    metadata['verification'] = {'all_source_pixels_checked': True,
                                'pixel_count': int(data.size),
                                'maximum_pixel_error': maximum_error,
                                'maximum_grid_error_voxels': maximum_grid_error}


def _convert(tool, input_dir, output_dir):
    command = [str(tool), '-b', 'n', '-z', 'n', '-f', 'volume', '-o', str(output_dir), str(input_dir)]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=CONVERTER_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as error:
        raise ValueError('DICOM conversion timed out. Choose a smaller single-series folder.') from error
    except OSError as error:
        raise ConverterUnavailable('The server could not start dcm2niix. Check its installation.') from error
    if completed.returncode:
        # Converter messages can contain DICOM identity; do not return or log them.
        raise ValueError('The converter could not read this DICOM series. Choose valid, uncompressed classic MR images.')
    outputs = list(output_dir.glob('*.nii'))
    if len(outputs) != 1:
        raise ValueError('The folder converted into multiple or no images. Choose only one complete MR series and image component.')
    return outputs[0]


def import_folder(store, files, folder_name=None):
    tool = converter_path()
    if not files or len(files) > MAX_FILES:
        raise ValueError('Choose a DICOM folder containing 1–1000 image files (up to 512 MB total).')
    if not _IMPORT_LOCK.acquire(blocking=False):
        raise ValueError('Another DICOM folder is being imported. Wait for it to finish and try again.')
    try:
        uploads = store.directory.parent / 'uploads'
        uploads.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='dicom-', dir=uploads) as temporary:
            temporary = Path(temporary)
            input_dir, output_dir = temporary / 'input', temporary / 'output'
            input_dir.mkdir(); output_dir.mkdir()
            paths, total = [], 0
            for index, uploaded in enumerate(files):
                basename = (uploaded.filename or '').replace('\\', '/').split('/')[-1]
                if basename == '.DS_Store' or basename.startswith('._'):
                    continue
                path = input_dir / f'image-{index:04d}.dcm'
                with path.open('wb') as stream:
                    while chunk := uploaded.file.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_BYTES:
                            raise ValueError('DICOM folders are limited to 512 MB total.')
                        stream.write(chunk)
                if not path.stat().st_size:
                    raise ValueError('The DICOM folder contains an empty file.')
                paths.append(path)
            records, metadata = inspect_series(paths)
            converted = _convert(tool, input_dir, output_dir)
            try:
                image = sitk.ReadImage(str(converted))
            except RuntimeError as error:
                raise ValueError('The converted DICOM image could not be read.') from error
            verify_conversion(image, records, metadata)
            # The converter can place scanner or identity text in NIfTI header
            # metadata. Geometry is intrinsic to the image, so clear all text
            # fields before creating the workstation's immutable source file.
            for key in image.GetMetaDataKeys():
                image.EraseMetaData(key)
            if metadata['shape_xyzt'][3] > 1:
                if image.GetDimension() != 4:
                    raise ValueError('The converter did not preserve the DICOM time axis.')
                image.SetSpacing((*image.GetSpacing()[:3], metadata['nominal_frame_interval_seconds']))
                image.SetOrigin((*image.GetOrigin()[:3], 0.))
            destination = temporary / 'dicom-volume.nii.gz'
            sitk.WriteImage(image, str(destination))
            metadata['converter'] = 'dcm2niix'
            metadata['converter_sha256'] = hashlib.sha256(tool.read_bytes()).hexdigest()
            name = (folder_name or 'DICOM study').replace('\\', '/').rstrip('/').split('/')[-1]
            name = ''.join(character for character in name if character.isprintable()).strip()[:120] or 'DICOM study'
            project = store.create(destination, name)
            try:
                width, height, depth, frames = metadata['shape_xyzt']
                note = f'Imported DICOM · {width} × {height} · {depth} layers · {frames} frames.'
                if metadata['warnings']:
                    note += ' ' + ' '.join(metadata['warnings'])
                project.provenance = {'dicom_import': metadata, 'note': note}
                project.save()
            except Exception:
                with store.lock:
                    store.projects.pop(project.id, None)
                shutil.rmtree(project.directory, ignore_errors=True)
                raise
            return project
    finally:
        _IMPORT_LOCK.release()
