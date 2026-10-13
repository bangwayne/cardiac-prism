"""DICOM import contracts, real conversion, native geometry and error atomicity."""
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
import SimpleITK as sitk
from fastapi import UploadFile
from fastapi.testclient import TestClient

from server.app import create_app
from server.dicom_import import (ConverterUnavailable, converter_path, import_folder,
                                 inspect_series, verify_conversion)
from server.domain import ProjectStore, require_same_geometry
from server.inference import ModelService


def make_series(directory, frames=4, depth=3, time_tag='trigger', rescale=True):
    """Asymmetric pixels distinguish every t,z,row,column, in an oblique basis."""
    directory.mkdir(parents=True, exist_ok=True)
    study, series = generate_uid(), generate_uid()
    angle = np.pi / 7
    row_axis = np.asarray([np.cos(angle), np.sin(angle), 0.])
    column_axis = np.asarray([-np.sin(angle), np.cos(angle), 0.])
    normal = np.cross(row_axis, column_axis)
    records = []
    for t in range(frames):
        for z in range(depth):
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0' * 128)
            ds.SOPClassUID = MRImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
            ds.PatientName, ds.PatientID = 'Synthetic^Fixture', 'SYNTHETIC-NO-PATIENT'
            ds.Modality, ds.Manufacturer = 'MR', 'SIEMENS'
            ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = '20000101'
            ds.StudyTime = ds.SeriesTime = ds.AcquisitionTime = '120000'
            ds.SeriesNumber, ds.AcquisitionNumber = 42, 1
            ds.InstanceNumber = t * depth + z + 1
            ds.ImageType = ['ORIGINAL', 'PRIMARY', 'M']
            ds.ScanningSequence, ds.SequenceVariant = 'GR', 'SK'
            ds.MRAcquisitionType = '2D'
            ds.Rows, ds.Columns = 7, 11
            ds.PixelSpacing, ds.SliceThickness, ds.SpacingBetweenSlices = [.8, 1.2], 4., 4.
            ds.ImageOrientationPatient = [float(value) for value in np.r_[row_axis, column_axis]]
            ds.ImagePositionPatient = [float(value) for value in np.asarray([13., -8., 21.]) + normal * z * 4.]
            ds.SliceLocation = z * 4.
            ds.RepetitionTime, ds.EchoTime, ds.FlipAngle = 3.246, 1.5, 30.
            if time_tag in ('trigger', 'both'):
                ds.TriggerTime = t * 50. + 10.
            if time_tag in ('tpi', 'both'):
                ds.TemporalPositionIdentifier = t + 1
                ds.NumberOfTemporalPositions = frames
            ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit, ds.PixelRepresentation = 15, 1
            pixels = (np.arange(77).reshape(7, 11) + t * 1000 + z * 100).astype('<i2')
            ds.PixelData = pixels.tobytes()
            if rescale:
                ds.RescaleSlope, ds.RescaleIntercept = 2.5, -12.25
            path = directory / f'extensionless-{frames * depth - len(records):04d}'
            ds.save_as(path, enforce_file_format=True)
            records.append(path)
    return records


def edit(path, **values):
    ds = pydicom.dcmread(path)
    for key, value in values.items():
        if value is None:
            delattr(ds, key)
        else:
            setattr(ds, key, value)
    ds.save_as(path, enforce_file_format=True)


class DicomImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = make_series(self.root / 'source')

    def tearDown(self):
        self.temporary.cleanup()

    def test_native_non_square_shape_time_and_spacing(self):
        records, meta = inspect_series(list(reversed(self.paths)))
        self.assertEqual(meta['shape_xyzt'], [11, 7, 3, 4])
        np.testing.assert_allclose(meta['phase_offsets_seconds'], [0., .05, .1, .15])
        self.assertAlmostEqual(meta['frame_interval_seconds'], .05)
        self.assertNotAlmostEqual(meta['frame_interval_seconds'], .003246)
        self.assertEqual({(record.time, record.layer) for record in records},
                         {(t, z) for t in range(4) for z in range(3)})
        np.testing.assert_allclose(meta['source_spacing_xyz_mm'], [1.2, .8, 4.])

    def test_3d_requires_no_time_tags(self):
        paths = make_series(self.root / '3d', frames=1, time_tag='none')
        _, meta = inspect_series(paths)
        self.assertEqual(meta['shape_xyzt'], [11, 7, 3, 1])
        self.assertEqual(meta['time_grouping'], 'single_frame')
        self.assertEqual(meta['timing_source'], 'not_applicable')
        self.assertIsNone(meta['frame_interval_seconds'])

    def test_tpi_known_order_unknown_duration_never_uses_tr(self):
        paths = make_series(self.root / 'unknown', time_tag='tpi')
        _, meta = inspect_series(paths)
        self.assertEqual(meta['time_grouping'], 'TemporalPositionIdentifier')
        self.assertFalse(meta['timing_known'])
        self.assertEqual(meta['nominal_frame_interval_seconds'], .1)
        self.assertIsNone(meta['phase_offsets_seconds'])
        self.assertIn('TR is not used', meta['warnings'][0])

    def test_repeated_layer_without_time_rejected(self):
        paths = make_series(self.root / 'no-time', time_tag='none')
        with self.assertRaisesRegex(ValueError, 'time identifiers are missing'):
            inspect_series(paths)

    def test_missing_and_duplicate_time_layer_rejected(self):
        with self.assertRaisesRegex(ValueError, 'grid is incomplete'):
            inspect_series(self.paths[:-1])
        duplicate = self.root / 'duplicate'
        duplicate.write_bytes(self.paths[0].read_bytes())
        with self.assertRaisesRegex(ValueError, 'duplicate DICOM'):
            inspect_series([*self.paths, duplicate])
        edit(duplicate, SOPInstanceUID=generate_uid())
        with self.assertRaisesRegex(ValueError, 'duplicate time/layer'):
            inspect_series([*self.paths, duplicate])

    def test_mixed_series_and_components_rejected(self):
        edit(self.paths[-1], SeriesInstanceUID=generate_uid())
        with self.assertRaisesRegex(ValueError, 'multiple studies or series'):
            inspect_series(self.paths)
        edit(self.paths[-1], SeriesInstanceUID=pydicom.dcmread(self.paths[0]).SeriesInstanceUID,
             ImageType=['ORIGINAL', 'PRIMARY', 'P'])
        with self.assertRaisesRegex(ValueError, 'components or echoes'):
            inspect_series(self.paths)

    def test_mixed_direction_and_in_plane_shift_rejected(self):
        edit(self.paths[-1], ImageOrientationPatient=[1, 0, 0, 0, 1, 0])
        with self.assertRaisesRegex(ValueError, 'directions'):
            inspect_series(self.paths)
        first = pydicom.dcmread(self.paths[0])
        edit(self.paths[-1], ImageOrientationPatient=first.ImageOrientationPatient,
             ImagePositionPatient=[14, -8, 29])
        with self.assertRaisesRegex(ValueError, 'shift within'):
            inspect_series(self.paths)

    def test_irregular_layer_spacing_rejected(self):
        for path in self.paths:
            ds = pydicom.dcmread(path)
            if float(ds.ImagePositionPatient[2]) == 29:
                edit(path, ImagePositionPatient=[13, -8, 30])
        with self.assertRaisesRegex(ValueError, 'spacing is irregular'):
            inspect_series(self.paths)

    def test_partial_and_contradictory_time_tags_rejected(self):
        edit(self.paths[-1], TriggerTime=None)
        with self.assertRaisesRegex(ValueError, 'TriggerTime values are incomplete'):
            inspect_series(self.paths)
        paths = make_series(self.root / 'both', time_tag='both')
        for path in paths:
            ds = pydicom.dcmread(path)
            edit(path, TriggerTime=200 - float(ds.TriggerTime))
        with self.assertRaisesRegex(ValueError, 'disagree about frame order'):
            inspect_series(paths)

    def test_enhanced_and_compressed_are_explicitly_rejected(self):
        from pydicom.uid import EnhancedMRImageStorage, JPEGLosslessSV1
        edit(self.paths[0], SOPClassUID=EnhancedMRImageStorage)
        with self.assertRaisesRegex(ValueError, 'Enhanced/multiframe'):
            inspect_series(self.paths)
        edit(self.paths[0], SOPClassUID=MRImageStorage)
        ds = pydicom.dcmread(self.paths[0], stop_before_pixels=True)
        ds.file_meta.TransferSyntaxUID = JPEGLosslessSV1
        ds.save_as(self.paths[0], enforce_file_format=True)
        with self.assertRaisesRegex(ValueError, 'Compressed DICOM'):
            inspect_series(self.paths)

    def test_unknown_files_are_not_silently_imported(self):
        path = self.root / 'readme'
        path.write_bytes(b'not a DICOM file')
        with self.assertRaisesRegex(ValueError, 'classic MR'):
            inspect_series([*self.paths, path])

    def test_missing_converter_is_clear(self):
        with patch.dict(os.environ, CARDIAC_PRISM_DCM2NIIX=str(self.root / 'missing')):
            with self.assertRaisesRegex(ConverterUnavailable, 'setup-dicom.sh'):
                converter_path()

    def test_count_size_empty_file_limits_cleanup(self):
        store = ProjectStore(self.root / 'workspace/projects')
        with self.assertRaisesRegex(ValueError, '1–1000'):
            import_folder(store, [])
        with patch('server.dicom_import.MAX_FILES', 1):
            with self.assertRaisesRegex(ValueError, '1–1000'):
                import_folder(store, [None, None])
        with patch('server.dicom_import.MAX_BYTES', 10):
            with self.assertRaisesRegex(ValueError, '512 MB'):
                import_folder(store, [UploadFile(io.BytesIO(b'x' * 11), filename='image')])
        with self.assertRaisesRegex(ValueError, 'empty file'):
            import_folder(store, [UploadFile(io.BytesIO(b''), filename='image')])
        self.assertEqual(list((self.root / 'workspace/uploads').iterdir()), [])
        self.assertEqual(store.list(), [])

    def test_declared_missing_frame_and_gap_rejected(self):
        paths = make_series(self.root / 'tpi', time_tag='tpi')
        with self.assertRaisesRegex(ValueError, 'all declared'):
            inspect_series(paths[:9])
        with self.assertRaisesRegex(ValueError, 'identifiers have gaps'):
            inspect_series([*paths[:3], *paths[6:]])

    def test_converter_timeout_and_multiple_outputs_are_safe(self):
        import subprocess
        from server.dicom_import import _convert
        output = self.root / 'converter-output'; output.mkdir()
        with patch('server.dicom_import.subprocess.run', side_effect=subprocess.TimeoutExpired('dcm2niix', 180)):
            with self.assertRaisesRegex(ValueError, 'timed out'):
                _convert(Path('/safe/tool'), self.root, output)
        (output / 'a.nii').touch(); (output / 'b.nii').touch()
        with patch('server.dicom_import.subprocess.run', return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaisesRegex(ValueError, 'multiple or no images'):
                _convert(Path('/safe/tool'), self.root, output)

    def test_actual_converter_4d_preserves_rescale_geometry_and_persistence(self):
        try:
            converter_path()
        except ConverterUnavailable:
            self.skipTest('Pinned dcm2niix must be installed for real conversion integration.')
        store = ProjectStore(self.root / 'workspace/projects')
        uploads = [UploadFile(io.BytesIO(path.read_bytes()), filename='folder/' + path.name)
                   for path in reversed(self.paths)]
        project = import_folder(store, uploads, 'Synthetic cine')
        self.assertEqual(project.voxels.shape, (4, 3, 7, 11))
        self.assertEqual(project.name, 'Synthetic cine')
        meta = project.provenance['dicom_import']
        self.assertTrue(meta['verification']['all_source_pixels_checked'])
        self.assertEqual(meta['verification']['pixel_count'], 924)
        self.assertAlmostEqual(project.image.GetSpacing()[3], .05)
        self.assertEqual(project.input_profile['input_state'], 'raw')
        reopened = ProjectStore(store.directory).get(project.id)
        self.assertEqual(reopened.provenance, project.provenance)
        np.testing.assert_array_equal(reopened.voxels, project.voxels)
        output = self.root / 'mask.nii.gz'
        output.write_bytes(project.export('labelmap')[0])
        require_same_geometry(project.image, sitk.ReadImage(str(output)))
        self.assertEqual(list((self.root / 'workspace/uploads').iterdir()), [])
        self.assertNotIn('SYNTHETIC-NO-PATIENT', str(project.provenance))
        self.assertNotIn('PatientName', str(project.provenance))

    def test_actual_converter_3d_and_unknown_timing_4d(self):
        try:
            converter_path()
        except ConverterUnavailable:
            self.skipTest('Pinned dcm2niix must be installed for real conversion integration.')
        store = ProjectStore(self.root / 'workspace/projects')
        for frames, time_tag in [(1, 'none'), (4, 'tpi')]:
            paths = make_series(self.root / f'source-{frames}', frames=frames, time_tag=time_tag)
            project = import_folder(store, [UploadFile(io.BytesIO(path.read_bytes()), filename=path.name) for path in paths])
            self.assertEqual(project.voxels.shape, (frames, 3, 7, 11))
            if frames == 1:
                self.assertEqual(project.image.GetDimension(), 3)
            else:
                self.assertAlmostEqual(project.image.GetSpacing()[3], .1)
                self.assertFalse(project.provenance['dicom_import']['timing_known'])

    def test_api_failure_leaves_no_project_or_upload_files(self):
        application = create_app(self.root / 'workspace', ModelService(lambda *args: None))
        with TestClient(application) as client:
            self.assertTrue(client.get('/api/health').json()['features']['dicom_import'])
            files = [('files', ('folder/' + path.name, path.read_bytes(), 'application/dicom')) for path in self.paths[:-1]]
            response = client.post('/api/projects/import-dicom', data={'folder_name': 'broken'}, files=files)
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn('incomplete', response.json()['detail'])
            self.assertEqual(client.get('/api/projects').json(), [])
        self.assertEqual(list((self.root / 'workspace/uploads').iterdir()), [])

    def test_unexpected_converter_outputs_and_wrong_pixels_fail_atomically(self):
        store = ProjectStore(self.root / 'workspace/projects')
        def wrong_image(tool, input_dir, output_dir):
            image = sitk.GetImageFromArray(np.zeros((4, 3, 7, 11), dtype=np.int16), isVector=False)
            direction = np.eye(4)
            first = pydicom.dcmread(self.paths[0])
            direction[:3, :3] = np.column_stack((np.asarray(first.ImageOrientationPatient[:3]),
                                               np.asarray(first.ImageOrientationPatient[3:]), [0, 0, 1]))
            image.SetDirection(direction.reshape(-1))
            image.SetSpacing((1.2, .8, 4, .05)); image.SetOrigin((13, -8, 21, 0))
            path = output_dir / 'wrong.nii'; sitk.WriteImage(image, str(path))
            return path
        with patch('server.dicom_import._convert', side_effect=wrong_image):
            with self.assertRaisesRegex(ValueError, 'pixel values or time ordering'):
                import_folder(store, [UploadFile(io.BytesIO(path.read_bytes()), filename=path.name) for path in self.paths])
        self.assertEqual(store.list(), [])
        self.assertEqual(list((self.root / 'workspace/uploads').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
