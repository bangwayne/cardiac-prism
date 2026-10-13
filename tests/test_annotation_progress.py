"""Frame/layer history must reflect effective edits without loading priors."""
import unittest

import numpy as np
import SimpleITK as sitk
import test_platform as fixtures
from server.annotation_progress import summary
from server.domain import ProjectStore


class AnnotationProgressTests(unittest.TestCase):
    setUp = fixtures.PlatformTests.setUp
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post
    brush = fixtures.PlatformTests.brush
    infer = fixtures.PlatformTests.infer
    wait_job = fixtures.PlatformTests.wait_job
    annotation = fixtures.PlatformTests.annotation

    def progress(self):
        response = self.client.get(self.prefix + '/progress')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_empty_then_scoped_myo_edit_has_exact_frame_layer_and_timestamp(self):
        empty = self.progress()
        self.assertEqual(empty['shape'], [2, 3])
        self.assertEqual(empty['mask_labels'], [0] * 6)
        self.assertEqual(empty['edited_labels'], [0] * 6)
        self.assertIsNone(empty['latest_edit_at'])
        self.brush(label=2, t=1, z=2)
        progress = self.progress()
        self.assertEqual(progress['mask_labels'], [0, 0, 0, 0, 0, 2])
        self.assertEqual(progress['edited_labels'], [0, 0, 0, 0, 0, 2])
        self.assertEqual(progress['edited_at'], [None] * 5 + [self.project.modified_at])
        self.assertEqual(progress['latest_edit_at'], self.project.modified_at)

    def test_brush_overwrite_tracks_both_structures_on_only_one_plane(self):
        self.brush(label=1, t=0, z=1)
        self.brush(label=2, t=0, z=1)
        progress = self.progress()
        self.assertEqual(progress['mask_labels'], [0, 2, 0, 0, 0, 0])
        self.assertEqual(progress['edited_labels'], [0, 3, 0, 0, 0, 0])

    def test_click_and_negative_click_are_local_and_automatic_initialization_resets_edited_status(self):
        self.assertEqual(self.infer(t=1, z=2)['state'], 'completed')
        self.assertEqual(self.progress()['edited_labels'], [0, 0, 0, 0, 0, 1])
        first = self.progress()['edited_at'][5]
        self.assertEqual(self.infer(t=1, z=2, point={'x': 6, 'y': 5, 'positive': False})['state'], 'completed')
        self.assertNotEqual(self.progress()['edited_at'][5], first)
        self.assertEqual(self.infer(t=1, z=2, mode='organ')['state'], 'completed')
        self.assertEqual(self.progress()['edited_labels'], [0] * 6)
        self.assertTrue(self.progress()['mask_labels'][5])
        self.post('undo')
        self.assertEqual(self.progress()['edited_labels'][5], 1)

    def test_clear_across_time_tracks_one_layer_even_when_masks_are_empty_and_undo_restores_it(self):
        self.brush(label=2, t=1, z=2); before = self.progress()
        self.post('clear-all-masks-layer', time=0, slice=2)
        progress = self.progress()
        self.assertEqual(progress['mask_labels'], [0] * 6)
        self.assertEqual(progress['edited_labels'], [0, 0, 7, 0, 0, 7])
        self.post('undo'); restored = self.progress()
        for key in ['mask_labels', 'edited_labels', 'edited_at', 'latest_edit_at']:
            self.assertEqual(restored[key], before[key], key)
        self.post('redo')
        self.assertEqual(self.progress()['edited_at'], progress['edited_at'])
        self.assertEqual(self.progress()['edited_labels'], progress['edited_labels'])

    def test_mask_import_is_baseline_and_binary_import_preserves_locked_manual_edits(self):
        self.brush(label=2, t=1, z=2)
        self.post('lock', organ=2, locked=True)
        mask = np.zeros(self.project.voxels.shape, np.uint8); mask[0, 1, 3:5, 4:6] = 1
        image = sitk.GetImageFromArray(mask, isVector=False); image.CopyInformation(self.project.image)
        self.assertEqual(self.annotation(image, binary=1).status_code, 200)
        progress = self.progress()
        self.assertEqual(progress['edited_labels'], [0, 0, 0, 0, 0, 2])
        self.assertEqual(progress['mask_labels'], [0, 1, 0, 0, 0, 2])
        self.post('lock', organ=2, locked=False)
        self.assertEqual(self.annotation(image).status_code, 200)
        self.assertEqual(self.progress()['edited_labels'], [0] * 6)

    def test_notes_lock_reference_and_clearing_prompts_do_not_create_extra_edited_planes(self):
        self.brush(); before = self.progress()
        self.client.post(self.prefix + '/notes', json={'revision': 0, 'text': '已完成。', 'review_status': 'complete'})
        self.post('lock', organ=3, locked=True)
        self.post('clear-points', time=1, slice=2, organ=2)
        reference = sitk.GetImageFromArray(np.zeros(self.project.voxels.shape, np.uint8), isVector=False)
        reference.CopyInformation(self.project.image)
        self.assertEqual(self.annotation(reference, target='reference').status_code, 200)
        after = self.progress()
        for key in ['mask_labels', 'edited_labels', 'edited_at', 'latest_edit_at']:
            self.assertEqual(after[key], before[key])

    def test_save_reopen_and_portable_backup_preserve_progress_without_prior_reads(self):
        self.brush(t=1, z=2); self.brush(label=3, t=0, z=1)
        self.post('undo'); self.post('redo'); before = self.progress(); self.post('save')
        reopened = ProjectStore(self.app.state.store.directory).get(self.id)
        self.assertEqual(summary(reopened), before); self.assertEqual(reopened.prior_store.reads, 0)
        backup = self.client.get(self.prefix + '/export?kind=study')
        restored = self.client.post('/api/projects/import-backup', files={'file': ('study.zip', backup.content)})
        self.assertEqual(restored.status_code, 200, restored.text)
        imported = self.client.get('/api/projects/' + restored.json()['id'] + '/progress').json()
        imported.pop('study_id'); before.pop('study_id'); self.assertEqual(imported, before)

    def test_legacy_events_reconstruct_effective_edits_and_undo_redo(self):
        self.brush(t=0, z=1); self.brush(label=2, t=1, z=2)
        self.post('clear-all-masks-layer', time=0, slice=1); self.post('undo')
        expected = self.progress()
        for event in self.project.events:
            event.pop('annotation_scope', None); event.pop('annotation_effect', None)
        self.project.progress_cache = None
        self.assertEqual(self.progress(), expected)
        self.post('redo'); after = self.progress()
        self.assertEqual(after['edited_labels'], [0, 7, 0, 0, 7, 2])

    def test_cached_progress_scans_only_the_changed_layer_and_no_priors(self):
        before = self.progress(); self.assertEqual(self.project.progress_last_scanned, 6)
        self.assertIs(summary(self.project), self.project.progress_cache)
        self.brush(t=1, z=2)
        after = self.progress(); self.assertEqual(self.project.progress_last_scanned, 1)
        self.assertEqual(self.project.prior_store.reads, 0)
        self.assertEqual(after['mask_labels'][:5], before['mask_labels'][:5])
        self.post('lock', organ=3, locked=True); self.progress()
        self.assertEqual(self.project.progress_last_scanned, 0)

    def test_failed_inference_does_not_record_annotation(self):
        before = self.progress(); self.predictor.fail = True
        with self.assertLogs('server.inference', level='ERROR') as logged:
            self.assertEqual(self.infer()['state'], 'failed')
        self.assertEqual([(record.getMessage(), str(record.exc_info[1])) for record in logged.records],
                         [('Inference task failed', 'Deliberate test prediction failure')])
        self.assertEqual(self.progress(), before)

    def test_missing_legacy_history_is_explicitly_reported(self):
        self.brush(); self.project.events.clear(); self.project.progress_cache = None
        progress = self.progress()
        self.assertFalse(progress['history_complete'])
        self.assertEqual(progress['edited_labels'], [0] * 6)
        self.assertTrue(progress['mask_labels'][1])

    def test_3d_progress_has_one_time_frame_and_original_non512_acquisition_layers(self):
        image = sitk.GetImageFromArray(np.ones((4, 24, 32), np.float32))
        path = self.root / 'image3d.nii.gz'; sitk.WriteImage(image, str(path))
        project = self.app.state.store.create(path, path.name)
        response = self.client.post('/api/projects/' + project.id + '/brush', json={'revision': 0, 'time': 0, 'slice': 3, 'organ': 3, 'points': [[12, 12]], 'radius': 0})
        self.assertEqual(response.status_code, 200, response.text)
        progress = self.client.get('/api/projects/' + project.id + '/progress').json()
        self.assertEqual(progress['shape'], [1, 4]); self.assertEqual(progress['edited_labels'], [0, 0, 0, 4])


if __name__ == '__main__':
    unittest.main()
