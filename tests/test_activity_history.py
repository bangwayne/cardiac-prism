"""Readable operation scope, pagination and persistence without array reads."""
from types import SimpleNamespace
import unittest

import test_platform as fixtures
import test_temporal_propagation as temporal
from server.activity_history import activity
from server.domain import ProjectStore


def fake_project(events, frames=5, depth=4):
    records = [dict(revision=index + 1, time='2026-10-12T12:00:00Z', **event)
               for index, event in enumerate(events)]
    return SimpleNamespace(id='metadata-only-study', revision=len(records), events=records,
                           voxels=SimpleNamespace(shape=(frames, depth, 12, 16)))


class ActivityMetadataTests(unittest.TestCase):
    def test_sparse_scope_filter_does_not_match_cartesian_frame_layer_combinations(self):
        project = fake_project([dict(action='segment_all_time_frames', organs=[1, 2, 3],
            annotation_targets=[[0, 1, 1], [0, 1, 2], [2, 3, 2]],
            skipped_locked_slices=[[1, 1], [2, 2]])])
        result = activity(project)
        entry = result['entries'][0]
        self.assertEqual(entry['time_frames'], [0, 2])
        self.assertEqual(entry['slices'], [1, 3])
        self.assertEqual(entry['target_count'], 2)
        self.assertEqual(entry['prediction_count'], 3)
        self.assertIn('2 slices across 2 time frames', entry['text'])
        self.assertIn('frame 2, layer 2', entry['text'])
        self.assertEqual(entry['skipped_locked_slices'], [[1, 1], [2, 2]])
        self.assertEqual(activity(project, time=0, slice=3)['entries'], [])
        self.assertEqual(len(activity(project, time=2, slice=3)['entries']), 1)

    def test_pagination_newest_first_including_filtered_pages(self):
        project = fake_project([dict(action='brush', time_frame=index % 2, slice=1, organ=2)
                                for index in range(11)])
        first = activity(project, limit=3)
        self.assertEqual([entry['revision'] for entry in first['entries']], [11, 10, 9])
        self.assertEqual(first['next_before_revision'], 9)
        second = activity(project, before_revision=9, limit=3)
        self.assertEqual([entry['revision'] for entry in second['entries']], [8, 7, 6])
        filtered = activity(project, time=1, slice=1, limit=2)
        self.assertEqual([entry['revision'] for entry in filtered['entries']], [10, 8])
        older = activity(project, time=1, slice=1, before_revision=8, limit=20)
        self.assertEqual([entry['revision'] for entry in older['entries']], [6, 4, 2])
        self.assertIsNone(older['next_before_revision'])
        self.assertTrue(first['history_complete'])

    def test_explicit_and_legacy_click_polarity_are_distinguished_from_unknown(self):
        project = fake_project([
            dict(action='inference', organ=2, time_frame=0, slice=1, mode='organ_click',
                 point={'x': 6, 'y': 5, 'positive': False}),
            dict(action='inference', organ=2, time_frame=1, slice=2, mode='click',
                 points=[{'x': 3, 'y': 7, 'positive': True}]),
            dict(action='inference', organ=2, time_frame=2, slice=0, mode='click',
                 points=[{'x': 2, 'y': 3}]),
            dict(action='inference', organ=2, time_frame=2, slice=0),
        ])
        rows = {entry['revision']: entry for entry in activity(project)['entries']}
        self.assertIn('negative', rows[1]['text'])
        self.assertIn('frame 1, layer 2', rows[1]['text'])
        self.assertFalse(rows[1]['click']['positive'])
        self.assertTrue(rows[2]['click']['positive'])
        self.assertIn('polarity not recorded', rows[3]['text'])
        self.assertIsNone(rows[3]['click']['positive'])
        self.assertNotIn('click', rows[4])
        self.assertIn('interaction mode was not recorded', rows[4]['text'])

    def test_propagation_lists_selected_references_and_actual_targets_only(self):
        project = fake_project([dict(action='propagate_time', slice=2, organs=[2],
            reference_times=[0, 2, 4], changed_times=[0, 1, 3], skipped_locked_slices=[[2, 2]])])
        entry = activity(project)['entries'][0]
        self.assertEqual(entry['time_frames'], [1, 3])
        self.assertEqual(entry['reference_times'], [0, 2, 4])
        self.assertIn('3 selected reference frames (1, 3, 5)', entry['text'])
        self.assertIn('layer 3 in 2 time frames (2, 4)', entry['text'])
        self.assertEqual(activity(project, time=0, slice=2)['entries'], [])

    def test_lock_and_legacy_undo_redo_descriptions_preserve_actual_scope(self):
        project = fake_project([
            dict(action='brush', time_frame=1, slice=3, organ=2),
            dict(action='lock_slices', changed_slices=[[0, 0], [2, 1]]),
            dict(action='unlock_slices', changed_slices=[[0, 0]]),
            dict(action='undo', description='Brush stroke'),
            dict(action='redo', description='Brush stroke'),
        ])
        rows = {entry['revision']: entry for entry in activity(project)['entries']}
        self.assertEqual(rows[4]['time_frames'], [1])
        self.assertEqual(rows[4]['slices'], [3])
        self.assertIn('Undid "Brush stroke" on frame 2, layer 4', rows[4]['text'])
        self.assertIn('Redid', rows[5]['text'])
        self.assertEqual(rows[2]['target_count'], 2)
        self.assertEqual(activity(project, time=0, slice=1)['entries'], [])

    def test_invalid_queries_and_missing_history_are_explicit(self):
        project = fake_project([dict(action='brush', time_frame=0, slice=0, organ=1)])
        for options in ({'limit': 0}, {'limit': 201}, {'before_revision': 0}, {'time': 5}, {'slice': -1}):
            with self.assertRaises(ValueError):
                activity(project, **options)
        project.events[0]['revision'] = 2; project.revision = 2
        self.assertFalse(activity(project)['history_complete'])
        project.events = [dict(action='undo', revision=2, changed_times=[0])]
        result = activity(project)
        self.assertFalse(result['history_complete'])
        self.assertFalse(result['entries'][0]['scope_complete'])
        self.assertIn('scope not recorded', result['entries'][0]['text'])


class ActivityAPITests(unittest.TestCase):
    setUp = fixtures.PlatformTests.setUp
    tearDown = fixtures.PlatformTests.tearDown
    post = fixtures.PlatformTests.post
    brush = fixtures.PlatformTests.brush
    infer = fixtures.PlatformTests.infer
    wait_job = fixtures.PlatformTests.wait_job

    def test_slice_lock_and_partial_batch_history_report_only_actual_changed_planes(self):
        self.brush(label=2, t=0, z=1)
        self.post('slice-locks', times=[0], slices=[1], locked=True)
        locked = activity(self.project)['entries'][0]
        self.assertEqual(locked['action'], 'lock_slices')
        self.assertEqual(locked['time_frames'], [0]); self.assertEqual(locked['slices'], [1])
        self.assertIn('Locked frame 1, layer 2', locked['text'])
        job = self.wait_job(self.post('segment-all-layers', time=0)['id'])
        self.assertEqual(job['state'], 'completed', job)
        batch = activity(self.project)['entries'][0]
        self.assertEqual(batch['slices'], [0, 2])
        self.assertEqual(batch['target_count'], 2)
        self.assertEqual(batch['skipped_locked_slices'], [[0, 1]])
        self.assertIn('Preserved 1 locked slice (frame 1, layer 2)', batch['text'])
        on_locked = activity(self.project, time=0, slice=1)
        self.assertEqual([entry['action'] for entry in on_locked['entries']], ['lock_slices', 'brush'])
        self.assertEqual(self.project.prior_store.reads, 0)

    def test_api_save_reopen_and_backup_preserve_all_recorded_operations_without_prior_reads(self):
        self.brush(label=2, t=1, z=2)
        self.assertEqual(self.infer(label=2, t=1, z=2,
            point={'x': 6, 'y': 5, 'positive': False})['state'], 'completed')
        self.post('undo'); self.post('redo')
        self.post('save')
        before = activity(self.project)
        count = len(before['entries'])
        self.post('save')
        self.assertEqual(len(activity(self.project)['entries']), count, 'Do not invent save audit events or revisions')
        self.assertEqual(self.project.revision, before['revision'])
        response = self.client.get(self.prefix + '/activity?limit=2&time=1&slice=2')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([entry['action'] for entry in response.json()['entries']], ['redo', 'undo'])
        reopened = ProjectStore(self.app.state.store.directory).get(self.id)
        self.assertEqual(activity(reopened), before)
        self.assertEqual(reopened.prior_store.reads, 0)
        self.app.state.store.projects[self.id] = reopened
        self.client.get(self.prefix + '/activity').raise_for_status()
        self.assertEqual(reopened.prior_store.reads, 0)
        backup = self.client.get(self.prefix + '/export?kind=study')
        restored = self.client.post('/api/projects/import-backup', files={'file': ('study.zip', backup.content)})
        self.assertEqual(restored.status_code, 200, restored.text)
        imported = self.client.get('/api/projects/' + restored.json()['id'] + '/activity').json()
        imported.pop('study_id'); before.pop('study_id')
        self.assertEqual(imported, before)


class PropagationActivityTests(unittest.TestCase):
    setUp = temporal.TemporalPropagationTests.setUp
    tearDown = temporal.TemporalPropagationTests.tearDown
    post = temporal.TemporalPropagationTests.post
    brush = temporal.TemporalPropagationTests.brush
    wait_job = temporal.TemporalPropagationTests.wait_job
    request = temporal.TemporalPropagationTests.request
    propagate = temporal.TemporalPropagationTests.propagate
    reference_predictor = temporal.TemporalPropagationTests.reference_predictor
    drain = temporal.TemporalPropagationTests.drain

    def test_failed_and_cancelled_jobs_are_not_misrepresented_as_successful_propagation(self):
        before = activity(self.project)
        self.predictor.fail = True
        self.assertEqual(self.propagate(organs=[2])['state'], 'failed')
        self.assertEqual(activity(self.project), before)
        self.predictor.fail = False; self.predictor.block = True
        self.predictor.entered.clear(); self.predictor.release.clear()
        job = self.post('propagate-time', **self.request(organs=[2]))
        self.assertTrue(self.predictor.entered.wait(5))
        self.client.post('/api/jobs/' + job['id'] + '/cancel').raise_for_status()
        self.predictor.release.set(); self.drain()
        self.assertEqual(activity(self.project), before)
        self.predictor.block = False
        self.assertEqual(self.propagate(organs=[2])['state'], 'completed')
        latest = activity(self.project)['entries'][0]
        self.assertEqual(latest['action'], 'propagate_time')
        self.assertEqual(latest['time_frames'], [1, 3, 5])
        self.assertEqual(latest['slices'], [1])
        self.assertEqual(latest['reference_times'], [0, 2, 4])


if __name__ == '__main__':
    unittest.main()
