"""Propagation lineage stays local, effective, persistent and independent of priors."""
import unittest

import test_temporal_propagation as fixtures
from server.annotation_progress import _history_state, summary
from server.domain import ProjectStore


class AnnotationOriginTests(unittest.TestCase):
    setUp = fixtures.TemporalPropagationTests.setUp
    tearDown = fixtures.TemporalPropagationTests.tearDown
    post = fixtures.TemporalPropagationTests.post
    brush = fixtures.TemporalPropagationTests.brush
    infer = fixtures.TemporalPropagationTests.infer
    wait_job = fixtures.TemporalPropagationTests.wait_job
    request = fixtures.TemporalPropagationTests.request
    propagate = fixtures.TemporalPropagationTests.propagate
    reference_predictor = fixtures.TemporalPropagationTests.reference_predictor
    drain = fixtures.TemporalPropagationTests.drain

    def test_failed_and_cancelled_propagation_add_no_current_or_historical_origin(self):
        self.predictor.fail = True
        self.assertEqual(self.propagate(organs=[2])['state'], 'failed')
        self.assertFalse(any(summary(self.project)['propagated_labels']))
        self.assertFalse(summary(self.project)['propagation_history'])
        self.predictor.fail = False
        self.predictor.block = True
        self.predictor.entered.clear(); self.predictor.release.clear()
        job = self.post('propagate-time', **self.request(organs=[2]))
        self.assertTrue(self.predictor.entered.wait(5))
        self.client.post('/api/jobs/' + job['id'] + '/cancel').raise_for_status()
        self.predictor.release.set()
        self.assertEqual(self.wait_job(job['id'])['state'], 'cancelled')
        self.drain()
        self.assertFalse(any(summary(self.project)['ever_propagated_labels']))
        self.assertFalse(summary(self.project)['propagation_history'])

    def test_propagation_current_lineage_survives_manual_edits_then_resets_on_replacement(self):
        self.assertEqual(self.propagate(organs=[2])['state'], 'completed')
        progress = summary(self.project)
        self.assertEqual(progress['propagated_labels'], [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0])
        self.assertEqual(progress['propagation_details']['1:1:2']['reference_times'], [0, 2, 4])
        self.assertTrue(progress['propagated_at'][4])
        self.brush(label=2, t=1, z=1)
        edited = summary(self.project)
        self.assertEqual(edited['propagated_labels'][4], 2)
        self.assertEqual(edited['edited_labels'][4] & 2, 2)
        self.assertEqual(self.infer(label=2, t=1, z=1, mode='organ')['state'], 'completed')
        replaced = summary(self.project)
        self.assertEqual(replaced['propagated_labels'][4], 0)
        self.assertEqual(replaced['ever_propagated_labels'][4], 2)
        self.assertNotIn('1:1:2', replaced['propagation_details'])
        self.assertEqual(len(replaced['propagation_history']['1:1:2']), 1)
        self.post('undo')
        self.assertEqual(summary(self.project)['propagation_details'], edited['propagation_details'])
        self.post('redo')
        self.assertEqual(summary(self.project)['propagation_details'], replaced['propagation_details'])

    def test_clear_resets_current_origin_and_undo_restores_without_prior_reads(self):
        self.assertEqual(self.propagate(organs=[2], target_times=[1])['state'], 'completed')
        before = summary(self.project)
        self.post('save')
        reads = self.project.prior_store.reads
        self.post('clear-slice', time=1, slice=1, organ=2)
        cleared = summary(self.project)
        self.assertEqual(cleared['propagated_labels'][4], 0)
        self.assertEqual(cleared['ever_propagated_labels'][4], 2)
        self.post('undo')
        self.assertEqual(summary(self.project)['propagation_details'], before['propagation_details'])
        self.assertEqual(self.project.prior_store.reads, reads)

    def test_sparse_support_details_and_save_backup_round_trip(self):
        self.post('clear-slice', time=2, slice=1, organ=2)
        self.assertEqual(self.propagate(organs=[2], target_times=[1])['state'], 'completed')
        before = summary(self.project)
        self.assertEqual(before['propagation_details']['1:1:2']['reference_times'], [0, 4])
        self.assertEqual(set(before['propagation_details']), {'1:1:2'})
        self.post('save')
        reopened = ProjectStore(self.app.state.store.directory).get(self.id)
        self.assertEqual(summary(reopened), before)
        self.assertEqual(reopened.prior_store.reads, 0)
        backup = self.client.get(self.prefix + '/export?kind=study')
        restored = self.client.post('/api/projects/import-backup', files={'file': ('study.zip', backup.content)})
        self.assertEqual(restored.status_code, 200, restored.text)
        imported = self.client.get('/api/projects/' + restored.json()['id'] + '/progress').json()
        imported.pop('study_id'); before.pop('study_id')
        self.assertEqual(imported, before)


class LegacyOriginReplayTests(unittest.TestCase):
    def test_legacy_changed_times_excludes_reference_and_incomplete_history_is_unknown(self):
        event = {'action': 'propagate_time', 'revision': 1, 'slice': 2,
                 'reference_times': [0], 'changed_times': [1, 3], 'organs': [2], 'time': '2026-10-12T12:00:00Z'}
        edits, complete, origins, traces, known = _history_state([event], 4, 3)
        self.assertTrue(complete and known)
        self.assertEqual(set(origins), {(1, 2, 2), (3, 2, 2)})
        self.assertEqual(set(traces), set(origins))
        incomplete = {key: value for key, value in event.items() if key != 'changed_times'}
        self.assertFalse(_history_state([incomplete], 4, 3)[4])
        self.assertFalse(_history_state([incomplete], 4, 3)[2])

    def test_lock_metadata_does_not_disturb_effective_propagation_undo_redo(self):
        event = {'action': 'propagate_time', 'revision': 1, 'slice': 0,
                 'reference_times': [0], 'changed_times': [1], 'organs': [1], 'time': 'stamp'}
        events = [event, {'action': 'lock_frames', 'revision': 2, 'changed_times': [1]},
                  {'action': 'unlock_frames', 'revision': 3, 'changed_times': [1]},
                  {'action': 'undo', 'revision': 4}]
        state = _history_state(events, 2, 1)
        self.assertFalse(state[2]); self.assertIn((1, 0, 1), state[3])
        events.append({'action': 'redo', 'revision': 5})
        self.assertIn((1, 0, 1), _history_state(events, 2, 1)[2])


if __name__ == '__main__':
    unittest.main()
