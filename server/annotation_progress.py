"""Small frame/layer edit summaries; no probability priors are read."""
import numpy as np

MANUAL = {'brush', 'erase', 'clear_slice', 'clear_all_masks_slice',
          'clear_all_masks_layer', 'resolve_overlaps'}
BASELINE = {'segment_all', 'segment_all_layers', 'segment_all_time_frames',
            'import_annotation', 'import_paired_annotation', 'import_evaluated_predictions', 'propagate_time'}
COMMITS = MANUAL | BASELINE | {'inference', 'clear_points'}


def effect(action, details):
    if action in MANUAL:
        return 'edited'
    if action == 'inference':
        return 'edited' if details.get('mode') != 'organ' and details.get('points') else 'baseline'
    return 'baseline' if action in BASELINE else 'none'


def event_details(project, snapshot, action, details):
    t, z, labels = snapshot.scope if snapshot.scope is not None else (None, None, (1, 2, 3))
    labels = (labels,) if isinstance(labels, int) else labels
    if action == 'import_annotation':
        binary = details.get('binary_organ', 0)
        labels = [label for label in labels if not project.locks[label] and (not binary or label == binary)]
    result = dict(annotation_scope=[t, z, list(labels)], annotation_effect=effect(action, details))
    if snapshot.planes is not None:
        result['annotation_targets'] = [list(key) for key in snapshot.planes]
    if action == 'propagate_time':
        # Capture only small lineage metadata before a later brush replaces the
        # per-plane context. Accessing context dictionaries never reads priors.
        targets = snapshot.planes or ()
        result['propagation_details'] = {
            ':'.join(map(str, key)): {
                **project.contexts.get(key, {}).get('propagation', {}),
                'checkpoint_sha256': details.get('checkpoint_sha256'),
            } for key in targets
        }
    return result


def legacy_scope(event, locked):
    action = event['action']
    labels = event.get('affected_organs', event.get('organs', [event.get('organ')] if event.get('organ') else [1, 2, 3]))
    t, z = event.get('time_frame'), event.get('slice')
    if action == 'import_annotation':
        binary = event.get('binary_organ', 0)
        labels = [label for label in (1, 2, 3) if label not in locked and (not binary or label == binary)]
    elif action == 'resolve_overlaps':
        labels = [label for label in (1, 2, 3) if label not in locked]
    return [t, z, labels]


def _targets(event, locked, frames, depth):
    """Yield actual changed planes, excluding propagation reference frames."""
    scope = event.get('annotation_scope', legacy_scope(event, locked))
    if not isinstance(scope, (list, tuple)) or len(scope) != 3:
        return [], False
    t, z, labels = scope
    valid_index = lambda value, limit: isinstance(value, int) and not isinstance(value, bool) and 0 <= value < limit
    if ((t is not None and not valid_index(t, frames))
            or (z is not None and not valid_index(z, depth))
            or not isinstance(labels, (list, tuple))):
        return [], False
    labels = [label for label in labels if label in (1, 2, 3)]
    if 'annotation_targets' in event:
        targets, valid = [], True
        for key in event['annotation_targets']:
            if (not isinstance(key, (list, tuple)) or len(key) != 3
                    or not valid_index(key[0], frames) or not valid_index(key[1], depth)
                    or key[2] not in (1, 2, 3)):
                valid = False
                continue
            targets.append(tuple(key))
        return targets, valid
    if event.get('action') == 'propagate_time':
        # Old propagation commits already recorded changed_times, but their
        # broad scope also included references; do not mark those as targets.
        times = event.get('changed_times')
        if z is None or not isinstance(times, list):
            return [], False
        times = [time for time in times if valid_index(time, frames)
                 and time not in event.get('reference_times', [])]
    else:
        times = range(frames) if t is None else (t,)
    return [(time, layer, label) for time in times
            for layer in (range(depth) if z is None else (z,)) for label in labels], True


def _history_state(events, frames, depth, contexts=None):
    """Replay tiny metadata states; no masks, priors or cine arrays are copied."""
    edits, origins, traces, undo, redo, locked = {}, {}, {}, [], [], set()
    complete = origin_complete = True
    previous_revision = 0
    for event in events:
        revision = event.get('revision', previous_revision + 1)
        if revision != previous_revision + 1:
            complete = origin_complete = False
        previous_revision = revision
        action = event.get('action')
        if action in ('lock', 'unlock'):
            locked.add(event['organ']) if action == 'lock' else locked.discard(event['organ'])
            continue
        if action in ('lock_frames', 'unlock_frames', 'lock_slices', 'unlock_slices'):
            continue
        if action in ('undo', 'redo'):
            source, destination = (redo, undo) if action == 'redo' else (undo, redo)
            if source:
                destination.append((edits.copy(), origins.copy()))
                edits, origins = source.pop()
            else:
                complete = origin_complete = False
            continue
        if action not in COMMITS and 'annotation_scope' not in event:
            continue
        undo.append((edits.copy(), origins.copy())); del undo[:-40]; redo.clear()
        mode = event.get('annotation_effect', effect(action, event))
        if mode == 'none':
            continue
        targets, valid = _targets(event, locked, frames, depth)
        complete = complete and valid
        origin_complete = origin_complete and valid
        for key in targets:
            if mode == 'baseline':
                edits.pop(key, None)
            elif mode == 'edited':
                edits[key] = event.get('time')
            if action == 'propagate_time':
                encoded = ':'.join(map(str, key))
                recorded = event.get('propagation_details', {}).get(encoded, {})
                # Earlier versions stored the exact nonempty support subset
                # only in contexts. Recover it when its source revision matches
                # this commit; a later replacement context must not be reused.
                if not recorded and contexts is not None:
                    candidate = contexts.get(key, {}).get('propagation', {})
                    if candidate.get('source_revision') == revision - 1:
                        recorded = candidate
                refs = recorded.get('reference_times', event.get('reference_times'))
                if not isinstance(refs, list) or not refs or any(
                        not isinstance(value, int) or not 0 <= value < frames for value in refs):
                    origins.pop(key, None)
                    origin_complete = False
                    continue
                lineage = {'reference_times': list(refs), 'slice': key[1],
                           'propagated_at': event.get('time'),
                           'source_revision': recorded.get('source_revision', event.get('source_revision', revision - 1))}
                if not recorded.get('reference_times') and len(refs) > 1:
                    # A legacy commit proves the selected frames and changed
                    # targets, but not which masks were nonempty per organ.
                    lineage['reference_times_complete'] = False
                    origin_complete = False
                checkpoint = recorded.get('checkpoint_sha256') or event.get('checkpoint_sha256')
                if checkpoint:
                    lineage['checkpoint_sha256'] = checkpoint
                origins[key] = lineage
                traces.setdefault(key, []).append(lineage)
            elif mode == 'baseline' or action in ('clear_slice', 'clear_all_masks_slice', 'clear_all_masks_layer'):
                origins.pop(key, None)
    return edits, complete, origins, traces, origin_complete


def edit_history(events, frames, depth):
    """Compatibility helper for consumers that only need effective edits."""
    edits, complete, _, _, _ = _history_state(events, frames, depth)
    return edits, complete


def summary(project):
    project.check_slice(0, 0)
    if project.progress_cache is not None and project.progress_cache['revision'] == project.revision:
        return project.progress_cache
    frames, depth = project.voxels.shape[:2]
    if project.progress_mask_labels is None:
        labels = np.zeros((frames, depth), np.uint8)
        for label in (1, 2, 3):
            labels |= np.any(project.masks[label], axis=(2, 3)).astype(np.uint8) << (label - 1)
        project.progress_mask_labels = labels
        project.progress_last_scanned = frames * depth
    else:
        project.progress_last_scanned = len(project.progress_dirty_planes)
        for t, z in project.progress_dirty_planes:
            project.progress_mask_labels[t, z] = sum((1 << (label - 1)) for label in (1, 2, 3)
                                                     if np.any(project.masks[label][t, z]))
    project.progress_dirty_planes.clear()
    edits, history_complete, origins, traces, origin_complete = _history_state(
        project.events, frames, depth, project.contexts)
    at_revision = (project.events[-1].get('revision', 0) if project.events else 0) == project.revision
    history_complete = history_complete and at_revision
    edited = np.zeros((frames, depth), np.uint8)
    stamps = [None] * (frames * depth)
    for (t, z, label), stamp in edits.items():
        edited[t, z] |= 1 << (label - 1)
        index = t * depth + z
        if stamp and (stamps[index] is None or stamp > stamps[index]):
            stamps[index] = stamp
    propagated = np.zeros((frames, depth), np.uint8)
    ever_propagated = np.zeros((frames, depth), np.uint8)
    propagated_at = [None] * (frames * depth)
    for (t, z, label), origin in origins.items():
        propagated[t, z] |= 1 << (label - 1)
        stamp, index = origin['propagated_at'], t * depth + z
        if stamp and (propagated_at[index] is None or stamp > propagated_at[index]):
            propagated_at[index] = stamp
    for t, z, label in traces:
        ever_propagated[t, z] |= 1 << (label - 1)
    project.progress_cache = dict(study_id=project.id, revision=project.revision,
                                 shape=[frames, depth], mask_labels=project.progress_mask_labels.reshape(-1).tolist(),
                                 edited_labels=edited.reshape(-1).tolist(), edited_at=stamps,
                                 locked_frames=sorted(getattr(project, 'locked_frames', set())),
                                 locked_slices=[list(key) for key in sorted(getattr(project, 'locked_slices', set()))],
                                 propagated_labels=propagated.reshape(-1).tolist(), propagated_at=propagated_at,
                                 ever_propagated_labels=ever_propagated.reshape(-1).tolist(),
                                 propagation_details={':'.join(map(str, key)): value for key, value in origins.items()},
                                 propagation_history={':'.join(map(str, key)): value for key, value in traces.items()},
                                 origin_history_complete=origin_complete and at_revision,
                                 history_complete=history_complete,
                                 latest_edit_at=max((stamp for stamp in stamps if stamp), default=None))
    return project.progress_cache
