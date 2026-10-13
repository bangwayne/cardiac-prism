"""Readable, paginated study operations from persisted annotation events.

This is a metadata view: it never reads masks, priors or image values, never
changes revisions, and does not invent navigation, save or failed-job events.
"""
import math
import heapq

from .annotation_progress import COMMITS

ORGAN_NAMES = {1: 'RV', 2: 'MYO', 3: 'LV'}
LOCK_ACTIONS = {'lock_slices', 'unlock_slices', 'lock_frames', 'unlock_frames', 'lock', 'unlock'}
GLOBAL_ACTIONS = {'initialize_paired_annotation', 'import_reference', 'save', 'save_study'}


def _index(value, size):
    return type(value) is int and 0 <= value < size


def _labels(event, locked):
    values = event.get('affected_organs', event.get('organs', [event['organ']] if event.get('organ') else [1, 2, 3]))
    if not isinstance(values, (list, tuple)):
        return [], False
    result = [label for label in values if type(label) is int and label in ORGAN_NAMES]
    if event.get('action') == 'import_annotation':
        binary = event.get('binary_organ', 0)
        result = [label for label in result if label not in locked and (not binary or label == binary)]
    return sorted(set(result)), len(result) == len(values) or event.get('action') == 'import_annotation'


def _coverage(event, frames, depth, locked):
    """Return exact (t,z)->organs coverage, not a Cartesian guess for sparse jobs."""
    action = event.get('action')
    labels, complete = _labels(event, locked)
    if 'annotation_targets' in event:
        result = {}
        targets = event['annotation_targets']
        if not isinstance(targets, list):
            return {}, False
        for key in targets:
            if (not isinstance(key, (list, tuple)) or len(key) != 3
                    or not _index(key[0], frames) or not _index(key[1], depth)
                    or type(key[2]) is not int or key[2] not in ORGAN_NAMES):
                complete = False
                continue
            result.setdefault(tuple(key[:2]), set()).add(key[2])
        return result, complete
    if 'changed_slices' in event:
        result = {}
        pairs = event['changed_slices']
        if not isinstance(pairs, list):
            return {}, False
        for pair in pairs:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or not _index(pair[0], frames) or not _index(pair[1], depth)):
                complete = False
                continue
            result[tuple(pair)] = set(labels)
        return result, complete
    if action in ('undo', 'redo') and 'annotation_scope' not in event:
        # Older undo records may identify frames only. Their exact layer scope
        # comes from replaying the matching original edit below, never from
        # assuming that every layer in a listed frame was restored.
        return {}, False
    scope = event.get('annotation_scope')
    if scope is not None:
        if not isinstance(scope, (list, tuple)) or len(scope) != 3 or not isinstance(scope[2], (list, tuple)):
            return {}, False
        t, z, values = scope
        labels = [label for label in values if type(label) is int and label in ORGAN_NAMES]
        complete = complete and len(labels) == len(values)
    else:
        t, z = event.get('time_frame'), event.get('slice')
    if (t is not None and not _index(t, frames)) or (z is not None and not _index(z, depth)):
        return {}, False
    if action == 'propagate_time':
        times = event.get('changed_times')
        if z is None or not isinstance(times, list):
            return {}, False
        references = event.get('reference_times', [])
        times = [value for value in times if value not in references]
    elif isinstance(event.get('changed_times'), list):
        times = event['changed_times']
    elif t is not None:
        times = [t]
    elif scope is not None or action in GLOBAL_ACTIONS | {'lock', 'unlock', 'import_annotation',
            'import_paired_annotation', 'import_evaluated_predictions', 'segment_all_time_frames',
            'clear_all_masks_layer'}:
        times = list(range(frames))
    else:
        # An old undo or unrecognized event without a recorded scope must not
        # claim to have changed every layer of the study.
        return {}, False
    valid_times = [value for value in times if _index(value, frames)]
    complete = complete and len(valid_times) == len(times)
    if z is None and scope is None and action in {'inference', 'segment_all', 'brush', 'erase',
                                                'clear_slice', 'clear_all_masks_slice', 'clear_points', 'resolve_overlaps'}:
        return {}, False
    layers = range(depth) if z is None else [z]
    return {(time, layer): set(labels) for time in valid_times for layer in layers}, complete


def _numbers(values):
    """Compact one-based UI numbering while keeping payload indices zero-based."""
    values = sorted(set(values))
    ranges, index = [], 0
    while index < len(values):
        first = last = values[index]
        while index + 1 < len(values) and values[index + 1] == last + 1:
            index += 1
            last = values[index]
        ranges.append(str(first + 1) if first == last else f'{first + 1}–{last + 1}')
        index += 1
    return ', '.join(ranges)


def _scope_text(coverage, depth):
    if not coverage:
        return 'scope not recorded'
    times = sorted({key[0] for key in coverage})
    layers = sorted({key[1] for key in coverage})
    if len(coverage) == 1:
        return f'frame {times[0] + 1}, layer {layers[0] + 1}'
    if len(times) == 1 and len(layers) == depth:
        return f'all {depth} layers of frame {times[0] + 1}'
    if len(layers) == 1:
        return f'layer {layers[0] + 1} in {len(times)} time frames ({_numbers(times)})'
    if len(coverage) == len(times) * len(layers):
        return f'{len(layers)} layers ({_numbers(layers)}) in {len(times)} time frame{"s" if len(times) != 1 else ""} ({_numbers(times)})'
    return (f'{len(coverage)} slices across {len(times)} time frames ({_numbers(times)}), '
            f'layers {_numbers(layers)}')


def _organ_text(values):
    return ', '.join(ORGAN_NAMES[label] for label in values) or 'structures not recorded'


def _click(event):
    point = event.get('point')
    if not isinstance(point, dict):
        points = event.get('points')
        point = points[-1] if isinstance(points, list) and points and isinstance(points[-1], dict) else {}
    result = {'positive': point.get('positive') if type(point.get('positive')) is bool else None}
    polarity = event.get('click_polarity')
    if result['positive'] is None and polarity in ('positive', 'negative'):
        result['positive'] = polarity == 'positive'
    for axis in ('x', 'y'):
        value = point.get(axis)
        if type(value) in (int, float) and math.isfinite(value):
            result[axis] = value
    return result


def _skipped(event, frames, depth):
    frame_values = event.get('skipped_locked_frames', [])
    frame_values = frame_values if isinstance(frame_values, list) else []
    skipped_frames = sorted({value for value in frame_values if _index(value, frames)})
    pairs = event.get('skipped_locked_slices', [])
    pairs = pairs if isinstance(pairs, list) else []
    skipped_slices = sorted({tuple(pair) for pair in pairs if isinstance(pair, (list, tuple)) and len(pair) == 2
                            and _index(pair[0], frames) and _index(pair[1], depth)})
    parts = []
    if skipped_frames:
        parts.append(f'Preserved locked time frames {_numbers(skipped_frames)}.')
    if skipped_slices:
        examples = '; '.join(f'frame {t + 1}, layer {z + 1}' for t, z in skipped_slices[:8])
        rest = f'; and {len(skipped_slices) - 8} more' if len(skipped_slices) > 8 else ''
        parts.append(f'Preserved {len(skipped_slices)} locked slice{"s" if len(skipped_slices) != 1 else ""} ({examples}{rest}).')
    return skipped_frames, [list(pair) for pair in skipped_slices], ' '.join(parts)


def _entry(event, revision, coverage, complete, frames, depth):
    action = str(event.get('action', 'unknown'))
    times = sorted({key[0] for key in coverage})
    layers = sorted({key[1] for key in coverage})
    organs = sorted({label for values in coverage.values() for label in values})
    scope, organ_text = _scope_text(coverage, depth), _organ_text(organs)
    selected = ORGAN_NAMES.get(event.get('organ'), organ_text)
    result = dict(id=f'event-{revision}', revision=revision, at=event.get('time'), action=action,
                  time_frames=times, slices=layers, organs=organs, target_count=len(coverage),
                  organ_plane_count=sum(len(values) for values in coverage.values()),
                  prediction_count=sum(len(values) for values in coverage.values()) if action in {
                      'inference', 'segment_all', 'segment_all_layers', 'segment_all_time_frames', 'propagate_time'} else 0,
                  scope_complete=complete,
                  outcome='committed' if action in COMMITS | {'undo', 'redo'} else 'recorded')
    has_click = event.get('mode') in ('click', 'organ_click') or isinstance(event.get('point'), dict) or bool(event.get('points'))
    if action == 'inference' and has_click:
        point = _click(event)
        result['click'] = point
        polarity = 'positive' if point['positive'] is True else 'negative' if point['positive'] is False else 'polarity not recorded'
        location = f' at native pixel (x {point["x"]:g}, y {point["y"]:g}; zero-based)' if 'x' in point and 'y' in point else ''
        text = (f'Added a {polarity} AI click{location} and refined {selected} on {scope}.'
                if point['positive'] is not None else
                f'Refined {selected} with an AI click ({polarity}){location} on {scope}.')
    elif action == 'inference' and event.get('mode') != 'organ':
        text = f'Ran AI segmentation for {selected} on {scope}; interaction mode was not recorded.'
    elif action in ('inference', 'segment_all', 'segment_all_layers', 'segment_all_time_frames'):
        text = f'Segmented {organ_text} on {scope}.'
    elif action in ('brush', 'erase'):
        text = f'{"Painted" if action == "brush" else "Erased"} {selected} with the {"brush" if action == "brush" else "eraser"} on {scope}.'
        others = [label for label in organs if label != event.get('organ')]
        if action == 'brush' and others:
            text += f' Also updated {_organ_text(others)} where the stroke replaced existing labels.'
    elif action == 'propagate_time':
        refs = event.get('reference_times', [])
        refs = [value for value in refs if _index(value, frames)] if isinstance(refs, list) else []
        result['reference_times'] = refs
        sources = f'{len(refs)} selected reference frame{"s" if len(refs) != 1 else ""} ({_numbers(refs)})' if refs else 'reference frames not recorded'
        text = f'Propagated {organ_text} from {sources} to {scope}.'
    elif action in ('clear_slice', 'clear_all_masks_slice', 'clear_all_masks_layer'):
        text = f'Cleared {organ_text} masks on {scope}.'
    elif action == 'clear_points':
        text = f'Cleared {organ_text} click prompts on {scope}; kept the existing mask.'
    elif action == 'resolve_overlaps':
        priority = ORGAN_NAMES.get(event.get('priority'), 'selected structure')
        text = f'Resolved overlapping labels on {scope}, giving {priority} priority among unlocked structures.'
    elif action in ('undo', 'redo'):
        description = str(event.get('description', 'annotation edit'))[:180]
        text = f'{"Undid" if action == "undo" else "Redid"} "{description}" on {scope}.'
    elif action in ('lock_slices', 'unlock_slices', 'lock_frames', 'unlock_frames'):
        text = f'{"Unlocked" if action.startswith("unlock") else "Locked"} {scope} · RV + MYO + LV.'
    elif action in ('lock', 'unlock'):
        text = f'{"Unlocked" if action == "unlock" else "Locked"} {selected} throughout the study.'
    elif action in ('import_annotation', 'import_paired_annotation', 'initialize_paired_annotation', 'import_evaluated_predictions', 'import_reference'):
        kind = {'import_annotation': 'annotation', 'import_paired_annotation': 'matched annotation',
                'initialize_paired_annotation': 'initial matched annotation', 'import_evaluated_predictions': 'evaluated example masks',
                'import_reference': 'reference overlay'}[action]
        filename = str(event.get('filename', ''))[:160]
        text = f'Loaded {kind}{f" ({filename})" if filename else ""} on {scope}.'
        if action == 'import_reference':
            text += ' Editable masks were unchanged.'
    elif action in ('save', 'save_study'):
        text = 'Saved the study.'
    else:
        text = f'Recorded operation "{action[:100]}"; detailed operation information is unavailable.'
    skipped_frames, skipped_slices, skipped_text = _skipped(event, frames, depth)
    result.update(skipped_locked_frames=skipped_frames, skipped_locked_slices=skipped_slices,
                  text=text + (' ' + skipped_text if skipped_text else ''))
    return result


def activity(project, before_revision=None, limit=50, time=None, slice=None):
    """Return newest-first persisted operations; query indices remain zero-based."""
    frames, depth = project.voxels.shape[:2]
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError('Activity page size must be between 1 and 200.')
    if before_revision is not None and (type(before_revision) is not int or before_revision < 1):
        raise ValueError('Activity before_revision must be a positive integer.')
    if time is not None and not _index(time, frames) or slice is not None and not _index(slice, depth):
        raise ValueError('Activity frame or layer is outside the image.')
    candidates, undo, redo, organ_locks = [], [], [], set()
    previous, complete = 0, True
    for index, event in enumerate(project.events):
        if not isinstance(event, dict):
            complete = False
            continue
        revision = event.get('revision', index + 1)
        if type(revision) is not int:
            complete = False
            continue
        if revision != previous + 1 or 'revision' not in event:
            complete = False
        previous = revision
        action = event.get('action')
        coverage, known = _coverage(event, frames, depth, organ_locks)
        if action in ('undo', 'redo'):
            source, destination = (redo, undo) if action == 'redo' else (undo, redo)
            if source:
                saved = source.pop()
                destination.append(saved)
                if 'annotation_scope' not in event and 'annotation_targets' not in event and 'changed_slices' not in event:
                    coverage, known = saved
            elif not known:
                complete = False
        elif action in COMMITS or 'annotation_scope' in event and action not in LOCK_ACTIONS:
            undo.append((coverage, known)); del undo[:-40]; redo.clear()
        if action in ('lock', 'unlock') and event.get('organ') in ORGAN_NAMES:
            organ_locks.add(event['organ']) if action == 'lock' else organ_locks.discard(event['organ'])
        complete = complete and known
        if (before_revision is not None and revision >= before_revision
                or time is not None and not any(t == time and (slice is None or z == slice) for t, z in coverage)
                or time is None and slice is not None and not any(z == slice for _, z in coverage)):
            continue
        candidate = (revision, index, event, coverage, known)
        if len(candidates) < limit + 1:
            heapq.heappush(candidates, candidate)
        else:
            heapq.heappushpop(candidates, candidate)
    complete = complete and previous == project.revision
    more = len(candidates) > limit
    entries = [_entry(event, revision, coverage, known, frames, depth)
               for revision, _, event, coverage, known in sorted(candidates, reverse=True)[:limit]]
    return dict(study_id=project.id, revision=project.revision, entries=entries,
                next_before_revision=entries[-1]['revision'] if more and entries else None,
                history_complete=complete)
