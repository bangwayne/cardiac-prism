"""Revision-checked study completion and deletion, isolated from inference."""
import logging
import shutil
import uuid

log = logging.getLogger(__name__)


def check(project, revision):
    project.check_revision(revision)
    if project.pending_job:
        raise ValueError('Wait for the active inference or cancel it before managing this study.')


def current_note(notes, project_id, revision):
    note = notes.get(project_id)
    if note['revision'] != revision:
        raise ValueError('This study note changed in another window. Refresh before managing this study.')
    return note


def set_completion(project, notes, revision, notes_revision, review_status):
    with project.lock, notes.lock:
        check(project, revision)
        current_note(notes, project.id, notes_revision)
    if review_status == 'complete':
        # save() takes save_lock before project.lock. Do not reverse that order.
        project.save()
    with project.lock, notes.lock:
        check(project, revision)
        note = current_note(notes, project.id, notes_revision)
        if review_status == 'complete' and project.saved_revision != revision:
            raise ValueError('Save the latest annotation before marking this study complete.')
        return notes.save(project.id, notes_revision, note['text'], review_status)


def delete_study(store, notes, project, revision, notes_revision):
    # Wait for any backup/save snapshot, then atomically remove the active folder.
    with project.save_lock, project.lock, notes.lock, store.lock:
        check(project, revision)
        current_note(notes, project.id, notes_revision)
        if project.directory.is_symlink() or project.directory.parent.resolve() != store.directory.resolve():
            raise ValueError('Invalid saved-study directory.')
        deleted = store.directory.parent / '.study-deletions'
        deleted.mkdir(exist_ok=True)
        target = deleted / (project.id + '-' + uuid.uuid4().hex)
        project.directory.rename(target)
        project.deleted = True
        store.projects.pop(project.id, None)
        return target


def cleanup_deleted(directory):
    try:
        shutil.rmtree(directory)
    except OSError:
        log.exception('Could not finish clearing detached study %s', directory)
