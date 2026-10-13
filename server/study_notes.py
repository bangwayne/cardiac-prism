"""Small, independently persisted study notes; never load or rewrite mask arrays."""
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import threading
import uuid


def read_notes(directory):
    path = Path(directory) / 'study-notes.json'
    if not path.exists():
        return dict(revision=0, text='', review_status='in_progress', updated_at=None)
    return json.loads(path.read_text())


class StudyNotesStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock = threading.RLock()

    def project_directory(self, project_id):
        if not re.fullmatch(r'[a-f0-9]{32}', project_id):
            raise KeyError('Unknown study.')
        directory = self.directory / project_id
        if not (directory / 'project.json').is_file():
            raise KeyError('Unknown study.')
        return directory

    def get(self, project_id):
        with self.lock:
            return read_notes(self.project_directory(project_id))

    def save(self, project_id, revision, text, review_status):
        with self.lock:
            directory = self.project_directory(project_id)
            previous = read_notes(directory)
            if revision != previous['revision']:
                raise ValueError('This study note changed in another window. Reload its saved note before retrying.')
            if previous['text'] == text and previous['review_status'] == review_status:
                return previous
            record = dict(revision=revision + 1, text=text, review_status=review_status,
                          updated_at=datetime.now(timezone.utc).isoformat())
            path = directory / 'study-notes.json'
            temporary = directory / f'.study-notes-{uuid.uuid4().hex}.tmp'
            try:
                temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            return record
