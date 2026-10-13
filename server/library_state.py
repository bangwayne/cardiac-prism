"""Catalog visibility and permanent exclusion; source scans remain intact."""
import json
from pathlib import Path
import threading
import uuid


class LibraryState:
    def __init__(self, workspace):
        self.path = Path(workspace) / 'library-state.json'
        self.lock = threading.RLock()

    def state(self):
        with self.lock:
            return json.loads(self.path.read_text()) if self.path.exists() else {}

    def hidden(self):
        return set(self.state().get('hidden_examples', []))

    def deleted(self):
        return set(self.state().get('deleted_examples', []))

    def write(self, hidden, deleted):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name('.library-' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_text(json.dumps({'hidden_examples': sorted(hidden),
                                            'deleted_examples': sorted(deleted)}, indent=2) + '\n')
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def set_hidden(self, name, hidden):
        with self.lock:
            names = self.hidden()
            deleted = self.deleted()
            if name in deleted:
                raise ValueError('This example was permanently removed from the library.')
            names.add(name) if hidden else names.discard(name)
            self.write(names, deleted)

    def remove_permanently(self, name):
        with self.lock:
            names, deleted = self.hidden(), self.deleted()
            if name not in names:
                raise ValueError('Remove this example before permanently removing it from the library.')
            names.remove(name); deleted.add(name)
            self.write(names, deleted)
