"""Release packaging rejects mismatched models and safely resumes draft uploads."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("assemble_release", ROOT / "scripts/assemble_release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseAssetsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.models = self.root / "models"
        self.output = self.root / "output"
        self.content = b"checkpoint tensors"
        self.entry = {
            "id": "model", "group": "weights", "path": "weight/model.pth",
            "profile": "weight/model.json", "git_ref": "models/v0.1.0",
            "size": len(self.content), "sha256": hashlib.sha256(self.content).hexdigest(),
        }
        self.manifest = {"schema_version": 1, "release": "v0.1.1", "assets": [self.entry]}
        for folder in (self.source / "assets", self.source / "weight", self.source / "docs", self.models / "weight"):
            folder.mkdir(parents=True)
        self.write_manifest()
        (self.source / self.entry["profile"]).write_text(json.dumps({"checkpoint_sha256": self.entry["sha256"]}))
        (self.models / self.entry["path"]).write_bytes(self.content)
        for name in ("LICENSE", "NOTICE", "docs/model-card.md"):
            (self.source / name).write_text("Research release information.\n")

    def write_manifest(self):
        (self.source / "assets/manifest.json").write_text(json.dumps(self.manifest))

    def assemble(self):
        return release.assemble(self.source, self.models, self.output, "v0.1.1")

    def test_source_release_can_reuse_an_immutable_model_revision(self):
        self.assertEqual(release.read_manifest(self.source, "v0.1.1")[1], "models/v0.1.0")
        self.assertEqual(self.assemble(), 7)
        self.assertEqual((self.output / "model.pth").read_bytes(), self.content)
        subprocess.run(["sha256sum", "--strict", "--check", "SHA256SUMS"], cwd=self.output,
                       capture_output=True, check=True)

    def test_bad_weight_or_profile_never_produces_uploadable_assets(self):
        path = self.models / self.entry["path"]
        path.write_bytes(b"wrong contents")
        with self.assertRaisesRegex(ValueError, "Weight size or SHA-256"):
            self.assemble()
        self.assertFalse(self.output.exists())
        path.write_bytes(self.content)
        (self.source / self.entry["profile"]).write_text('{"checkpoint_sha256": "wrong"}')
        with self.assertRaisesRegex(ValueError, "profile checksum"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_wrong_source_tag_and_mutable_model_branch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "source tag"):
            release.read_manifest(self.source, "v0.1.2")
        self.entry["git_ref"] = "model-assets"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "models/v"):
            self.assemble()

    def test_duplicate_attachment_names_cannot_overwrite_earlier_assets(self):
        self.manifest["assets"].append({**self.entry, "id": "another-model"})
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "unique"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def prepare_publication(self, draft=None):
        self.assemble()
        self.remote = self.root / "remote"
        self.remote.mkdir()
        self.state = self.root / "state.json"
        self.state.write_text(json.dumps({"exists": draft is not None, "draft": draft, "mutations": []}))
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        gh = fake_bin / "gh"
        gh.write_text("#!" + sys.executable + "\n" + '''
import json, os, pathlib, shutil, sys
state_path = pathlib.Path(os.environ['TEST_RELEASE_STATE'])
state = json.loads(state_path.read_text())
remote = pathlib.Path(os.environ['TEST_RELEASE_REMOTE'])
args = sys.argv[1:]
assert args[0] == 'release'
action = args[1]
if action == 'view':
    if not state['exists']: sys.exit(1)
    print(str(state['draft']).lower() if 'isDraft' in args else '{}')
elif action == 'create':
    assert not state['exists'] and '--draft' in args
    state.update(exists=True, draft=True)
    state['mutations'].append('create')
elif action == 'upload':
    assert state['draft'], 'Published assets must never be replaced'
    for name in args[3:]:
        if not name.startswith('--'): shutil.copyfile(name, remote / pathlib.Path(name).name)
    state['mutations'].append('upload')
elif action == 'download':
    destination = pathlib.Path(args[args.index('--dir') + 1])
    for path in remote.iterdir(): shutil.copyfile(path, destination / path.name)
    if os.environ.get('TEST_RELEASE_CORRUPT') == '1':
        (destination / 'model.pth').write_bytes(b'corrupt upload')
elif action == 'edit':
    assert '--draft=false' in args
    state['draft'] = False
    state['mutations'].append('publish')
else:
    raise AssertionError(args)
state_path.write_text(json.dumps(state))
''')
        gh.chmod(0o755)
        return {**os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
                "TEST_RELEASE_STATE": str(self.state), "TEST_RELEASE_REMOTE": str(self.remote)}

    def publish(self, environment):
        return subprocess.run(["bash", str(ROOT / "scripts/publish_release.sh"), "v0.1.1",
                               str(self.output), str(self.source / "NOTICE")], env=environment,
                              capture_output=True, text=True)

    def test_first_publication_and_rerun_verify_without_replacing_published_files(self):
        environment = self.prepare_publication()
        result = self.publish(environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads(self.state.read_text())
        self.assertFalse(state["draft"])
        self.assertEqual(state["mutations"], ["create", "upload", "publish"])
        result = self.publish(environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.state.read_text()), state)

    def test_interrupted_draft_is_resumed(self):
        environment = self.prepare_publication(draft=True)
        (self.remote / "model.pth").write_bytes(b"partial upload")
        result = self.publish(environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.state.read_text())["mutations"], ["upload", "publish"])

    def test_corrupt_upload_never_promotes_draft(self):
        environment = self.prepare_publication()
        result = self.publish({**environment, "TEST_RELEASE_CORRUPT": "1"})
        self.assertNotEqual(result.returncode, 0)
        state = json.loads(self.state.read_text())
        self.assertTrue(state["draft"])
        self.assertNotIn("publish", state["mutations"])

    def test_mismatched_published_release_fails_without_mutation(self):
        environment = self.prepare_publication(draft=False)
        for path in self.output.iterdir():
            shutil.copyfile(path, self.remote / path.name)
        (self.remote / "model.pth").write_bytes(b"different published model")
        result = self.publish(environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(self.state.read_text())["mutations"], [])


if __name__ == "__main__":
    unittest.main()
