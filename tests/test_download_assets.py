"""Release assets must be verified before replacing files on disk."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


SPEC = importlib.util.spec_from_file_location(
    "download_assets", Path(__file__).resolve().parents[1] / "scripts/download_assets.py")
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class Response(io.BytesIO):
    def __init__(self, content, size=None, url="https://example.org/model.pth"):
        super().__init__(content)
        self.headers = {} if size is None else {"Content-Length": str(size)}
        self.url = url

    def geturl(self):
        return self.url


class DownloadAssetsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.content = b"release weight contents"
        self.asset = {
            "id": "model", "group": "weights", "path": "weight/model.pth",
            "url": "https://example.org/model.pth", "size": len(self.content),
            "sha256": hashlib.sha256(self.content).hexdigest(),
        }
        self.target = assets.destination(self.root, self.asset)

    def assert_no_partial_files(self):
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_verified_download_is_published_atomically(self):
        with patch.object(assets, "build_opener") as opener:
            opener.return_value.open.return_value = Response(self.content, len(self.content))
            assets.download(self.asset, self.target, retries=1, timeout=1)
        self.assertEqual(self.target.read_bytes(), self.content)
        self.assertTrue(assets.validate(self.target, self.asset))
        self.assert_no_partial_files()

    def test_bad_checksum_preserves_previous_file(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b"previous installed model")
        bad_content = bytes([self.content[0] ^ 1]) + self.content[1:]
        with patch.object(assets, "build_opener") as opener:
            opener.return_value.open.return_value = Response(bad_content, len(bad_content))
            with self.assertRaisesRegex(assets.AssetError, "SHA-256 mismatch"):
                assets.download(self.asset, self.target, retries=1, timeout=1)
        self.assertEqual(self.target.read_bytes(), b"previous installed model")
        self.assert_no_partial_files()

    def test_incomplete_stream_is_not_installed(self):
        with patch.object(assets, "build_opener") as opener:
            opener.return_value.open.return_value = Response(self.content[:-1])
            with self.assertRaises(assets.AssetError):
                assets.download(self.asset, self.target, retries=1, timeout=1)
        self.assertFalse(self.target.exists())
        self.assert_no_partial_files()

    def test_oversized_response_is_rejected(self):
        with patch.object(assets, "build_opener") as opener:
            opener.return_value.open.return_value = Response(self.content + b"x")
            with self.assertRaisesRegex(assets.AssetError, "exceeds"):
                assets.download(self.asset, self.target, retries=1, timeout=1)
        self.assertFalse(self.target.exists())
        self.assert_no_partial_files()

    def test_failed_download_can_retry(self):
        with patch.object(assets, "build_opener") as opener, patch.object(assets.time, "sleep"):
            opener.return_value.open.side_effect = [OSError("connection reset"), Response(self.content)]
            assets.download(self.asset, self.target, retries=2, timeout=1)
            self.assertEqual(opener.return_value.open.call_count, 2)
        self.assertTrue(assets.validate(self.target, self.asset))
        self.assert_no_partial_files()

    def test_offline_copy_validates_before_replacing(self):
        source = self.root / "local.pth"
        source.write_bytes(b"unrelated weights")
        self.target.parent.mkdir()
        self.target.write_bytes(self.content)
        with self.assertRaisesRegex(assets.AssetError, "supplied weight"):
            assets.import_weights(source, self.target, self.asset)
        self.assertEqual(self.target.read_bytes(), self.content)
        source.write_bytes(self.content)
        assets.import_weights(source, self.target, self.asset)
        self.assertTrue(assets.validate(self.target, self.asset))
        self.assert_no_partial_files()

    def test_verify_never_accesses_network(self):
        self.target.parent.mkdir()
        self.target.write_bytes(self.content)
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "assets": [self.asset]}))
        with patch.object(assets, "build_opener") as opener:
            self.assertEqual(assets.main(["--verify", "--manifest", str(manifest), "--output-root", str(self.root)]), 0)
            opener.assert_not_called()
            self.target.unlink()
            self.assertEqual(assets.main(["--verify", "--manifest", str(manifest), "--output-root", str(self.root)]), 1)
            opener.assert_not_called()

    def test_destinations_cannot_escape_installation(self):
        for path in ("../model.pth", "/tmp/model.pth"):
            with self.subTest(path=path), self.assertRaises(assets.AssetError):
                assets.destination(self.root, {**self.asset, "path": path})
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(assets.AssetError):
            assets.destination(self.root, {**self.asset, "path": "escape/model.pth"})

    def test_http_response_or_redirect_is_rejected(self):
        with patch.object(assets, "build_opener") as opener:
            opener.return_value.open.return_value = Response(self.content, url="http://example.org/model.pth")
            with self.assertRaisesRegex(assets.AssetError, "HTTPS"):
                assets.download(self.asset, self.target, retries=1, timeout=1)
        with self.assertRaisesRegex(assets.AssetError, "HTTPS"):
            assets.HTTPSRedirectHandler().redirect_request(None, None, 302, "Found", {}, "http://example.org/model.pth")
        self.assert_no_partial_files()

    def test_private_release_errors_request_authenticated_fallback(self):
        for code in (401, 403, 404):
            with self.subTest(code=code), patch.object(assets, "build_opener") as opener:
                opener.return_value.open.side_effect = HTTPError(self.asset["url"], code, "Private", {}, None)
                with self.assertRaises(assets.AssetAccessError):
                    assets.download(self.asset, self.target, retries=3, timeout=1)
                self.assertEqual(opener.return_value.open.call_count, 1)
        self.assert_no_partial_files()

    def test_private_import_uses_origin_and_validates_lfs_file(self):
        commands = []
        self.asset["git_ref"] = "models/v0.1.0"

        def git_run(arguments, **kwargs):
            commands.append(arguments)
            self.assertEqual(kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")
            if arguments[-3:] == ["remote", "get-url", "origin"]:
                return assets.subprocess.CompletedProcess(arguments, 0, stdout="git@example.org:team/project.git\n")
            if arguments[1] == "clone":
                clone = Path(arguments[-1])
                local = clone / self.asset["path"]
                local.parent.mkdir(parents=True)
                local.write_bytes(self.content)
            return assets.subprocess.CompletedProcess(arguments, 0, stdout="")

        with patch.object(assets.subprocess, "run", side_effect=git_run):
            assets.import_private_weights(self.asset, self.target, self.root)
        self.assertTrue(assets.validate(self.target, self.asset))
        self.assertTrue(any("models/v0.1.0" in command for command in commands))
        self.assertTrue(any(command[1:3] == ["lfs", "pull"] for command in commands))

    def test_private_errors_do_not_echo_git_credentials(self):
        failure = assets.subprocess.CalledProcessError(1, ["git"], stderr="https://secret-token@example.org")
        with patch.object(assets.subprocess, "run", side_effect=failure):
            with self.assertRaises(assets.AssetError) as caught:
                assets.import_private_weights(self.asset, self.target, self.root)
        self.assertNotIn("secret-token", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
