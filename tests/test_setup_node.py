"""Node's npm links must work on Python 3.10 without relaxing archive safety."""

import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "setup_node", Path(__file__).resolve().parents[1] / "scripts/setup_node.py")
setup_node = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup_node)


def runtime_archive(link="../lib/node_modules/npm/bin/npm-cli.js", kind=tarfile.SYMTYPE):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        content = b"#!/usr/bin/env node\n"
        entry = tarfile.TarInfo("node/lib/node_modules/npm/bin/npm-cli.js")
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))
        entry = tarfile.TarInfo("node/bin")
        entry.type = tarfile.DIRTYPE
        archive.addfile(entry)
        entry = tarfile.TarInfo("node/bin/npm")
        entry.type, entry.linkname = kind, link
        archive.addfile(entry)
    output.seek(0)
    return tarfile.open(fileobj=output, mode="r")


class SetupNodeTests(unittest.TestCase):
    def test_relative_npm_link_works_without_tarfile_symlink_filter(self):
        original = tarfile.data_filter

        def distribution_filter(member, destination):
            if member.issym():
                raise tarfile.FilterError("Distribution symlink backport")
            return original(member, destination)

        with tempfile.TemporaryDirectory() as directory, runtime_archive() as archive:
            with patch.dict(tarfile._NAMED_FILTERS, {"data": distribution_filter}):
                setup_node.extract_runtime(archive, directory)
            installed = Path(directory) / "node/bin/npm"
            self.assertTrue(installed.is_symlink())
            self.assertEqual(installed.read_bytes(), b"#!/usr/bin/env node\n")

    def test_absolute_and_escaping_links_are_rejected(self):
        for link in ("/etc/passwd", "../../../outside"):
            with self.subTest(link=link), tempfile.TemporaryDirectory() as directory:
                with runtime_archive(link) as archive, self.assertRaisesRegex(ValueError, "unsafe executable link"):
                    setup_node.extract_runtime(archive, directory)
                self.assertFalse((Path(directory) / "node/bin/npm").is_symlink())

    def test_hardlinks_are_not_extracted(self):
        with tempfile.TemporaryDirectory() as directory, runtime_archive(kind=tarfile.LNKTYPE) as archive:
            with self.assertRaisesRegex(ValueError, "unsupported file type"):
                setup_node.extract_runtime(archive, directory)


if __name__ == "__main__":
    unittest.main()
