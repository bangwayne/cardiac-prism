#!/usr/bin/env python3
"""Assemble checksum-verified GitHub Release attachments from a source manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe release source: {name}")
    return path


def read_manifest(root, tag):
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?", tag):
        raise ValueError("A version tag such as v0.1.2 is required.")
    manifest = json.loads(relative_file(root, "assets/manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("release") != tag:
        raise ValueError("The selected source tag does not match its release manifest.")
    weights = [entry for entry in manifest["assets"] if entry["group"] == "weights"]
    refs = {entry.get("git_ref") for entry in weights}
    if len(refs) != 1:
        raise ValueError("Release weights must specify one immutable version tag.")
    model_ref = refs.pop()
    if not isinstance(model_ref, str) or not re.fullmatch(r"(?:models/)?v[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?", model_ref):
        raise ValueError("Release weights must use an immutable version tag (vX.Y.Z or legacy models/vX.Y.Z).")
    return weights, model_ref


def assemble(root, models, output, tag):
    weights, _ = read_manifest(root, tag)
    files = []
    for entry in weights:
        path = relative_file(models, entry.get("git_path", entry["path"]))
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Weight size or SHA-256 mismatch: {entry['id']}")
        profile = relative_file(root, entry["profile"])
        if json.loads(profile.read_text()).get("checkpoint_sha256") != entry["sha256"]:
            raise ValueError(f"Model profile checksum mismatch: {entry['id']}")
        files.extend([(path, Path(entry["path"]).name), (profile, profile.name)])
    for name in ("LICENSE", "NOTICE", "docs/model-card.md", "assets/manifest.json"):
        path = relative_file(root, name)
        files.append((path, path.name))
    names = [name for _, name in files]
    if len(names) != len(set(names)) or "SHA256SUMS" in names:
        raise ValueError("Release attachment filenames must be unique.")
    output.mkdir(parents=True, exist_ok=False)
    for source, name in files:
        shutil.copyfile(source, output / name)
    (output / "SHA256SUMS").write_text("".join(
        f"{sha256(output / name)}  {name}\n" for name in sorted(names)))
    return len(files) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--models", type=Path, default=Path("model-files"))
    parser.add_argument("--output", type=Path, default=Path("release-assets"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model-ref", action="store_true")
    args = parser.parse_args()
    if args.model_ref:
        print(read_manifest(args.root, args.tag)[1])
    else:
        count = assemble(args.root, args.models, args.output, args.tag)
        print(f"Verified {count} release attachments in {args.output}.")


if __name__ == "__main__":
    main()
