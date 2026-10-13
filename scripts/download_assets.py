#!/usr/bin/env python3
"""Download the versioned model and optional ACDC examples using Python's stdlib."""

from __future__ import annotations

import argparse
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 1024 * 1024


class AssetError(RuntimeError):
    """An asset could not be installed or verified."""


class AssetAccessError(AssetError):
    """A release asset requires authenticated repository access."""


class HTTPSRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        if urlsplit(new_url).scheme != "https":
            raise AssetError("Refusing an asset download redirect outside HTTPS.")
        return super().redirect_request(request, file_pointer, code, message, headers, new_url)


def destination(root: Path, asset: dict) -> Path:
    """Keep manifest paths, including symlinks, inside the chosen installation."""
    path = Path(asset["path"])
    if path.is_absolute() or ".." in path.parts:
        raise AssetError(f"Unsafe asset destination: {asset['path']}")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise AssetError(f"Asset destination escapes installation: {asset['path']}")
    return target


def validate(path: Path, asset: dict) -> bool:
    if not path.is_file() or path.stat().st_size != asset["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest() == asset["sha256"]


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("assets"), list):
        raise AssetError("Unsupported asset manifest.")
    seen = set()
    for asset in manifest["assets"]:
        try:
            identifier, name = asset["id"], asset["path"]
            digest = asset["sha256"]
            valid = (isinstance(identifier, str) and identifier not in seen
                     and asset["group"] in {"weights", "sample"}
                     and isinstance(name, str)
                     and isinstance(asset["size"], int) and asset["size"] > 0
                     and isinstance(digest, str) and len(digest) == 64
                     and all(char in "0123456789abcdef" for char in digest)
                     and urlsplit(asset["url"]).scheme == "https"
                     and bool(urlsplit(asset["url"]).hostname))
        except (KeyError, TypeError, ValueError) as error:
            raise AssetError("Invalid asset manifest entry.") from error
        if not valid:
            raise AssetError(f"Invalid manifest entry: {identifier}")
        seen.add(identifier)
    return manifest


def temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
    os.close(descriptor)
    return Path(name)


def publish(temporary: Path, target: Path, asset: dict) -> None:
    if not validate(temporary, asset):
        raise AssetError(f"Size or SHA-256 mismatch for {asset['id']}; no file was installed.")
    temporary.chmod(0o644)
    os.replace(temporary, target)


def import_weights(source: Path, target: Path, asset: dict) -> None:
    if not validate(source, asset):
        raise AssetError("The supplied weight file does not match this release's size and SHA-256.")
    if source.resolve() == target.resolve():
        return
    temporary = temporary_path(target)
    try:
        with source.open("rb") as src, temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK_SIZE)
            dst.flush()
            os.fsync(dst.fileno())
        publish(temporary, target, asset)
    finally:
        temporary.unlink(missing_ok=True)


def import_private_weights(asset: dict, target: Path, repository: Path) -> None:
    """Use the clone's existing Git credentials without exposing them in output."""
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}

    def run(arguments: list[str], description: str, cwd: Path | None = None) -> str:
        try:
            result = subprocess.run(arguments, cwd=cwd, env=environment, capture_output=True,
                                    text=True, timeout=300, check=True)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            raise AssetError(
                f"Model download failed during {description}. Install Git LFS, verify the Git "
                "remote and GitHub connectivity, and retry. Public HTTPS downloads need no "
                "account; SSH remotes require a working SSH key. Alternatively use "
                "--weights-file with the matching checkpoint from the GitHub Release."
            ) from error

    run(["git", "lfs", "version"], "Git LFS detection")
    remote = run(["git", "-C", str(repository), "remote", "get-url", "origin"], "repository discovery")
    if not remote or remote.startswith("-"):
        raise AssetError("A valid Git origin is required to download versioned model assets.")
    reference = asset.get("git_ref")
    if not isinstance(reference, str) or not reference or reference.startswith("-"):
        raise AssetError("A versioned model Git reference is required in the asset manifest.")
    print("Release asset unavailable; retrieving the verified checkpoint from its versioned Git tag.", flush=True)
    with tempfile.TemporaryDirectory(prefix="cardiac-prism-model-") as directory:
        clone = Path(directory) / "repository"
        run(["git", "clone", "--quiet", "--depth", "1", "--single-branch", "--branch", reference,
             "--", remote, str(clone)], "versioned model checkout")
        run(["git", "lfs", "pull", "--include", asset["path"], "--exclude", ""], "model transfer", cwd=clone)
        source = destination(clone, asset)
        import_weights(source, target, asset)


def download(asset: dict, target: Path, retries: int, timeout: float) -> None:
    opener = build_opener(HTTPSRedirectHandler())
    for attempt in range(1, retries + 1):
        temporary = temporary_path(target)
        try:
            request = Request(asset["url"], headers={"User-Agent": "Cardiac-Prism-Asset-Installer/1.0"})
            with opener.open(request, timeout=timeout) as response, temporary.open("wb") as stream:
                if urlsplit(response.geturl()).scheme != "https":
                    raise AssetError("Asset response did not use HTTPS.")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != asset["size"]:
                    raise AssetError(f"Unexpected download size for {asset['id']}.")
                received = 0
                next_report = 10 * CHUNK_SIZE
                while chunk := response.read(CHUNK_SIZE):
                    received += len(chunk)
                    if received > asset["size"]:
                        raise AssetError(f"Download exceeds the expected size for {asset['id']}.")
                    stream.write(chunk)
                    if received >= next_report:
                        print(f"  {asset['id']}: {received / asset['size']:.0%}", flush=True)
                        next_report += 10 * CHUNK_SIZE
                stream.flush()
                os.fsync(stream.fileno())
            publish(temporary, target, asset)
            return
        except HTTPError as error:
            if error.code in {401, 403, 404} and asset["group"] == "weights":
                raise AssetAccessError("Release asset is not accessible without repository authentication.") from error
            if attempt == retries:
                raise AssetError(f"Download failed for {asset['id']} (HTTP {error.code}).") from error
            print(f"  Attempt {attempt} failed (HTTP {error.code}); retrying...", file=sys.stderr, flush=True)
            time.sleep(min(2 ** (attempt - 1), 8))
        except (AssetError, OSError, URLError, HTTPException, ValueError) as error:
            if attempt == retries:
                raise AssetError(f"Download failed for {asset['id']} after {retries} attempt(s): {error}") from error
            print(f"  Attempt {attempt} failed: {error}; retrying...", file=sys.stderr, flush=True)
            time.sleep(min(2 ** (attempt - 1), 8))
        finally:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", "--model", action="store_true", help="download the release model (the default)")
    parser.add_argument("--sample", action="store_true", help="download the official ACDC 4D cine, 3D volume and reference mask")
    parser.add_argument("--example", choices=["acdc"], help="alias for --sample")
    parser.add_argument("--all", action="store_true", help="download both the model and ACDC examples")
    parser.add_argument("--verify", action="store_true", help="verify selected files locally without downloading")
    parser.add_argument("--weights-file", type=Path, help="install an already-downloaded release weight file")
    parser.add_argument("--output-root", type=Path, default=ROOT, help="installation directory (default: repository root)")
    parser.add_argument("--manifest", type=Path, default=ROOT / "assets/manifest.json", help="versioned asset manifest")
    parser.add_argument("--retries", type=int, default=3, help="download attempts per file (default: 3)")
    parser.add_argument("--timeout", type=float, default=60, help="network operation timeout in seconds (default: 60)")
    args = parser.parse_args(argv)
    if args.retries < 1 or args.timeout <= 0:
        parser.error("--retries and --timeout must be positive")
    if args.verify and args.weights_file:
        parser.error("--weights-file cannot be combined with --verify")
    sample = args.all or args.sample or args.example is not None
    weights = args.all or args.weights or args.weights_file is not None or not sample
    groups = ({"weights"} if weights else set()) | ({"sample"} if sample else set())
    try:
        assets = [asset for asset in load_manifest(args.manifest)["assets"] if asset["group"] in groups]
        if sample:
            print("ACDC examples: CC BY-NC-SA 4.0 and ACDC research/citation terms; see docs/sample-data.md.")
        for asset in assets:
            target = destination(args.output_root, asset)
            if args.weights_file and asset["group"] == "weights":
                import_weights(args.weights_file, target, asset)
            if validate(target, asset):
                print(f"Verified: {asset['path']}")
                continue
            if args.verify:
                raise AssetError(f"Missing or invalid asset: {asset['path']}. Run this command without --verify to install it.")
            print(f"Downloading {asset['id']} ({asset['size'] / 1024**2:.1f} MiB)...", flush=True)
            try:
                download(asset, target, args.retries, args.timeout)
            except AssetAccessError:
                import_private_weights(asset, target, ROOT)
            print(f"Verified: {asset['path']}")
        return 0
    except (AssetError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
