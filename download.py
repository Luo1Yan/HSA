"""Acquire original datasets and frozen encoder weights for HSA experiments.

The default prints the acquisition plan. Add --fetch to download the selection.
Existing user-provided datasets can be checked with extract.py --check-inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import ssl
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile

from protocols import CONFIG_ROOT, read_json, safe_path, select, selection_arguments


def checksum(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_file(url, target, sha256=None, size=None):
    """Resume HTTP downloads; validate byte count/hash before publishing a file."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if size is not None and target.stat().st_size != size:
            raise ValueError(f"Existing file has a wrong size: {target}")
        if sha256 and checksum(target) != sha256:
            raise ValueError(f"Existing file has a wrong SHA256: {target}")
        if target.stat().st_size > 0:
            return target
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "HSA-reproduction/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        resumed = offset > 0 and response.status == 206
        if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError(f"Server returned an unexpected byte range: {url}")
        length = response.headers.get("Content-Length")
        expected = int(length) + (offset if resumed else 0) if length else size
        with partial.open("ab" if resumed else "wb") as stream:
            shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
    if expected is not None and partial.stat().st_size != expected:
        raise IOError(f"Incomplete download: {partial}")
    if size is not None and partial.stat().st_size != size:
        raise ValueError(f"Download size mismatch: {partial}")
    if sha256 and checksum(partial) != sha256:
        raise ValueError(f"SHA256 mismatch: {partial}")
    partial.replace(target)
    return target


def unpack(path, destination, strip_first=False):
    """Extract regular files only, refusing traversal paths and archive symlinks."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ("." + Path(path).name + ".extracted")
    if marker.is_file():
        return
    def relative(name):
        name = name.replace("\\", "/")
        if strip_first:
            pieces = name.split("/", 1)
            name = pieces[1] if len(pieces) == 2 else ""
        return safe_path(destination, name) if name else None
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                target = relative(item.filename)
                if target is None or item.is_dir():
                    continue
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Archive symlink refused: {item.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    else:
        with tarfile.open(path, "r:*") as archive:
            for item in archive:
                target = relative(item.name)
                if target is None or item.isdir():
                    continue
                if not item.isfile():
                    raise ValueError(f"Archive special file refused: {item.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    marker.write_text("complete\n", encoding="utf-8")


def hf_download(resource, destination):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub==0.36.0 to download public Hub files") from exc
    return Path(snapshot_download(repo_id=resource["repo"], repo_type=resource.get("repo_type", "dataset"),
                                  revision=resource["revision"], allow_patterns=resource["patterns"],
                                  local_dir=str(destination)))


class ArchiveParts(io.RawIOBase):
    """Read split archives in order without creating a second full-size archive."""
    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.stream = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.stream is None:
                try:
                    self.stream = open(next(self.paths), "rb")
                except StopIteration:
                    return 0
            count = self.stream.readinto(buffer)
            if count:
                return count
            self.stream.close()
            self.stream = None

    def close(self):
        if self.stream is not None:
            self.stream.close()
        super().close()


def unpack_parts(paths, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".multipart.extracted"
    if marker.exists():
        return
    with io.BufferedReader(ArchiveParts(paths)) as stream, tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            target = safe_path(destination, member.name.replace("\\", "/"))
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Archive special file refused: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    marker.write_text("complete\n", encoding="utf-8")


def acquire(resource, destination, do_unpack=True):
    kind = resource["kind"]
    if kind == "http":
        path = fetch_file(resource["url"], safe_path(destination, resource["filename"]),
                          resource.get("sha256"), resource.get("bytes"))
        if do_unpack and resource.get("unpack"):
            unpack(path, path.parent / resource.get("extract_to", "extracted"), resource.get("strip_first", False))
    elif kind == "hf":
        root = hf_download(resource, destination)
        if do_unpack and resource.get("unpack"):
            for path in sorted(root.rglob("*")):
                if path.is_file() and (path.name.endswith(".zip") or path.name.endswith(".tar.gz")):
                    unpack(path, path.parent / (path.stem if path.suffix == ".zip" else "extracted"))
    elif kind == "manual":
        raise RuntimeError(resource["instructions"] + "\n" + resource["url"])
    elif kind == "multipart":
        paths = [fetch_file(part["url"], safe_path(destination, part["filename"])) for part in resource["parts"]]
        if do_unpack:
            unpack_parts(paths, Path(destination) / "extracted")
    else:
        raise ValueError(f"Unknown resource type: {kind}")


def plan(args):
    recipes = select(args)
    config = read_json(CONFIG_ROOT / "downloads.json")
    selected = []
    if not args.models_only:
        for dataset in sorted({r["dataset"] for r in recipes}):
            collections = {r["collection"] for r in recipes if r["dataset"] == dataset}
            for resource in config["datasets"][dataset]:
                if "collections" not in resource or collections.intersection(resource["collections"]):
                    selected.append((dataset, resource, args.data_root / "raw" / dataset))
    if not args.data_only:
        for backbone in sorted({r["backbone"] for r in recipes}):
            modalities = {m for r in recipes if r["backbone"] == backbone for m in (r["a"], r["hub"], r["b"])}
            for resource in config["models"][backbone]:
                resource = dict(resource)
                if resource.get("repo") == "LanguageBind/LanguageBind_Image" and "image" not in modalities:
                    resource["patterns"] = [p for p in resource["patterns"] if p != "pytorch_model.bin"]
                    resource.pop("modality", None)
                if resource.get("modality") not in (None, "text") and resource["modality"] not in modalities:
                    continue
                if resource.get("variant") == "msrvtt" and not any(r["dataset"] == "msrvtt" for r in recipes):
                    continue
                selected.append((backbone, resource, args.model_root / backbone / resource.get("directory", "")))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection_arguments(parser)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    exclusive = parser.add_mutually_exclusive_group()
    exclusive.add_argument("--models-only", action="store_true")
    exclusive.add_argument("--data-only", action="store_true")
    parser.add_argument("--fetch", action="store_true", help="Execute the displayed download plan")
    parser.add_argument("--no-unpack", action="store_true", help="Keep downloaded original archives compressed")
    parser.add_argument("--skip-manual", action="store_true",
                        help="Download public resources while leaving access-controlled datasets to the user")
    args = parser.parse_args()
    try:
        selected = plan(args)
        for name, resource, destination in selected:
            origin = resource.get("url") or ("https://huggingface.co/" + ("datasets/" if resource.get("repo_type", "dataset") == "dataset" else "") + resource["repo"])
            print(f"{name}: {resource.get('label', resource['kind'])}\n  {origin}\n  -> {destination}", flush=True)
            if resource["kind"] == "manual":
                print("  " + resource["instructions"], flush=True)
        if not args.fetch or args.list:
            print("Review dataset terms and available disk space; add --fetch to download this selection.")
            return
        failures = []
        for name, resource, destination in selected:
            if resource["kind"] == "manual" and args.skip_manual:
                print(f"Manual dataset: {name}; use the access instructions above and place files in {destination}", flush=True)
                continue
            try:
                acquire(resource, destination, not args.no_unpack)
            except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
                failures.append(f"{name}: {exc}")
        if failures:
            parser.exit(1, "\n".join(failures) + "\n")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
