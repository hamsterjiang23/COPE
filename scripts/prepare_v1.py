"""Download pinned official assets with resumable files and integrity checks."""

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path, algorithm="sha256", git_blob=False):
    h = hashlib.new(algorithm)
    if git_blob:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(item, proxy):
    target = Path(item["destination"])
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if not target.exists():
        command = [
            "curl",
            "-fLsS",
            "--retry",
            "5",
            "--connect-timeout",
            "30",
            "--speed-time",
            "120",
            "--speed-limit",
            "1024",
            "-C",
            "-",
            "-o",
            str(partial),
        ]
        if proxy and not item.get("mirror_url"):
            command += ["--proxy", proxy]
        print("download", str(target), flush=True)
        subprocess.run(command + [item.get("mirror_url", item["url"])], check=True)
        candidate = partial
    else:
        candidate = target
    if candidate.stat().st_size != item["size"]:
        raise ValueError(f"size mismatch: {candidate}")
    if item["sha256"]:
        actual = digest(candidate)
        expected = item["sha256"]
    else:
        actual = digest(candidate, "sha1", git_blob=True)
        expected = item["git_blob_sha1"]
    if actual != expected:
        raise ValueError(f"checksum mismatch: {candidate}; preserve for investigation")
    if candidate == partial:
        os.replace(partial, target)
    print("verified", str(target), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--proxy", default="")
    args = parser.parse_args()
    items = json.loads(Path(args.manifest).read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: download(item, args.proxy), items))
    print("PREPARATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
