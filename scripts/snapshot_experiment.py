"""Record an exact local code archive, including uncommitted and untracked files."""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "source"
    output.mkdir(parents=True, exist_ok=True)
    files = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
        )
        .decode()
        .split("\0")
    )
    files = sorted({p for p in files if p and (root / p).is_file()})
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}
    content_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    directory = output / content_hash
    directory.mkdir(exist_ok=False)
    patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
    (directory / "working-tree.patch").write_bytes(patch)
    archive = directory / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name in files:
            handle.add(root / name, arcname=name, recursive=False)
    record = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
        .decode()
        .strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root).decode(),
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "content_hash": content_hash,
        "files": hashes,
        "archive": str(archive),
        "patch": str(directory / "working-tree.patch"),
    }
    (directory / "source-record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(directory)


if __name__ == "__main__":
    main()
