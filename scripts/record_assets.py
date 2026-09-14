"""Record local student files and validate the pinned prepared teacher/data inventory."""

import argparse
import json
from pathlib import Path

from cope.experiment_data import sha256_file, write_new_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--downloads", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    inventory = {}
    for item in json.loads(Path(args.downloads).read_text()):
        path = Path(item["destination"])
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"missing/incomplete pinned asset: {path}")
        checksum = sha256_file(path)
        if item["sha256"] and checksum != item["sha256"]:
            raise ValueError(f"pinned asset hash mismatch: {path}")
        inventory[str(path)] = {
            "sha256": checksum,
            "size": path.stat().st_size,
            "official_url": item["url"],
            "transport_url": item.get("mirror_url", item["url"]),
        }
    for path in sorted(Path(config["student"]["path"]).iterdir()):
        if path.is_file() and path.suffix in (".json", ".safetensors", ".txt", ".jinja"):
            inventory[str(path)] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    write_new_json(
        args.output,
        {
            "teacher_revision": config["teacher"]["revision"],
            "dataset_revision": config["dataset"]["revision"],
            "files": inventory,
        },
    )


if __name__ == "__main__":
    main()
