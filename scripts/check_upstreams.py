"""Verify that checked-out third-party submodules match documented commits."""

from __future__ import annotations

import subprocess
from pathlib import Path

EXPECTED_COMMITS = {
    "third_party/C2C": "113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a",
    "third_party/MRL": "7ccb42df6be05f3d21d0648aa03099bba46386bf",
    "third_party/LLM-to-SLM": "5ef4eda6e0a078bbec26bc6eceef0c294b5c7a92",
}


def checked_out_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    mismatches: list[str] = []
    for relative_path, expected in EXPECTED_COMMITS.items():
        path = repository_root / relative_path
        if not path.exists():
            mismatches.append(f"{relative_path}: missing; run git submodule update --init")
            continue
        actual = checked_out_commit(path)
        if actual != expected:
            mismatches.append(f"{relative_path}: expected {expected}, got {actual}")
        else:
            print(f"PASS {relative_path} {actual}")
    if mismatches:
        for mismatch in mismatches:
            print(f"FAIL {mismatch}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
