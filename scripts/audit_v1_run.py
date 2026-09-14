"""Append CPU-only integrity evidence to a finalized v1 run; never rescore outputs."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    metrics = json.loads((root / "metrics.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    data = json.loads(Path(manifest["source"]["prepared_data_path"]).read_text())
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    arms = manifest["config"]["training"]["arms"]
    train_ids = {row["id"] for row in data["train"]}
    holdout_ids = [row["id"] for row in data["holdout"]]
    sequences, training = {}, {}
    for arm in arms:
        steps = [
            e for e in events if e["kind"] == "step" and e["phase"] == "train" and e["arm"] == arm
        ]
        sequences[arm] = [row["id"] for e in steps for row in e["examples"]]
        assert len(steps) == 128 and len(sequences[arm]) == 1024
        assert set(sequences[arm]) == train_ids
        assert all(e["grad_norm"] > 0 for e in steps)
        if arm == "mismatched":
            assert all(
                r["donor_id"] != r["id"] and r["donor_id"] in train_ids
                for e in steps
                for r in e["examples"]
            )
        state = torch.load(
            root / "checkpoints" / arm / "step-0128.pt", map_location="cpu", weights_only=False
        )
        lora_b = [p for n, p in state["weights"].items() if "lora_B" in n]
        # PEFT initializes LoRA B to zero: nonzero final values show actual updates.
        assert lora_b and all(bool(torch.isfinite(p).all()) for p in lora_b)
        changed = sum(bool(torch.count_nonzero(p)) for p in lora_b)
        assert changed == len(lora_b)
        training[arm] = {
            "steps": len(steps),
            "sample_exposures": len(sequences[arm]),
            "unique_samples": len(set(sequences[arm])),
            "lora_b_tensors_updated": changed,
            "sequence_sha256": hashlib.sha256(json.dumps(sequences[arm]).encode()).hexdigest(),
        }
        del state
    assert all(order == sequences[arms[0]] for order in sequences.values())
    assert not train_ids.intersection(holdout_ids)
    prediction_checks = {}
    for path in sorted((root / "predictions").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["id"] for r in rows] == holdout_ids
        assert sum(r["correct"] for r in rows) / len(rows) == metrics["arms"][path.stem]["accuracy"]
        if path.stem in ("mismatched", "correct_test_mismatched"):
            assert all(r["id"] != r["donor_id"] and r["donor_id"] in holdout_ids for r in rows)
        prediction_checks[path.stem] = {
            "n": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "parse_failures": sum(r["prediction"] is None for r in rows),
            "truncated": sum(r["truncated"] for r in rows),
        }
    post = root / "postprocess"
    post.mkdir(exist_ok=True)
    task_id = str(manifest["pueue_task_id"])
    log = post / f"pueue-{task_id}.log"
    with log.open("xb") as handle:
        subprocess.run(
            ["/root/.local/bin/pueue", "log", "--full", task_id], stdout=handle, check=True
        )
    paths = [
        root / name for name in ("manifest.json", "metrics.json", "artifacts.json", "events.jsonl")
    ]
    paths += sorted((root / "predictions").glob("*.jsonl"))
    paths += sorted((root / "checkpoints").glob("*/*.pt"))
    paths.append(log)
    inventory = {str(p): {"size": p.stat().st_size, "sha256": digest(p)} for p in paths}
    result = {
        "run_id": root.name,
        "audit_script_sha256": digest(Path(__file__)),
        "same_training_order": True,
        "train_holdout_disjoint": True,
        "mismatched_donors_disjoint_from_receivers": True,
        "training": training,
        "predictions": prediction_checks,
        "common_milestones": [e["step"] for e in events if e["kind"] == "common_milestone"],
        "learning_probe": next(e for e in events if e["kind"] == "learning_probe"),
        "files": inventory,
    }
    with (post / "integrity-audit.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "audit": str(post / "integrity-audit.json"),
                "checks": "passed",
                "training": training,
                "predictions": prediction_checks,
            }
        )
    )


if __name__ == "__main__":
    main()
