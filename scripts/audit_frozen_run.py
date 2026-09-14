"""CPU-only post-run audit for frozen-endpoint experiments; preserve original scores."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch

from cope.experiment_data import extract_answer, paired_diagnostics, sha256_file, write_new_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    manifest = json.loads((root / "manifest.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    config, source = manifest["config"], manifest["source"]
    assert metrics["status"] == "completed_exploratory"
    assert config["training_scope"] == "interface_only" and "lora" not in config
    assert all(config[role]["frozen"] for role in ("teacher", "student"))
    assert metrics["parameter_counts"]["student_trainable"] == 0
    checkout = Path(source["checkout"])
    for name, sha in source["files"].items():
        assert sha256_file(checkout / name) == sha, name
    assert sha256_file(source["archive"]) == source["archive_sha256"]
    assert sha256_file(source["patch"]) == source["patch_sha256"]
    assert sha256_file(source["prepared_data_path"]) == source["prepared_data_sha256"]
    assets = json.loads(Path(source["assets_record"]).read_text())
    assert sha256_file(source["assets_record"]) == source["assets_record_sha256"]
    for name, info in assets["files"].items():
        assert sha256_file(name) == info["sha256"], name
    data = json.loads(Path(source["prepared_data_path"]).read_text())
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    train_ids = {row["id"] for row in data["train"]}
    holdout_ids = [row["id"] for row in data["holdout"]]
    assert not train_ids.intersection(holdout_ids)
    milestones = [e["step"] for e in events if e["kind"] == "common_milestone"]
    assert milestones == [32, 64, 96, 128]
    sequences, training = {}, {}
    for arm in config["training"]["arms"]:
        steps = [
            e for e in events if e["kind"] == "step" and e["phase"] == "train" and e["arm"] == arm
        ]
        sequences[arm] = [r["id"] for e in steps for r in e["examples"]]
        assert len(steps) == 128 and len(sequences[arm]) == len(train_ids) == 1024
        assert set(sequences[arm]) == train_ids
        assert all(
            math.isfinite(e["loss"]) and math.isfinite(e["grad_norm"]) and e["grad_norm"] > 0
            for e in steps
        )
        if arm == "mismatched":
            assert all(
                r["donor_id"] != r["id"] and r["donor_id"] in train_ids
                for e in steps
                for r in e["examples"]
            )
        early = torch.load(
            root / "checkpoints" / arm / "step-0032.pt", map_location="cpu", weights_only=False
        )
        final = torch.load(
            root / "checkpoints" / arm / "step-0128.pt", map_location="cpu", weights_only=False
        )
        weights = final["weights"]
        assert all(
            n.startswith(("private_projectors.", "readers.", "controller.", "reducer."))
            for n in weights
        )
        assert not any("lora" in n.lower() for n in weights)
        assert all(bool(torch.isfinite(p).all()) for p in weights.values())
        changed = [n for n, p in weights.items() if not torch.equal(p, early["weights"][n])]
        expected = (
            ("controller.", "readers.")
            if arm == "constant"
            else ("private_projectors.", "readers.")
        )
        assert all(n.startswith(expected) for n in changed)
        assert all(any(n.startswith(prefix) for n in changed) for prefix in expected)
        optimized = [p for n, p in weights.items() if n.startswith(expected)]
        assert sum(len(g["params"]) for g in final["optimizer"]["param_groups"]) == len(optimized)
        training[arm] = {
            "steps": 128,
            "exposures": 1024,
            "sequence_sha256": hashlib.sha256(json.dumps(sequences[arm]).encode()).hexdigest(),
            "parameters_in_optimizer": sum(p.numel() for p in optimized),
            "changed_interface_tensors_32_to_128": changed,
            "backbone_or_lora_tensors_in_checkpoint": 0,
        }
        del early, final, weights, optimized
    assert all(s == sequences["correct"] for s in sequences.values())
    predictions, prediction_checks = {}, {}
    for path in sorted((root / "predictions").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["id"] for r in rows] == holdout_ids
        for r in rows:
            assert extract_answer(r["text"]) == r["prediction"]
            assert r["correct"] == (r["prediction"] is not None and r["prediction"] == r["gold"])
            if r["donor_id"] is not None:
                assert r["id"] != r["donor_id"] and r["donor_id"] in holdout_ids
        assert sum(r["correct"] for r in rows) / len(rows) == metrics["arms"][path.stem]["accuracy"]
        predictions[path.stem] = rows
        prediction_checks[path.stem] = {
            "n": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "parse_failures": sum(r["prediction"] is None for r in rows),
            "truncated": sum(r["truncated"] for r in rows),
        }
    assert len(predictions) == 19
    for name, result in metrics["comparisons"].items():
        assisted, baseline = name.split("_vs_")
        if assisted == "test_mismatched":
            assisted = "correct_test_mismatched"
        assert paired_diagnostics(predictions[baseline], predictions[assisted]) == result, name
    # The original process checks every parameter's requires_grad/grad at build
    # and every optimization step. Its frozen source and success are evidence;
    # we cannot retrospectively inspect every in-memory backbone tensor.
    source_text = (checkout / "src/cope/real_experiment.py").read_text()
    assert "assert_frozen_backbones(self.system)" in source_text
    assert events[-1]["kind"] == "finished" and events[-1]["status"] == metrics["status"]
    post = root / "postprocess"
    post.mkdir(exist_ok=True)
    task = str(manifest["pueue_task_id"])
    log = post / f"pueue-{task}.log"
    with log.open("xb") as handle:
        subprocess.run(["/root/.local/bin/pueue", "log", "--full", task], stdout=handle, check=True)
    paths = [root / n for n in ("manifest.json", "metrics.json", "artifacts.json", "events.jsonl")]
    paths += (
        sorted((root / "predictions").glob("*.jsonl"))
        + sorted((root / "checkpoints").glob("*/*.pt"))
        + [log]
    )
    result = {
        "run_id": root.name,
        "audit_script_sha256": sha256_file(__file__),
        "frozen_source_and_assets_hashes_verified": True,
        "frozen_endpoints": {
            "teacher_trainable": 0,
            "student_trainable": 0,
            "evidence": (
                "frozen source configuration and all-step requires_grad/grad assertions; "
                "model asset hashes unchanged"
            ),
            "full_in_memory_weight_comparison_after_training": False,
        },
        "train_holdout_disjoint": True,
        "same_training_order": True,
        "all_donors_differ_from_receivers": True,
        "all_recorded_paired_comparisons_recomputed": True,
        "common_milestones": milestones,
        "training": training,
        "predictions": prediction_checks,
        "learning_probe": next(e for e in events if e["kind"] == "learning_probe"),
        "files": {str(p): {"size": p.stat().st_size, "sha256": sha256_file(p)} for p in paths},
    }
    write_new_json(post / "integrity-audit.json", result)
    print(
        json.dumps(
            {
                "audit": str(post / "integrity-audit.json"),
                "checks": "passed",
                "predictions": prediction_checks,
            }
        )
    )


if __name__ == "__main__":
    main()
