"""CPU audit of a completed v3 search; run with its frozen checkout on PYTHONPATH.

Writes supplemental evidence exclusively. Does not replace metrics or rescore with
a new parser. Checkpoints must be trusted outputs of this repository.
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import torch

from cope.experiment_data import extract_answer, paired_diagnostics, sha256_file, write_new_json
from cope.interface_search import numeric_answer
from cope.search_experiment import SearchExperiment


def read(path):
    import json

    return json.loads(Path(path).read_text())


def lines(path):
    import json

    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    manifest, metrics = read(root / "manifest.json"), read(root / "metrics.json")
    config, source = manifest["config"], manifest["source"]
    assert config["version"] == "v3" and metrics["status"] == "completed"
    assert all(config[r]["frozen"] for r in ("teacher", "student"))
    assert config["training_scope"] == "interface_only" and "lora" not in config
    for name, digest in source["files"].items():
        assert sha256_file(args.checkout / name) == digest, name
    for name, key in (("source.tar.gz", "archive_sha256"), ("working-tree.patch", "patch_sha256")):
        assert sha256_file(args.source_dir / name) == source[key]
    for name, info in source["assets"]["files"].items():
        assert sha256_file(name) == info["sha256"], name
    assert sha256_file(source["prepared_data_path"]) == source["prepared_data_sha256"]
    data = read(source["prepared_data_path"])
    split_ids = {s: [r["id"] for r in data[s]] for s in ("train", "calibration", "dev", "holdout")}
    all_ids = [sid for ids in split_ids.values() for sid in ids]
    assert len(all_ids) == len(set(all_ids)) == 864
    old = read(args.checkout / config["dataset"]["exclude_samples"])
    old_ids = {r["id"] for s in ("train", "holdout") for r in old[s]}
    assert not set(all_ids) & old_ids
    events = lines(root / "events.jsonl")
    loaded = next(e for e in events if e["kind"] == "models_loaded")
    assert loaded["teacher_trainable"] == loaded["student_trainable"] == 0
    probes = [e for e in events if e["kind"] == "real_probe"]
    assert len(probes) == 12 and all(e["teacher_calls_this_batch"] == 1 for e in probes)
    learning = [e for e in events if e["kind"] == "learning_probe"]
    assert len(learning) == 6 and all(e["after"] < e["before"] for e in learning)
    training, sequences = {}, []
    labels = list(read(root / "artifacts.json")["checkpoints"])
    assert len(labels) == 8
    for label in labels:
        steps = [e for e in events if e["kind"] == "step" and e["label"] == label]
        assert [e["step"] for e in steps] == list(range(1, 129))
        seq = [r["id"] for e in steps for r in e["examples"]]
        assert Counter(seq) == Counter({sid: 2 for sid in split_ids["train"]})
        sequences.append(seq)
        assert all(math.isfinite(e["loss"]) and e["grad_norm"] > 0 for e in steps)
        for e in steps:
            for r in e["examples"]:
                if r["donor_id"] is not None:
                    assert r["donor_id"] != r["id"] and r["donor_id"] in split_ids["train"]
        states = []
        for step in (32, 64, 96, 128):
            state = torch.load(
                root / "checkpoints" / label / f"step-{step:04d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            assert state["step"] == step and state["optimizer"]["state"]
            assert all(k in state for k in ("python_rng", "numpy_rng", "torch_rng", "cuda_rng"))
            states.append(state["bridge"])
        changed = [n for n in states[-1] if not torch.equal(states[0][n], states[-1][n])]
        assert changed and all(n.startswith(("projector.", "reader.", "constant")) for n in changed)
        start = next(e for e in events if e["kind"] == "training_start" and e["label"] == label)
        training[label] = dict(
            steps=128,
            exposures=len(seq),
            changed_tensors=len(changed),
            active_parameters=start["active_parameters"],
            step_seconds=sum(e["seconds"] for e in steps),
        )
    assert all(seq == sequences[0] for seq in sequences)
    predictions, latency = {}, {}
    for path in sorted((root / "predictions").glob("*.jsonl")):
        rows = lines(path)
        split = path.stem.split("_")[0]
        assert [r["id"] for r in rows] == split_ids[split]
        for r in rows:
            assert numeric_answer(r["text"]) == r["prediction"]
            assert (r["prediction"] == r["gold"]) == r["correct"]
            assert extract_answer(r["text"]) == r["strict_prediction"]
            if r["donor_id"] is not None:
                assert r["donor_id"] != r["id"] and r["donor_id"] in split_ids[split]
        assert SearchExperiment.summarize(rows) == metrics["evaluations"][path.stem]
        predictions[path.stem] = rows
        latency[path.stem] = {
            key: sum(r[key] for r in rows) / len(rows)
            for key in ("encode_ms", "reader_transfer_ms", "generate_ms", "payload_bytes")
        }
    assert len(predictions) == 20
    for base in ("student", "constant", "mismatched", "test_mismatch"):
        assert (
            paired_diagnostics(predictions["holdout_" + base], predictions["holdout_correct"])
            == (metrics["comparisons"]["correct_vs_" + base])
        )
    scores = {}
    for candidate in config["candidates"]:
        name = candidate["id"]
        correct = predictions["dev_" + name]
        b = paired_diagnostics(predictions["dev_student"], correct)
        m = paired_diagnostics(predictions["dev_" + name + "_test_mismatch"], correct)
        scores[name] = min(b["net_accuracy_change"], m["net_accuracy_change"])
        assert metrics["screen"][name]["score"] == scores[name]
    winner = sorted((n for n in scores if scores[n] > 0), key=lambda n: (-scores[n], n))[0]
    assert winner == read(root / "selection.json")["candidate"]["id"] == metrics["selected"]["id"]
    assert (root / "selection.json").stat().st_mtime < (
        root / "predictions/holdout_student.jsonl"
    ).stat().st_mtime
    supported = all(c["paired_bootstrap_95"][0] > 0 for c in metrics["comparisons"].values())
    assert metrics["conclusion"] == (
        "promising_single_seed" if supported else "inconclusive_or_no_gain"
    )
    assert metrics["budget_used_seconds"] < config["budget"]["seconds"]
    inventory = {
        str(p.relative_to(root)): sha256_file(p)
        for p in root.rglob("*")
        if p.is_file() and "postprocess" not in p.parts
    }
    output = dict(
        status="passed",
        run_id=root.name,
        training=training,
        latency=latency,
        prediction_files=20,
        source_hash=source["content_hash"],
        file_sha256=inventory,
        scope="CPU evidence audit; runtime freeze checks plus unchanged model assets, "
        "not a post-exit comparison of in-memory backbone tensors",
        auditor_sha256=sha256_file(__file__),
    )
    target = root / "postprocess/integrity-audit.json"
    target.parent.mkdir(exist_ok=True)
    write_new_json(target, output)
    print(target)


if __name__ == "__main__":
    main()
