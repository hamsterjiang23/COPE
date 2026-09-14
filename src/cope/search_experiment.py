"""Registered v3: six frozen-backbone recipes, dev selection, untouched evaluation."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch

from cope.experiment_data import (
    donor_indices,
    epoch_order,
    extract_answer,
    paired_diagnostics,
    prompt_ids,
    sha256_file,
    write_new_json,
)
from cope.interface_search import SearchBridge, conditioned_inputs, numeric_answer
from cope.real_experiment import (
    BudgetExpired,
    Experiment,
    cpu_tree,
    load_tokenizers,
    seed_all,
    synchronize,
)


def validate(config):
    assert config["version"] == "v3"
    assert config["training_scope"] == "interface_only" and "lora" not in config
    assert all(config[r]["frozen"] for r in ("teacher", "student"))
    assert len(config["candidates"]) == 6
    assert len({c["id"] for c in config["candidates"]}) == 6
    assert {(c["kind"], c["layer"]) for c in config["candidates"]} == {
        (kind, layer) for kind in ("embedding", "kv_add", "kv_replace") for layer in (20, -1)
    }


def prepare(config, path):
    import pyarrow.parquet as pq

    teacher, student = load_tokenizers(config)
    old = json.loads(Path(config["dataset"]["exclude_samples"]).read_text())
    excluded = {r["id"] for key in ("train", "holdout") for r in old[key]}
    rows, seen = [], set()
    for index, row in enumerate(pq.read_table(config["dataset"]["path"]).to_pylist()):
        question = row["question"].strip()
        key = " ".join(question.split())
        sid = hashlib.sha256(key.encode()).hexdigest()
        if sid in seen or sid in excluded:
            continue
        seen.add(sid)
        rows.append(dict(row, id=sid, source_index=index))
    random.Random(config["dataset"]["split_seed"]).shuffle(rows)
    data, offset = {}, 0
    # Reserve evaluation before any answer-length training exclusion.
    for split in ("holdout", "dev", "calibration"):
        n = config["dataset"][split + "_size"]
        data[split] = rows[offset : offset + n]
        offset += n
    data["train"], data["train_exclusions"] = [], []
    for row in rows:
        row["teacher_ids"] = prompt_ids(teacher, row["question"])
        row["prompt_ids"] = prompt_ids(student, row["question"])
        row["target_ids"] = student.encode(row["answer"], add_special_tokens=False) + [
            student.eos_token_id
        ]
    for row in rows[offset:]:
        if (
            max(len(row["teacher_ids"]), len(row["prompt_ids"])) > 512
            or len(row["prompt_ids"]) + len(row["target_ids"]) > 1024
        ):
            data["train_exclusions"].append(row["id"])
        elif len(data["train"]) < config["dataset"]["train_size"]:
            data["train"].append(row)
    assert len(data["train"]) == config["dataset"]["train_size"]
    all_ids = [r["id"] for split in ("train", "dev", "calibration", "holdout") for r in data[split]]
    assert len(set(all_ids)) == len(all_ids) and not (set(all_ids) & excluded)
    # All replacement candidates use the same existing final chat-prefix token as start.
    starts = {
        r["prompt_ids"][-1]
        for split in ("train", "dev", "calibration", "holdout")
        for r in data[split]
    }
    assert len(starts) == 1
    data.update(
        config=config,
        start_token_id=starts.pop(),
        source_sha256=sha256_file(config["dataset"]["path"]),
    )
    write_new_json(path, data)


class SearchExperiment(Experiment):
    def build(self):
        import subprocess

        from transformers import AutoModelForCausalLM

        self.metrics = {"status": "running", "evaluations": {}, "comparisons": {}, "screen": {}}
        for line in subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines():
            index, memory, utilization = map(int, line.split(","))
            if index in (0, 1) and (memory > 1024 or utilization > 10):
                raise RuntimeError(f"GPU {index} is occupied; no automatic competing launch")
        self.teacher_tokenizer, self.student_tokenizer = load_tokenizers(self.config)
        self.models = {}
        for role in ("teacher", "student"):
            self.models[role] = (
                AutoModelForCausalLM.from_pretrained(
                    self.config[role]["path"],
                    torch_dtype=torch.bfloat16,
                    device_map=self.config[role]["device"],
                    attn_implementation="sdpa",
                    local_files_only=True,
                )
                .requires_grad_(False)
                .eval()
            )
        self.student = self.models["student"]
        self.teacher_model = self.models["teacher"]
        self.metrics = {"status": "running", "evaluations": {}, "comparisons": {}, "screen": {}}
        self.bridge = None
        self.freeze_check()
        self.event("models_loaded", teacher_trainable=0, student_trainable=0)

    def freeze_check(self):
        for model in self.models.values():
            assert not any(p.requires_grad or p.grad is not None for p in model.parameters())

    def create_bridge(self, candidate, mode, seed):
        seed_all(seed)
        c = self.config["interface"]
        self.bridge = SearchBridge(
            self.config["teacher"]["hidden_size"],
            self.student.config,
            candidate["kind"],
            c["slots"],
            c["width"],
            c["kv_rank"],
        ).place(self.config["teacher"]["device"], self.config["student"]["device"])
        self.candidate, self.mode = candidate, mode
        # Inactive parameters are frozen too, excluding unused constant/projector state.
        for name, p in self.bridge.named_parameters():
            p.requires_grad_(
                name == "constant" or name.startswith("reader.")
                if mode == "constant"
                else name != "constant"
            )
        self.params = {n: p for n, p in self.bridge.named_parameters() if p.requires_grad}
        return torch.optim.AdamW(
            list(self.params.values()), lr=self.config["training"]["lr"], weight_decay=0
        )

    def latent(self, row, donor=None):
        if self.mode == "constant":
            return self.bridge.constant
        chosen = donor if donor is not None else row
        ids = torch.tensor([chosen["teacher_ids"]], device=self.config["teacher"]["device"])
        with torch.no_grad():
            output = self.teacher_model.model(
                input_ids=ids, output_hidden_states=True, use_cache=False
            )
            hidden = output.hidden_states[self.candidate["layer"]]
        self.teacher_calls += 1
        return self.bridge.encode(hidden)

    def loss(self, row, donor=None):
        z = self.latent(row, donor)
        memory = self.bridge.decode(z, self.student)
        inputs = conditioned_inputs(
            self.student, self.bridge, memory, row["prompt_ids"], row["target_ids"]
        )
        return self.student(**inputs, use_cache=True).loss

    def save(self, label, step, optimizer):
        path = self.root / "checkpoints" / label / f"step-{step:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        state = dict(
            candidate=self.candidate,
            mode=self.mode,
            step=step,
            bridge=cpu_tree(self.bridge.state_dict()),
            optimizer=cpu_tree(optimizer.state_dict()),
            python_rng=random.getstate(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            budget_used_seconds=self.prior_seconds + time.monotonic() - self.start,
        )
        import numpy as np

        state["numpy_rng"] = np.random.get_state()
        torch.save(state, path.with_suffix(".partial"))
        os.replace(path.with_suffix(".partial"), path)
        self.latest[label] = str(path)
        self.event("checkpoint", label=label, step=step, path=str(path))

    def load(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        optimizer = self.create_bridge(state["candidate"], state["mode"], self.config["seed"])
        self.bridge.load_state_dict(state["bridge"])
        optimizer.load_state_dict(state["optimizer"])
        return state, optimizer

    def train_candidate(self, candidate, rows, mode, label, steps, probe=False):
        optimizer = self.create_bridge(candidate, mode, self.config["seed"])
        self.event(
            "training_start",
            label=label,
            candidate=candidate,
            mode=mode,
            active_parameters=sum(p.numel() for p in self.params.values()),
        )
        accumulation = self.config["training"]["accumulation"]
        completed = 0
        try:
            for step in range(steps):
                self.check_budget(self.config["budget"]["evaluation_reserve_seconds"])
                optimizer.zero_grad(set_to_none=True)
                total, ids = 0.0, []
                synchronize()
                tick = time.monotonic()
                for part in range(accumulation):
                    self.check_budget()
                    absolute = step * accumulation + part
                    epoch = absolute // len(rows)
                    index = epoch_order(len(rows), self.config["seed"], epoch)[absolute % len(rows)]
                    donor = donor_indices(len(rows), self.config["seed"] + 1009, epoch)[index]
                    row = rows[index]
                    loss = self.loss(row, rows[donor] if mode == "mismatched" else None)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite loss")
                    (loss / accumulation).backward()
                    total += loss.item() / accumulation
                    ids.append(
                        {
                            "id": row["id"],
                            "donor_id": rows[donor]["id"] if mode == "mismatched" else None,
                        }
                    )
                self.freeze_check()
                grads = {
                    n: p.grad.float().norm().item()
                    for n, p in self.params.items()
                    if p.grad is not None
                }
                if not grads or not any(v > 0 for v in grads.values()):
                    raise RuntimeError("interface has no gradient")
                norm = torch.nn.utils.clip_grad_norm_(
                    list(self.params.values()), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                synchronize()
                completed = step + 1
                self.event(
                    "step",
                    label=label,
                    probe=probe,
                    step=completed,
                    loss=total,
                    grad_norm=float(norm),
                    seconds=time.monotonic() - tick,
                    examples=ids,
                )
                if not probe and completed % 32 == 0:
                    self.save(label, completed, optimizer)
        except BudgetExpired:
            if not probe and completed and completed % 32:
                self.save(label, completed, optimizer)
            raise
        if not probe and completed % 32:
            self.save(label, completed, optimizer)

    @torch.no_grad()
    def generate(self, model, inputs):
        inputs = dict(inputs)
        inputs.pop("labels", None)
        output = []
        eos = model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos)
        length = inputs["attention_mask"].shape[1]
        for _ in range(self.config["generation"]["max_new_tokens"]):
            self.check_budget()
            result = model(**inputs, use_cache=True, logits_to_keep=1)
            token = result.logits[:, -1].argmax(-1, keepdim=True)
            value = token.item()
            output.append(value)
            if value in eos:
                break
            inputs = dict(
                input_ids=token,
                past_key_values=result.past_key_values,
                attention_mask=torch.ones((1, length + 1), device=token.device, dtype=torch.long),
                position_ids=torch.tensor([[length]], device=token.device),
            )
            length += 1
        return output, bool(output and output[-1] not in eos)

    @torch.no_grad()
    def evaluate(self, rows, label, assisted=False, mismatch=False, teacher=False):
        records = []
        path = self.root / "predictions" / f"{label}.jsonl"
        path.parent.mkdir(exist_ok=True)
        donors = donor_indices(len(rows), self.config["seed"] + 2003)
        with path.open("x") as handle:
            for index, row in enumerate(rows):
                self.check_budget()
                synchronize()
                start = time.monotonic()
                z = (
                    self.latent(row, rows[donors[index]] if mismatch else None)
                    if assisted
                    else None
                )
                synchronize()
                encoded = time.monotonic()
                model = self.teacher_model if teacher else self.student
                tokenizer = self.teacher_tokenizer if teacher else self.student_tokenizer
                if teacher:
                    inputs = dict(
                        input_ids=torch.tensor([row["teacher_ids"]], device=model.device),
                        attention_mask=torch.ones(
                            (1, len(row["teacher_ids"])), device=model.device, dtype=torch.long
                        ),
                    )
                else:
                    memory = self.bridge.decode(z, self.student) if assisted else None
                    inputs = conditioned_inputs(
                        model, self.bridge if assisted else None, memory, row["prompt_ids"]
                    )
                synchronize()
                decoded = time.monotonic()
                tokens, truncated = self.generate(model, inputs)
                synchronize()
                end = time.monotonic()
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                prediction, gold = numeric_answer(text), numeric_answer(row["answer"])
                if gold is None:
                    raise ValueError("unparsed reference answer")
                if not teacher:
                    memory = self.bridge.decode(z, model) if assisted else None
                    loss_inputs = conditioned_inputs(
                        model,
                        self.bridge if assisted else None,
                        memory,
                        row["prompt_ids"],
                        row["target_ids"],
                    )
                    nll = model(**loss_inputs, use_cache=True).loss.item()
                    target_tokens = len(row["target_ids"])
                else:
                    target = tokenizer.encode(row["answer"], add_special_tokens=False) + [
                        tokenizer.eos_token_id
                    ]
                    loss_inputs = conditioned_inputs(model, None, None, row["teacher_ids"], target)
                    nll = model(**loss_inputs, use_cache=True).loss.item()
                    target_tokens = len(target)
                record = dict(
                    id=row["id"],
                    donor_id=rows[donors[index]]["id"] if mismatch else None,
                    prediction=prediction,
                    gold=gold,
                    correct=prediction == gold,
                    strict_prediction=extract_answer(text),
                    format_success=extract_answer(text) is not None,
                    text=text,
                    output_tokens=len(tokens),
                    truncated=truncated,
                    nll_sum=nll * target_tokens,
                    target_tokens=target_tokens,
                    online_ms=1000 * (end - start),
                    encode_ms=1000 * (encoded - start),
                    reader_transfer_ms=1000 * (decoded - encoded),
                    generate_ms=1000 * (end - decoded),
                    payload_bytes=self.bridge.slots * self.bridge.width * 4 if assisted else 0,
                )
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
        self.metrics["evaluations"][label] = self.summarize(records)
        self.event("evaluated", label=label, **self.metrics["evaluations"][label])
        return records

    @staticmethod
    def summarize(records):
        n = len(records)
        return dict(
            n=n,
            accuracy=sum(r["correct"] for r in records) / n,
            parse_failures=sum(r["prediction"] is None for r in records),
            format_success_rate=sum(r["format_success"] for r in records) / n,
            truncation_rate=sum(r["truncated"] for r in records) / n,
            nll=sum(r["nll_sum"] for r in records) / sum(r["target_tokens"] for r in records),
            mean_output_tokens=sum(r["output_tokens"] for r in records) / n,
            mean_online_ms=sum(r["online_ms"] for r in records) / n,
        )

    @torch.no_grad()
    def native_cache_check(self, row):
        # Validate in FP32: BF16 full-prompt and incremental kernels round differently.
        # A separate copy avoids casting the experimental model's rotary buffers.
        reference = copy.deepcopy(self.student).float()
        ids = torch.tensor([row["prompt_ids"]], device=reference.device)
        native = reference(input_ids=ids, use_cache=True, logits_to_keep=1)
        prefix = reference(input_ids=ids[:, :-1], use_cache=True, logits_to_keep=1)
        resumed = reference(
            input_ids=ids[:, -1:],
            past_key_values=prefix.past_key_values,
            attention_mask=torch.ones_like(ids),
            position_ids=torch.tensor([[ids.shape[1] - 1]], device=ids.device),
            use_cache=True,
            logits_to_keep=1,
        )
        delta = (native.logits - resumed.logits).abs().max().item()
        assert torch.allclose(native.logits, resumed.logits, atol=1e-4, rtol=1e-4)
        assert native.logits.argmax(-1).item() == resumed.logits.argmax(-1).item()
        self.event(
            "native_cache_check",
            max_abs_logit_error=delta,
            same_greedy_token=True,
            validation_dtype="float32_copy",
            experimental_dtype="bfloat16_unchanged",
        )

    def run(self, data):
        self.build()
        self.native_cache_check(data["calibration"][0])
        self.native_cache_check(data["calibration"][1])
        # Exercise every actual two-GPU path before any full training or evaluation sweep.
        for candidate in self.config["candidates"]:
            self.create_bridge(candidate, "correct", self.config["seed"])
            for row in data["train"][:2]:
                self.check_budget()
                self.bridge.zero_grad(set_to_none=True)
                calls = self.teacher_calls
                loss = self.loss(row)
                loss.backward()
                self.freeze_check()
                assert self.teacher_calls == calls + 1
                if not torch.isfinite(loss):
                    raise FloatingPointError("real probe nonfinite loss")
                for family in ("projector.", "reader."):
                    grads = [p.grad for n, p in self.params.items() if n.startswith(family)]
                    assert any(g is not None and g.abs().sum() > 0 for g in grads)
                    assert all(g is None or torch.isfinite(g).all() for g in grads)
                self.event(
                    "real_probe",
                    candidate=candidate["id"],
                    loss=loss.item(),
                    teacher_calls_this_batch=1,
                )
        # Calibration outputs are evidence only; no automatic parser or prompt tuning.
        self.evaluate(data["calibration"], "calibration_student")
        baseline = self.evaluate(data["dev"], "dev_student")
        for candidate in self.config["candidates"]:
            label = candidate["id"]
            self.create_bridge(candidate, "correct", self.config["seed"])
            with torch.no_grad():
                before = sum(self.loss(r).item() for r in data["train"][:16]) / 16
            self.train_candidate(
                candidate, data["train"][:16], "correct", label + "_probe", 16, probe=True
            )
            with torch.no_grad():
                after = sum(self.loss(r).item() for r in data["train"][:16]) / 16
            self.event("learning_probe", label=label, before=before, after=after)
            if after >= before:
                self.metrics["screen"][label] = {
                    "status": "learning_probe_failed",
                    "before": before,
                    "after": after,
                }
                continue
            self.train_candidate(
                candidate, data["train"], "correct", label, self.config["training"]["steps"]
            )
            correct = self.evaluate(data["dev"], "dev_" + label, assisted=True)
            wrong = self.evaluate(
                data["dev"], "dev_" + label + "_test_mismatch", assisted=True, mismatch=True
            )
            b = paired_diagnostics(baseline, correct)
            m = paired_diagnostics(wrong, correct)
            score = min(b["net_accuracy_change"], m["net_accuracy_change"])
            self.metrics["screen"][label] = dict(
                status="screened",
                score=score,
                vs_student=b,
                vs_mismatch=m,
                accuracy=self.summarize(correct)["accuracy"],
            )
        eligible = [
            c
            for c in self.config["candidates"]
            if self.metrics["screen"][c["id"]].get("score", -1) > 0
        ]
        if not eligible:
            self.metrics.update(
                status="completed_no_candidate",
                conclusion="No candidate passed dev point-estimate gate; holdout remains unopened.",
            )
            return
        winner = sorted(
            eligible, key=lambda c: (-self.metrics["screen"][c["id"]]["score"], c["id"])
        )[0]
        self.metrics["selected"] = winner
        write_new_json(
            self.root / "selection.json",
            dict(candidate=winner, screen=self.metrics["screen"], rule=self.config["selection"]),
        )
        for mode in ("constant", "mismatched"):
            self.train_candidate(
                winner,
                data["train"],
                mode,
                winner["id"] + "_" + mode,
                self.config["training"]["steps"],
            )
        results = {"student": self.evaluate(data["holdout"], "holdout_student")}
        for mode in ("correct", "constant", "mismatched"):
            label = winner["id"] + ("" if mode == "correct" else "_" + mode)
            self.load(self.latest[label])
            results[mode] = self.evaluate(data["holdout"], "holdout_" + mode, assisted=True)
            if mode == "correct":
                results["test_mismatch"] = self.evaluate(
                    data["holdout"], "holdout_test_mismatch", assisted=True, mismatch=True
                )
        results["teacher"] = self.evaluate(data["holdout"], "holdout_teacher", teacher=True)
        for base in ("student", "constant", "mismatched", "test_mismatch"):
            self.metrics["comparisons"]["correct_vs_" + base] = paired_diagnostics(
                results[base], results["correct"]
            )
        groups = {}
        for t in (False, True):
            for s in (False, True):
                indices = [
                    i
                    for i in range(len(data["holdout"]))
                    if results["teacher"][i]["correct"] == t
                    and results["student"][i]["correct"] == s
                ]
                groups[f"teacher_{int(t)}_student_{int(s)}"] = dict(
                    n=len(indices), correct=sum(results["correct"][i]["correct"] for i in indices)
                )
        self.metrics["ability_groups"] = groups
        supported = all(
            v["paired_bootstrap_95"][0] > 0 for v in self.metrics["comparisons"].values()
        )
        self.metrics.update(
            status="completed",
            conclusion="promising_single_seed" if supported else "inconclusive_or_no_gain",
        )

    def finish(self, error=None):
        self.metrics.setdefault("evaluations", {})
        if error:
            self.metrics.update(
                status="budget_exhausted" if isinstance(error, BudgetExpired) else "failed",
                error=f"{type(error).__name__}: {error}",
            )
        self.metrics.update(
            budget_used_seconds=self.prior_seconds + time.monotonic() - self.start,
            teacher_encoding_calls=self.teacher_calls,
            peak_allocated_bytes=[
                torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
            ],
            parameter_scope="both backbones frozen; interface only",
        )
        # Partial predictions remain explicitly incomplete, never used as confirmation.
        for path in (self.root / "predictions").glob("*.jsonl"):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if rows and path.stem not in self.metrics["evaluations"]:
                self.metrics["evaluations"][path.stem] = dict(self.summarize(rows), complete=False)
        self.event("finished", status=self.metrics["status"])
        write_new_json(self.root / "metrics.json", self.metrics)
        write_new_json(
            self.root / "artifacts.json",
            dict(
                root=str(self.root),
                checkpoints=self.latest,
                events=str(self.root / "events.jsonl"),
                predictions=str(self.root / "predictions"),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["validate", "prepare", "run"])
    parser.add_argument("--config", default="goal/v3/config.json")
    parser.add_argument("--data")
    parser.add_argument("--run-id")
    parser.add_argument("--source-record")
    parser.add_argument("--assets")
    parser.add_argument("--prior-run")
    parser.add_argument("--diagnostic-seconds", type=float, default=0)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate(config)
    if args.stage == "validate":
        print(json.dumps(config["candidates"], indent=2))
        return
    if args.stage == "prepare":
        prepare(config, args.data)
        return
    data = json.loads(Path(args.data).read_text())
    assert data["config"] == config
    if not args.run_id or not args.run_id.replace("-", "").isalnum():
        raise ValueError("unique alphanumeric/hyphen run ID required")
    source = json.loads(Path(args.source_record).read_text())
    if not args.assets:
        parser.error("run requires a freshly verified --assets inventory")
    source["assets"] = json.loads(Path(args.assets).read_text())
    source["assets_sha256"] = sha256_file(args.assets)
    if args.diagnostic_seconds < 0:
        raise ValueError("negative diagnostic budget")
    source["diagnostic_seconds_charged"] = args.diagnostic_seconds
    if args.prior_run:
        prior = json.loads((Path(args.prior_run) / "metrics.json").read_text())
        if prior["status"] != "failed" or prior.get("screen") or prior.get("evaluations"):
            raise ValueError("Only pre-training startup failures can be restarted by this entry")
    source.update(prepared_data_sha256=sha256_file(args.data), prepared_data_path=args.data)
    lock_path = Path(config["server"]["workspace"]) / "outputs" / "cope-gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = SearchExperiment(config, args.run_id, source, resume_from=args.prior_run)
        run.prior_seconds += args.diagnostic_seconds
        run.deadline -= args.diagnostic_seconds
        run.event("budget_inherited", prior_seconds=run.prior_seconds)
        try:
            run.run(data)
        except BudgetExpired as exc:
            run.finish(exc)
        except BaseException as exc:
            run.finish(exc)
            raise
        else:
            run.finish()


if __name__ == "__main__":
    main()
