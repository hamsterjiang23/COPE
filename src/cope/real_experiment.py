"""Versioned, budgeted real-model COPE experiments. Run with --help for stages."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import importlib.metadata
import json
import os
import random
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from cope.config import ProtocolConfig
from cope.experiment_data import (
    donor_indices,
    epoch_order,
    extract_answer,
    paired_diagnostics,
    prepare_examples,
    sha256_file,
    summarize_records,
    write_new_json,
)
from cope.factory import build_huggingface_system


class BudgetExpired(Exception):
    """A controlled stop, not evidence against the research hypothesis."""


def validate_recipe(config):
    """Reject obsolete adaptation recipes before downloading or loading models."""
    if config.get("training_scope") != "interface_only" or "lora" in config:
        raise ValueError(
            "Teacher and student must be frozen: use an interface_only recipe without LoRA. "
            "Historical v1 is preserved in its recorded source archive."
        )
    if not all(config[role].get("frozen") is True for role in ("teacher", "student")):
        raise ValueError("Both model records must explicitly set frozen=true")
    if config["training"].get("width_weighting") != "uniform":
        raise ValueError("This runner implements uniform multi-width loss weighting")
    arms = config["training"]["arms"]
    if set(arms) != {"correct", "full_only", "constant", "mismatched"} or len(arms) != 4:
        raise ValueError(
            "Train correct/full_only/constant/mismatched; no_latent is evaluation-only"
        )
    widths = config["protocol"]["widths"]
    if not widths or len(set(widths)) != len(widths):
        raise ValueError("unique nonempty widths required")
    if max(widths) != config["protocol"]["max_width"] or min(widths) <= 0:
        raise ValueError("widths must include max_width and be positive")
    if config["protocol"]["teacher_hidden_state_index"] != -1:
        raise ValueError("teacher input to the protocol must use the last hidden layer")


def assert_frozen_backbones(system):
    for role, module in (("teacher", system.teacher), ("student", system.students)):
        if any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise AssertionError(f"{role} parameters must stay frozen and gradient-free")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synchronize():
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def padded(sequences, pad, device, left=False):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        start = width - len(seq) if left else 0
        ids[i, start : start + len(seq)] = torch.tensor(seq, device=device)
        mask[i, start : start + len(seq)] = 1
    return {"input_ids": ids, "attention_mask": mask}


def student_batch(rows, tokenizer, device, training=True):
    sequences = [r["prompt_ids"] + r["target_ids"] if training else r["prompt_ids"] for r in rows]
    batch = padded(sequences, tokenizer.pad_token_id, device, left=not training)
    if training:
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for index, row in enumerate(rows):
            labels[index, : len(row["prompt_ids"])] = -100
        batch["labels"] = labels
    return batch


def load_tokenizers(config):
    from transformers import AutoTokenizer

    result = []
    for role in ("teacher", "student"):
        tokenizer = AutoTokenizer.from_pretrained(config[role]["path"], local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        result.append(tokenizer)
    return result


def prepare(config, output):
    import pyarrow.parquet as pq

    teacher_tokenizer, student_tokenizer = load_tokenizers(config)
    rows = pq.read_table(config["dataset"]["path"]).to_pylist()
    data = prepare_examples(rows, teacher_tokenizer, student_tokenizer, config)
    data["dataset"] = config["dataset"]
    data["source_sha256"] = sha256_file(config["dataset"]["path"])
    data["config"] = config
    write_new_json(output, data)
    return data


class Experiment:
    def __init__(self, config, run_id, source_record, resume_from=None):
        self.config = config
        self.run_id = run_id
        self.root = Path(config["server"]["output_root"]) / run_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.resume = Path(resume_from) if resume_from else None
        self.prior_seconds = 0.0
        if self.resume:
            prior = json.loads((self.resume / "manifest.json").read_text())
            if prior["config"] != config:
                raise ValueError(
                    "resume requires identical config; create a new version for recipe changes"
                )
            events = [
                json.loads(line) for line in (self.resume / "events.jsonl").read_text().splitlines()
            ]
            self.prior_seconds = max((e.get("budget_used_seconds", 0) for e in events), default=0)
        self.start = time.monotonic()
        self.deadline = self.start + max(0, config["budget"]["seconds"] - self.prior_seconds)
        self.stop_requested = False
        self.teacher_calls = 0
        self.latest = {}
        self.initial = None
        self.system = None
        self.metrics = {"status": "running", "arms": {}, "comparisons": {}}
        write_new_json(
            self.root / "manifest.json",
            {
                "run_id": run_id,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "config": config,
                "resume_from": str(self.resume) if self.resume else None,
                "source": source_record,
                "pueue_task_id": os.environ.get("PUEUE_TASK_ID"),
                "packages": {
                    d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                },
                "hostname": os.uname().nodename,
                "budget_prior_seconds": self.prior_seconds,
                "hardware": subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ),
            },
        )
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        self.event("run_created")

    def request_stop(self, *_):
        self.stop_requested = True

    def check_budget(self, reserve=0):
        # Leave a small margin for flushing evidence and safe checkpoint writes.
        if self.stop_requested or time.monotonic() + max(reserve, 15) >= self.deadline:
            raise BudgetExpired("budget or stop request reached")

    def generation_guard(self):
        from transformers import StoppingCriteria, StoppingCriteriaList

        experiment = self

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                experiment.check_budget()
                return False

        return StoppingCriteriaList([Deadline()])

    def event(self, kind, **fields):
        record = {
            "time": dt.datetime.now(dt.timezone.utc).isoformat(),
            "kind": kind,
            "budget_used_seconds": self.prior_seconds + time.monotonic() - self.start,
            **fields,
        }
        with (self.root / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
        print(json.dumps(record, allow_nan=False), flush=True)

    def build(self):
        from transformers import AutoModelForCausalLM

        c = self.config
        occupancy = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for line in occupancy.strip().splitlines():
            index, memory, utilization = map(int, line.split(","))
            if index in (0, 1) and (memory > 1024 or utilization > 10):
                raise RuntimeError(
                    f"GPU {index} is occupied: {memory} MiB, {utilization}%; "
                    "retry as a new run when free"
                )
        self.teacher_tokenizer, self.student_tokenizer = load_tokenizers(c)
        self.event("loading_models")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            c["teacher"]["path"],
            torch_dtype=torch.bfloat16,
            device_map=c["teacher"]["device"],
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self.teacher_model.requires_grad_(False).eval()
        student = AutoModelForCausalLM.from_pretrained(
            c["student"]["path"],
            torch_dtype=torch.bfloat16,
            device_map=c["student"]["device"],
            attn_implementation="sdpa",
            local_files_only=True,
        )
        seed_all(c["seed"])
        student.requires_grad_(False).eval()
        pc = c["protocol"]
        self.system = build_huggingface_system(
            ProtocolConfig(
                c["teacher"]["hidden_size"], pc["max_width"], tuple(pc["widths"]), pc["num_slots"]
            ),
            self.teacher_model.model,
            {"student": student},
            projector_sharing="private",
            reducer_kind="masked_mean",
            projector_hidden_size=pc["projector_hidden_size"],
            freeze_student_backbones=True,
        )
        # Only the backbone is called for teacher hidden states; avoid all-position LM logits.
        self.system.private_projectors.float().to(c["teacher"]["device"])
        self.system.controller.float().to(c["teacher"]["device"])
        self.system.readers.float().to(c["student"]["device"])
        self.params = {n: p for n, p in self.system.named_parameters() if p.requires_grad}
        assert_frozen_backbones(self.system)
        allowed = ("private_projectors.", "readers.", "controller.", "reducer.")
        if any(not n.startswith(allowed) for n in self.params):
            raise AssertionError("optimizer must contain only intermediate protocol parameters")
        self.initial = {n: p.detach().cpu().clone() for n, p in self.params.items()}
        self.teacher_probe = self.teacher_model.model.embed_tokens.weight[:2].detach().clone()
        self.frozen_student_probe = student.get_input_embeddings().weight[:2].detach().clone()
        self.system.teacher.register_forward_hook(self.count_teacher)
        self.event(
            "models_loaded", trainable_parameters=sum(p.numel() for p in self.params.values())
        )
        self.check_budget()

    def count_teacher(self, *_):
        self.teacher_calls += 1

    def load_weights(self, weights):
        with torch.no_grad():
            for name, parameter in self.params.items():
                parameter.copy_(weights[name].to(parameter))

    def optimizer(self, arm):
        if arm == "no_latent":
            raise ValueError("Frozen no-latent baseline has no optimizer")
        interface = [
            p
            for n, p in self.params.items()
            if (
                (arm == "constant" and (n.startswith("controller.") or n.startswith("readers.")))
                or (
                    arm in ("correct", "full_only", "mismatched")
                    and not n.startswith("controller.")
                )
            )
        ]
        return torch.optim.AdamW(
            interface, lr=self.config["training"]["interface_lr"], weight_decay=0.0
        )

    def checkpoint(self, arm, step, optimizer):
        target = self.root / "checkpoints" / arm / f"step-{step:04d}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        state = {
            "arm": arm,
            "step": step,
            "weights": {n: p.detach().cpu().clone() for n, p in self.params.items()},
            "optimizer": cpu_tree(optimizer.state_dict()),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        partial = target.with_suffix(".partial")
        torch.save(state, partial)
        os.replace(partial, target)
        self.latest[arm] = str(target)
        self.event("checkpoint", arm=arm, step=step, path=str(target))
        return target

    def restore(self, arm, path=None):
        self.load_weights(self.initial)
        optimizer = self.optimizer(arm)
        seed_all(self.config["seed"])
        if not path:
            return optimizer, 0
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["arm"] != arm:
            raise ValueError("checkpoint arm mismatch")
        self.load_weights(state["weights"])
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        return optimizer, state["step"]

    def teacher_batch(self, rows):
        return padded(
            [r["teacher_ids"] for r in rows],
            self.teacher_tokenizer.pad_token_id,
            self.config["teacher"]["device"],
        )

    def latents(self, rows, arm, donors=None, timed=False):
        if arm == "no_latent":
            return None, (0.0, 0.0)
        if timed:
            synchronize()
        started = time.perf_counter()
        if arm == "constant":
            z = self.system.controller.constant.expand(len(rows), -1, -1)
            encoded = time.perf_counter()
        else:
            inputs = self.teacher_batch(donors if arm == "mismatched" else rows)
            states = self.system.teacher(inputs)
            if timed:
                synchronize()
            encoded = time.perf_counter()
            reduced = self.system.reducer(states, inputs["attention_mask"])
            z = self.system._project(reduced, {"student"})["student"]
        if timed:
            synchronize()
        ended = time.perf_counter()
        return z, ((encoded - started) * 1000, (ended - encoded) * 1000)

    def prefixes(self, rows, arm, donors=None, timed=False, width=None):
        z, (teacher_ms, interface_ms) = self.latents(rows, arm, donors, timed)
        if z is None:
            return None, (teacher_ms, interface_ms)
        start = time.perf_counter()
        u = self.system.read_latent("student", z, width or self.config["protocol"]["max_width"])
        if timed:
            synchronize()
        return u, (teacher_ms, interface_ms + (time.perf_counter() - start) * 1000)

    def loss(self, rows, arm, donors=None):
        # Form one maximum latent, then supervise every nested prefix through
        # the same frozen student. Do not use no_grad around the student path.
        z, _ = self.latents(rows, arm, donors)
        batch = student_batch(rows, self.student_tokenizer, self.config["student"]["device"])
        if z is None:
            return self.system.students["student"].compute_loss(None, batch)
        widths = (
            [self.config["protocol"]["max_width"]]
            if arm == "full_only"
            else self.config["protocol"]["widths"]
        )
        losses = [
            self.system.students["student"].compute_loss(
                self.system.read_latent("student", z, width), batch
            )
            for width in widths
        ]
        return torch.stack(losses).mean()

    def train_segment(self, arm, rows, target_step, phase="train", reserve=0):
        optimizer, step = self.restore(arm, self.latest.get(arm) if phase == "train" else None)
        c = self.config["training"]
        micro, accumulation = c["microbatch"], c["gradient_accumulation"]
        per_step = micro * accumulation
        segment_start = time.monotonic()
        self.system.train()
        completed = step
        try:
            for current in range(step, target_step):
                self.check_budget(reserve)
                synchronize()
                tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                self.system.zero_grad(set_to_none=True)
                losses, pairs = [], []
                for part in range(accumulation):
                    self.check_budget()
                    absolute = current * per_step + part * micro
                    epoch = absolute // len(rows)
                    order = epoch_order(len(rows), self.config["seed"], epoch)
                    indices = [order[(absolute + k) % len(rows)] for k in range(micro)]
                    donor_map = donor_indices(len(rows), self.config["seed"] + 1009, epoch)
                    selected = [rows[i] for i in indices]
                    donors = [rows[donor_map[i]] for i in indices]
                    loss = self.loss(selected, arm, donors)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite training loss")
                    (loss / accumulation).backward()
                    losses.append(loss.item())
                    pairs.extend(
                        {
                            "id": rows[i]["id"],
                            "donor_id": rows[donor_map[i]]["id"] if arm == "mismatched" else None,
                        }
                        for i in indices
                    )
                grads = {
                    n: float(p.grad.float().norm())
                    for n, p in self.params.items()
                    if p.grad is not None
                }
                if not grads or not all(np.isfinite(v) for v in grads.values()):
                    raise FloatingPointError("missing or nonfinite gradients")
                assert_frozen_backbones(self.system)
                if arm in ("correct", "full_only", "mismatched"):
                    for family in ("private_projectors.", "readers."):
                        if not any(v > 0 for n, v in grads.items() if n.startswith(family)):
                            raise RuntimeError(f"no gradient in {family}")
                norm = torch.nn.utils.clip_grad_norm_(
                    self.params.values(), c["max_grad_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                synchronize()
                completed = current + 1
                self.event(
                    "step",
                    phase=phase,
                    arm=arm,
                    step=completed,
                    loss=float(np.mean(losses)),
                    grad_norm=float(norm),
                    seconds=time.perf_counter() - tick,
                    examples=pairs,
                )
            if phase == "train":
                self.checkpoint(arm, completed, optimizer)
        except BudgetExpired:
            if phase == "train" and completed > step:
                self.checkpoint(arm, completed, optimizer)
            raise
        self.event(
            "segment_finished",
            phase=phase,
            arm=arm,
            step=completed,
            seconds=time.monotonic() - segment_start,
            peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)],
        )

    def probe(self, rows):
        self.restore("correct")
        self.system.train()
        calls = self.teacher_calls
        probe_losses = []
        for index in range(2):
            self.check_budget()
            self.system.zero_grad(set_to_none=True)
            loss = self.loss(rows[index * 2 : index * 2 + 2], "correct")
            loss.backward()
            if not torch.isfinite(loss):
                raise FloatingPointError("probe loss")
            probe_losses.append(float(loss.detach()))
        if self.teacher_calls - calls != 2:
            raise AssertionError("teacher must encode exactly once per probe batch")
        if any(p.grad is not None for p in self.teacher_model.parameters()):
            raise AssertionError("teacher received gradients")
        assert_frozen_backbones(self.system)
        self.event(
            "two_batch_probe",
            losses=probe_losses,
            teacher_calls=2,
            peak_memory_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)],
        )
        smoke_rows = rows[: self.config["training"]["smoke_examples"]]
        self.load_weights(self.initial)
        self.system.eval()
        with torch.no_grad():
            before = float(
                np.mean(
                    [
                        float(self.loss(smoke_rows[i : i + 2], "correct"))
                        for i in range(0, len(smoke_rows), 2)
                    ]
                )
            )
        self.train_segment(
            "correct",
            smoke_rows,
            self.config["training"]["smoke_steps"],
            phase="smoke",
        )
        self.system.eval()
        with torch.no_grad():
            after = float(
                np.mean(
                    [
                        float(self.loss(smoke_rows[i : i + 2], "correct"))
                        for i in range(0, len(smoke_rows), 2)
                    ]
                )
            )
        self.event("learning_probe", before_nll=before, after_nll=after, decreased=after < before)
        if not np.isfinite(after) or after >= before:
            raise RuntimeError(
                "16-example learning probe did not reduce loss; diagnose before formal arms"
            )
        if not torch.equal(self.teacher_probe, self.teacher_model.model.embed_tokens.weight[:2]):
            raise AssertionError("teacher changed")
        if not torch.equal(
            self.frozen_student_probe,
            self.system.students["student"].model.get_input_embeddings().weight[:2],
        ):
            raise AssertionError("student backbone changed")
        self.load_weights(self.initial)

    @torch.no_grad()
    def evaluate(self, rows, name, arm="no_latent", teacher=False, width=None):
        self.system.eval()
        self.teacher_model.eval()
        path = self.root / "predictions" / f"{name}.jsonl"
        path.parent.mkdir(exist_ok=True)
        results = []
        donors = donor_indices(len(rows), self.config["seed"] + 9001)
        batch_size = self.config["generation"]["batch_size"]
        with path.open("x") as handle:
            for offset in range(0, len(rows), batch_size):
                self.check_budget()
                batch_rows = rows[offset : offset + batch_size]
                donor_rows = [rows[donors[i]] for i in range(offset, offset + len(batch_rows))]
                # Individual likelihoods avoid assigning a batch-average loss to each sample.
                nlls = []
                if teacher:
                    for row in batch_rows:
                        target = self.teacher_tokenizer.encode(
                            row["answer"], add_special_tokens=False
                        ) + [self.teacher_tokenizer.eos_token_id]
                        ids = row["teacher_ids"] + target
                        inp = torch.tensor([ids], device=self.config["teacher"]["device"])
                        labels = inp.clone()
                        labels[:, : len(row["teacher_ids"])] = -100
                        out = self.teacher_model(input_ids=inp, labels=labels, use_cache=False)
                        nlls.append((float(out.loss) * len(target), len(target)))
                else:
                    # NLL and online generation are separate measurements. Each uses one
                    # fresh encoding; reference targets never enter deployment encoding.
                    prefix_for_nll, _ = self.prefixes(batch_rows, arm, donor_rows, width=width)
                    for k, row in enumerate(batch_rows):
                        sb = student_batch(
                            [row], self.student_tokenizer, self.config["student"]["device"]
                        )
                        loss = self.system.students["student"].compute_loss(
                            None if prefix_for_nll is None else prefix_for_nll[k : k + 1], sb
                        )
                        nlls.append((float(loss) * len(row["target_ids"]), len(row["target_ids"])))
                synchronize()
                start = time.perf_counter()
                if teacher:
                    inputs = padded(
                        [r["teacher_ids"] for r in batch_rows],
                        self.teacher_tokenizer.pad_token_id,
                        self.config["teacher"]["device"],
                        left=True,
                    )
                    gen_start = time.perf_counter()
                    generated = self.teacher_model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=self.config["generation"]["max_new_tokens"],
                        pad_token_id=self.teacher_tokenizer.pad_token_id,
                        use_cache=True,
                        stopping_criteria=self.generation_guard(),
                    )
                    generated = generated[:, inputs["input_ids"].shape[1] :]
                    teacher_ms, interface_ms = 0.0, 0.0
                    tokenizer = self.teacher_tokenizer
                else:
                    prefix, (teacher_ms, interface_ms) = self.prefixes(
                        batch_rows, arm, donor_rows, timed=True, width=width
                    )
                    batch = student_batch(
                        batch_rows,
                        self.student_tokenizer,
                        self.config["student"]["device"],
                        training=False,
                    )
                    synchronize()
                    gen_start = time.perf_counter()
                    generated = self.system.students["student"].generate(
                        prefix,
                        batch,
                        do_sample=False,
                        max_new_tokens=self.config["generation"]["max_new_tokens"],
                        pad_token_id=self.student_tokenizer.pad_token_id,
                        use_cache=True,
                        stopping_criteria=self.generation_guard(),
                    )
                    tokenizer = self.student_tokenizer
                synchronize()
                end = time.perf_counter()
                count = len(batch_rows)
                for k, (row, tokens) in enumerate(zip(batch_rows, generated.tolist(), strict=True)):
                    eos_ids = (
                        self.teacher_model.generation_config.eos_token_id
                        if teacher
                        else self.system.students["student"].model.generation_config.eos_token_id
                    )
                    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
                    stop = next(
                        (i + 1 for i, token in enumerate(tokens) if token in eos_ids), len(tokens)
                    )
                    tokens = tokens[:stop]
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    prediction = extract_answer(text)
                    record = {
                        "id": row["id"],
                        "prediction": prediction,
                        "text": text,
                        "gold": extract_answer(row["answer"]),
                        "correct": prediction is not None
                        and prediction == extract_answer(row["answer"]),
                        "output_tokens": len(tokens),
                        "truncated": len(tokens) >= self.config["generation"]["max_new_tokens"]
                        and (not tokens or tokens[-1] not in eos_ids),
                        "nll_sum": nlls[k][0],
                        "target_tokens": nlls[k][1],
                        "online_ms": (end - start) * 1000 / count,
                        "teacher_prefill_ms": teacher_ms / count,
                        "interface_ms": interface_ms / count,
                        "generate_ms": (end - gen_start) * 1000 / count,
                        "donor_id": donor_rows[k]["id"] if arm == "mismatched" else None,
                    }
                    results.append(record)
                    handle.write(json.dumps(record) + "\n")
                handle.flush()
                self.event("evaluation_batch", name=name, completed=len(results), total=len(rows))
        summary = summarize_records(results)
        summary["complete"] = len(results) == len(rows)
        self.metrics["arms"][name] = summary
        self.event("evaluation_finished", name=name, summary=summary)
        return results

    def run(self, data):
        validate_recipe(self.config)
        self.check_budget()
        self.build()
        rows, holdout = data["train"], data["holdout"]
        self.event("data_loaded", train=len(rows), holdout=len(holdout))
        self.probe(rows)
        arms = self.config["training"]["arms"]
        previous = 0
        if self.resume:
            # Resume only from a milestone completed by every arm.
            candidates = {
                arm: {
                    int(p.stem.split("-")[1]): p
                    for p in (self.resume / "checkpoints" / arm).glob("step-*.pt")
                }
                for arm in arms
            }
            shared_steps = set.intersection(*(set(v) for v in candidates.values()))
            previous = max(shared_steps, default=0)
            if previous:
                self.latest = {arm: str(candidates[arm][previous]) for arm in arms}
        reserve = self.config["budget"]["evaluation_reserve_seconds"]
        self.event("training_started", resumed_common_step=previous)
        common_paths = dict(self.latest)
        try:
            for target in range(previous + 32, self.config["training"]["max_steps"] + 1, 32):
                for arm in arms:
                    self.train_segment(arm, rows, target, reserve=reserve)
                previous = target
                common_paths = dict(self.latest)
                self.event("common_milestone", step=target, checkpoints=common_paths)
        except BudgetExpired:
            self.event("training_budget_boundary", common_step=previous)
        self.metrics["common_step"] = previous
        if not previous:
            self.metrics["status"] = "incomplete_no_common_milestone"
            return
        all_records = {}
        # Complete matched comparisons before spending remaining time on teacher diagnostics.
        for arm in ("no_latent", "correct", "full_only", "constant", "mismatched"):
            if arm != "no_latent":
                self.restore(arm, common_paths[arm])
            all_records[arm] = self.evaluate(holdout, arm, arm)
        self.restore("correct", common_paths["correct"])
        all_records["correct_test_mismatched"] = self.evaluate(
            holdout, "correct_test_mismatched", "mismatched"
        )
        # Both denote exactly the same frozen model and prompt; no extra evaluation.
        all_records["raw_student"] = all_records["no_latent"]
        self.metrics["arms"]["raw_student"] = dict(self.metrics["arms"]["no_latent"])
        self.metrics["baseline_aliases"] = {"raw_student": "no_latent"}
        all_records["teacher"] = self.evaluate(holdout, "teacher", teacher=True)
        for arm in ("correct", "full_only", "constant", "mismatched"):
            self.metrics["comparisons"][f"{arm}_vs_no_latent"] = paired_diagnostics(
                all_records["no_latent"], all_records[arm]
            )
        self.metrics["comparisons"]["correct_vs_full_only"] = paired_diagnostics(
            all_records["full_only"], all_records["correct"]
        )
        self.metrics["comparisons"]["correct_vs_constant"] = paired_diagnostics(
            all_records["constant"], all_records["correct"]
        )
        self.metrics["comparisons"]["correct_vs_mismatched"] = paired_diagnostics(
            all_records["mismatched"], all_records["correct"]
        )
        self.metrics["comparisons"]["test_mismatched_vs_correct"] = paired_diagnostics(
            all_records["correct"], all_records["correct_test_mismatched"]
        )
        groups = {}
        for t in (False, True):
            for s in (False, True):
                selected = [
                    i
                    for i, (a, b) in enumerate(
                        zip(all_records["teacher"], all_records["raw_student"], strict=True)
                    )
                    if a["correct"] == t and b["correct"] == s
                ]
                groups[f"teacher_{int(t)}_raw_student_{int(s)}"] = {
                    "n": len(selected),
                    "accuracy": {
                        name: sum(records[i]["correct"] for i in selected) / len(selected)
                        if selected
                        else None
                        for name, records in all_records.items()
                    },
                }
        self.metrics["initial_ability_groups"] = groups
        for arm in arms:
            self.restore(arm, common_paths[arm])
            for width in self.config["protocol"]["widths"]:
                if width == self.config["protocol"]["max_width"]:
                    continue
                name = f"{arm}_d{width}"
                records = self.evaluate(holdout, name, arm, width=width)
                self.metrics["comparisons"][f"{name}_vs_no_latent"] = paired_diagnostics(
                    all_records["no_latent"], records
                )
        self.metrics["status"] = "completed_exploratory"
        self.metrics["interpretation"] = "single_seed_small_holdout_not_confirmatory"

    def finish(self, error=None):
        if error:
            self.metrics["status"] = (
                "budget_exhausted" if isinstance(error, BudgetExpired) else "failed"
            )
            self.metrics["error"] = f"{type(error).__name__}: {error}"
        # Preserve useful paired diagnostics even if later evaluations hit the deadline.
        records = {}
        for path in (self.root / "predictions").glob("*.jsonl"):
            values = [json.loads(line) for line in path.read_text().splitlines()]
            records[path.stem] = values
            summary = summarize_records(values)
            summary["complete"] = len(values) == self.config["dataset"]["holdout_size"]
            self.metrics["arms"][path.stem] = summary
        for arm in ("correct", "full_only", "constant", "mismatched"):
            if (
                records.get("no_latent")
                and records.get(arm)
                and [r["id"] for r in records["no_latent"]] == [r["id"] for r in records[arm]]
            ):
                self.metrics["comparisons"][f"{arm}_vs_no_latent"] = paired_diagnostics(
                    records["no_latent"], records[arm]
                )
        for width in self.config["protocol"]["widths"]:
            suffix = "" if width == self.config["protocol"]["max_width"] else f"_d{width}"
            for baseline in ("full_only", "constant", "mismatched"):
                a, b = f"correct{suffix}", f"{baseline}{suffix}"
                if (
                    records.get(a)
                    and records.get(b)
                    and [r["id"] for r in records[a]] == [r["id"] for r in records[b]]
                ):
                    self.metrics["comparisons"][f"{a}_vs_{b}"] = paired_diagnostics(
                        records[b], records[a]
                    )
        self.metrics["budget_used_seconds"] = self.prior_seconds + time.monotonic() - self.start
        self.metrics["teacher_encoding_calls"] = self.teacher_calls
        events = [
            json.loads(line) for line in (self.root / "events.jsonl").read_text().splitlines()
        ]
        self.metrics["training"] = {
            arm: {
                "optimizer_steps_this_run": sum(
                    e["kind"] == "step" and e.get("phase") == "train" and e.get("arm") == arm
                    for e in events
                ),
                "measured_step_seconds": sum(
                    e["seconds"]
                    for e in events
                    if e["kind"] == "step" and e.get("phase") == "train" and e.get("arm") == arm
                ),
                "matched_at_step": self.metrics.get("common_step"),
            }
            for arm in self.config["training"]["arms"]
        }
        self.metrics["peak_allocated_bytes"] = [
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        ]
        self.metrics["payload"] = {
            "slots": self.config["protocol"]["num_slots"],
            "width": self.config["protocol"]["max_width"],
            "element_bytes": 4,
            "bytes_per_sample": self.config["protocol"]["num_slots"]
            * self.config["protocol"]["max_width"]
            * 4,
            "transport": "in_process_cuda_peer_copy",
            "wire_serialization": "not_used",
            "bytes_per_sample_by_width": {
                str(width): self.config["protocol"]["num_slots"] * width * 4
                for width in self.config["protocol"]["widths"]
            },
        }
        self.metrics["parameter_counts"] = (
            {
                "student_trainable": 0,
                "projector": sum(
                    p.numel() for n, p in self.params.items() if n.startswith("private_projectors.")
                ),
                "reader": sum(
                    p.numel() for n, p in self.params.items() if n.startswith("readers.")
                ),
                "constant": sum(
                    p.numel() for n, p in self.params.items() if n.startswith("controller.")
                ),
            }
            if hasattr(self, "params")
            else None
        )
        self.event("finished", status=self.metrics["status"])
        write_new_json(self.root / "metrics.json", self.metrics)
        write_new_json(
            self.root / "artifacts.json",
            {
                "server": self.config["server"]["host"],
                "root": str(self.root),
                "events": str(self.root / "events.jsonl"),
                "predictions": str(self.root / "predictions"),
                "latest_checkpoints": self.latest,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--config")
    parser.add_argument("--data")
    parser.add_argument("--run-dir")
    parser.add_argument("--run-id")
    parser.add_argument("--source-record")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    if args.stage == "summarize":
        if not args.run_dir:
            parser.error("summarize requires --run-dir")
        print((Path(args.run_dir) / "metrics.json").read_text())
        return
    if not args.config or not args.data:
        parser.error("prepare/run require --config and --data")
    config = json.loads(Path(args.config).read_text())
    validate_recipe(config)
    if args.stage == "prepare":
        prepare(config, args.data)
        return
    if not args.run_id or not args.source_record:
        parser.error("run requires --run-id and --source-record")
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run ID must contain only letters, digits, hyphens and underscores")
    data = json.loads(Path(args.data).read_text())
    if data["config"] != config:
        raise ValueError("prepared data config mismatch")
    output_root = Path(config["server"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_record = json.loads(Path(args.source_record).read_text())
        source_record["prepared_data_sha256"] = sha256_file(args.data)
        source_record["prepared_data_path"] = str(Path(args.data).resolve())
        run = Experiment(config, args.run_id, source_record, args.resume_from)
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
