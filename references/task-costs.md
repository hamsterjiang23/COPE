# Task Costs and I/O Dependencies

Measured resource costs for COPE development tasks. `[verified: preflight-check workflow]`

## Cost Lookup Table

| Task | Category | Runtime | Memory | External resources | Notes |
|---|---|---|---|---|---|
| Ruff and compile checks | Light | <1 min | <1 GB | None | Reads source and writes ignored caches. `[verified: 2026-09-08 run]` |
| Unit tests with toy tensors | Light | ~6 sec for module slice | <2 GB | CPU | No model or dataset downloads. `[verified: 2026-09-08 probe]` |
| v2 frozen-model training and evaluation | Heavy | 148.81 min | GPU allocated peaks 28.392 / 17.458 GiB | Two A800 80GB, pinned models and data | Four arms × 128 steps; 19 evaluations × 128 questions. See v2 evidence. |

## I/O Dependency Table

| Task | Reads | Writes |
|---|---|---|
| Ruff and compile checks | `src/`, `tests/`, `examples/` | Ignored cache directories |
| Unit tests with toy tensors | `src/`, `tests/` | `.pytest_cache/`, coverage/cache files |
| Real-model COPE training | Model weights, datasets, recipes | `outputs/`, `checkpoints/`, tracking backend |

`outputs/`, `checkpoints/`, and `wandb/` are ignored and must not be shared by concurrent training jobs unless each job has a unique output directory. `[verified: .gitignore]`

## v1 real-model experiment

- Cost classification: Heavy (14B BF16 frozen teacher plus 1.7B student LoRA); actual GPU peak and runtime must come from the two-batch probe.
- Reads pinned model files, `goal/v1/config.json`, and prepared disjoint GSM8K splits. Writes only `outputs/v1/<run-id>/` and the version evidence index.
- GPU 0: teacher/projector; GPU 1: student/reader. Only one COPE run at a time using an execution lock and dedicated pueue group. Recheck external GPU occupancy before loading.
- Training/evaluation budget: six hours including probes; installs/downloads excluded. Reserve 90 minutes for evaluation and compare only common milestones.

## v2 frozen-model multi-width experiment

- Heavy: same frozen 14B teacher and 1.7B student, four student-width loss graphs per multi-width batch. Expect higher student activation memory than v1; start with the built-in two-batch probe and verify actual peaks.
- Reference v1 whole-run peaks: 28.392 GiB on GPU 0 and 8.367 GiB on GPU 1. v2 conservatively reserves up to 40 GiB on GPU 1 before probing. Estimated end-to-end 2–4 hours for all width evaluations; hard limit six hours.
- Reads the same immutable model files and GSM8K source, plus `goal/v2/config.json`; writes only `data/v2-prepared.json`, `outputs/v2/<run-id>/` and v2 evidence. Dedicated `cope-v2` pueue group, one job. Never write v1 outputs.
- Checkpoint storage estimated below 8 GiB; 2026-09-13 preflight found both GPUs idle and 22 GiB free on the data volume. Recheck before launch. No model downloads are needed.

### v2 observed cost (completed 2026-09-13)

- Run `v2-20260913T100946Z-r01`: 8,928.52 seconds for loading, probes, training, checkpoints and evaluation. All four arms reached 128 steps; 19 prediction files each contain 128 questions. Teacher encoding calls: 3,346, including separate NLL and generation calls.
- Whole-process peak allocated bytes: GPU 0 = 30,485,697,024 (28.392 GiB); GPU 1 = 18,745,912,320 (17.458 GiB). These are shared-process high-water marks, not independently measured arm peaks.
- Training-step totals: correct 214.48 s, full-only 77.10 s, constant 183.59 s, mismatched 214.87 s; excludes checkpoint/restore overhead. Multi-width supervision has greater compute than full-only despite matched optimizer steps and sample exposure.
- Transport audit: actual copied tensor is full FP32 `[batch,8,2048]` before reader slicing, hence 65,536 bytes per sample at every width. Width-dependent payload in metrics is logical prefix size only. No measured communication saving or equal-quality speedup claim.
- Evidence: `goal/v2/runs/v2-20260913T100946Z-r01/{metrics,integrity-audit,transport-audit}.json`; full cost interpretation in `goal/v2/results.md`.

## v3 interface screening

Heavy; frozen 14B/1.7B on two A800s, single-example microbatches, six sequential candidates and at most two extra control trainings. Reads only pinned model/data assets and v2 sample exclusion IDs; writes data/v3-prepared.json and outputs/v3/<run-id>. Estimated under 6 hours with 2-hour evaluation reserve; actual probe required. 32×512 FP32 latent crosses devices (64 KiB/sample); KV expansion happens on student device. New checkpoint budget expected below 4 GiB; preflight available disk 16 GiB. Separate cope-v3 group plus global GPU execution lock.

### v3 observed cost (completed 2026-09-14)

- Cumulative budget: 10,576.90 seconds (176.28 minutes), including failed startup r01 and 60 seconds charged for diagnostics. Eight formal trainings × 128 steps, 20 complete evaluation files.
- Peak allocated: GPU 0 = 30,474,784,256 bytes; GPU 1 = 10,385,927,680 bytes. Includes independent FP32 student cache validation copy, not per-arm peaks.
- Interface active parameters: embedding 3,946,496; KV 5,247,872; embedding constant 1,067,008. Equal exposure and optimizer steps, not equal FLOPs.
- Correct holdout mean online latency 2,889.60 ms with 110.11 generated tokens; student 5,503.83 ms with 215.77 tokens. Different output lengths, quality and parse rates preclude equal-quality speedup claims. Each assisted path copies 65,536 bytes of FP32 latent; expansion happens on student GPU.
- Full phase timings, parameter counts and checksum audit: `goal/v3/results.md` and `goal/v3/runs/v3-20260913T140917Z-r02/integrity-audit.json`.
