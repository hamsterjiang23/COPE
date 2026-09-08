# Task Costs and I/O Dependencies

Measured resource costs for COPE development tasks. `[verified: preflight-check workflow]`

## Cost Lookup Table

| Task | Category | Runtime | Memory | External resources | Notes |
|---|---|---|---|---|---|
| Ruff and compile checks | Light | <1 min | <1 GB | None | Reads source and writes ignored caches. `[verified: 2026-09-08 run]` |
| Unit tests with toy tensors | Light | ~6 sec for module slice | <2 GB | CPU | No model or dataset downloads. `[verified: 2026-09-08 probe]` |
| Real-model COPE training | Unknown | Not measured | Not measured | GPU, model weights, datasets | Run a one-batch probe before classification. `[opinion]` |

## I/O Dependency Table

| Task | Reads | Writes |
|---|---|---|
| Ruff and compile checks | `src/`, `tests/`, `examples/` | Ignored cache directories |
| Unit tests with toy tensors | `src/`, `tests/` | `.pytest_cache/`, coverage/cache files |
| Real-model COPE training | Model weights, datasets, recipes | `outputs/`, `checkpoints/`, tracking backend |

`outputs/`, `checkpoints/`, and `wandb/` are ignored and must not be shared by concurrent training jobs unless each job has a unique output directory. `[verified: .gitignore]`
