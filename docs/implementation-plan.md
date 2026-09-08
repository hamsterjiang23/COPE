# COPE Implementation Plan

## Summary

Build a testable research framework for one teacher, a reusable maximum-width latent, nested width prefixes, and multiple student readers. `[opinion]`

Use C2C as the engineering reference, MRL as the ordered-prefix objective reference, and LLM-to-SLM as the embedding-injection baseline reference. `[verified: upstream source inspections recorded in AGENTS.md]`

The riskiest assumption is whether a student-independent latent ordering survives joint optimization across students; framework tests can validate contracts but only GPU experiments can validate that research claim. `[opinion]`

## Decisions Most Likely To Change

### Upstream repositories

- Choice: pin C2C, MRL, and the LLM-to-SLM reproduction as Git submodules under `third_party/`. `[opinion]`
- Alternative: vendor their full source trees into this repository. `[opinion]`
- Change cost: low before downstream patches, high after local modifications diverge from upstream. `[opinion]`

### Core protocol boundary

- Choice: implement COPE in a separate `src/cope/` package and leave submodules unmodified. `[opinion]`
- Alternative: fork and modify C2C's `RosettaModel` directly. `[opinion]`
- Change cost: high because C2C's wrapper is organized around per-layer KV-cache fusion rather than a student-independent maximum latent. `[verified: C2C rosetta/model/wrapper.py and projector.py]`

### First student injection path

- Choice: prefix embeddings through a Hugging Face-compatible adapter; keep the reader interface open for later KV injection. `[opinion]`
- Alternative: make KV-cache injection the only path. `[opinion]`
- Change cost: moderate because reducers, projectors, and objectives remain unchanged while only readers/adapters differ. `[opinion]`

### Multi-width training

- Choice: exact all-width loss is the default faithful MRL mode; sampled-width training is an explicit approximation. `[verified: MRL MRL.py L18-L30]`
- Alternative: sample one width from the first implementation. `[opinion]`
- Change cost: low in code but high for experimental interpretation if the two modes are not distinguished. `[opinion]`

## Known Unknowns

- Real teacher and student model families are not fixed; adapters use Hugging Face-compatible duck typing and tests use deterministic fake models. `[opinion]`
- Sequence reduction is unsettled; the framework provides masked mean and learned-query reducers behind one contract. `[opinion]`
- The first implementation injects prefix embeddings; KV readers remain a named extension point and C2C remains an external baseline. `[opinion]`
- No benchmark data or checkpoints are downloaded in the framework build; real-task recipes remain subsequent experimental work. `[opinion]`

## Mechanical Work

1. Add pinned upstream submodules and document license/reuse boundaries.
2. Add package metadata and core tensor/config contracts.
3. Implement reducers, shared/private ordered projectors, MRL/MRL-E-style readers, and Hugging Face adapters.
4. Implement one-prefill multi-student/multi-width loss aggregation, branch planning, controls, and optimizer step.
5. Add four factor-arm configs, causal-control manifests, and a deterministic toy smoke example.
6. Add unit tests for nesting, sharing, controls, loss aggregation, one-prefill reuse, freezing, and parameter updates.
7. Update `AGENTS.md` and `README.md` with completed, incomplete, and next work.
8. Run formatting and static checks; keep runtime PyTorch tests authored but deferred per user instruction, then commit and push.

## Acceptance Criteria

- Three upstream repositories are pinned and their reuse boundaries are explicit.
- A single teacher encoding is reused across every requested student-width branch.
- Shared and private projector modes and full-only versus multi-width branch plans are executable.
- Nested prefixes are exact slices of one maximum latent.
- MRL-style weighted all-width generation loss is covered by tests.
- Teacher freezing and trainable protocol parameters are covered by tests.
- Documentation distinguishes framework completion from unrun model experiments.
