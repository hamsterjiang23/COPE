# Third-Party Repositories

The repositories below are pinned as Git submodules. Their source is not COPE source code. `[verified: .gitmodules L1-L9]`

## C2C

- Path: `third_party/C2C`. `[verified: .gitmodules L1-L3]`
- Upstream: <https://github.com/thu-nics/C2C>. `[verified: .gitmodules L1-L3]`
- Pin: `113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a`. `[verified: git submodule status]`
- License: Apache-2.0. `[verified: third_party/C2C/LICENSE L1-L3]`
- Reuse boundary: model loading, freezing, configuration, datasets, distributed training, evaluation, and the external shared-full-width baseline are reusable references. `[verified: C2C source tree]`
- Excluded from COPE core: `RosettaModel` and `C2CProjector` perform layer-wise KV-cache projection and fusion into one fixed base model; they do not provide a student-independent ordered maximum latent. `[verified: C2C rosetta/model/wrapper.py and projector.py]`

## Matryoshka Representation Learning

- Path: `third_party/MRL`. `[verified: .gitmodules L4-L6]`
- Upstream: <https://github.com/RAIVNLab/MRL>. `[verified: .gitmodules L4-L6]`
- Pin: `7ccb42df6be05f3d21d0648aa03099bba46386bf`. `[verified: git submodule status]`
- License: MIT. `[verified: third_party/MRL/LICENSE L1-L3]`
- Reuse boundary: nested width lists, weighted all-width loss, independent width heads, MRL-E truncated shared heads, and their tests define the ordered-prefix semantics reused by COPE. `[verified: MRL MRL.py and tests/test_MRL.py]`
- Excluded from COPE core: the ResNet50, ImageNet, FFCV, classification, and retrieval pipeline is domain-specific and is not imported by COPE. `[verified: MRL README]`

## LLM-to-SLM Reproduction

- Path: `third_party/LLM-to-SLM`. `[verified: .gitmodules L7-L9]`
- Upstream: <https://github.com/tosiyuki/LLM-to-SLM>. `[verified: .gitmodules L7-L9]`
- Pin: `5ef4eda6e0a078bbec26bc6eceef0c294b5c7a92`. `[verified: git submodule status]`
- License: Apache-2.0. `[verified: third_party/LLM-to-SLM/LICENSE L1-L3]`
- Provenance: this repository describes itself as a reproduced implementation, not the paper authors' official release. `[verified: LLM-to-SLM README]`
- Reuse boundary: projected teacher hidden states added to student embeddings define one private-full-width baseline and motivate the prefix-embedding student adapter. `[verified: llm_to_slm/model/llm_to_slm_arch.py]`
- Excluded from COPE core: its T5 encoder, GPT-2 specialization, Alpaca-only recipe, and token-aligned addition do not support the multi-student ordered protocol. `[verified: LLM-to-SLM source tree]`

## COPE-Native Functionality

The following functionality is implemented in `src/cope/` and is not supplied by any one upstream repository. `[verified: src/cope source tree]`

- One teacher prefill reused across multiple student-width losses. `[verified: src/cope/system.py L94-L157]`
- A student-independent maximum latent and exact leading-dimension prefixes. `[verified: src/cope/projectors.py L8-L42]`
- Shared and student-private projector modes. `[verified: src/cope/system.py L26-L92]`
- MRL-style independent readers and MRL-E-style truncated shared readers for autoregressive students. `[verified: src/cope/readers.py L31-L76]`
- Exact all-width objectives and unbiased sampled-width approximations. `[verified: src/cope/objectives.py L34-L89]`
- No-latent, zero, learned-constant, and cross-sample mismatched controls. `[verified: src/cope/controls.py L9-L46]`
- Prefix-embedding adapters for Hugging Face-compatible causal language models. `[verified: src/cope/adapters.py L71-L165]`
