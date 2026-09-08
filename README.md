# COPE

**A Composable Ordered-Prefix Latent Protocol for Cross-Model Reasoning**

一个教师执行一次输入 Prefill，经共享 reducer/projector 形成最大 latent；不同宽度的有序前缀由学生专用 reader 映射后，供多个同家族、不同规模的学生生成答案。`[verified: 项目发起者给定 COPE 定义；src/cope/system.py L26-L157]`

本研究受 [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) 启发，核心待验证命题是同一坐标顺序能否同时服务不同学生与通信预算。`[opinion]`

## Current Status

当前仓库已从研究说明扩展为可组装的 PyTorch 研究框架；尚未完成真实模型训练、benchmark 评测或论文结论验证。`[verified: src/cope source tree；仓库中无训练结果目录]`

已经实现：

- 一个 teacher encoding 复用于多个 student-width loss 分支。`[verified: src/cope/system.py L94-L157]`
- shared/private projector，以及最大 latent 的严格 leading-prefix 截取。`[verified: src/cope/system.py L70-L92；src/cope/projectors.py L8-L42]`
- MRL-E 风格共享截断 reader 和 MRL 风格独立宽度 reader。`[verified: src/cope/readers.py L31-L76]`
- 全宽度精确加权目标、full-only 目标和无偏 sampled-width 近似。`[verified: src/cope/objectives.py L34-L89]`
- no-latent、zero、learned-constant 和跨样本 mismatched controls。`[verified: src/cope/controls.py L9-L46]`
- Hugging Face-compatible teacher hidden-state 与 causal student prefix-embedding adapters。`[verified: src/cope/adapters.py L34-L165]`
- 四格核心实验及 control manifests、纠错/伤害指标和逐学生 Pareto 统计。`[verified: configs/arms/*.json L1-L6；src/cope/evaluation.py L10-L109]`

## Upstream Sources

初始化 submodule 后可在本仓库直接查看三个固定版本。`[verified: .gitmodules]`

```powershell
git clone --recurse-submodules git@github.com:hamsterjiang23/COPE.git
git submodule update --init --recursive
```

| Repository | Role | Reuse boundary |
|---|---|---|
| [C2C](https://github.com/thu-nics/C2C) | 工程与 shared full-width baseline | 复用模型加载、冻结、训练、数据和评测语义；其逐层 KV-cache fusion 不作为 COPE 核心。`[verified: THIRD_PARTY.md L5-L12]` |
| [MRL](https://github.com/RAIVNLab/MRL) | ordered-prefix 方法规范 | 复用 nesting、加权全宽度 loss、独立 head 和 MRL-E 截断 head 语义；不复用 ImageNet/FFCV pipeline。`[verified: THIRD_PARTY.md L14-L21]` |
| [LLM-to-SLM reproduction](https://github.com/tosiyuki/LLM-to-SLM) | private full-width baseline | 复用 projected hidden-state embedding injection 语义；T5/GPT-2 专用实现不作为主框架。`[verified: THIRD_PARTY.md L23-L31]` |

完整许可证、commit pin 和代码归属边界见 [THIRD_PARTY.md](THIRD_PARTY.md)。`[verified: THIRD_PARTY.md L1-L43]`

## Layout

```text
src/cope/                 COPE-native protocol implementation
configs/arms/             decisive factor arms and causal controls
examples/toy_train.py     deterministic tensor-level usage example
tests/                    authored contract tests
third_party/              pinned upstream Git submodules
docs/implementation-plan.md
references/task-costs.md
scripts/check_upstreams.py
```

## Setup

基础安装需要 PyTorch；连接真实 Hugging Face 模型时安装 `hf` extra。`[verified: pyproject.toml L5-L24]`

```powershell
uv sync --extra hf
```

上游 commit 可在不加载 PyTorch 的情况下校验。`[verified: scripts/check_upstreams.py L1-L46]`

```powershell
python scripts/check_upstreams.py
```

核心组装入口是 `build_huggingface_system`；调用方提供已加载的 teacher/student models，因此框架不会隐式下载权重。`[verified: src/cope/factory.py L43-L115]`

```python
from cope import ProtocolConfig, build_huggingface_system

config = ProtocolConfig(
    teacher_hidden_size=3584,
    max_width=1024,
    widths=(128, 256, 512, 1024),
    num_slots=8,
)

system = build_huggingface_system(
    config,
    teacher_model,
    {"small": small_student, "medium": medium_student},
    projector_sharing="shared",
    reader_kind="truncated_shared",
)
```

分支由 `BranchPlanner` 显式决定；`ALL_WIDTHS` 对应 faithful MRL-style objective，`SAMPLE_WIDTH` 是带逆概率校正的计算近似。`[verified: src/cope/objectives.py L34-L74]`

## Not Completed

- 尚未选择和下载实际 teacher/student checkpoints。`[verified: 仓库中无 checkpoint 配置或权重]`
- 尚未把 C2C 原始方法接入统一 COPE evaluator；C2C 当前作为固定 submodule baseline。`[verified: third_party/C2C；src/cope 中无 C2C adapter]`
- 尚未实现 KV-cache reader、LoRA/PEFT 专用 adapter、跨设备模型放置和真实分布式训练 recipe。`[verified: src/cope 当前接口范围]`
- 尚未接入 GSM8K、知识问答和工具调用 benchmark，也未生成 quality-width-cost 实验结果。`[verified: 仓库中无 benchmark recipe 或结果文件]`
- 21 个 contract tests 已写入；遵照用户要求，本轮未执行全套 PyTorch 测试。`[verified: tests 中 test function 统计；用户最新指令]`

## Next Work

1. 固定首组同家族 teacher/student checkpoints 和 student 可见输入协议。`[opinion]`
2. 接入一个小型数学数据 slice，先复现 private full-width 正对照。`[opinion]`
3. 运行 `private/shared × full-only/multi-width` 四格核心实验。`[opinion]`
4. 加入 constant/mismatched controls，随后扩展知识问答、工具调用与端到端成本测量。`[opinion]`

研究假设、全部结果分支、实验门控和表述边界见 [AGENTS.md](AGENTS.md)。`[verified: AGENTS.md L1-L356]`
