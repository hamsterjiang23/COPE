# COPE

> **当前方向：** 优先寻找冻结大小模型间能带来有效信息增益的接口。v3 比较保留学生问题 Prefill 与直接 Decode 的候选；v2 仅验证连续前缀注入，不能作为直接 Decode 路线的证据。

**A Composable Ordered-Prefix Latent Protocol for Cross-Model Reasoning**

一个教师执行一次输入 Prefill，经共享 reducer/projector 形成最大 latent；不同宽度的有序前缀由学生专用 reader 映射后，供多个同家族、不同规模的学生生成答案。`[verified: 项目发起者给定 COPE 定义；src/cope/system.py L26-L157]`

本研究受 [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) 启发，核心待验证命题是同一坐标顺序能否同时服务不同学生与通信预算。`[opinion]`

**阅读入口：[当前目标、全部代码改动与后续路线](docs/project-status-and-roadmap.md) · [版本索引](goal/README.md) · [V3 最终结果](goal/v3/results.md)**

## Experiment versions

所有实验按 `goal/v1`、`goal/v2`、`goal/v3` 迭代，配置、过程和结果均归档于对应版本。见 [版本索引](goal/README.md)。v2 入口为 `python -m cope.real_experiment`，v3 入口为 `python -m cope.search_experiment`；历史 v1 须使用该 run 冻结源码，当前执行状态以版本索引为准。

## Current Status

当前固定教师与学生，只训练中间接口。[v3](goal/v3/README.md) 已完成六种接口/教师层位组合的筛选及独立确认：正确组 55.86%，训练错配 59.38%，推理错配 56.25%，尚未确认输入相关信息增益，详见 [v3 结果](goal/v3/results.md)。[v2](goal/v2/README.md) 实现该训练边界和多前缀监督，真实模型实验已完成：短前缀有局部正信号，但正确 latent 未可靠优于常量或错配，详见 [v2 结果](goal/v2/results.md)。已完成的 [v1](goal/v1/results.md) 使用学生 LoRA 且只训练满宽，偏离当前目标，保留历史证据但不作为双冻结套娃方法的验证。

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
goal/v1/, v2/, v3/        versioned recipes and immutable run evidence
docs/project-status-and-roadmap.md  current goals, changes, findings and next steps
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

## Framework scaffold snapshot (2026-09-08)

以下为框架初建时状态；后续实现与实验进度以 `goal/` 版本记录为准。

### Not Completed

- 尚未选择和下载实际 teacher/student checkpoints。`[verified: 仓库中无 checkpoint 配置或权重]`
- 尚未把 C2C 原始方法接入统一 COPE evaluator；C2C 当前作为固定 submodule baseline。`[verified: third_party/C2C；src/cope 中无 C2C adapter]`
- 尚未实现 KV-cache reader、LoRA/PEFT 专用 adapter、跨设备模型放置和真实分布式训练 recipe。`[verified: src/cope 当前接口范围]`
- 尚未接入 GSM8K、知识问答和工具调用 benchmark，也未生成 quality-width-cost 实验结果。`[verified: 仓库中无 benchmark recipe 或结果文件]`
- 21 个 contract tests 已写入；遵照用户要求，本轮未执行全套 PyTorch 测试。`[verified: tests 中 test function 统计；用户最新指令]`

## Next Work

已执行配方见 [v2](goal/v2/README.md)：教师与学生均冻结，四档前缀联合监督，仅更新中间模块，比较 full-only truncate、常量、错配与冻结学生基线。真实模型实验已完成。后续优先解决评分格式混杂、满宽退化及先复制满宽再截取的通信问题，见 [v2 结果与建议](goal/v2/results.md)；尚未启动 v3。

研究假设、全部结果分支、实验门控和表述边界见 [AGENTS.md](AGENTS.md)。`[verified: AGENTS.md L1-L356]`

## v3：有效信息传递优先

用户最新确认的主目标是冻结大小模型间的有效信息传递；学生无问题 Prefill 是候选分支。已注册 [v3](goal/v3/README.md)，筛选连续前缀、附加 KV、替代问题 KV 与教师中间/末层的 6 个组合，开发集选择后独立留出确认。当前准备状态以版本索引为准。
