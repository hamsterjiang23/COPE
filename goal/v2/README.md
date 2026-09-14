# v2：冻结教师与学生，只训练套娃接口

> **方法范围更正（用户本轮明确）：** 用户要求学生跳过原问题的 embedding/Prefill，直接使用套娃接口提供的状态进行 Decode。v2 实际把 latent 与问题 embeddings 拼接，仍执行学生 Prefill，因此本页结果仅适用于连续前缀注入变体，**不能验证用户要求的直接 Decode 方法**。原始 run 证据保留不变。

状态：2026-09-13 20:38 CST 实验完成，run `v2-20260913T100946Z-r01`，pueue 任务 `16` 成功退出。四组均完成 128 步，耗时 148.81 分钟；短前缀有局部正信号，但输入相关教师收益未获支持。详见 [结果](results.md)。

## 相对 v1 的变化

v1 更新了学生 LoRA，且只监督满宽；其结果属于该历史设置，不能用来判断“教师、学生均冻结，只训练中间套娃模块”的有效性。v1 原始配置、指标、检查点和源码快照全部保留。

v2 不安装或启用学生 LoRA，不优化学生或教师的任何参数。无 latent 就是原始冻结学生，仅评测，没有对应的训练步骤或优化器。训练中的学生前向保留输入梯度，让生成损失经冻结学生传回中间模块。

```text
问题与固定提示 → 冻结教师一次 Prefill → 最后一层 H [n,5120]
    → 固定序列汇聚为 8 个 slot
    → 可训练共享 MLP projector → Z [8,2048]
    → 截取 Z[:,:d] → 可训练、跨宽度共享的线性 reader
    → U_d [8,2048] → 冻结学生生成解题过程与答案
```

“最后一层 hidden”指输入序列的最后一层表示，不等同于只取最后一个 token，也不包含参考答案或教师生成的 CoT。当前保留一次 Prefill 与双方看完整问题的设定。中间模块输出适配学生输入 embedding 空间的连续表示，注入位置沿用前缀；尚未比较学生中间层注入，也未额外采用 hidden-state MSE 对齐损失。

## 多宽度目标与对照

执行前配方见 [config.json](config.json)。暂定宽度为 **256、512、1024、2048**，这是本版实施选择，不是用户指定的数值；固定 m=8、D=2048。每个 microbatch 只编码教师一次，生成同一个最大 Z，对四个 leading prefix 分别计算参考解答 NLL，等权平均后反向传播。

`L = (NLL_256 + NLL_512 + NLL_1024 + NLL_2048) / 4`

| 组别 | 监督宽度 | 可训练部分 |
|---|---|---|
| correct | 全部四档，教师编码当前问题 | projector、reader |
| full_only | 只监督 2048，评测时截取全部四档 | projector、reader |
| constant | 全部四档，共用一个零初始化后学习的最大 latent | 常量 latent、reader |
| mismatched | 全部四档，教师编码其他问题 | projector、reader |
| no_latent / raw_student | 仅评测，同一组结果 | 无 |

所有训练组从同一接口初始化开始、用相同样本顺序和 128 优化步；full_only 计算量小于多宽度组，报告实际计算成本，不声称等训练 FLOPs。每个宽度分别比较 correct 与 full_only、constant、mismatched、冻结无 latent。另对 correct 的最终模型做满宽推理时错配，单列教师独立生成。

## 暂沿用的工程配方

- Qwen3-14B → Qwen3-1.7B，骨干 BF16、接口 FP32，两张 A800 分别放教师/projector 与学生/reader。
- 沿用 v1 固定模型文件、GSM8K revision 和 seed=42，1,024 训练样本、128 留出题；**该留出集在 v1 已见，本版仅作探索性诊断，不称全新独立确认**。严格 `#### 数值` 评分的格式混杂仍须单列，不能把冻结学生的格式失败当作能力不足。
- 接口 AdamW 学习率 1e-3、无 weight decay，microbatch=2、累积=4、梯度裁剪 1；只在参考解答 token 上计算损失。
- 两批真实模型检查、16 条样本最多 32 步学习检查通过后，从初始权重重新开始正式训练；保存 32/64/96/128 步状态。
- 配置保留六小时计算上限；计时从本次 run 初始化开始，包含真实探测、训练和评测。失败时不能通过解冻教师或学生来让检查通过。

## 运行入口

新入口拒绝旧的 LoRA 配方。v1 的历史重现只能使用其 manifest 中记录的冻结源码快照，不能让新代码重新解释旧配置。

```sh
python -m cope.real_experiment prepare --config goal/v2/config.json --data data/v2-prepared.json
python -m cope.real_experiment run --config goal/v2/config.json --data data/v2-prepared.json --run-id <unique-run-id> --source-record <source-record.json>
```

真实运行前另建源码快照和独立 run ID，确认 GPU/磁盘资源；本次独立 pueue 任务已完成，完成跟踪随归档关闭。真实结果记入 [results.md](results.md)，过程追加 [journal.md](journal.md)。
