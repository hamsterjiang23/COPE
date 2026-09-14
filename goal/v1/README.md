# v1：单教师、单学生满宽可行性

> 2026-09-13 用户纠正：目标方法要求教师与学生均完全冻结，仅训练中间套娃模块。本版实际训练了学生 LoRA，且只监督满宽，偏离该设定；以下结果仅描述历史 v1，不能作为双冻结套娃方法的验证。原始配置、指标和源码保留，修正版本见 [v2](../v2/README.md)。

2026-09-13：四组均已完成 128 步，全部评测完成，累计 54.68 分钟。实现检查通过，但输入相关教师收益尚无充分证据，研究结论为不确定。见 [结果与限制](results.md)；v2 未启动。

## 问题与范围

Qwen3-14B 的一次问题 Prefill 表示，经 8 个 mean-pooling slot、2048 维 projector 和学生 reader，是否能改善 Qwen3-1.7B 的 GSM8K 回答？本版是实现检查和探索性实证，不是 MRL 前缀监督或跨学生共享的验证，也不宣称复现了某篇论文。

这是首个实验版本，无前序版本。结果允许有效、无收益、不确定或未完成。

## 固定配方

以 [config.json](config.json) 为准。教师冻结，BF16；学生冻结骨干，在 q_proj/v_proj 上训练 r=16、alpha=32、dropout=0 的 LoRA。接口参数 FP32。教师 hidden size=5120，学生 hidden size=2048，m=8，D=d=2048。双方看相同问题，教师编码不含参考答案。

官方 Qwen3-14B revision 与 GSM8K revision 固定于配置及 [downloads.json](downloads.json)，下载时核验官方文件尺寸及 SHA256/Git blob hash。现有学生权重运行前登记 SHA256。

GSM8K main/train 按规范化问题去重，seed=42，先抽取 128 条独立留出，再从其余样本选择 1024 条符合长度约束的训练样本。参考答案长度仅用于训练样本过滤；不据此筛选留出集。官方 test 不参与本版。

四组独立训练：correct、no_latent、learned constant、mismatched。相同初始化、样本顺序、LoRA、microbatch=2、累积=4、最多 128 步。32 步为共同里程碑。错配 donor 在同一数据 split 内产生且永不自配。smoke 完成后正式各组重新初始化。

原始学生、教师独立生成及 correct 模型推理时错配作为诊断。所有输出使用相同固定提示、非 thinking 模式、贪心生成、512 新 token 上限；严格解析最终 `#### 数值`，解析失败计错，单列截断率。

## 执行顺序

1. 校验资产、检查 GPU/磁盘及其它任务，准备独立环境和固定数据划分。
2. 小型功能测试；两批真实模型前后向；16 条训练样本最多 32 步学习检查。
3. 各组依次完成共同的 32/64/96/128 步里程碑，保存接口、LoRA、优化器和随机状态。
4. 对共同完成的最大里程碑评测并汇总。训练/评测累计最多六小时，默认至少预留 90 分钟评测；不保证预算内完成全部评测。
5. 完成后更新 [results.md](results.md) 和 [journal.md](journal.md)。本版不会自动扩大数据、添加种子或进入 v2。

## 命令入口

以下为历史命令，仅用于 manifest 记录的 v1 冻结源码快照；当前入口已拒绝 LoRA 配方。历史运行使用独立环境：

```sh
python -m cope.real_experiment prepare --config goal/v1/config.json --data data/v1-prepared.json
python -m cope.real_experiment run --config goal/v1/config.json --data data/v1-prepared.json --run-id <unique-run-id> --source-record outputs/source-record.json
python -m cope.real_experiment summarize --run-dir outputs/v1/<run-id>
```

恢复必须传入新的 run ID 和 `--resume-from outputs/v1/<prior-run-id>`。只恢复所有组共同完成的里程碑，延续累计预算；配方不允许改变。原 run 保留。

## 指标解释

报告准确率、目标 token 加权 NLL、输出长度、解析/截断率、配对 bootstrap 区间及初始能力四分组。纠错、伤害和净收益以匹配训练 no-latent 组为基线，原始学生分组仅为诊断。零分母比率记为 null。

在线延迟包括新教师 Prefill、汇聚/投影、CUDA 设备复制、reader 和学生生成；教师独立生成另列，NLL 前向不计入在线生成延迟。此版为单机双 GPU，不模拟网络序列化；FP32 紧密 payload 为 8×2048×4=65,536 bytes/题，不将其等同于端到端加速。
