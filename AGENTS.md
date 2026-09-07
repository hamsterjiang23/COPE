# COPE 研究约定

## 标题与核心命题

**COPE: A Composable Ordered-Prefix Latent Protocol for Cross-Model Reasoning**

**用于跨模型推理的可组合有序前缀潜表征协议**。`[verified: 用户给定标题]`

**一个教师、一个共享投影器，通过不同宽度的有序 latent 前缀，适配多个同家族、不同规模的学生。** 教师在每次请求中仅执行一次输入 Prefill，学生负责自回归生成推理过程或答案。`[verified: 用户给定 COPE 核心定义]`

本文固定研究方向与创新假设；性能、泛化与成本收益均须通过实验验证。`[opinion]`

## 方法来源：从 MRL 到跨模型生成接口

用户明确指定 **Matryoshka Representation Learning（MRL）** 为本 idea 的来源。参考笔记实际路径为 `E:\Obsidian_Note\Papers\深度学习理论\Matryoshka Representation Learning.md`。`[verified: 用户对 idea 来源的说明；该文件读取结果 L75–97]`

MRL 对同一 embedding 的多个嵌套前缀分别施加任务监督，使一次 encoder 输出支持多档表示容量；MRL-E 进一步共享不同宽度的读取权重。参见 [MRL 原文 §3](https://arxiv.org/html/2205.13147v4#S3)。`[verified: MRL §3 式1及 MRL-E 段落]`

COPE 将上述思想扩展到跨模型生成：**同一教师表示的不同前缀，不仅对应不同通信预算，还要能被不同能力的学生读取。** 研究新增的变量是接收模型，目标是学习跨学生有效的统一信息排序。`[opinion]`

## 核心创新假设

| 研究维度 | COPE 要验证的命题 |
|---|---|
| 有序前缀生成接口 | 将 MRL 的多前缀监督用于学生自回归生成，使每档 latent 宽度都接受下游任务约束；验证短前缀可用、长前缀能补充有效信息。`[opinion]` |
| 跨学生共享信息排序 | 在“学生 × 宽度”上联合训练同一最大 projector；验证同一坐标顺序可被不同学生利用，并量化相对学生专用接口的损失。`[opinion]` |
| 能力与预算匹配规律 | 测量不同学生达到给定质量所需的最小宽度，以及增加宽度的边际收益；据此识别通信受限与学生能力受限的部署区域。`[opinion]` |

本项目将 **Composable** 操作性定义为：教师侧编码不依赖学生身份或所选宽度，同一 latent 可截取后与不同学生读取器组合使用。冻结协议后接入新学生是更强的扩展证据，不是首轮验证的前提。`[opinion]`

## 方法与分析的贡献重心

当前建议按 **Method 约 40%、分析与实证约 60%** 组织研究；这是预期贡献重心的主观判断，不是已验证结论、论文篇幅配额或必须保持的比例。`[opinion]`

方法贡献集中于共享 projector、有序前缀及“学生 × 宽度”联合训练。分析贡献需回答共享排序的代价、信息容量与学生可读性的区别，以及目标质量下的学生与预算选择；普通准确率表和超参数扫描主要验证方法，不能自动视为独立的分析贡献。`[opinion]`

质量曲线饱和时，结合教师信号、读取器容量与学生专用接口对照再归因，不能仅由曲线变平就断言学生能力不足。先验证这些问题；若发现稳定的共享冲突，再设计针对性机制，无需为了增加方法占比预先堆叠模块与损失。`[opinion]`

## 最小接口与训练约束

建议以以下接口组织实现；序列汇聚与接入位置属于待比较的设计选择。`[opinion]`

```text
输入 x → 教师一次 Prefill → H [n, h_T]
       → 共享序列汇聚 R → 共享最大宽度 projector P
       → Z [m, D] → 截取 Z[:, :d]
       → 学生读取器 A_s → 学生 S_s → 推理过程 / 答案
```

`n` 为输入位置数，`m` 为 latent slot 数，`D` 为最大通信宽度，`d` 为本次选择的前缀宽度，`h_T` 和 `h_s` 为教师与学生的 hidden size。必须分别记录这些量；`d` 不要求等于 `h_s`。`[opinion]`

教师侧汇聚器与 projector 在学生之间共享。每个学生可拥有自己的轻量读取器，但同一学生的读取器应跨宽度共享参数。线性读取基线可写为 `U_s = Z[:, :d] @ W_s[:d, :] + b_s`，输出形状为 `[m, h_s]`。`[opinion]`

训练从多学生、多宽度的生成损失开始：`L = Σ_s Σ_d λ[s,d] · NLL(S_s(y | x_s, A_s(Z[:, :d])))`。所有分支更新同一教师侧接口；学生采样、宽度采样与损失权重必须记录。教师冻结、学生骨干冻结或使用 LoRA，作为明确的实验配置比较。`[opinion]`

训练与评估遵守以下约束：

- **输入可用性：** 生成部署 latent 的教师输入仅含部署时可得的问题与上下文；参考答案和 CoT 可作监督目标，不能混入该编码输入。`[opinion]`
- **信息可见性：** 明确学生输入 `x_s`。学生仍看完整上下文时评估附加指导；学生只看问题与 latent 时，才评估被移除上下文的替代效果。`[opinion]`
- **一次编码复用：** 对相同输入先形成同一个最大表示，再提供各前缀；不随学生身份重算另一份表示。不同输入或变化后的上下文不属于同一次编码复用。`[opinion]`
- **前缀监督：** 只训练最大宽度再截断是对照组，不能代替 MRL 式多宽度训练。`[opinion]`

## 首轮证据与对照

优先顺序是单学生有效信号、多宽度前缀、跨学生共享，最后扩展新学生接入；先用小规模训练与独立留出集验证链路，再展开完整矩阵。`[opinion]`

| 对照 / 实验 | 要隔离的问题 |
|---|---|
| 无 latent 的匹配训练基线；常量与跨样本错配 latent | 收益是否来自输入相关的教师信号，而非普通微调或额外参数。`[opinion]` |
| 共享 projector 仅训练满宽，再截断 | 多前缀联合监督是否必要。`[opinion]` |
| 每个学生独立训练自己的多宽度 projector | 跨学生共享排序的性能代价。`[opinion]` |
| 每个宽度独立训练一套跨学生接口 | 跨宽度参数共享的性能代价。`[opinion]` |
| COPE：同一 projector 联合训练学生与宽度 | 能否兼顾前缀可用性、跨学生质量与接口维护成本。`[opinion]` |
| 留出学生与中间宽度的组合 | 已学到的表示能否支持未直接监督的组合；该学生需有其他宽度训练覆盖。`[opinion]` |
| 冻结协议，仅训练新学生读取器 | 扩展实验：轻量适配后的接入质量与样本效率。`[opinion]` |

各组固定教师信号、数据划分、学生可见输入与适配配方，分别报告训练数据曝光和实际训练成本。主结果展示每个学生的完整质量—宽度曲线，不能只报告跨学生平均分；关键差异需有重复实验与不确定性估计。`[opinion]`

## 相较已有论文的核心差异

**贡献概括：COPE 将 MRL 的嵌套前缀学习扩展为跨模型生成协议，通过学生与宽度的联合监督，研究一套教师 latent 的信息排序能否在不同能力的学生之间复用，并建立质量、通信预算与学生规模之间的实证匹配关系。** 前两项是方法设计，匹配规律是待实验支持的分析贡献。`[opinion]`

| 对比论文 | 已有工作的着力点 | COPE 的研究增量 |
|---|---|---|
| [MRL](https://arxiv.org/html/2205.13147v4#S3) | 对同一表示的嵌套维度施加任务监督，支持多档表示容量。`[verified: §3 式1]` | 把前缀读取者扩展为多个不同规模的自回归学生，学习跨学生有效的排序。`[opinion]` |
| [LLM-to-SLM](https://arxiv.org/html/2402.16844v2#S3) | 大模型一次编码；projector 将表征映射到学生 embedding 空间，再由学生解码。`[verified: §3.1–3.2]` | 将通信宽度与学生 hidden size 分开，使同一教师表示同时支持学生选择与预算选择。`[opinion]` |
| [Latent-Guided Reasoning](https://proceedings.iclr.cc/paper_files/paper/2026/file/6958e9d0f5a76d54ff97da8c45f4d52e-Paper-Conference.pdf) | 训练教师形成 latent guidance，并经 MLP 指导学生生成可读推理。`[verified: §3.2–3.4]` | 研究指导表示内部的嵌套宽度与跨学生联合复用，重点验证每档前缀的生成效用。`[opinion]` |
| [C2C](https://arxiv.org/html/2510.03215v2#A5.SS2) | KV cache 转换与融合；附录已探索共享 latent projector、多接收器和联合训练。`[verified: 附录 A.5.2及表14]` | 在共享接口上增加有序前缀预算轴，检验同一排序跨学生的有效性；不能将多接收器或共享投影本身作为新增贡献。`[opinion]` |
| [OverFill](https://arxiv.org/abs/2508.08446) | 完整模型 Prefill 后切换到 dense-pruned 模型解码。`[verified: Abstract]` | 通过可学习的通信接口接入学生，研究 latent 预算的伸缩，不要求学生由教师剪枝得到。`[opinion]` |
| [MentorPulse](https://arxiv.org/html/2608.20927#S1) | 增量处理学生新生成内容，刷新 latent memory，解决长生成中指导过期。`[verified: §1]` | 当前版本固定一次 Prefill，研究表示宽度与接收能力的匹配；宽度扩展不替代信息刷新。`[opinion]` |
| [ReGuLaR](https://arxiv.org/abs/2601.23184) | 用 rendered CoT 的视觉语义表征正则化变分 latent reasoning。`[verified: Abstract]` | 关注教师到多个学生的可伸缩通信接口及复用代价。`[opinion]` |

核心技术问题是：**对不同学生都有用的信息，能否被排进同一套前缀顺序？** 对照学生专用 MRL projector 可以检验共享排序的代价，对照共享但只训练满宽的 projector 可以检验前缀监督的价值；二者共同构成核心证据。`[opinion]`

## 参考论文的实验数据集

下表按原文区分训练、主要评测和扩展实验。数据集名称相同不代表训练划分、提示格式或评分方式相同，复现时需固定具体版本与样本划分。`[opinion]`

| 论文 | 训练 / 适配数据 | 主要评测与扩展实验 |
|---|---|---|
| [MRL](https://arxiv.org/html/2205.13147v4) | ImageNet-1K、JFT-300M、ALIGN 图文数据；BERT 实验使用 English Wikipedia、BooksCorpus。 | 分类与检索：ImageNet-1K、ImageNet-4K；鲁棒性：ImageNetV2、ImageNet-A、ImageNet-R、ImageNet-Sketch、ObjectNet；长尾：FLUID；另有 BERT masked language modeling 实验。`[verified: §4.1，附录 B/C/G]` |
| [LLM-to-SLM](https://arxiv.org/html/2402.16844v2) | 翻译使用 WMT 数据与教师生成标签；摘要使用 CNN/Daily Mail；指令微调使用 Alpaca。 | 翻译：WMT14 英→德、英→法，WMT16 英→罗马尼亚语；摘要：CNN/Daily Mail；指令跟随：MT-Bench。`[verified: §4.1–4.3]` |
| [Latent-Guided Reasoning](https://proceedings.iclr.cc/paper_files/paper/2026/file/6958e9d0f5a76d54ff97da8c45f4d52e-Paper-Conference.pdf) | GSM8K 与论文划分的 BBH 训练部分。 | 域内：GSM8K、BBH；域外：AGIEval、ARC-E、ARC-C、Odyssey-Math、SVAMP、AQuA；额外使用 ELI5-Test 评估长问答与解释质量。`[verified: 正式版 §4.1、表2、附录 D]` |
| [C2C](https://arxiv.org/html/2510.03215v2) | 主实验：OpenHermes-2.5 前 50 万条；规模和模型组合实验：MMLU auxiliary_train；长上下文实验：LongBench-E 的训练划分。 | 主表：OpenBookQA、MMLU-Redux、ARC-C、C-Eval；长上下文：LongBench-E；补充 agent 协作实验：GSM8K。`[verified: §4.1–4.3，附录 A.3.5、A.5.3]` |
| [OverFill](https://arxiv.org/html/2508.08446) | 主实验：Infinity-Instruct，经英文过滤；剪枝比例扫描：OpenHermes-2.5。 | GSM8K-CoT、ARC-C、MMLU、MATH、WMT16 德→英、IFEval、Natural Questions；另用 ZeroEval 设置的 MMLU-Redux、CRUXEval 测较长推理输出。不同模型规模的表格覆盖范围不同。`[verified: §4.1–4.3，表2–4，附录表6]` |
| [ReGuLaR](https://arxiv.org/html/2601.23184) | 主实验：GSM8K-Aug；极端压缩另使用 GSM8K-Aug-NL、AQUA-RAT、MATH；多模态扩展使用 MolReasoner 的分子描述数据。 | 主表：GSM8K-Aug、GSM-Hard、SVAMP、MultiArith；极端压缩：GSM8K-Aug-NL、AQUA-RAT、MATH；多模态：MolReasoner molecule captioning。`[verified: §4.1、§4.3、§5，附录 A.1]` |
| [MentorPulse](https://arxiv.org/html/2608.20927) | 自建混合训练集 fuse_v3：开放指令来源、代码指令、可验证约束指令和文档生成任务，经教师离线生成与过滤，混合长输出和短答案。 | 13 个主评测集：MMLU-Pro、GPQA Diamond、AGIEval-MCQ、MATH-500、OlympiadBench、LiveCodeBench、IFEval、WritingBench、LongBench v2、QuALITY、GovReport、MultiNews、LongBench-Write。`[verified: 附录 A.1 表3、D.1]` |

### 数据划分与口径注意事项

- **Latent-Guided Reasoning：** 8 个推理基准由 2 个域内和 6 个域外基准组成，ARC-E 与 ARC-C 分开计数；ELI5-Test 是额外评测。BBH 按论文划分使用，不能默认整套 BBH 都是未见测试数据。`[verified: 正式版 §4.1、表2、附录 D]`
- **C2C：** 长上下文主文称 LongBenchV1，附录明确使用 LongBench-E 的 13 个子数据集，并按数据索引随机划分 3/4 训练、1/4 评估；该结果不能当作完全未见长上下文任务的零样本评测。规模实验另用 15,000 条 MMLU auxiliary_train 样本训练。`[verified: §4.3，附录 A.3.4–A.3.5]`
- **MentorPulse：** 独立诊断集为 Multi-IF、QMSum、HelloBench、BigCodeBench、ARC-C，用于区分指导过期与能力限制，不计入 13 个主评测集。主评测中的 QuALITY 使用 dev split；LiveCodeBench 使用 2025 年 7 月之后发布的题目。`[verified: §3.1，附录 A.1 表3]`
- **ReGuLaR：** GSM8K-Aug 去掉推理链中的自然语言描述、保留数学表达式；GSM8K-Aug-NL 保留自然语言解释，两者均由 GSM8K 扩增得到，不能直接当作相同的原始 GSM8K 配方。`[verified: 附录 A.1]`
- **跨论文对比：** MMLU、MMLU-Redux、MMLU-Pro，MATH、MATH-500，以及 LongBench-E、LongBench v2 分别保留名称；不能将不同版本或不同提示协议的分数放在同一列直接比较。`[opinion]`

## 按数据集类型研究协同表现

研究主线按用户提出的 **数学推理、知识问答、工具调用** 组织，比较不同信息需求下的协同收益；不预设数学必然提升或知识问答必然无效。先使用前述已调研论文中的数据集，严格 KBQA 与工具调用基准属于后续待补充项，不将新候选自动视为已确定方案。`[opinion]`

| 类型 | 已调研论文中的数据集 | 研究问题与范围 |
|---|---|---|
| 数学推理 | GSM8K 及其已列变体、SVAMP、MultiArith、AQuA/AQUA-RAT、Odyssey-Math、MATH、MATH-500、OlympiadBench。 | 按难度、推理步骤和学生规模比较前缀收益；检查教师指导是否改善问题建模与执行。`[opinion]` |
| 事实知识问答 | Natural Questions（OverFill）。 | 检验教师独立答对、学生独立答错的事实问题能否经 latent 改善；需明确闭卷或提供证据的输入设置。`[opinion]` |
| 知识与推理混合问答 | OpenBookQA、MMLU/MMLU-Redux/MMLU-Pro、ARC-E/ARC-C、C-Eval、AGIEval、GPQA Diamond。 | 按学科与题目所需操作分组；这些任务同时涉及知识和推理，不能把平均增益直接解释为纯知识传递。`[opinion]` |
| 综合推理与上下文理解 | BBH；LongBench-E、LongBench v2、QuALITY。 | 区分综合推理与给定文档中的证据利用；数据集内部按任务类型拆分，不能把整个集合标为纯数学或闭卷知识问答。`[opinion]` |
| 代码能力与工具调用边界 | CRUXEval、LiveCodeBench、BigCodeBench；此前汇总未包含专门的工具调用基准。 | 代码理解、生成或使用库函数不等于多轮工具交互；C2C 的 GSM8K agent 协作也不直接证明工具调用能力。工具调用后续单独选型，分别测调用正确性、执行结果与动态状态下的后续决策。`[opinion]` |
| 辅助生成任务 | IFEval、Multi-IF、MT-Bench、ELI5-Test；WMT、CNN/Daily Mail、QMSum、GovReport、MultiNews、WritingBench、HelloBench、LongBench-Write。 | 用于补充指令约束、翻译、摘要、解释和长输出边界；不作为当前知识传递主张的替代证据。MRL 的视觉实验与 ReGuLaR 的分子描述扩展保留为方法参考。`[opinion]` |

上述归类依据前一节已核验的数据集用途；它是实验组织方式，不声称每个数据集只考察一种能力。严格 KBQA 需要明确知识库访问、实体链接或查询执行协议，此前汇总的知识考试题不能直接代替该设置。`[opinion]`

候选评测集的 test/dev 部分不用于训练或选择超参数；若使用某数据集的训练部分，记录划分并将对应评测标为域内。训练数据与评测数据独立登记，上表不作为混合训练清单。跨类型先固定教师、学生、读取器与适配预算，报告每个学生的质量—宽度—端到端成本曲线；结合题目难度、输入输出长度与教师/学生初始差距分组，避免仅凭不同数据集均分归因于知识类型。`[opinion]`

MRL 主要提供前缀曲线、独立宽度对照、未训练宽度与任务分组分析的实验范式；其视觉数据集无需直接迁移到 COPE。准确率、输出 token 数和端到端成本分开报告，避免把前缀压缩比当作整体加速比。`[opinion]`

## 教师知识能否经 latent 传给学生

**关键假设：教师可回答、学生独立回答失败的事实问题，能否由同一教师的一次 Prefill 表示，经共享有序前缀接口帮助学生答对？** 这里研究推理期的信息利用，不等于知识已经永久写入学生参数。`[opinion]`

需区分三个条件：教师独立生成时能回答；所选 Prefill 状态包含可供读取的事实信息；压缩后的表示能被目标学生利用。第一个条件不自动保证后两个条件。教师单次答对只能定义观测到的能力差异，不能作为“教师必然知道”或“latent 已含正确知识”的证明；必要时用问题改写与重复评估检查稳定性。`[opinion]`

### 最小诊断设计

1. **确定能力差异。** 在预先固定的独立评测集上运行教师和匹配适配条件的学生基线，按两者答对/答错分成四组；保留全量结果，重点分析教师对、学生错的一组。分组只用于分析，不用于测试集调参。`[opinion]`
2. **比较信息通道。** 对同题比较无 latent、常量/错配 latent、各宽度 COPE、学生专用接口，以及保留更多 slot 或更多原始 hidden states 的高预算接口。改变输入形状或容量的接口需在相应条件下训练；高预算接口是诊断对照，不是理论上界。`[opinion]`
3. **加入文本提示对照。** 允许教师额外生成事实提示给学生，并单独计入教师解码成本；若提示直接包含答案，只能说明学生能消费该答案，不能据此证明推理迁移。该分支不计作 COPE 的一次 Prefill 部署路径。`[opinion]`
4. **区分知识来源。** 分别测试双方闭卷、双方获得相同证据，以及教师获得证据而学生仅获得问题与 latent。最后一种检验外部证据压缩传输，不能当成教师参数知识传递的证据；参考答案不得进入主实验的教师编码输入。`[opinion]`
5. **报告收益与损害。** 报告各分组样本量、学生错误被纠正的比例、学生原本正确却被协同改错的比例，以及全量净增益和成本。教师与学生都答错的一组也保留，不能把教师独立准确率视为协同系统的硬上限。`[opinion]`

### 失败结果如何影响适用性

| 观测结果 | 可支持的判断与下一步 |
|---|---|
| 高预算或长前缀有效，短前缀无效 | 优先定位宽度、slot 汇聚或前缀训练的瓶颈；该结果不否定知识可以经 latent 传递。`[opinion]` |
| 学生专用接口有效，共享接口无效 | 直接挑战 COPE 的跨学生共享主张；应报告共享代价，并定位学生间的冲突。`[opinion]` |
| 教师文本提示有效，已充分训练的 Prefill latent 接口仍无效 | 表明当前状态提取与读取方案未实现同等可用的信息传递；还需区分表示、接口及优化问题，不能外推为所有 latent 方法都不可能。`[opinion]` |
| 事实知识问答无收益，数学推理有稳定收益 | 收窄为经验证的推理指导适用范围，不宣称通用知识通信；知识问答的失败是应用边界。`[opinion]` |
| 仅教师可见外部证据时有收益 | 支持上下文信息压缩与传输，不足以证明教师参数知识迁移。`[opinion]` |
| 质量提高，但相同质量下总成本不占优 | 功能有效，效率价值尚未成立；尤其检查短答案任务的教师 Prefill 固定开销。`[opinion]` |
| 在目标任务上，经输入、训练和接口检查后仍无稳定净收益，或持续被匹配成本的基线支配 | 当前 COPE 配置不适合该部署范围，应调整方案或停止扩展；不能用潜在的未测场景掩盖失败。`[opinion]` |

## 结果表述边界

解释实验时保持以下边界：

- 多前缀损失继承自 MRL；单次 Prefill、latent 通信、共享 projector 与多接收器均有相关先例。贡献是上述跨学生有序接口的具体设计与验证；未完成更全面检索和实验前，不宣称“首次”、性能优越或普适规律。`[opinion]`
- 多宽度监督不保证每个样本随宽度增加都更准确，也不预定义“前几维是计划、后几维是细节”。MRL 原文已观察到部分样本在低维下更准确。`[verified: MRL §5 “Disagreement across Dimensions”段落]`
- 普通上下文 hidden states 不能直接称为教师已完成的推理计划；答案提升或可读 CoT 也不足以证明忠实传递了教师推理过程。`[opinion]`
- 不预设学生越小就应使用越短前缀；能力与预算的匹配方向由实验决定。`[opinion]`
- 同家族结果只支持所测范围；新学生训练读取器后接入，应表述为轻量适配，不能称为零训练泛化。`[opinion]`

## 成本口径

端到端延迟包含教师 Prefill、汇聚与投影、序列化与传输、学生 Prefill、学生自回归解码；另报训练成本、可训练参数量与新增学生接入成本。`[opinion]`

未压缩的紧密 latent payload 按 `m × d × 每元素字节数` 计算，并单独记录协议头和复制等开销。固定 `m`、映射回固定 `h_s` 后，缩短 `d` 不会自动降低学生 Transformer 的解码宽度或 KV cache 大小；先计算满宽再截断也不会减少教师 Prefill 成本。`[opinion]`

性能与成本须在明确硬件、批量、输入输出长度和通信条件下比较。参数共享减少接口数量，不等价于已证明训练更快或端到端加速。`[opinion]`
