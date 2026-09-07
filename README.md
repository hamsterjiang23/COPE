# COPE

**A Composable Ordered-Prefix Latent Protocol for Cross-Model Reasoning**

**用于跨模型推理的可组合有序前缀潜表征协议**。`[verified: 项目发起者给定标题]`

一个教师、一个共享投影器，通过不同宽度的有序 latent 前缀，适配多个同家族、不同规模的学生。教师对输入执行一次 Prefill，学生读取 latent 后自回归生成推理过程或答案。`[verified: 项目发起者给定 COPE 定义]`

本研究受 [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) 启发，探索将嵌套前缀表示用于跨模型生成接口：同一教师表示能否支持不同学生与不同通信预算。`[opinion]`

研究重点包括有序前缀的独立可用性、跨学生共享相对专用接口的性能代价，以及学生能力、latent 宽度与端到端成本之间的匹配关系。上述内容是待验证的研究目标。`[opinion]`

核心创新假设、接口约定、实验对照与结论边界见 [AGENTS.md](AGENTS.md)。`[verified: AGENTS.md 中“核心创新假设”“最小接口与训练约束”“首轮证据与对照”“创新边界与结果表述”]`
