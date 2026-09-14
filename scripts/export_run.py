"""Import immutable small run evidence and render the version's latest status."""

import argparse
import json
from pathlib import Path

from cope.experiment_data import write_new_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--version-dir", required=True)
    args = parser.parse_args()
    source, version = Path(args.run_dir), Path(args.version_dir)
    destination = version / "runs" / source.name
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "metrics.json", "artifacts.json"):
        if not (source / name).exists():
            continue
        value = json.loads((source / name).read_text())
        target = destination / name
        if target.exists():
            if json.loads(target.read_text()) != value:
                raise ValueError(f"refusing to replace prior evidence: {target}")
        else:
            write_new_json(target, value)
    if not (destination / "metrics.json").exists():
        print("Manifest imported; run has not finalized metrics.")
        return
    metrics = json.loads((destination / "metrics.json").read_text())
    manifest = json.loads((destination / "manifest.json").read_text())
    version_name = manifest["config"]["version"]
    lines = [
        f"# {version_name} 自动汇总（研究判断见 results.md）",
        "",
        f"最新 run：`{source.name}`",
        "",
        f"状态：`{metrics['status']}`；共同训练步数：{metrics.get('common_step', '未完成')}。",
        "",
        "| 组别 | 样本数 | 准确率 | NLL | 平均输出 token | 解析失败率 | 截断率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in metrics.get("arms", {}).items():
        values = [name, result["n"]] + [
            result.get(key)
            for key in (
                "accuracy",
                "nll",
                "mean_output_tokens",
                "parse_failure_rate",
                "truncation_rate",
            )
        ]
        lines.append(
            "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in values) + " |"
        )
    lines += ["", "## 配对比较", ""]
    for name, result in metrics.get("comparisons", {}).items():
        lines.append(
            f"- `{name}`：净准确率变化 {result['net_accuracy_change']:.4f}，"
            f"95% 配对 bootstrap 区间 {result['paired_bootstrap_95']}；"
            f"纠错 {result['corrected']}，伤害 {result['harmed']}。"
        )
    lines += [
        "",
        "## 成本与边界",
        "",
        f"- 累计预算使用：{metrics.get('budget_used_seconds', 0):.1f} 秒。",
        f"- 两卡峰值 allocated bytes：{metrics.get('peak_allocated_bytes')}。",
        "- 完整参数量、逐组耗时、逐题预测位置和版本证据见 runs 下的 JSON。",
        "- 单种子、128 条训练集内部留出样本，仅支持初探；不是官方 test 成绩。",
        f"- 配置宽度：{manifest['config']['protocol']['widths']}；完成范围以逐组指标为准。",
        "- 单学生结果不得据此宣称跨学生共享排序成立；不自动启动下一版本。",
        "",
    ]
    # Do not overwrite the human interpretation or an earlier finalized export.
    with (destination / "results.generated.md").open("x") as handle:
        handle.write("\n".join(lines))
    print(destination)


if __name__ == "__main__":
    main()
