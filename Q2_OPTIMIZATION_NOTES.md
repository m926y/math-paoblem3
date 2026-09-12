# 第二问优化：严格因果的同星期预测

## 中文摘要

本次修改针对第二问原最近7天预测没有识别星期周期的问题。旧模型给前一天40%权重，却只给前7天2%权重，导致星期日负荷被系统性低估；星期日约占旧模型紧急购电费用的92%。优化后使用前7、14、21、28天的同星期曲线，权重为0.50、0.25、0.15、0.10，并用目标日前历史误差形成单侧安全裕度。现有 `run_q2.py` 直接运行优化模型，没有新增第二问入口，也没有修改原始附件、储能约束或5倍紧急购电费用口径。

默认 `net/0.90` 版本将2月1日至12月31日总费用从18,696,518.4879元降至14,558,771.6146元，下降4,137,746.8733元，即22.13%；紧急购电量从405,289.3713 kWh降至123,296.7784 kWh。六组敏感性实验的全年描述性最低值为 `net/0.85` 的14,489,331.8246元；默认仍采用 `net/0.90`，因为它在预先划分的2—6月选择期费用最低，7—12月作为留出期。统一验证160/160项通过。

建议提交信息：

```text
feat(q2): use weekday-aware causal forecasts
```

建议Pull Request标题：

```text
优化第二问日前购电的同星期因果预测
```

---

## English details

## Suggested commit

```text
feat(q2): use weekday-aware causal forecasts
```

## Suggested pull request title

```text
Optimize Q2 day-ahead purchases with weekday-aware forecasts
```

## Problem

The previous Q2 model predicted each target day from the immediately preceding seven days and assigned 40% weight to yesterday. The data have a strong weekday/weekend pattern: Sundays generated about 92% of the previous model's emergency-purchase cost. A Saturday-dominated forecast therefore systematically underestimated Sunday demand, which was especially expensive because emergency electricity costs five times the normal tariff.

## Changes

- Add causal forecasts from the previous 7, 14, 21, and 28 days with weights 0.50, 0.25, 0.15, and 0.10.
- Add two one-sided safety-margin modes:
  - `net`: apply the historical quantile to underestimated net load;
  - `separate`: apply separate margins to underestimated load and overestimated PV.
- Keep all forecast inputs strictly before the target day.
- Update the existing `run_q2.py` entry point instead of adding a second optimized runner.
- Preserve `run_q2_causal.py` and `run_q2_legacy.py` as comparison baselines.
- Continue writing the official `result2.xlsx` template and the existing JSON, text, CSV, and plot outputs.
- Extend unified validation to cover the optimized Q2 result.

## Reproduction

Run the selected main configuration:

```powershell
python run_q2.py
```

This is equivalent to:

```powershell
python run_q2.py --quantile 0.90 --safety-mode net
```

Run the complete six-case sensitivity comparison by changing the two arguments and assigning a distinct `--output-dir` when separate workbooks are required.

## Results

All costs cover 2025-02-01 through 2025-12-31. Emergency purchases are charged at five times the time-varying normal tariff.

| Safety mode | Quantile | Normal cost (CNY) | Emergency cost (CNY) | Emergency energy (kWh) | Spill (kWh) | Total cost (CNY) |
|---|---:|---:|---:|---:|---:|---:|
| net | 0.80 | 13,012,574.1428 | 1,608,928.8545 | 277,413.7540 | 2,347,121.4663 | 14,621,502.9974 |
| net | 0.85 | 13,351,438.7518 | 1,137,893.0728 | 197,548.0631 | 2,737,444.5841 | **14,489,331.8246** |
| net | 0.90 | 13,855,599.0460 | 703,172.5686 | 123,296.7784 | 3,362,443.5276 | 14,558,771.6146 |
| separate | 0.80 | 13,199,342.8703 | 1,313,811.6540 | 229,820.1593 | 2,568,533.9654 | 14,513,154.5242 |
| separate | 0.85 | 13,582,871.8249 | 934,968.5156 | 164,676.9086 | 3,043,960.5129 | 14,517,840.3405 |
| separate | 0.90 | 14,147,765.5385 | 597,328.2857 | 105,627.7244 | 3,770,833.8710 | 14,745,093.8243 |

The full-period descriptive minimum is `net/0.85`. The default remains `net/0.90` because it had the lowest cost on the pre-specified February-June selection period; July-December was retained as a holdout period. This avoids choosing the final parameter solely from the full-year minimum.

Increasing the quantile consistently lowers emergency purchases but raises normal purchases and spilled energy. The total cost is therefore U-shaped rather than monotonic. The separate mode becomes increasingly conservative because load and PV margins can accumulate.

Compared with the previous causal seven-day model, the selected `net/0.90` configuration reduces total cost from 18,696,518.4879 CNY to 14,558,771.6146 CNY, a reduction of 4,137,746.8733 CNY or 22.13%. Emergency energy falls from 405,289.3713 kWh to 123,296.7784 kWh.

## Validation

- Unified validation: 160/160 checks passed.
- SOC remained within 1,200-10,800 kWh.
- Maximum energy-balance residual: `4.55e-13 kWh`.
- Simultaneous actual charge/discharge: `0 kWh`.
- The official result2 workbook agrees with its JSON summary.
- Input file hashes and the result2 template hash are recorded in `output/q2/summary.json`.

## Files changed

- `forecast.py`: weekday forecasts and safety-margin modes.
- `q2_core.py`: optimized causal annual simulation.
- `run_q2.py`: existing Q2 entry now runs the optimized model.
- `validate_project.py`: optimized Q2 output added to validation.
- `README.md`: model definition, sensitivity results, and selected configuration.
- `output/q2/*`: regenerated main Q2 result workbook, summary, checks, CSV, and plots.
- `output/validation_report.json` and `output/validation_report.md`: refreshed validation evidence.
- `output/model_versions/model_comparison.csv`, `.json`, and `.xlsx`: refreshed comparison index.
- `../求解/实验/experiment_contract.yaml`: expanded Q2 experiment contract.
- `../求解/实验/experiment_report.md`: six-case sensitivity table and interpretation.
- `../求解/实验/events.jsonl`: machine-readable experiment events.

No raw attachment was modified. No new Q2 execution entry point was added.
