"""问题1：使用附件1单日数据求解并生成 result1.xlsx。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import DT_H, solve_day_dispatch
from xlsx_io import read_sheet, write_xlsx


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
OUT = ROOT / "output" / "q1"
SEED = 2026


def interval_labels() -> list[str]:
    labels = []
    for t in range(144):
        start = t * 10
        end = (t + 1) * 10
        sh, sm = divmod(start, 60)
        eh, em = divmod(end, 60)
        start_text = f"{sh}:{sm:02d}"
        end_text = f"{eh}:{em:02d}" if eh < 24 else "0:00+1"
        labels.append(f"{start_text}-{end_text}")
    return labels


def main() -> None:
    np.random.seed(SEED)  # 本模型无随机步骤，保留种子便于后续扩展
    rows = read_sheet(str(DATA / "附件1.xlsx"), 0)
    if len(rows) != 145 or len(rows[0]) < 4:
        raise ValueError("附件1应为表头加144行、至少4列")
    price = np.asarray([float(row[1]) for row in rows[1:]], dtype=float)
    load = np.asarray([float(row[2]) for row in rows[1:]], dtype=float) * DT_H
    pv = np.asarray([float(row[3]) for row in rows[1:]], dtype=float) * DT_H
    result = solve_day_dispatch(price, load, pv, initial_energy=6000.0, terminal_energy=6000.0)

    labels = interval_labels()
    plan_rows = [["时间段", "购电量"]] + [[label, float(value)] for label, value in zip(labels, result["grid_plan"])]
    block_rows = [["时间段", "充电量", "放电量", "时刻", "储电量"]]
    for block in range(6):
        left, right = block * 24, (block + 1) * 24
        block_rows.append([f"{block*4}:00-{(block+1)*4}:00", float(np.sum(result["charge"][left:right])), float(np.sum(result["discharge"][left:right])), "", ""])
    block_rows[1][3], block_rows[1][4] = "0:00", float(result["soc"][0])
    block_rows[2][3], block_rows[2][4] = "24:00", float(result["soc"][-1])
    OUT.mkdir(parents=True, exist_ok=True)
    write_xlsx(str(OUT / "result1.xlsx"), {"计划购电量": plan_rows, "充放电量": block_rows})

    balance_residual = result["grid_plan"] + pv + result["discharge"] - load - result["charge"] - result["spill"]
    report = [
        f"objective_cost={result['cost']:.10f}",
        f"max_balance_residual={np.max(np.abs(balance_residual)):.10e}",
        f"initial_soc={result['soc'][0]:.10f}",
        f"terminal_soc={result['soc'][-1]:.10f}",
        f"max_charge_discharge_overlap={result['overlap_max']:.10e}",
    ]
    (OUT / "check.txt").write_text("\n".join(report), encoding="utf-8")

    x = np.arange(144)
    plt.figure(figsize=(11, 4))
    plt.plot(x, result["grid_plan"], label="Grid purchase")
    plt.plot(x, result["charge"], label="Charge")
    plt.plot(x, result["discharge"], label="Discharge")
    plt.xlabel("10-minute interval")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "energy_balance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 4))
    plt.plot(np.arange(145), result["soc"])
    plt.xlabel("Time point")
    plt.ylabel("Stored energy (kWh)")
    plt.tight_layout()
    plt.savefig(OUT / "soc.png", dpi=180)
    plt.close()
    print(f"问题1完成：{OUT / 'result1.xlsx'}")


if __name__ == "__main__":
    main()
