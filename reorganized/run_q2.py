"""问题2：历史滚动预测、日前计划与实际紧急购电仿真。"""

from __future__ import annotations

from collections import OrderedDict
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dispatch import DT_H, EMAX, EMIN, simulate_receding_plan, solve_day_dispatch
from forecast import safe_forecast
from xlsx_io import read_sheet, write_xlsx


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
OUT = ROOT / "output" / "q2"
HISTORY_DAYS = 31  # 1月实际调度初始化，模板从2月1日开始输出
WINDOW = 7
EMERGENCY_MULTIPLIER = 5.0
RESERVE_FRACTION = 0.20  # 下一日高价时段预测净负荷的保留比例
SEED = 2026


def _numeric_matrix(rows: list[list[object]]) -> np.ndarray:
    if len(rows) < 2 or len(rows[0]) < 145:
        raise ValueError("附件2工作表应为表头加365行、日期列加144个数据列")
    matrix = np.asarray([[float(value) for value in row[1:145]] for row in rows[1:]], dtype=float)
    return matrix


def interval_labels() -> list[str]:
    result = []
    for t in range(144):
        start = t * 10
        end = (t + 1) * 10
        sh, sm = divmod(start, 60)
        eh, em = divmod(end, 60)
        result.append(f"{sh}:{sm:02d}-{eh}:{em:02d}" if eh < 24 else f"{sh}:{sm:02d}-0:00+1")
    return result


def day_blocks(values: np.ndarray, soc: np.ndarray, current_date: date, headers: bool = True) -> list[list[object]]:
    rows = []
    for block in range(6):
        left, right = block * 24, (block + 1) * 24
        rows.append([
            current_date.isoformat() if block == 0 else "",
            f"{block*4}:00-{(block+1)*4}:00",
            float(np.sum(values[0][left:right])),
            float(np.sum(values[1][left:right])),
            "0:00" if block == 0 else ("24:00" if block == 1 else ""),
            float(soc[0]) if block == 0 else (float(soc[-1]) if block == 1 else ""),
        ])
    return rows


def reserve_target_soc(price: np.ndarray, load_safe_kw: np.ndarray, pv_safe_kw: np.ndarray) -> float:
    """根据下一日安全预测的高价时段净负荷确定跨日储能保留目标。"""
    net_need = np.maximum(load_safe_kw - pv_safe_kw, 0.0) * DT_H
    expensive = price >= np.quantile(price, 0.75)
    target = EMIN + RESERVE_FRACTION * float(np.sum(net_need[expensive]))
    return float(np.clip(target, EMIN, EMAX))


def main() -> None:
    np.random.seed(SEED)  # 当前预测和优化均为确定性计算
    attachment1 = read_sheet(str(DATA / "附件1.xlsx"), 0)
    price = np.asarray([float(row[1]) for row in attachment1[1:]], dtype=float)
    if len(price) != 144:
        raise ValueError("附件1电价必须有144个十分钟点")

    attachment2 = str(DATA / "附件2.xlsx")
    load_kw = _numeric_matrix(read_sheet(attachment2, 0))
    pv_kw = _numeric_matrix(read_sheet(attachment2, 1))
    if load_kw.shape != (365, 144) or pv_kw.shape != (365, 144):
        raise ValueError(f"附件2数据规模错误: load={load_kw.shape}, pv={pv_kw.shape}")
    load_kwh = load_kw * DT_H
    pv_kwh = pv_kw * DT_H
    actual_kw = np.stack([load_kw, pv_kw])

    plan_header = ["日期\\时间"] + interval_labels() + ["全天购电量", "全天购电费"]
    plan_rows = [plan_header]
    storage_rows = [["日期", "时间段", "充电量", "放电量", "时刻", "储电量"]]
    emergency_rows = [["日期", "购电时间段", "购电量"]]
    daily_rows = [["日期", "正常购电成本", "紧急购电成本", "紧急购电量", "弃电量", "日总成本", "日终计划SOC", "日终实际SOC", "实际再调度电量"]]
    all_plans, all_plan_charges, all_plan_discharges = [], [], []
    all_emergencies, all_socs = [], []

    # 先用1月实际数据做状态初始化：每一天都根据当日实际负荷、光伏和电价
    # 求解一次调度，日终储电量传递给下一天。1月31日的日终储电量就是
    # 2月1日优化的初始储电量，而不是人为再次固定为6000 kWh。
    energy = 6000.0
    january_initial_energy = energy
    january_warmup_rows = [["日期", "实际调度正常购电成本", "日终SOC", "下一日保留目标"]]
    january_warmup_cost = 0.0
    for day_index in range(HISTORY_DAYS):
        current_date = date(2025, 1, 1) + timedelta(days=day_index)
        if day_index + 1 >= WINDOW:
            _, _, next_load_safe_kw, next_pv_safe_kw = safe_forecast(
                actual_kw, day_index + 1, alpha=0.8, window=WINDOW
            )
        else:
            next_load_safe_kw = load_kw[day_index]
            next_pv_safe_kw = pv_kw[day_index]
        terminal_target = reserve_target_soc(price, next_load_safe_kw, next_pv_safe_kw)
        january_plan = solve_day_dispatch(
            price,
            load_kwh[day_index],
            pv_kwh[day_index],
            energy,
            terminal_energy=terminal_target,
        )
        january_cost = float(np.sum(price * np.asarray(january_plan["grid_plan"])))
        energy = float(january_plan["soc"][-1])
        january_warmup_cost += january_cost
        january_warmup_rows.append([current_date.isoformat(), january_cost, energy, terminal_target])
    february_initial_energy = energy

    for day_index in range(HISTORY_DAYS, 365):
        current_date = date(2025, 1, 1) + timedelta(days=day_index)
        _, _, load_safe_kw, pv_safe_kw = safe_forecast(actual_kw, day_index, alpha=0.8, window=WINDOW)
        plan = solve_day_dispatch(price, load_safe_kw * DT_H, pv_safe_kw * DT_H, energy, terminal_energy=None)
        simulation = simulate_receding_plan(plan, load_kwh[day_index], pv_kwh[day_index], energy)
        grid = np.asarray(plan["grid_plan"])
        charge = np.asarray(simulation["charge"])
        discharge = np.asarray(simulation["discharge"])
        soc = np.asarray(simulation["soc"])
        emergency = np.asarray(simulation["emergency"])
        actual_spill = np.asarray(simulation["actual_spill"])
        normal_cost = float(np.sum(price * grid))
        emergency_cost = float(np.sum(EMERGENCY_MULTIPLIER * price * emergency))
        total_cost = normal_cost + emergency_cost

        plan_rows.append([current_date.isoformat()] + [float(x) for x in grid] + [float(np.sum(grid)), normal_cost])
        storage_rows.extend(day_blocks(np.vstack([charge, discharge]), soc, current_date))
        for t, quantity in enumerate(emergency):
            if quantity > 1e-8:
                emergency_rows.append([current_date.isoformat(), interval_labels()[t], float(quantity)])
        planned_soc = np.asarray(plan["soc"])
        recourse_energy = float(
            np.sum(np.abs(charge - np.asarray(plan["charge"])))
            + np.sum(np.abs(discharge - np.asarray(plan["discharge"])))
        )
        daily_rows.append([
            current_date.isoformat(), normal_cost, emergency_cost,
            float(np.sum(emergency)), float(np.sum(actual_spill)), total_cost,
            float(planned_soc[-1]), float(soc[-1]), recourse_energy,
        ])
        all_plans.append(grid)
        all_plan_charges.append(np.asarray(plan["charge"]))
        all_plan_discharges.append(np.asarray(plan["discharge"]))
        all_emergencies.append(emergency)
        all_socs.append(soc)
        energy = float(soc[-1])

    OUT.mkdir(parents=True, exist_ok=True)
    write_xlsx(str(OUT / "result2.xlsx"), OrderedDict([
        ("计划购电量", plan_rows),
        ("充放电量", storage_rows),
        ("紧急购电量", emergency_rows),
        ("每日汇总", daily_rows),
    ]))

    plan_array = np.asarray(all_plans)
    plan_charge_array = np.asarray(all_plan_charges)
    plan_discharge_array = np.asarray(all_plan_discharges)
    emergency_array = np.asarray(all_emergencies)
    soc_array = np.asarray(all_socs)
    daily = np.asarray(daily_rows[1:], dtype=object)
    (OUT / "daily_summary.csv").write_text(
        "\n".join(",".join(str(item) for item in row) for row in daily_rows), encoding="utf-8"
    )
    checks = [
        f"output_days={len(all_plans)}",
        f"first_date=2025-02-01",
        f"last_date=2025-12-31",
        f"january_initial_soc={january_initial_energy:.10f}",
        f"february_initial_soc={february_initial_energy:.10f}",
        f"january_warmup_normal_cost={january_warmup_cost:.10f}",
        f"january_reserve_fraction={RESERVE_FRACTION:.10f}",
        f"annual_normal_cost={sum(float(row[1]) for row in daily_rows[1:]):.10f}",
        f"annual_emergency_cost={sum(float(row[2]) for row in daily_rows[1:]):.10f}",
        f"annual_emergency_energy={sum(float(row[3]) for row in daily_rows[1:]):.10f}",
        f"planned_charge_energy={plan_charge_array.sum():.10f}",
        f"planned_discharge_energy={plan_discharge_array.sum():.10f}",
        f"soc_min={soc_array.min():.10f}",
        f"soc_max={soc_array.max():.10f}",
        f"total_actual_recourse_energy={sum(float(row[8]) for row in daily_rows[1:]):.10f}",
        f"plan_min={plan_array.min():.10f}",
        f"emergency_nonzero_intervals={int(np.sum(emergency_array > 1e-8))}",
    ]
    (OUT / "check.txt").write_text("\n".join(checks), encoding="utf-8")
    (OUT / "january_warmup.csv").write_text(
        "\n".join(",".join(str(item) for item in row) for row in january_warmup_rows),
        encoding="utf-8",
    )

    dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(HISTORY_DAYS, 365)]
    daily_normal = np.asarray([float(row[1]) for row in daily_rows[1:]])
    daily_emergency = np.asarray([float(row[2]) for row in daily_rows[1:]])
    plt.figure(figsize=(11, 4))
    plt.plot(dates, daily_normal, label="Normal purchase cost")
    plt.plot(dates, daily_emergency, label="Emergency purchase cost")
    plt.ylabel("Cost (CNY)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "daily_cost.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 4))
    plt.plot(np.arange(soc_array.shape[0]), soc_array[:, -1])
    plt.ylabel("End-of-day SOC (kWh)")
    plt.xlabel("Output day index")
    plt.tight_layout()
    plt.savefig(OUT / "daily_soc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 4))
    plt.plot(np.asarray(daily_rows[1:], dtype=object)[:, 0], np.asarray(daily_rows[1:], dtype=object)[:, 3].astype(float))
    plt.xticks(rotation=45)
    plt.ylabel("Emergency purchase (kWh)")
    plt.tight_layout()
    plt.savefig(OUT / "daily_emergency.png", dpi=180)
    plt.close()
    print(f"问题2完成：{OUT / 'result2.xlsx'}")


if __name__ == "__main__":
    main()
