"""共享的单日线性储能调度模型。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


ETA = 0.9
EMIN = 1200.0
EMAX = 10800.0
P_MAX_KW = 5000.0
DT_H = 1.0 / 6.0
Q_MAX_KWH = P_MAX_KW * DT_H


def _slice(start: int, length: int) -> slice:
    return slice(start, start + length)


def solve_day_dispatch(price: np.ndarray, load_kwh: np.ndarray, pv_kwh: np.ndarray, initial_energy: float, terminal_energy: float | None = None) -> dict[str, np.ndarray | float]:
    """求解一天的购电、充放电和SOC计划。

    采用线性规划并允许弃电。由于电价均为正且没有反向售电收益，
    同时充放电不会降低目标值；结果中仍会进行 overlap 检查。
    """
    price = np.asarray(price, dtype=float)
    load_kwh = np.asarray(load_kwh, dtype=float)
    pv_kwh = np.asarray(pv_kwh, dtype=float)
    if not (price.shape == load_kwh.shape == pv_kwh.shape):
        raise ValueError("price/load/pv 的长度必须相同")
    if np.any(price < 0) or np.any(load_kwh < 0) or np.any(pv_kwh < 0):
        raise ValueError("电价、负荷和光伏电量不能为负")
    if not EMIN - 1e-8 <= initial_energy <= EMAX + 1e-8:
        raise ValueError("初始SOC超出允许范围")

    T = len(price)
    g0, c0, d0, spill0, e0 = 0, T, 2 * T, 3 * T, 4 * T
    n = 4 * T + T + 1
    objective = np.zeros(n)
    objective[_slice(g0, T)] = price

    bounds_lower = np.zeros(n)
    bounds_upper = np.full(n, np.inf)
    bounds_upper[_slice(c0, T)] = Q_MAX_KWH
    bounds_upper[_slice(d0, T)] = Q_MAX_KWH
    bounds_lower[_slice(e0, T + 1)] = EMIN
    bounds_upper[_slice(e0, T + 1)] = EMAX
    # 使用 linprog 在不同 SciPy 版本都支持的 (lower, upper) 列表格式。
    variable_bounds = list(zip(bounds_lower.tolist(), bounds_upper.tolist()))

    eq_rows = []
    rhs = []
    for t in range(T):
        balance = np.zeros(n)
        balance[g0 + t] = 1.0
        balance[c0 + t] = -1.0
        balance[d0 + t] = 1.0
        balance[spill0 + t] = -1.0
        eq_rows.append(balance)
        rhs.append(load_kwh[t] - pv_kwh[t])

        storage = np.zeros(n)
        storage[e0 + t + 1] = 1.0
        storage[e0 + t] = -1.0
        storage[c0 + t] = -ETA
        storage[d0 + t] = 1.0 / ETA
        eq_rows.append(storage)
        rhs.append(0.0)

    start = np.zeros(n)
    start[e0] = 1.0
    eq_rows.append(start)
    rhs.append(float(initial_energy))
    if terminal_energy is not None:
        end = np.zeros(n)
        end[e0 + T] = 1.0
        eq_rows.append(end)
        rhs.append(float(terminal_energy))

    result = linprog(
        objective,
        A_eq=np.asarray(eq_rows),
        b_eq=np.asarray(rhs),
        bounds=variable_bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"单日调度求解失败: {result.message}")

    x = result.x
    values = {
        "grid_plan": x[_slice(g0, T)],
        "charge": x[_slice(c0, T)],
        "discharge": x[_slice(d0, T)],
        "spill": x[_slice(spill0, T)],
        "soc": x[_slice(e0, T + 1)],
        "cost": float(result.fun),
    }
    values["overlap_max"] = float(np.minimum(values["charge"], values["discharge"]).max(initial=0.0))
    return values


def simulate_fixed_plan(plan: dict[str, np.ndarray | float], load_real_kwh: np.ndarray, pv_real_kwh: np.ndarray) -> dict[str, np.ndarray | float]:
    """固定执行日前计划，按实际负荷/光伏计算紧急购电和弃电。"""
    grid = np.asarray(plan["grid_plan"], dtype=float)
    charge = np.asarray(plan["charge"], dtype=float)
    discharge = np.asarray(plan["discharge"], dtype=float)
    load_real_kwh = np.asarray(load_real_kwh, dtype=float)
    pv_real_kwh = np.asarray(pv_real_kwh, dtype=float)
    residual = load_real_kwh + charge - pv_real_kwh - discharge - grid
    emergency = np.maximum(residual, 0.0)
    spill = np.maximum(-residual, 0.0)
    return {"emergency": emergency, "actual_spill": spill, "shortage_max": float(emergency.max(initial=0.0))}


def simulate_receding_plan(
    plan: dict[str, np.ndarray | float],
    load_real_kwh: np.ndarray,
    pv_real_kwh: np.ndarray,
    initial_energy: float,
) -> dict[str, np.ndarray | float]:
    """按日前购电计划运行，并根据实际负荷/光伏对储能做有限再调度。

    正常购电量保持日前计划不变。实际偏差优先通过取消相反方向的计划动作、
    再使用剩余充放电能力进行修正，最后才产生紧急购电或弃电。SOC按实际动作
    闭环更新，日终实际SOC可传递给下一天。
    """
    grid = np.asarray(plan["grid_plan"], dtype=float)
    charge_plan = np.asarray(plan["charge"], dtype=float)
    discharge_plan = np.asarray(plan["discharge"], dtype=float)
    load_real_kwh = np.asarray(load_real_kwh, dtype=float)
    pv_real_kwh = np.asarray(pv_real_kwh, dtype=float)
    if not (grid.shape == charge_plan.shape == discharge_plan.shape == load_real_kwh.shape == pv_real_kwh.shape):
        raise ValueError("日前计划和实际数据长度必须相同")
    if not EMIN - 1e-8 <= initial_energy <= EMAX + 1e-8:
        raise ValueError("实际运行初始SOC超出允许范围")

    actual_charge = np.zeros_like(charge_plan)
    actual_discharge = np.zeros_like(discharge_plan)
    emergency = np.zeros_like(grid)
    spill = np.zeros_like(grid)
    balance_residual = np.zeros_like(grid)
    soc = np.empty(len(grid) + 1, dtype=float)
    soc[0] = initial_energy

    for t in range(len(grid)):
        energy = float(soc[t])
        charge = float(charge_plan[t])
        discharge = float(discharge_plan[t])
        # 日前计划基于预测SOC，实际SOC可能不同；先把基线动作截断到当前可行范围。
        charge = min(charge, max((EMAX - energy) / ETA, 0.0))
        discharge = min(discharge, max((energy - EMIN) * ETA, 0.0))
        residual = float(load_real_kwh[t] + charge - pv_real_kwh[t] - discharge - grid[t])

        if residual > 0.0:
            # 实际负荷偏大或光伏偏小时，先取消原计划充电，再增加放电。
            cancelled = min(charge, residual)
            charge -= cancelled
            residual -= cancelled
            energy_after_baseline = energy + ETA * charge - discharge / ETA
            extra_discharge = min(
                Q_MAX_KWH - discharge,
                residual,
                max((energy_after_baseline - EMIN) * ETA, 0.0),
            )
            discharge += extra_discharge
            residual -= extra_discharge
        elif residual < 0.0:
            # 实际负荷偏小或光伏偏大时，先取消原计划放电，再增加充电。
            surplus = -residual
            cancelled = min(discharge, surplus)
            discharge -= cancelled
            surplus -= cancelled
            extra_charge = min(
                Q_MAX_KWH - charge,
                surplus,
                max((EMAX - energy - ETA * charge + discharge / ETA) / ETA, 0.0),
            )
            charge += extra_charge
            surplus -= extra_charge
            residual = -surplus

        actual_charge[t] = charge
        actual_discharge[t] = discharge
        emergency[t] = max(residual, 0.0)
        spill[t] = max(-residual, 0.0)
        soc[t + 1] = energy + ETA * charge - discharge / ETA
        soc[t + 1] = min(max(soc[t + 1], EMIN), EMAX)
        balance_residual[t] = grid[t] + pv_real_kwh[t] + discharge + emergency[t] - load_real_kwh[t] - charge - spill[t]

    return {
        "grid": grid,
        "charge": actual_charge,
        "discharge": actual_discharge,
        "emergency": emergency,
        "actual_spill": spill,
        "soc": soc,
        "balance_residual_max": float(np.abs(balance_residual).max(initial=0.0)),
        "shortage_max": float(emergency.max(initial=0.0)),
    }
