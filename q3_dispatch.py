"""第三问共享调度模型：24小时购电MPC与十分钟级储能闭环执行。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class DispatchParams:
    """全部能量单位均为kWh。"""

    eta: float = 0.90
    e_min: float = 1200.0
    e_max: float = 10800.0
    e_initial: float = 6000.0
    power_max_kw: float = 5000.0
    dt_h: float = 1.0 / 6.0
    cycle_tiebreak: float = 1e-6
    soc_guard: float = 1e-6
    feasibility_tol: float = 1e-8

    @property
    def q_max(self) -> float:
        return self.power_max_kw * self.dt_h

    @property
    def operating_min(self) -> float:
        return self.e_min + self.soc_guard

    @property
    def operating_max(self) -> float:
        return self.e_max - self.soc_guard


def _validate_series(*arrays: np.ndarray) -> int:
    arrays = tuple(np.asarray(a, dtype=float) for a in arrays)
    if not arrays:
        raise ValueError("至少需要一个数组")
    n = len(arrays[0])
    if any(a.ndim != 1 or len(a) != n for a in arrays):
        raise ValueError("输入序列必须是一维且长度一致")
    if any(np.any(~np.isfinite(a)) for a in arrays):
        raise ValueError("输入序列包含NaN或无穷值")
    return n


def solve_mpc(
    price: np.ndarray,
    load_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    initial_soc: float,
    params: DispatchParams,
    plan_reference: np.ndarray | None = None,
    terminal_value_per_kwh: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """求解未来24小时线性规划。

    plan_reference为有限值的位置相对0点计划结算调整费用；NaN位置代表
    尚未形成次日正式计划，只承担正常购电费用。
    """

    price = np.asarray(price, dtype=float)
    load_kwh = np.asarray(load_kwh, dtype=float)
    pv_kwh = np.asarray(pv_kwh, dtype=float)
    n = _validate_series(price, load_kwh, pv_kwh)
    if np.any(price < 0) or np.any(load_kwh < 0) or np.any(pv_kwh < 0):
        raise ValueError("电价、负荷和光伏均不能为负")
    if not params.e_min - params.feasibility_tol <= initial_soc <= params.e_max + params.feasibility_tol:
        raise ValueError(f"初始SOC越界：{initial_soc}")
    if not np.isfinite(terminal_value_per_kwh) or terminal_value_per_kwh < 0:
        raise ValueError("terminal_value_per_kwh必须为非负有限数")

    if plan_reference is None:
        reference = np.full(n, np.nan)
    else:
        reference = np.asarray(plan_reference, dtype=float)
        if reference.shape != (n,):
            raise ValueError("plan_reference长度错误")
    adjusted = np.isfinite(reference)

    # x=[g,c,r,spill,E,u,v]
    g0, c0, r0, s0, e0, u0, v0 = [i * n for i in range(7)]
    n_var = 7 * n
    n_eq = 2 * n + int(adjusted.sum())
    a_eq = lil_matrix((n_eq, n_var), dtype=float)
    b_eq = np.zeros(n_eq)
    row = 0

    for t in range(n):
        # g+pv+r=load+c+spill
        a_eq[row, g0 + t] = 1.0
        a_eq[row, r0 + t] = 1.0
        a_eq[row, c0 + t] = -1.0
        a_eq[row, s0 + t] = -1.0
        b_eq[row] = load_kwh[t] - pv_kwh[t]
        row += 1

    for t in range(n):
        # E_t=E_(t-1)+eta*c-r/eta
        a_eq[row, e0 + t] = 1.0
        a_eq[row, c0 + t] = -params.eta
        a_eq[row, r0 + t] = 1.0 / params.eta
        if t == 0:
            b_eq[row] = float(initial_soc)
        else:
            a_eq[row, e0 + t - 1] = -1.0
        row += 1

    for t in np.flatnonzero(adjusted):
        # g-g0=u-v
        a_eq[row, g0 + t] = 1.0
        a_eq[row, u0 + t] = -1.0
        a_eq[row, v0 + t] = 1.0
        b_eq[row] = reference[t]
        row += 1

    objective = np.zeros(n_var)
    objective[g0:g0+n] = price
    objective[c0:c0+n] = params.cycle_tiebreak
    objective[r0:r0+n] = params.cycle_tiebreak
    objective[u0:u0+n] = 0.5 * price
    objective[v0:v0+n] = 0.5 * price
    # 预测窗口末端剩余电量的机会价值；为0时严格保持原模型。
    objective[e0 + n - 1] -= terminal_value_per_kwh

    bounds: list[tuple[float, float | None]] = []
    bounds.extend([(0.0, None)] * n)
    bounds.extend([(0.0, params.q_max)] * n)
    bounds.extend([(0.0, params.q_max)] * n)
    bounds.extend([(0.0, None)] * n)
    bounds.extend([(params.operating_min, params.operating_max)] * n)
    bounds.extend([(0.0, None) if adjusted[t] else (0.0, 0.0) for t in range(n)])
    bounds.extend([(0.0, None) if adjusted[t] else (0.0, 0.0) for t in range(n)])

    result = linprog(
        objective,
        A_eq=a_eq.tocsr(),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
            "ipm_optimality_tolerance": 1e-10,
        },
    )
    if not result.success:
        raise RuntimeError(f"MPC求解失败：{result.message}")

    x = result.x
    soc = x[e0:e0+n]
    if soc.min() < params.e_min - params.feasibility_tol or soc.max() > params.e_max + params.feasibility_tol:
        raise RuntimeError(f"MPC返回真实SOC越界：min={soc.min()}, max={soc.max()}")
    soc = np.clip(soc, params.e_min, params.e_max)

    balance = x[g0:g0+n] + pv_kwh + x[r0:r0+n] - load_kwh - x[c0:c0+n] - x[s0:s0+n]
    prev = np.r_[initial_soc, soc[:-1]]
    soc_residual = soc - prev - params.eta * x[c0:c0+n] + x[r0:r0+n] / params.eta
    return {
        "grid": x[g0:g0+n],
        "charge_plan": x[c0:c0+n],
        "discharge_plan": x[r0:r0+n],
        "spill_plan": x[s0:s0+n],
        "soc_plan": soc,
        "increase": x[u0:u0+n],
        "decrease": x[v0:v0+n],
        "objective": float(result.fun),
        "terminal_value_per_kwh": float(terminal_value_per_kwh),
        "balance_residual_max": float(np.max(np.abs(balance))),
        "soc_residual_max": float(np.max(np.abs(soc_residual))),
    }


def execute_causal_block(
    grid: np.ndarray,
    charge_plan: np.ndarray,
    discharge_plan: np.ndarray,
    load_actual_kwh: np.ndarray,
    pv_actual_kwh: np.ndarray,
    initial_soc: float,
    params: DispatchParams,
) -> dict[str, np.ndarray | float]:
    """按当前十分钟真实测量值闭环修正储能，不读取后续实际数据。

    缺电时先取消充电、再增加放电；过剩时先取消放电、再增加充电；
    剩余缺口才紧急购电。正常购电grid不在块内改变。
    """

    grid = np.asarray(grid, dtype=float)
    charge_plan = np.asarray(charge_plan, dtype=float)
    discharge_plan = np.asarray(discharge_plan, dtype=float)
    load_actual_kwh = np.asarray(load_actual_kwh, dtype=float)
    pv_actual_kwh = np.asarray(pv_actual_kwh, dtype=float)
    n = _validate_series(grid, charge_plan, discharge_plan, load_actual_kwh, pv_actual_kwh)

    charge = np.zeros(n)
    discharge = np.zeros(n)
    emergency = np.zeros(n)
    spill = np.zeros(n)
    soc = np.empty(n + 1)
    soc[0] = initial_soc
    recourse = np.zeros(n)

    for t in range(n):
        energy = float(soc[t])
        c_planned = float(charge_plan[t])
        r_planned = float(discharge_plan[t])
        c = min(c_planned, params.q_max, max((params.operating_max - energy) / params.eta, 0.0))
        r = min(r_planned, params.q_max, max((energy - params.operating_min) * params.eta, 0.0))
        residual = float(load_actual_kwh[t] + c - pv_actual_kwh[t] - r - grid[t])

        if residual > 0.0:
            cancelled = min(c, residual)
            c -= cancelled
            residual -= cancelled
            energy_after = energy + params.eta * c - r / params.eta
            extra = min(
                params.q_max - r,
                residual,
                max((energy_after - params.operating_min) * params.eta, 0.0),
            )
            r += extra
            residual -= extra
        elif residual < 0.0:
            surplus = -residual
            cancelled = min(r, surplus)
            r -= cancelled
            surplus -= cancelled
            extra = min(
                params.q_max - c,
                surplus,
                max((params.operating_max - energy - params.eta * c + r / params.eta) / params.eta, 0.0),
            )
            c += extra
            surplus -= extra
            residual = -surplus

        next_soc = energy + params.eta * c - r / params.eta
        if next_soc < params.e_min - params.feasibility_tol or next_soc > params.e_max + params.feasibility_tol:
            raise RuntimeError(f"实际执行SOC越界：t={t}, SOC={next_soc}")
        if next_soc < params.operating_min:
            next_soc = params.operating_min
        elif next_soc > params.operating_max:
            next_soc = params.operating_max

        charge[t] = c
        discharge[t] = r
        emergency[t] = max(residual, 0.0)
        spill[t] = max(-residual, 0.0)
        soc[t + 1] = next_soc
        recourse[t] = abs(c - c_planned) + abs(r - r_planned)

    balance = grid + pv_actual_kwh + discharge + emergency - load_actual_kwh - charge - spill
    soc_residual = soc[1:] - soc[:-1] - params.eta * charge + discharge / params.eta
    return {
        "charge": charge,
        "discharge": discharge,
        "emergency": emergency,
        "spill": spill,
        "soc": soc,
        "recourse": recourse,
        "balance_residual_max": float(np.max(np.abs(balance))),
        "soc_residual_max": float(np.max(np.abs(soc_residual))),
    }
