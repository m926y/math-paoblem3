"""问题2基线与严格因果版本的共享年度仿真。"""

from __future__ import annotations

from datetime import date

import numpy as np

from dispatch import DT_H, EMAX, EMIN, simulate_receding_plan, solve_day_dispatch
from forecast import (
    safe_forecast,
    safe_forecast_expanding,
    safe_weekday_forecast,
    weekday_prediction_series,
)


HISTORY_DAYS = 31
WINDOW = 7
EMERGENCY_MULTIPLIER = 5.0
RESERVE_FRACTION = 0.20


def reserve_target_soc(price: np.ndarray, load_safe_kw: np.ndarray, pv_safe_kw: np.ndarray) -> float:
    net_need = np.maximum(load_safe_kw - pv_safe_kw, 0.0) * DT_H
    expensive = price >= np.quantile(price, 0.75)
    target = EMIN + RESERVE_FRACTION * float(np.sum(net_need[expensive]))
    return float(np.clip(target, EMIN, EMAX))


def _run_day(price: np.ndarray, load_real_kwh: np.ndarray, pv_real_kwh: np.ndarray,
             initial_soc: float, load_safe_kw: np.ndarray, pv_safe_kw: np.ndarray) -> dict:
    plan = solve_day_dispatch(
        price, load_safe_kw * DT_H, pv_safe_kw * DT_H,
        initial_soc, terminal_energy=None,
    )
    simulation = simulate_receding_plan(plan, load_real_kwh, pv_real_kwh, initial_soc)
    grid = np.asarray(plan["grid_plan"])
    charge = np.asarray(simulation["charge"])
    discharge = np.asarray(simulation["discharge"])
    emergency = np.asarray(simulation["emergency"])
    spill = np.asarray(simulation["actual_spill"])
    soc = np.asarray(simulation["soc"])
    recourse = float(
        np.sum(np.abs(charge - np.asarray(plan["charge"])))
        + np.sum(np.abs(discharge - np.asarray(plan["discharge"])))
    )
    return {
        "grid": grid,
        "charge": charge,
        "discharge": discharge,
        "soc": soc,
        "emergency": emergency,
        "spill": spill,
        "plan_charge": np.asarray(plan["charge"]),
        "plan_discharge": np.asarray(plan["discharge"]),
        "normal_cost": float(np.sum(price * grid)),
        "emergency_cost": float(np.sum(EMERGENCY_MULTIPLIER * price * emergency)),
        "recourse_energy": recourse,
        "balance_residual_max": float(simulation["balance_residual_max"]),
    }


def _zero_cold_start(price: np.ndarray, load_real_kwh: np.ndarray, pv_real_kwh: np.ndarray,
                     initial_soc: float) -> dict:
    zeros = np.zeros_like(price, dtype=float)
    plan = {"grid_plan": zeros, "charge": zeros, "discharge": zeros}
    simulation = simulate_receding_plan(plan, load_real_kwh, pv_real_kwh, initial_soc)
    return {
        "grid": zeros.copy(),
        "charge": np.asarray(simulation["charge"]),
        "discharge": np.asarray(simulation["discharge"]),
        "soc": np.asarray(simulation["soc"]),
        "emergency": np.asarray(simulation["emergency"]),
        "spill": np.asarray(simulation["actual_spill"]),
        "plan_charge": zeros.copy(),
        "plan_discharge": zeros.copy(),
        "normal_cost": 0.0,
        "emergency_cost": float(np.sum(EMERGENCY_MULTIPLIER * price * simulation["emergency"])),
        "recourse_energy": float(np.sum(simulation["charge"]) + np.sum(simulation["discharge"])),
        "balance_residual_max": float(simulation["balance_residual_max"]),
    }


def simulate_legacy(price: np.ndarray, load_kw: np.ndarray, pv_kw: np.ndarray) -> tuple[list[dict], dict]:
    """保持工程修复前数学逻辑，作为不可混写的基线。"""
    load_kwh, pv_kwh = load_kw * DT_H, pv_kw * DT_H
    actual_kw = np.stack([load_kw, pv_kw])
    energy = 6000.0
    january_cost = 0.0
    for day in range(HISTORY_DAYS):
        if day + 1 >= WINDOW:
            _, _, next_load, next_pv = safe_forecast(actual_kw, day + 1, 0.8, WINDOW)
        else:
            next_load, next_pv = load_kw[day], pv_kw[day]
        terminal = reserve_target_soc(price, next_load, next_pv)
        plan = solve_day_dispatch(
            price, load_kwh[day], pv_kwh[day], energy, terminal_energy=terminal
        )
        january_cost += float(np.sum(price * np.asarray(plan["grid_plan"])))
        energy = float(np.asarray(plan["soc"])[-1])
    february_initial = energy

    records = []
    for day in range(HISTORY_DAYS, 365):
        _, _, load_safe, pv_safe = safe_forecast(actual_kw, day, 0.8, WINDOW)
        record = _run_day(price, load_kwh[day], pv_kwh[day], energy, load_safe, pv_safe)
        records.append(record)
        energy = float(record["soc"][-1])
    checks = {
        "model_version": "engineering_baseline_legacy_january",
        "january_initial_soc_kwh": 6000.0,
        "february_initial_soc_kwh": february_initial,
        "january_warmup_cost_cny": january_cost,
        "january_reserve_fraction": RESERVE_FRACTION,
    }
    return records, checks


def simulate_causal(price: np.ndarray, baseline_load_kw: np.ndarray, baseline_pv_kw: np.ndarray,
                    load_kw: np.ndarray, pv_kw: np.ndarray,
                    cold_start: str) -> tuple[list[dict], dict]:
    """从1月1日起严格按可获得信息运行，返回2月至12月记录。"""
    if cold_start not in {"zero", "baseline"}:
        raise ValueError("cold_start必须为zero或baseline")
    load_kwh, pv_kwh = load_kw * DT_H, pv_kw * DT_H
    actual_kw = np.stack([load_kw, pv_kw])
    energy = 6000.0
    all_records: list[dict] = []

    if cold_start == "zero":
        first = _zero_cold_start(price, load_kwh[0], pv_kwh[0], energy)
    else:
        first = _run_day(
            price, load_kwh[0], pv_kwh[0], energy,
            np.asarray(baseline_load_kw), np.asarray(baseline_pv_kw),
        )
    all_records.append(first)
    energy = float(first["soc"][-1])

    for day in range(1, 365):
        if day < HISTORY_DAYS:
            _, _, load_safe, pv_safe = safe_forecast_expanding(actual_kw, day, 0.8, WINDOW)
        else:
            _, _, load_safe, pv_safe = safe_forecast(actual_kw, day, 0.8, WINDOW)
        record = _run_day(price, load_kwh[day], pv_kwh[day], energy, load_safe, pv_safe)
        all_records.append(record)
        energy = float(record["soc"][-1])

    january = all_records[:HISTORY_DAYS]
    checks = {
        "model_version": f"causal_january_{cold_start}_cold_start",
        "cold_start": cold_start,
        "january_initial_soc_kwh": 6000.0,
        "february_initial_soc_kwh": float(january[-1]["soc"][-1]),
        "january_warmup_cost_cny": float(sum(r["normal_cost"] + r["emergency_cost"] for r in january)),
    }
    return all_records[HISTORY_DAYS:], checks


def simulate_weekday_optimized(
    price: np.ndarray,
    baseline_load_kw: np.ndarray,
    baseline_pv_kw: np.ndarray,
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    cold_start: str = "zero",
    alpha: float = 0.90,
    safety_mode: str = "net",
) -> tuple[list[dict], dict]:
    """严格因果同星期预测版本，默认采用留出实验选出的0.90分位数。"""
    if cold_start not in {"zero", "baseline"}:
        raise ValueError("cold_start必须为zero或baseline")
    if safety_mode not in {"net", "separate"}:
        raise ValueError("safety_mode必须为net或separate")
    if not 0.5 <= alpha < 1.0:
        raise ValueError("alpha必须位于[0.5,1)内")

    load_kw = np.asarray(load_kw, dtype=float)
    pv_kw = np.asarray(pv_kw, dtype=float)
    if load_kw.shape != (365, 144) or pv_kw.shape != (365, 144):
        raise ValueError("负荷和光伏必须为365×144")
    load_kwh, pv_kwh = load_kw * DT_H, pv_kw * DT_H
    actual_kw = np.stack([load_kw, pv_kw])
    predictions_kw = np.stack([
        weekday_prediction_series(load_kw),
        weekday_prediction_series(pv_kw),
    ])

    energy = 6000.0
    all_records: list[dict] = []
    if cold_start == "zero":
        first = _zero_cold_start(price, load_kwh[0], pv_kwh[0], energy)
    else:
        first = _run_day(
            price,
            load_kwh[0],
            pv_kwh[0],
            energy,
            np.asarray(baseline_load_kw),
            np.asarray(baseline_pv_kw),
        )
    all_records.append(first)
    energy = float(first["soc"][-1])

    for day in range(1, 365):
        _, _, load_safe, pv_safe = safe_weekday_forecast(
            actual_kw,
            predictions_kw,
            day,
            alpha=alpha,
            safety_mode=safety_mode,
        )
        record = _run_day(
            price,
            load_kwh[day],
            pv_kwh[day],
            energy,
            load_safe,
            pv_safe,
        )
        all_records.append(record)
        energy = float(record["soc"][-1])

    january = all_records[:HISTORY_DAYS]
    checks = {
        "model_version": "causal_weekday_four_lag_forecast",
        "cold_start": cold_start,
        "forecast_lags_days": [7, 14, 21, 28],
        "forecast_weights": [0.50, 0.25, 0.15, 0.10],
        "safety_mode": safety_mode,
        "safety_quantile": alpha,
        "january_initial_soc_kwh": 6000.0,
        "february_initial_soc_kwh": float(january[-1]["soc"][-1]),
        "january_warmup_cost_cny": float(
            sum(r["normal_cost"] + r["emergency_cost"] for r in january)
        ),
    }
    return all_records[HISTORY_DAYS:], checks


def summarize(records: list[dict], checks: dict) -> dict:
    soc = np.concatenate([np.asarray(record["soc"]) for record in records])
    result = {
        **checks,
        "output_dates": ["2025-02-01", "2025-12-31"],
        "output_days": len(records),
        "normal_energy_kwh": float(sum(np.sum(r["grid"]) for r in records)),
        "normal_cost_cny": float(sum(r["normal_cost"] for r in records)),
        "emergency_energy_kwh": float(sum(np.sum(r["emergency"]) for r in records)),
        "emergency_cost_cny": float(sum(r["emergency_cost"] for r in records)),
        "spill_energy_kwh": float(sum(np.sum(r["spill"]) for r in records)),
        "actual_charge_energy_kwh": float(sum(np.sum(r["charge"]) for r in records)),
        "actual_discharge_energy_kwh": float(sum(np.sum(r["discharge"]) for r in records)),
        "actual_recourse_energy_kwh": float(sum(r["recourse_energy"] for r in records)),
        "soc_min_kwh": float(np.min(soc)),
        "soc_max_kwh": float(np.max(soc)),
        "max_balance_residual_kwh": float(max(r["balance_residual_max"] for r in records)),
        "max_simultaneous_charge_discharge_kwh": float(max(
            np.max(np.minimum(r["charge"], r["discharge"])) for r in records
        )),
    }
    result["total_cost_cny"] = result["normal_cost_cny"] + result["emergency_cost_cny"]
    return result
