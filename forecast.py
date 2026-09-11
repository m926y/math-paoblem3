"""问题2的历史滚动预测和单侧安全裕度。"""

from __future__ import annotations

import numpy as np


WEIGHTS = np.array([0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02], dtype=float)


def weighted_forecast(history_kw: np.ndarray, target_day: int, window: int = 7) -> np.ndarray:
    """使用目标日前 window 天的同一时刻实际值预测目标日。"""
    if window != len(WEIGHTS):
        raise ValueError("当前实现的固定权重长度为 7")
    if target_day < window:
        raise ValueError("目标日前历史数据不足 7 天")
    recent = history_kw[target_day - window:target_day][::-1]
    return np.tensordot(WEIGHTS, recent, axes=(0, 0))


def safe_forecast(actual_kw: np.ndarray, target_day: int, alpha: float = 0.8, window: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """同时返回原始预测、负荷安全预测、光伏安全预测和两组裕度。"""
    load_actual = actual_kw[0]
    pv_actual = actual_kw[1]
    load_pred = weighted_forecast(load_actual, target_day, window)
    pv_pred = weighted_forecast(pv_actual, target_day, window)

    load_errors = []
    pv_errors = []
    for past_day in range(window, target_day):
        past_load_pred = weighted_forecast(load_actual, past_day, window)
        past_pv_pred = weighted_forecast(pv_actual, past_day, window)
        load_errors.append(np.maximum(load_actual[past_day] - past_load_pred, 0.0))
        pv_errors.append(np.maximum(past_pv_pred - pv_actual[past_day], 0.0))

    if load_errors:
        load_margin = np.quantile(np.asarray(load_errors), alpha, axis=0)
        pv_margin = np.quantile(np.asarray(pv_errors), alpha, axis=0)
    else:
        load_margin = np.zeros(load_pred.shape)
        pv_margin = np.zeros(pv_pred.shape)

    load_safe = np.maximum(load_pred + load_margin, 0.0)
    pv_safe = np.maximum(pv_pred - pv_margin, 0.0)
    return load_pred, pv_pred, load_safe, pv_safe
