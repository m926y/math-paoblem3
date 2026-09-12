"""第三问预测模块：历史负荷预测、光伏插值和因果误差校准。"""

from __future__ import annotations

import numpy as np


DT_H = 1.0 / 6.0
T = 144
ISSUE_INDEX = (0, 36, 72, 108)
WEIGHTS = np.array([0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02], dtype=float)


def repeat_profile(profile: np.ndarray, start: int, length: int = T) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    return np.array([profile[(start + i) % len(profile)] for i in range(length)], dtype=float)


def weighted_past_profile(values_kw: np.ndarray, day: int, fallback: np.ndarray) -> np.ndarray:
    """只使用当前日期以前已经完整观测的负荷。"""

    n = min(len(WEIGHTS), day)
    if n == 0:
        return np.asarray(fallback, dtype=float).copy()
    weights = WEIGHTS[:n].copy()
    weights /= weights.sum()
    indices = day - np.arange(1, n + 1)
    return np.sum(values_kw[indices] * weights[:, None], axis=0)


def build_load_unfavourable_errors(load_kw: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    errors = np.zeros_like(load_kw, dtype=float)
    for day in range(len(load_kw)):
        prediction = weighted_past_profile(load_kw, day, fallback)
        errors[day] = np.maximum(load_kw[day] - prediction, 0.0)
    return errors


def _shrunken_quantile(
    local: np.ndarray,
    global_samples: np.ndarray,
    quantile: float,
    shrink_k: float,
) -> float:
    local = np.asarray(local, dtype=float)
    global_samples = np.asarray(global_samples, dtype=float)
    local = local[np.isfinite(local)]
    global_samples = global_samples[np.isfinite(global_samples)]
    if len(global_samples) == 0:
        return 0.0
    global_q = float(np.quantile(global_samples, quantile))
    if len(local) == 0:
        return global_q
    local_q = float(np.quantile(local, quantile))
    weight = len(local) / (len(local) + shrink_k)
    return weight * local_q + (1.0 - weight) * global_q


def load_profiles(
    day: int,
    load_kw: np.ndarray,
    fallback: np.ndarray,
    errors: np.ndarray,
    quantile: float,
    shrink_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """返回原始负荷预测和带单侧安全裕度的负荷预测。"""

    raw = weighted_past_profile(load_kw, day, fallback)
    if day == 0:
        return np.maximum(raw, 0.0), np.maximum(raw, 0.0)
    history = errors[:day]
    global_samples = history.ravel()
    margin = np.array([
        _shrunken_quantile(history[:, t], global_samples, quantile, shrink_k)
        for t in range(T)
    ])
    return np.maximum(raw, 0.0), np.maximum(raw + margin, 0.0)


def corrected_load_horizon(
    day: int,
    start: int,
    raw_profile_kw: np.ndarray,
    safe_profile_kw: np.ndarray,
    actual_load_kw: np.ndarray,
    decay_hours: float = 6.0,
    bias_clip_fraction: float = 0.15,
) -> tuple[np.ndarray, float]:
    """利用发布时间前最近1小时实测负荷做衰减偏差修正。

    只读取区间[0,start)内已经发生的数据；start=0时不修正，因而不存在
    当前日未来信息泄漏。修正叠加在安全预测上，并随预测提前量指数衰减。
    """

    horizon = repeat_profile(safe_profile_kw, start)
    if start == 0:
        return horizon, 0.0
    left = max(0, start - 6)
    observed_error = actual_load_kw[day, left:start] - raw_profile_kw[left:start]
    bias = float(np.median(observed_error)) if len(observed_error) else 0.0
    scale = max(float(np.median(raw_profile_kw)), 1.0)
    limit = bias_clip_fraction * scale
    bias = float(np.clip(bias, -limit, limit))
    lead_hours = np.arange(1, T + 1) * DT_H
    decay = np.exp(-lead_hours / max(decay_hours, DT_H))
    return np.maximum(horizon + bias * decay, 0.0), bias


def build_pv_hourly_errors(pv_kw: np.ndarray, forecasts_kw: np.ndarray) -> np.ndarray:
    """按发布日期、发布时间、提前小时生成历史光伏高估误差。"""

    days = len(pv_kw)
    actual_flat = pv_kw.ravel()
    errors = np.full((days, 4, 24), np.nan)
    for day in range(days):
        for issue_no, issue_index in enumerate(ISSUE_INDEX):
            boundary = day * T + issue_index
            for lead_hour in range(24):
                target_end = boundary + 6 * (lead_hour + 1) - 1
                if target_end < len(actual_flat):
                    errors[day, issue_no, lead_hour] = max(
                        forecasts_kw[day, issue_no, lead_hour] - actual_flat[target_end],
                        0.0,
                    )
    return errors


def pv_safe_horizon(
    day: int,
    issue_no: int,
    pv_actual_kw: np.ndarray,
    forecasts_kw: np.ndarray,
    errors: np.ndarray,
    quantile: float,
    shrink_k: float,
) -> np.ndarray:
    """把24个整点预报插值为144个十分钟预测，并减去历史单侧裕度。"""

    issue_index = ISSUE_INDEX[issue_no]
    boundary = day * T + issue_index
    actual_flat = pv_actual_kw.ravel()
    anchor = actual_flat[boundary - 1] if boundary > 0 else 0.0
    node_minutes = np.arange(25) * 60
    node_values = np.r_[anchor, forecasts_kw[day, issue_no]]
    lead_minutes = np.arange(1, T + 1) * 10
    interpolated = np.interp(lead_minutes, node_minutes, node_values)

    history = errors[:day, issue_no, :]
    global_samples = history.ravel()
    margins = np.array([
        _shrunken_quantile(
            history[:, h] if day else np.array([]),
            global_samples,
            quantile,
            shrink_k,
        )
        for h in range(24)
    ])
    groups = np.minimum((lead_minutes - 1) // 60, 23).astype(int)
    return np.maximum(interpolated - margins[groups], 0.0)
