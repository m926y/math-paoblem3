"""问题2的历史滚动预测和单侧安全裕度。"""

from __future__ import annotations

import numpy as np


WEIGHTS = np.array([0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02], dtype=float)
WEEKDAY_LAGS = (7, 14, 21, 28)
WEEKDAY_WEIGHTS = np.array([0.50, 0.25, 0.15, 0.10], dtype=float)


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


def weighted_forecast_expanding(history_kw: np.ndarray, target_day: int, window: int = 7) -> np.ndarray:
    """年初使用所有可用历史，达到window天后退化为固定七日预测。"""
    history_kw = np.asarray(history_kw, dtype=float)
    if history_kw.ndim != 2 or history_kw.shape[1] != 144:
        raise ValueError("历史数据必须为天数×144的二维数组")
    n = min(window, target_day)
    if n <= 0:
        raise ValueError("目标日前没有可用历史；第1天必须显式指定冷启动")
    weights = WEIGHTS[:n].copy()
    weights /= weights.sum()
    recent = history_kw[target_day - n:target_day][::-1]
    return np.tensordot(weights, recent, axes=(0, 0))


def safe_forecast_expanding(actual_kw: np.ndarray, target_day: int,
                            alpha: float = 0.8, window: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """严格使用目标日前数据的扩展窗口预测，供1月因果初始化使用。"""
    actual_kw = np.asarray(actual_kw, dtype=float)
    if actual_kw.shape[0] != 2 or actual_kw.ndim != 3 or actual_kw.shape[2] != 144:
        raise ValueError("actual_kw必须为2×天数×144")
    if target_day <= 0:
        raise ValueError("第1天没有历史，应由调用方选择冷启动方案")
    load_actual, pv_actual = actual_kw
    load_pred = weighted_forecast_expanding(load_actual, target_day, window)
    pv_pred = weighted_forecast_expanding(pv_actual, target_day, window)
    load_errors, pv_errors = [], []
    for past_day in range(1, target_day):
        past_load = weighted_forecast_expanding(load_actual, past_day, window)
        past_pv = weighted_forecast_expanding(pv_actual, past_day, window)
        load_errors.append(np.maximum(load_actual[past_day] - past_load, 0.0))
        pv_errors.append(np.maximum(past_pv - pv_actual[past_day], 0.0))
    if load_errors:
        load_margin = np.quantile(np.asarray(load_errors), alpha, axis=0)
        pv_margin = np.quantile(np.asarray(pv_errors), alpha, axis=0)
    else:
        load_margin = np.zeros(144)
        pv_margin = np.zeros(144)
    return (
        load_pred,
        pv_pred,
        np.maximum(load_pred + load_margin, 0.0),
        np.maximum(pv_pred - pv_margin, 0.0),
    )


def weekday_weighted_forecast(history_kw: np.ndarray, target_day: int) -> np.ndarray:
    """使用前1—4个同星期日预测；历史不足7天时退回扩展历史均值。"""
    history_kw = np.asarray(history_kw, dtype=float)
    if history_kw.ndim != 2 or history_kw.shape[1] != 144:
        raise ValueError("历史数据必须为天数×144的二维数组")
    if not 0 < target_day < len(history_kw):
        raise ValueError("target_day必须有至少1天历史且位于数据范围内")
    available = [
        (lag, weight)
        for lag, weight in zip(WEEKDAY_LAGS, WEEKDAY_WEIGHTS)
        if target_day - lag >= 0
    ]
    if not available:
        return np.mean(history_kw[:target_day], axis=0)
    total_weight = sum(weight for _, weight in available)
    return sum(
        weight * history_kw[target_day - lag]
        for lag, weight in available
    ) / total_weight


def weekday_prediction_series(history_kw: np.ndarray) -> np.ndarray:
    """为全年逐日建立严格因果的同星期预测序列。第1天由冷启动处理。"""
    history_kw = np.asarray(history_kw, dtype=float)
    if history_kw.ndim != 2 or history_kw.shape[1] != 144:
        raise ValueError("历史数据必须为天数×144的二维数组")
    predictions = np.zeros_like(history_kw, dtype=float)
    for target_day in range(1, len(history_kw)):
        predictions[target_day] = weekday_weighted_forecast(history_kw, target_day)
    return predictions


def safe_weekday_forecast(
    actual_kw: np.ndarray,
    predictions_kw: np.ndarray,
    target_day: int,
    alpha: float = 0.90,
    safety_mode: str = "net",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用目标日前误差构造同星期安全预测。

    safety_mode="net"时直接对净负荷低估误差取分位数；
    safety_mode="separate"时分别对负荷低估和光伏高估取分位数。
    """
    actual_kw = np.asarray(actual_kw, dtype=float)
    predictions_kw = np.asarray(predictions_kw, dtype=float)
    if actual_kw.shape != predictions_kw.shape or actual_kw.ndim != 3:
        raise ValueError("actual_kw和predictions_kw必须同为2×天数×144")
    if actual_kw.shape[0] != 2 or actual_kw.shape[2] != 144:
        raise ValueError("actual_kw和predictions_kw必须同为2×天数×144")
    if not 0.5 <= alpha < 1.0:
        raise ValueError("alpha必须位于[0.5,1)内")
    if not 0 < target_day < actual_kw.shape[1]:
        raise ValueError("target_day必须有至少1天历史且位于数据范围内")
    if safety_mode not in {"net", "separate"}:
        raise ValueError("safety_mode必须为net或separate")

    load_actual, pv_actual = actual_kw
    load_pred, pv_pred = predictions_kw
    history = slice(1, target_day)

    if safety_mode == "net":
        net_pred = load_pred[target_day] - pv_pred[target_day]
        if target_day == 1:
            net_margin = np.zeros(144)
        else:
            net_errors = (
                load_actual[history] - pv_actual[history]
                - load_pred[history] + pv_pred[history]
            )
            net_margin = np.quantile(np.maximum(net_errors, 0.0), alpha, axis=0)
        safe_net = net_pred + net_margin
        load_safe = np.maximum(safe_net, 0.0)
        pv_safe = np.maximum(-safe_net, 0.0)
    else:
        if target_day == 1:
            load_margin = np.zeros(144)
            pv_margin = np.zeros(144)
        else:
            load_margin = np.quantile(
                np.maximum(load_actual[history] - load_pred[history], 0.0),
                alpha,
                axis=0,
            )
            pv_margin = np.quantile(
                np.maximum(pv_pred[history] - pv_actual[history], 0.0),
                alpha,
                axis=0,
            )
        load_safe = np.maximum(load_pred[target_day] + load_margin, 0.0)
        pv_safe = np.maximum(pv_pred[target_day] - pv_margin, 0.0)

    return load_pred[target_day], pv_pred[target_day], load_safe, pv_safe
