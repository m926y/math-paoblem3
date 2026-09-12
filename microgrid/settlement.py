"""唯一费用定义。电量为kWh，价格为元/kWh。"""
from __future__ import annotations

import numpy as np


def settle(price, grid, emergency, reference=None):
    price, grid, emergency = (np.asarray(x, float) for x in (price, grid, emergency))
    if not (price.shape == grid.shape == emergency.shape):
        raise ValueError('计费数组形状不一致')
    if any(not np.isfinite(x).all() or np.any(x < -1e-7) for x in (price, grid, emergency)):
        raise ValueError('计费输入须为有限非负数')
    delta = np.zeros_like(grid) if reference is None else grid - np.asarray(reference, float)
    if not np.isfinite(delta).all() or delta.shape != grid.shape:
        raise ValueError('参考计划无效')
    increase, decrease = np.maximum(delta, 0), np.maximum(-delta, 0)
    normal = float(np.sum(price * grid))
    adjustment = float(.5 * np.sum(price * (increase + decrease)))
    urgent = float(5 * np.sum(price * emergency))
    return dict(normal_cost_cny=normal, adjustment_cost_cny=adjustment,
                emergency_cost_cny=urgent, total_cost_cny=normal+adjustment+urgent,
                increase_energy_kwh=float(increase.sum()), decrease_energy_kwh=float(decrease.sum()),
                increase_adjustment_cost_cny=float(.5*np.sum(price*increase)),
                decrease_adjustment_cost_cny=float(.5*np.sum(price*decrease)))


def revision_fee(price, previous, revised):
    """逐次调整收费的独立敏感性口径，不替换主口径。"""
    return float(.5*np.dot(np.asarray(price), np.abs(np.asarray(revised)-previous)))
