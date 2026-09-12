from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    name: str = 'net_adaptive'
    forecast: str = 'adaptive'
    mode: str = 'net'
    quantile: float = .65
    history_days: int = 56
    half_life: float = 28
    scenarios: int = 5
    cvar_weight: float = 0
    cvar_beta: float = .9
    reserve_factor: float = 0
    reserve_price: float = .3
    update_hours: tuple = (0, 6, 12, 18)
    update_pv: bool = True
    update_load: bool = True
    trigger: bool = False
    lexicographic: bool = True

    def __post_init__(self):
        if self.forecast not in ('adaptive','weighted','persistence','weekly','ridge'):
            raise ValueError('未知预测方法')
        if self.mode not in ('net','separate','scenario') or not 0 < self.quantile < 1:
            raise ValueError('风险模型或分位数无效')
        if self.history_days < 1 or self.half_life <= 0 or self.scenarios < 1:
            raise ValueError('历史窗口和场景参数必须为正')
        if self.cvar_weight < 0 or not 0 < self.cvar_beta < 1:
            raise ValueError('CVaR参数无效')
        if tuple(sorted(set(self.update_hours))) != self.update_hours or 0 not in self.update_hours:
            raise ValueError('更新时间必须有序、唯一且包含0')
        if any(h not in (0,6,12,18) for h in self.update_hours):
            raise ValueError('购电只允许在0/6/12/18点更新')
        if self.reserve_factor < 0 or self.reserve_price < 0:
            raise ValueError('备用参数不能为负')

    def to_dict(self):
        return asdict(self)
