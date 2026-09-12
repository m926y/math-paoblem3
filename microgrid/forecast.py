"""按预测原点缓存的因果预测；跨日部分只用同一原点以前的实测。"""
from __future__ import annotations

from datetime import date, timedelta
import numpy as np
from .config import Config

WEIGHTS = np.array([.40,.25,.15,.10,.05,.03,.02])
METHODS = ('persistence','weekly','weighted','ridge')


def weighted_quantile(samples, quantile, weights):
    samples = np.asarray(samples, float)
    if samples.ndim != 2 or len(samples) != len(weights):
        raise ValueError('分位数样本形状错误')
    order = np.argsort(samples, axis=0, kind='stable')
    values = np.take_along_axis(samples, order, axis=0)
    w = np.asarray(weights, float)[order]
    cumulative = np.cumsum(w, axis=0)
    idx = np.argmax(cumulative >= quantile*cumulative[-1], axis=0)
    return values[idx, np.arange(samples.shape[1])]


class ForecastEngine:
    def __init__(self, load, pv, forecasts, fallback_load, fallback_pv):
        self.actual = {'load': np.asarray(load), 'pv': np.asarray(pv)}
        self.fallback = {'load': np.asarray(fallback_load,float), 'pv': np.asarray(fallback_pv,float)}
        self.forecasts = forecasts
        self.cache = {}
        self.model_cache = {}
        self.raw_cache = {}
        self.selection = {}

    @staticmethod
    def calendar(day):
        dt = date(2025,1,1)+timedelta(days=int(day))
        angle = 2*np.pi*dt.timetuple().tm_yday/365
        return (float(dt.weekday() >= 5), np.sin(angle), np.cos(angle))

    def choose(self, kind, origin):
        key = kind, origin
        if key not in self.selection:
            if origin < 21:
                best = 'weighted'
            else:
                days = range(max(7,origin-21), origin)
                scores = {m: np.mean([np.mean(np.abs(self.actual[kind][j]-self.profile(kind,j,j,m)))
                                      for j in days]) for m in METHODS}
                best = min(scores, key=scores.get)
            self.selection[key] = best
        return self.selection[key]

    def profile(self, kind, origin, target, method):
        if method == 'adaptive':
            method = self.choose(kind, origin)
        key = kind, origin, target, method
        if key in self.cache:
            return self.cache[key]
        if origin == 0:
            result = self.fallback[kind].copy()
        else:
            def past(day):
                if day < 0:
                    return self.fallback[kind]
                if day < origin:
                    return self.actual[kind][day]
                return self.profile(kind, origin, day, method)
            if method == 'persistence':
                result = past(target-1).copy()
            elif method == 'weekly':
                result = past(target-7 if target >= 7 else target-1).copy()
            elif method == 'weighted' or origin < 21:
                n = min(7,target)
                w = WEIGHTS[:n]/WEIGHTS[:n].sum()
                result = sum(w[i]*past(target-i-1) for i in range(n))
            else:
                model_key = kind, origin
                if model_key not in self.model_cache:
                    days = np.arange(max(7,origin-84),origin)
                    scale = max(float(np.median(self.actual[kind][:origin])),1000.)
                    x = np.array([self._features(kind,j,lambda k:self.actual[kind][k],scale) for j in days])
                    y = self.actual[kind][days]/scale
                    w = np.exp2((days-origin)/42)
                    gram = np.einsum('dtf,dtg,d->tfg',x,x,w)
                    rhs = np.einsum('dtf,dt,d->tf',x,y,w)
                    penalty = np.diag([1e-6,.5,.5,.5,.5,.5,.5])
                    coef = np.linalg.solve(gram+penalty, rhs[...,None])[...,0]
                    self.model_cache[model_key] = coef, scale
                coef, scale = self.model_cache[model_key]
                result = np.einsum('tf,tf->t',self._features(kind,target,past,scale),coef)*scale
        # 无外部装机容量数据；只施加非负约束，不凭未来全年的极值截断。
        result = np.maximum(result,0)
        self.cache[key] = result
        return result

    def _features(self, kind, target, past, scale):
        recent = sum(WEIGHTS[i]*past(target-i-1) for i in range(7))
        cal = self.calendar(target)
        return np.column_stack([np.ones(144), past(target-1)/scale,past(target-7)/scale,
                                recent/scale, *[np.full(144,x) for x in cal]])

    def raw(self, question, day, issue, cfg):
        key = (question,day,issue,cfg.forecast,cfg.update_pv,cfg.update_load)
        if key in self.raw_cache:
            return self.raw_cache[key]
        start = issue*6
        load_day = self.profile('load',day,day,cfg.forecast)
        if start:
            load_next = self.profile('load',day,day+1,cfg.forecast)
            load = np.r_[load_day[start:],load_next[:start]]
        else:
            load = load_day.copy()
        if question == 3 and issue and cfg.update_load:
            observed = self.actual['load'][day,start-6:start]
            bias = np.median(observed-load_day[start-6:start])
            limit = .15*max(float(np.median(load_day)),1.)
            load += np.clip(bias,-limit,limit)*np.exp(-np.arange(1,145)/36)
        if question == 2:
            pv = self.profile('pv',day,day,cfg.forecast).copy()
        else:
            source_issue = issue if cfg.update_pv else 0
            boundary = day*144+source_issue*6
            anchor = self.actual['pv'].ravel()[boundary-1] if boundary else 0.
            nodes = np.r_[anchor,self.forecasts[day,source_issue//6]]
            pv = np.interp(np.arange(1,145)*10,np.arange(25)*60,nodes)
            if not cfg.update_pv and start:
                pv = np.r_[pv[start:],pv[:start]]
        result = np.maximum(load,0), np.maximum(pv,0)
        self.raw_cache[key] = result
        return result

    def horizon(self, question, day, issue, cfg):
        load,pv = self.raw(question,day,issue,cfg)
        left = max(0,day-cfg.history_days)
        le,ve = [],[]
        for past in range(left,day):
            begin = past*144+issue*6
            end = begin+144
            # 只有目标已观测完毕的历史预测才可校准。
            if end > day*144+issue*6:
                continue
            pl,pv_pred = self.raw(question,past,issue,cfg)
            le.append(self.actual['load'].ravel()[begin:end]-pl)
            ve.append(self.actual['pv'].ravel()[begin:end]-pv_pred)
        le = np.asarray(le) if le else np.zeros((1,144))
        ve = np.asarray(ve) if ve else np.zeros((1,144))
        errors = le-ve
        weights = np.exp2((np.arange(len(errors))-len(errors)+1)/cfg.half_life)
        if cfg.mode == 'separate':
            margin = (weighted_quantile(np.maximum(le,0),cfg.quantile,weights)
                      +weighted_quantile(np.maximum(-ve,0),cfg.quantile,weights))
        else:
            margin = weighted_quantile(errors,cfg.quantile,weights)
        return dict(load=load, pv=pv, net=load-pv, margin=margin,
                    errors=errors, weights=weights/weights.sum(),
                    load_method=self.choose('load',day) if cfg.forecast=='adaptive' else cfg.forecast)

    @staticmethod
    def scenarios(horizon, count):
        # 以整条历史误差轨迹为场景，按净需求总偏差分层选代表，不打乱段内相关性。
        errors,weights = horizon['errors'],horizon['weights']
        order = np.argsort(errors.sum(axis=1))
        groups = np.array_split(order,min(count,len(order)))
        representatives,probs = [],[]
        for group in groups:
            local = weights[group]
            mid = group[np.searchsorted(np.cumsum(local),local.sum()/2)]
            representatives.append(errors[mid])
            probs.append(local.sum())
        return horizon['net'][None,:]+np.asarray(representatives), np.asarray(probs)
