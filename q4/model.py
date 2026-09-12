"""Causal variable-price forecast, B dispatch, and paired-scenario C comparison."""
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from time import perf_counter
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix, vstack
from microgrid.config import Config
from microgrid.forecast import WEIGHTS
from microgrid.dispatch import solve as base_solve, execute, EMIN, EMAX, QMAX, ETA, GUARD
from microgrid.settlement import settle


@dataclass(frozen=True)
class Settings:
    price_method: str = 'weighted'
    quantile: float = .65
    variant: str = 'B'
    scenarios: int = 5
    beta: float = .9
    risk_weight: float = 0.

    def __post_init__(self):
        if self.price_method not in ('persistence','weighted') or self.variant not in ('B','C_fixed','C0','C15'):
            raise ValueError('Unknown fourth-question configuration')
        if not 0 < self.quantile < 1 or not 0 < self.beta < 1 or self.risk_weight < 0:
            raise ValueError('Invalid risk parameters')


class PriceEngine:
    def __init__(self, actual):
        self.actual = np.asarray(actual, float)
        if self.actual.ndim != 2 or self.actual.shape[1] != 144 or not np.isfinite(self.actual).all() or self.actual.min() <= 0:
            raise ValueError('Price must be a finite positive day by 144 matrix')
        self.cache = {}
        self.profile_cache = {}

    def profile(self, origin, target, method):
        key = origin, target, method
        if key in self.profile_cache:
            return self.profile_cache[key]
        if origin == 0:
            result = np.full(144, .7)  # declared prior; first day has no normal orders
        else:
            def past(j):
                return self.actual[j] if j < origin else self.profile(origin,j,method)
            n = min(7,target)
            if method == 'persistence':
                result = past(target-1).copy()
            else:
                weights = WEIGHTS[:n]/WEIGHTS[:n].sum()
                result = sum(weights[j]*past(target-j-1) for j in range(n))
        self.profile_cache[key] = result
        return result

    def horizon(self, day, issue, method):
        key = day, issue, method
        if key in self.cache:
            return self.cache[key]
        start = issue*6
        current = self.profile(day,day,method)
        result = current.copy() if not start else np.r_[current[start:],self.profile(day,day+1,method)[:start]]
        if start:
            bias = np.median(self.actual[day,start-6:start]-current[start-6:start])
            bound = .25*np.median(self.actual[day,:start])
            result = result+np.clip(bias,-bound,bound)*np.exp(-np.arange(1,145)/36)
        result = np.maximum(result,1e-6)  # numerical positive floor; no full-year calibration
        self.cache[key] = result
        return result


def paired_scenarios(engine, prices, day, issue, cfg, horizon):
    ids = list(range(max(0,day-56),day))
    forecast = prices.horizon(day,issue,cfg.price_method)
    if not ids:
        return horizon['net'][None,:]/6, forecast[None,:], np.ones(1)
    # ForecastEngine errors use precisely these completed same-issue windows.
    log_errors = []
    for j in ids:
        begin = j*144+issue*6
        actual = prices.actual.ravel()[begin:begin+144]
        if len(actual) != 144:
            raise ValueError('Unrealized historical window')
        log_errors.append(np.log(actual/prices.horizon(j,issue,cfg.price_method)))
    ps = forecast[None,:]*np.exp(np.asarray(log_errors))
    ns = (horizon['net'][None,:]+horizon['errors'])/6
    if len(ns) != len(ids):
        raise ValueError('Unpaired price/net residuals')
    # Same representative selection for C_fixed and joint-price variants:
    # rank joint stress by price-weighted positive net demand, keep whole trajectories.
    score = np.sum(ps*np.maximum(ns,0),axis=1)
    groups = np.array_split(np.argsort(score,kind='stable'),min(cfg.scenarios,len(ids)))
    reps,probs = [],[]
    weights = horizon['weights']
    for group in groups:
        local = weights[group]
        reps.append(group[np.searchsorted(np.cumsum(local),local.sum()/2)])
        probs.append(float(local.sum()))
    selected_prices = ps[reps] if cfg.variant != 'C_fixed' else np.tile(forecast,(len(reps),1))
    return ns[reps],selected_prices,np.asarray(probs)


def scenario_solve(price,net,probs,energy,reference,cfg):
    """Shared nonanticipative storage and orders, scenario residual recourse; total-cost CVaR."""
    price,net,probs = np.asarray(price),np.asarray(net),np.asarray(probs)
    S,n = net.shape
    if price.shape != net.shape or probs.shape != (S,) or not np.isclose(probs.sum(),1) or np.any(price<=0):
        raise ValueError('Scenario shapes or probabilities invalid')
    g,c,r,E = [np.arange(k*n,(k+1)*n) for k in range(4)]
    urgent = np.arange(4*n,(4+S)*n).reshape(S,n)
    spill = np.arange((4+S)*n,(4+2*S)*n).reshape(S,n)
    a = np.arange((4+2*S)*n,(5+2*S)*n)
    z = int(a[-1]+1); tails = np.arange(z+1,z+1+S); nv = int(tails[-1]+1)
    eq = lil_matrix(((S+1)*n,nv)); rhs = np.r_[net.ravel(),energy,np.zeros(n-1)]
    for s in range(S):
        for t in range(n):
            eq[s*n+t,[g[t],c[t],r[t],urgent[s,t],spill[s,t]]] = [1,-1,1,1,-1]
    for t in range(n):
        eq[S*n+t,[E[t],c[t],r[t]]] = [1,-ETA,1/ETA]
        if t: eq[S*n+t,E[t-1]] = -1
    active = np.flatnonzero(np.isfinite(reference))
    ub = lil_matrix((2*len(active)+S,nv)); b = np.zeros(ub.shape[0])
    for k,t in enumerate(active):
        ub[2*k,[g[t],a[t]]] = [1,-1]; b[2*k] = reference[t]
        ub[2*k+1,[g[t],a[t]]] = [-1,-1]; b[2*k+1] = -reference[t]
    losses = np.zeros((S,nv))
    for s in range(S):
        losses[s,g] = price[s]
        losses[s,a[active]] = .5*price[s,active]
        losses[s,urgent[s]] = 5*price[s]
        ub[2*len(active)+s,:] = losses[s]
        ub[2*len(active)+s,z] = -1; ub[2*len(active)+s,tails[s]] = -1
    obj = probs@losses
    obj[z] = cfg.risk_weight
    obj[tails] = cfg.risk_weight*probs/(1-cfg.beta)
    bounds = [(0,None)]*nv
    for i in np.r_[c,r]: bounds[i] = (0,QMAX)
    for i in E: bounds[i] = (EMIN+GUARD,EMAX-GUARD)
    for t in np.flatnonzero(~np.isfinite(reference)): bounds[a[t]] = (0,0)
    bounds[z] = (None,None)
    eq,ub = eq.tocsr(),ub.tocsr()
    options = {'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
    first = linprog(obj,A_eq=eq,b_eq=rhs,A_ub=ub,b_ub=b,bounds=bounds,method='highs',options=options)
    if not first.success: raise RuntimeError(first.message)
    secondary = np.zeros(nv); secondary[c] = 1; secondary[r] = 1
    result = linprog(secondary,A_eq=eq,b_eq=rhs,A_ub=vstack([ub,csr_matrix(obj[None,:])]),
                     b_ub=np.r_[b,first.fun+1e-6],bounds=bounds,method='highs',options=options)
    if not result.success: raise RuntimeError(result.message)
    x = result.x
    if np.minimum(x[c],x[r]).max() > 1e-6: raise RuntimeError('Simultaneous charging/discharging')
    return dict(grid=x[g],charge=x[c],discharge=x[r],energy=x[E],objective=float(obj@x),
                constraint_residual=float(np.max(np.abs(eq@x-rhs))))


def simulate(engine,prices,question,cfg,days=None,february_energy=None,snapshots=False,progress=False):
    start_time = perf_counter()
    count = len(prices.actual) if days is None else days
    net_cfg = Config(quantile=cfg.quantile,forecast='adaptive',mode='net')
    energy = 6000.; records=[]; plans=[]
    for day in range(count):
        if day == 31 and february_energy is not None: energy=float(february_energy)
        grid=np.zeros(144);cp=np.zeros(144);rp=np.zeros(144)
        values={k:np.zeros(144) for k in ('charge','discharge','emergency','spill','load_prediction','pv_prediction','margin','price_prediction')}
        energies=np.empty(145);energies[0]=energy;max_residual=0.
        for issue in ((0,) if question==2 else (0,6,12,18)):
            start=issue*6;take=144 if question==2 else 36
            horizon=engine.horizon(question,day,issue,net_cfg)
            prediction=prices.horizon(day,issue,cfg.price_method)
            reference=np.full(144,np.nan) if not issue else np.r_[grid0[start:],np.full(start,np.nan)]
            if day==0:
                solution=dict(grid=np.zeros(144),charge=np.zeros(144),discharge=np.zeros(144),constraint_residual=0.)
            elif cfg.variant=='B':
                solution=base_solve(prediction,((horizon['net']+horizon['margin'])/6)[None,:],[1.],energy,net_cfg,reference)
            else:
                net,price_s,prob=paired_scenarios(engine,prices,day,issue,cfg,horizon)
                solution=scenario_solve(price_s,net,prob,energy,reference,cfg)
            grid[start:]=solution['grid'][:144-start]
            cp[start:]=solution['charge'][:144-start];rp[start:]=solution['discharge'][:144-start]
            if issue==0: grid0=grid.copy()
            max_residual=max(max_residual,solution['constraint_residual'])
            if snapshots:
                plans.append(dict(day=day,issue=issue,energy_kwh=energy,grid=solution['grid'].copy(),
                                  charge=solution['charge'].copy(),discharge=solution['discharge'].copy(),
                                  price_prediction=prediction.copy(),load=horizon['load'].copy(),pv=horizon['pv'].copy()))
            sl=slice(start,start+take)
            block=execute(grid[sl],cp[sl],rp[sl],engine.actual['load'][day,sl]/6,engine.actual['pv'][day,sl]/6,energy)
            for k in ('charge','discharge','emergency','spill'): values[k][sl]=block[k]
            for k,source in [('load_prediction','load'),('pv_prediction','pv'),('margin','margin')]: values[k][sl]=horizon[source][:take]
            values['price_prediction'][sl]=prediction[:take]
            energies[start+1:start+take+1]=block['energy'][1:];energy=float(block['energy'][-1])
        fees=settle(prices.actual[day],grid,values['emergency'],grid0 if question==3 else None)
        records.append(dict(day=day,date=date(2025,1,1)+timedelta(days=day),grid=grid.copy(),grid0=grid0,
                            plan_charge=cp,plan_discharge=rp,energy=energies,load=engine.actual['load'][day]/6,
                            pv=engine.actual['pv'][day]/6,price=prices.actual[day].copy(),max_plan_residual_kwh=max_residual,**values,**fees))
        if progress and ((day+1)%60==0 or day+1==count):
            print(f'Q4-{question} {cfg.variant} {day+1}/{count}',flush=True)
    selected=records[31:] if len(records)>31 else records
    daily=np.array([r['total_cost_cny'] for r in selected])
    summary=dict(question=question,config=asdict(cfg),output_days=len(selected),
                 total_cost_cny=float(daily.sum()),january_warmup_cost_cny=float(sum(r['total_cost_cny'] for r in records[:31])),
                 february_initial_energy_kwh=float(selected[0]['energy'][0]),terminal_energy_kwh=float(selected[-1]['energy'][-1]),
                 worst_daily_cost_cny=float(daily.max()),daily_cost_p95_cny=float(np.quantile(daily,.95)),
                 runtime_seconds=perf_counter()-start_time,initial_energy_policy='fixed_B_february' if february_energy is not None else 'continuous_january',
                 price_mae=float(np.mean([np.mean(np.abs(r['price']-r['price_prediction'])) for r in selected])),
                 max_plan_residual_kwh=max(r['max_plan_residual_kwh'] for r in records))
    for k in ('normal_cost_cny','adjustment_cost_cny','emergency_cost_cny'):
        summary[k]=float(sum(r[k] for r in selected))
    for k in ('grid','charge','discharge','emergency','spill'):
        summary[k+'_energy_kwh']=float(sum(r[k].sum() for r in selected))
    summary['monthly']=[dict(month=m,total_cost_cny=float(sum(r['total_cost_cny'] for r in selected if r['date'].month==m))) for m in range(2,13)]
    return records,plans,summary
