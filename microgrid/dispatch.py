"""稀疏风险LP：所有场景共享购电和储能计划，只有即时缺口可按场景变化。"""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix, vstack
from .config import Config

ETA, EMIN, EMAX, QMAX = .9, 1200., 10800., 5000/6
GUARD = 1e-6


@lru_cache(maxsize=32)
def structure(n, scenarios):
    # x = g,c,r,E,emergency[S,n],spill[S,n],increase,decrease,z,tail[S],reserve_shortfall
    g,c,r,e = [np.arange(k*n,(k+1)*n) for k in range(4)]
    urgent = np.arange(4*n,(4+scenarios)*n).reshape(scenarios,n)
    spill = np.arange((4+scenarios)*n,(4+2*scenarios)*n).reshape(scenarios,n)
    u = np.arange((4+2*scenarios)*n,(5+2*scenarios)*n)
    v = u+n
    z = int(v[-1]+1)
    tail = np.arange(z+1,z+1+scenarios)
    reserve = int(tail[-1]+1)
    nvar = reserve+1
    matrix = lil_matrix(((scenarios+1)*n,nvar))
    for s in range(scenarios):
        for t in range(n):
            row = s*n+t
            matrix[row,[g[t],c[t],r[t],urgent[s,t],spill[s,t]]] = [1,-1,1,1,-1]
    for t in range(n):
        row = scenarios*n+t
        matrix[row,[e[t],c[t],r[t]]] = [1,-ETA,1/ETA]
        if t: matrix[row,e[t-1]] = -1
    return matrix.tocsr(), (g,c,r,e,urgent,spill,u,v,z,tail,reserve,nvar)


def solve(price, net_scenarios_kwh, probabilities, energy, cfg: Config, reference=None,
          reserve_target=EMIN, fixed_grid=None):
    price = np.asarray(price,float)
    net = np.atleast_2d(np.asarray(net_scenarios_kwh,float))
    probs = np.asarray(probabilities,float)
    ns,n = net.shape
    if price.shape != (n,) or probs.shape != (ns,) or not np.isclose(probs.sum(),1):
        raise ValueError('调度输入形状或场景概率错误')
    if not np.isfinite(net).all() or not np.isfinite(price).all() or np.any(price<=0):
        raise ValueError('需有限净负荷和正电价')
    if np.any(probs<0) or not np.isfinite(probs).all() or not EMIN-1e-7 <= energy <= EMAX+1e-7:
        raise ValueError('场景概率或储电量无效')
    base,idx = structure(n,ns)
    g,c,r,e,urgent,spill,u,v,z,tail,reserve,nvar = idx
    ref = np.full(n,np.nan) if reference is None else np.asarray(reference,float)
    if ref.shape!=(n,) or np.isinf(ref).any():
        raise ValueError('参考计划无效')
    active = np.flatnonzero(np.isfinite(ref))
    adjust = lil_matrix((len(active),nvar))
    for row,t in enumerate(active):
        adjust[row,[g[t],u[t],v[t]]] = [1,-1,1]
    aeq = vstack([base,adjust.tocsr()],format='csr')
    beq = np.r_[net.ravel(),energy,np.zeros(n-1),ref[active]]
    obj = np.zeros(nvar)
    obj[g] = price
    obj[urgent] = probs[:,None]*5*price
    obj[u[active]] = obj[v[active]] = .5*price[active]
    obj[z] = cfg.cvar_weight
    obj[tail] = cfg.cvar_weight*probs/(1-cfg.cvar_beta)
    obj[reserve] = cfg.reserve_price*np.median(price) if cfg.reserve_factor else 0
    aub = lil_matrix((ns+1,nvar))
    for s in range(ns):
        aub[s,urgent[s]] = 5*price
        aub[s,z] = -1
        aub[s,tail[s]] = -1
    aub[ns,e[-1]] = aub[ns,reserve] = -1
    bub = np.r_[np.zeros(ns),-np.clip(reserve_target,EMIN,EMAX)]
    bounds = [(0,None)]*nvar
    for i in np.r_[c,r]: bounds[i] = (0,QMAX)
    for i in e: bounds[i] = (EMIN+GUARD,EMAX-GUARD)
    if cfg.mode != 'scenario':
        for i in urgent.ravel(): bounds[i] = (0,0)
    for t in np.flatnonzero(~np.isfinite(ref)):
        bounds[u[t]]=bounds[v[t]]=(0,0)
    if fixed_grid is not None:
        fixed = np.asarray(fixed_grid)
        if fixed.shape != (n,): raise ValueError('固定计划长度错误')
        for t in np.flatnonzero(np.isfinite(fixed)):
            bounds[g[t]]=(float(fixed[t]),float(fixed[t]))
    options = {'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
    solution = linprog(obj,A_eq=aeq,b_eq=beq,A_ub=aub.tocsr(),b_ub=bub,bounds=bounds,
                       method='highs',options=options)
    if not solution.success: raise RuntimeError(solution.message)
    primary = float(solution.fun)
    if cfg.lexicographic:
        secondary = np.zeros(nvar)
        secondary[c]=secondary[r]=1.
        # 主目标最多偏离一百万分之一元；再最小化母线侧吞吐量。
        lex = linprog(secondary,A_eq=aeq,b_eq=beq,
                      A_ub=vstack([aub.tocsr(),csr_matrix(obj[None,:])],format='csr'),
                      b_ub=np.r_[bub,primary+1e-6],bounds=bounds,method='highs',options=options)
        if not lex.success: raise RuntimeError('二级吞吐量求解失败: '+lex.message)
        solution = lex
    x = solution.x
    if np.minimum(x[c],x[r]).max() > 1e-6:
        raise RuntimeError('共享储能计划出现同时充放电；需审查约束或启用MILP')
    return dict(grid=x[g],charge=x[c],discharge=x[r],energy=x[e],
                objective=float(obj@x),primary_objective=primary,
                expected_emergency_kwh=float(np.sum(probs[:,None]*x[urgent])),
                reserve_target_kwh=float(reserve_target),
                constraint_residual=float(np.max(np.abs(aeq@x-beq))))


def execute(grid,charge_plan,discharge_plan,load,pv,initial_energy):
    """可执行反馈：每段仅观察本段净负荷，不使用未来实际值。"""
    grid,charge_plan,discharge_plan,load,pv = [np.asarray(x,float) for x in (grid,charge_plan,discharge_plan,load,pv)]
    n=len(grid)
    if any(x.shape != (n,) or not np.isfinite(x).all() for x in (grid,charge_plan,discharge_plan,load,pv)):
        raise ValueError('执行输入无效')
    charge,discharge,emergency,spill = [np.zeros(n) for _ in range(4)]
    energy=np.empty(n+1); energy[0]=initial_energy
    for t in range(n):
        before=energy[t]
        c=min(charge_plan[t],QMAX,max((EMAX-GUARD-before)/ETA,0))
        r=min(discharge_plan[t],QMAX,max((before-EMIN-GUARD)*ETA,0))
        residual=load[t]+c-pv[t]-r-grid[t]
        if residual>0:
            cancel=min(c,residual); c-=cancel; residual-=cancel
            extra=min(QMAX-r,residual,max((before+ETA*c-r/ETA-EMIN-GUARD)*ETA,0))
            r+=extra
        elif residual<0:
            cancel=min(r,-residual); r-=cancel; residual+=cancel
            extra=min(QMAX-c,-residual,max((EMAX-GUARD-before-ETA*c+r/ETA)/ETA,0))
            c+=extra
        charge[t],discharge[t]=c,r
        residual=load[t]+c-pv[t]-r-grid[t]
        emergency[t],spill[t]=max(residual,0),max(-residual,0)
        energy[t+1]=before+ETA*c-r/ETA
    return dict(charge=charge,discharge=discharge,emergency=emergency,spill=spill,energy=energy)
