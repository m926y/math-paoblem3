"""日初提交、规定时刻更新、实时反馈与全期评估。"""
from __future__ import annotations
from dataclasses import replace
from datetime import date,timedelta
from time import perf_counter
import numpy as np
from .dispatch import solve,execute,EMIN,EMAX,ETA
from .settlement import settle,revision_fee


def optimize(engine, question, day, issue, energy, price, cfg, reference=None, fixed=None):
    horizon=engine.horizon(question,day,issue,cfg)
    start=issue*6
    tariff=np.roll(price,-start)
    if cfg.mode=='scenario':
        net,prob=engine.scenarios(horizon,cfg.scenarios)
    else:
        net=(horizon['net']+horizon['margin'])[None,:]
        prob=np.ones(1)
    expensive=tariff>=np.quantile(tariff,.75)
    reserve=EMIN+cfg.reserve_factor*np.sum(np.maximum(horizon['net'][expensive],0))/6/ETA
    solution=solve(tariff,net/6,prob,energy,cfg,reference,
                   reserve_target=float(np.clip(reserve,EMIN,EMAX-1e-6)),fixed_grid=fixed)
    return solution,horizon


def simulate(engine, question, price, cfg, *, days=None, february_energy=None,
             snapshots=False, progress=False):
    started=perf_counter()
    records,plans=[],[]
    energy=6000.
    count=len(engine.actual['load']) if days is None else days
    if count>len(engine.actual['load']) or count<1: raise ValueError('仿真天数无效')
    for day in range(count):
        if day==31 and february_energy is not None: energy=float(february_energy)
        initial=energy
        grid=np.zeros(144); cp=np.zeros(144); rp=np.zeros(144)
        values={key:np.zeros(144) for key in ('charge','discharge','emergency','spill','load_prediction','pv_prediction','margin')}
        energies=np.empty(145); energies[0]=energy
        base=None; base_grid=None; active_horizon=None
        changes=0; revision=0.; max_residual=0.; forecasts_used=[]
        for issue in ((0,) if question==2 else (0,6,12,18)):
            start=issue*6
            take=144 if question==2 else 36
            if issue==0 or issue in cfg.update_hours:
                reference=None if issue==0 or question==2 else np.r_[base_grid[start:],np.full(start,np.nan)]
                if question==2 and day==0:
                    solution=dict(grid=np.zeros(144),charge=np.zeros(144),discharge=np.zeros(144),
                                  objective=0.,constraint_residual=0.)
                    horizon=engine.horizon(question,day,issue,cfg)
                else:
                    solution,horizon=optimize(engine,question,day,issue,energy,price,cfg,reference)
                accepted=True
                if cfg.trigger and issue:
                    # 同一新预测下比较保留旧购电与改购电；不会读取未来实测。
                    fixed=np.r_[grid[start:],np.full(start,np.nan)]
                    alternative,_=optimize(engine,question,day,issue,energy,price,
                                           replace(cfg,mode='scenario'),reference,fixed)
                    if alternative['objective']<=solution['objective']+.01:
                        solution=alternative; accepted=False
                if issue:
                    revision+=revision_fee(price[start:],grid[start:],solution['grid'][:144-start])
                    changes+=int(accepted)
                grid[start:]=solution['grid'][:144-start]
                cp[start:]=solution['charge'][:144-start]
                rp[start:]=solution['discharge'][:144-start]
                active_horizon=horizon; active_start=start
                if issue==0: base_grid=grid.copy()
                max_residual=max(max_residual,solution['constraint_residual'])
                forecasts_used.append(horizon['load_method'])
                if snapshots:
                    plans.append(dict(day=day,issue=issue,accepted=accepted,energy_kwh=energy,
                                      grid=solution['grid'].copy(),charge=solution['charge'].copy(),
                                      discharge=solution['discharge'].copy(),
                                      load=horizon['load'].copy(),pv=horizon['pv'].copy(),
                                      reference=np.full(144,np.nan) if reference is None else reference.copy()))
            offset=start-active_start
            sl=slice(start,start+take)
            actual_load=engine.actual['load'][day,sl]/6
            actual_pv=engine.actual['pv'][day,sl]/6
            block=execute(grid[sl],cp[sl],rp[sl],actual_load,actual_pv,energy)
            for key in ('charge','discharge','emergency','spill'): values[key][sl]=block[key]
            values['load_prediction'][sl]=active_horizon['load'][offset:offset+take]
            values['pv_prediction'][sl]=active_horizon['pv'][offset:offset+take]
            values['margin'][sl]=active_horizon['margin'][offset:offset+take]
            energies[start+1:start+take+1]=block['energy'][1:]
            energy=float(block['energy'][-1])
        fees=settle(price,grid,values['emergency'],base_grid if question==3 else None)
        records.append(dict(day=day,date=date(2025,1,1)+timedelta(days=day),grid=grid,
                            grid0=base_grid,plan_charge=cp,plan_discharge=rp,energy=energies,
                            load=engine.actual['load'][day]/6,pv=engine.actual['pv'][day]/6,
                            changes=changes,revision_fee_cny=revision,
                            max_plan_residual_kwh=max_residual,forecast_methods=forecasts_used,
                            **values,**fees))
        if progress and ((day+1)%30==0 or day+1==count):
            print(f'{cfg.name}: {day+1}/{count}天，日末储电{energy:.2f} kWh',flush=True)
    summary=summarize(records,price,question)
    summary.update(config=cfg.to_dict(),runtime_seconds=perf_counter()-started,
                   initial_energy_policy='fixed_february' if february_energy is not None else 'continuous_january')
    return records,plans,summary


def summarize(records,price,question):
    selected=[r for r in records if r['day']>=31]
    january=[r for r in records if r['day']<31]
    if not selected: selected=records
    def cost(rs): return float(sum(r['total_cost_cny'] for r in rs))
    energy=np.concatenate([r['energy'] for r in selected])
    balance=max(float(np.max(np.abs(r['grid']+r['pv']+r['discharge']+r['emergency']-
                                        r['load']-r['charge']-r['spill']))) for r in selected)
    storage=max(float(np.max(np.abs(np.diff(r['energy'])-ETA*r['charge']+r['discharge']/ETA))) for r in selected)
    err=np.array([(r['load']-r['pv'])*6-(r['load_prediction']-r['pv_prediction']) for r in selected])
    high=price>=np.quantile(price,.75)
    result=dict(question=question,output_days=len(selected),output_dates=[str(selected[0]['date']),str(selected[-1]['date'])],
                total_cost_cny=cost(selected),january_warmup_cost_cny=cost(january),full_year_cost_cny=cost(records),
                february_initial_energy_kwh=float(selected[0]['energy'][0]),
                terminal_energy_kwh=float(selected[-1]['energy'][-1]),
                normal_energy_kwh=float(sum(r['grid'].sum() for r in selected)),
                emergency_energy_kwh=float(sum(r['emergency'].sum() for r in selected)),
                unused_supply_energy_kwh=float(sum(r['spill'].sum() for r in selected)),
                charge_energy_kwh=float(sum(r['charge'].sum() for r in selected)),
                discharge_energy_kwh=float(sum(r['discharge'].sum() for r in selected)),
                energy_min_kwh=float(energy.min()),energy_max_kwh=float(energy.max()),
                max_balance_residual_kwh=balance,max_storage_residual_kwh=storage,
                max_plan_residual_kwh=max(r['max_plan_residual_kwh'] for r in selected),
                max_charge_discharge_overlap_kwh=max(float(np.minimum(r['charge'],r['discharge']).max()) for r in selected),
                net_mae_kw=float(np.abs(err).mean()),net_rmse_kw=float(np.sqrt(np.mean(err**2))),
                net_bias_kw=float(err.mean()),peak_net_mae_kw=float(np.abs(err[:,high]).mean()),
                empirical_margin_coverage=float(np.mean(err<=np.array([r['margin'] for r in selected]))),
                worst_daily_cost_cny=max(r['total_cost_cny'] for r in selected),
                updates_accepted=sum(r['changes'] for r in selected),
                sequential_adjustment_fee_cny=float(sum(r['revision_fee_cny'] for r in selected)))
    for key in ('normal_cost_cny','adjustment_cost_cny','emergency_cost_cny'):
        result[key]=float(sum(r[key] for r in selected))
    result['sequential_fee_total_cost_cny']=result['normal_cost_cny']+result['emergency_cost_cny']+result['sequential_adjustment_fee_cny']
    return result
