"""一月选参，二月至十二月留出评价；实验只输出汇总，不复制整套工作簿。"""
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
from .config import Config
from .simulation import simulate
from .reporting import save_json,csv_file


def candidates():
    base=Config()
    return [replace(base,name='separate_weighted',mode='separate',forecast='weighted',quantile=.8),
            replace(base,name='net_weighted_65',forecast='weighted'),
            replace(base,name='net_adaptive_50',quantile=.5),
            replace(base,name='net_adaptive_65'),
            replace(base,name='net_adaptive_80',quantile=.8),
            replace(base,name='net_window28',history_days=28,half_life=14),
            replace(base,name='scenario5',mode='scenario'),
            replace(base,name='scenario5_cvar',mode='scenario',cvar_weight=.15),
            replace(base,name='reserve05',reserve_factor=.05),
            replace(base,name='reserve20',reserve_factor=.2)]


def select_config(engine,question,price,destination):
    rows=[]
    for cfg in candidates():
        _,_,summary=simulate(engine,question,price,cfg,days=31)
        rows.append(summary)
        print(f"Q{question}一月选参 {cfg.name}: {summary['total_cost_cny']:.2f}元",flush=True)
    minimum=min(x['january_warmup_cost_cny'] for x in rows)
    # 一分钱以内视为费用平局，优先不用备用/CVaR等额外机制，避免浮点选模。
    eligible=[x for x in rows if x['january_warmup_cost_cny']<=minimum+.01]
    winner=min(eligible,key=lambda x:(x['config']['reserve_factor']!=0,
                                      x['config']['cvar_weight']!=0,
                                      x['config']['mode']=='scenario'))
    payload=winner['config'].copy();payload['update_hours']=tuple(payload['update_hours'])
    cfg=Config(**payload)
    save_json(Path(destination)/f'q{question}_selection.json',
              dict(selection_period=['2025-01-01','2025-01-31'],evaluation_period=['2025-02-01','2025-12-31'],
                   criterion='january_warmup_cost_cny; ties within 0.01 CNY prefer simpler controls',
                   chosen=cfg.to_dict(),candidates=rows))
    return cfg


def run_suite(engine,price,configs,main_summaries,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    rows=[]
    checkpoint=destination/'ablation.json'
    tasks=[]
    for q in (2,3):
        base=configs[q]
        for candidate in candidates():
            if candidate.name in ('separate_weighted','net_weighted_65','net_adaptive_65','scenario5','scenario5_cvar','reserve05','reserve20'):
                tasks.append((q,'factor_'+candidate.name,candidate,None))
    base=configs[3]
    for n in range(4):
        for subset in combinations((6,12,18),n):
            hours=(0,)+subset
            cfg=replace(base,name='hours_'+'_'.join(map(str,hours)),update_hours=hours)
            tasks.append((3,cfg.name+'_continuous',cfg,None))
            tasks.append((3,cfg.name+'_fixed',cfg,main_summaries[3]['february_initial_energy_kwh']))
    for name,pv,load in [('neither',False,False),('pv_only',True,False),('load_only',False,True),('both',True,True)]:
        tasks.append((3,name,replace(base,name=name,update_pv=pv,update_load=load),main_summaries[3]['february_initial_energy_kwh']))
    tasks.append((3,'trigger',replace(base,name='trigger',mode='scenario',trigger=True),None))
    for q,label,cfg,february in tasks:
        print(f'对照实验 Q{q} {label}',flush=True)
        _,_,summary=simulate(engine,q,price,cfg,february_energy=february)
        rows.append(dict(label=label,**summary))
        save_json(checkpoint,rows)
    fields=['question','label','total_cost_cny','normal_cost_cny','adjustment_cost_cny','emergency_cost_cny',
            'emergency_energy_kwh','unused_supply_energy_kwh','net_mae_kw','peak_net_mae_kw',
            'february_initial_energy_kwh','terminal_energy_kwh','runtime_seconds']
    csv_file(destination/'ablation.csv',fields,([r[k] for k in fields] for r in rows))
    lines=['# 改进对照实验','',
           '主模型仅按1月费用选择；此处2—12月结果用于解释各因素，不据此重新选择主模型。',
           'continuous：自洽1月热身；fixed：在2月1日使用主模型共同初值，1月费用不参与受控比较。','',
           '| 问题 | 实验 | 总费用（元） | 紧急电量（kWh） | 未利用供给（kWh） |',
           '|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['question']} | {r['label']} | {r['total_cost_cny']:.2f} | {r['emergency_energy_kwh']:.2f} | {r['unused_supply_energy_kwh']:.2f} |")
    (destination/'ablation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return rows
