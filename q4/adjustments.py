"""B Q4-3 control: same midnight PV information, frozen normal orders all day.

Added to answer the inherited Q3 question about the value of later forecasts.
This is a fixed-policy explanatory ablation, never a main-policy selection trial.
"""
from pathlib import Path
import json
from .runner import inputs,write_run,event
from .model import Settings,simulate


class MidnightPV:
    def __init__(self,base):self.base=base;self.actual=base.actual
    def horizon(self,question,day,issue,cfg):
        assert issue==0
        return self.base.horizon(3,day,0,cfg)


def main():
    root=Path(__file__).resolve().parents[1]
    engine,prices,paths=inputs(root)
    b=json.loads((root/'output/q4-3/summary.json').read_text(encoding='utf8'))
    # Reuse Q4-2's freeze/execution schedule, but preserve Q4-3 midnight forecasts.
    records,_,summary=simulate(MidnightPV(engine),prices,2,Settings(**b['config']),
                              february_energy=b['february_initial_energy_kwh'],progress=True)
    summary['question']=3
    summary['control']='Q4-3 midnight PV forecast; no intraday order/forecast updates; fixed B February energy'
    summary['saving_from_full_updates_cny']=summary['total_cost_cny']-b['total_cost_cny']
    write_run(root/'output/q4_experiments/q3_B_no_updates',records,[],summary,3,paths)
    event(root/'output/q4_experiments',dict(stage='explanatory_update_ablation',summary=summary))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
