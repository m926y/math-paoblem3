"""Read saved artifacts, verify against protected inputs, optionally clean-rerun B."""
import argparse,json
from pathlib import Path
from datetime import date,timedelta
import numpy as np
from .runner import inputs
from .model import Settings,simulate
from .audit import audit
from microgrid.reporting import save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rerun',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    engine,prices,_=inputs(root);result={}
    for q in (2,3):
        folder=root/f'output/q4-{q}'
        summary=json.loads((folder/'summary.json').read_text(encoding='utf8'))
        with np.load(folder/'trace.npz',allow_pickle=False) as data:
            arrays={k:data[k] for k in data.files}
        records=[dict(day=d,date=date(2025,1,1)+timedelta(days=d),**{k:v[d] for k,v in arrays.items()}) for d in range(365)]
        with np.load(folder/'plans.npz',allow_pickle=False) as data:
            plans=[{k:data[k][i] for k in data.files} for i in range(len(data['day']))]
        check=audit(records,summary,q,plans,folder/f'result4-{q}.xlsx')
        assert np.array_equal(arrays['price'],prices.actual)
        assert np.array_equal(arrays['load'],engine.actual['load']/6)
        assert np.array_equal(arrays['pv'],engine.actual['pv']/6)
        check['input_arrays_match_original']=True
        if args.rerun:
            # Fresh predictor instances: no shared caches with the producing process.
            engine2,prices2,_=inputs(root)
            rec,plans2,s=simulate(engine2,prices2,q,Settings(**summary['config']),snapshots=True,progress=True)
            audit(rec,s,q,plans2)
            maximum=max(float(np.max(np.abs(np.array([r[k] for r in rec])-arrays[k]))) for k in arrays)
            assert maximum<=1e-6,(q,maximum)
            assert abs(s['total_cost_cny']-summary['total_cost_cny'])<=1e-5
            check['clean_rerun_max_abs_difference']=maximum
            check['clean_rerun_cost_difference']=s['total_cost_cny']-summary['total_cost_cny']
        result[str(q)]=check
        print(f'Q4-{q} saved artifacts verified',flush=True)
    save_json(root/'output/q4_experiments/reproducibility.json',result)


if __name__=='__main__':main()
