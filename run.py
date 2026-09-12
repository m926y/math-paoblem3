"""统一入口：q1、q2、q3、all、experiments、validate、baseline。"""
import argparse,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    if len(sys.argv)>1 and sys.argv[1]=='q1':
        sys.argv.pop(1)
        from microgrid.q1 import main as q1
        return q1()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('q2','q3','all','experiments','validate','baseline'))
    parser.add_argument('--attachment-dir',type=Path,default=ROOT/'附件')
    parser.add_argument('--experiments',action='store_true')
    parser.add_argument('--question',type=int,choices=(2,3),default=2)
    parser.add_argument('--config',type=Path,help='指定已选配置JSON（含chosen或config字段，或直接配置对象）')
    parser.add_argument('--output-dir',type=Path,help='仅q2/q3可指定输出目录')
    args=parser.parse_args()
    from microgrid.io import attachment_paths,read_attachment1,read_attachment2,read_attachment3
    from microgrid.config import Config
    from microgrid.forecast import ForecastEngine
    from microgrid.simulation import simulate
    from microgrid.experiments import select_config,run_suite
    from microgrid.reporting import write_outputs,save_json
    from microgrid.validation import validate_output
    if args.command=='validate':
        for q in (2,3):
            result=validate_output(ROOT/f'output/q{q}',q)
            print(f'Q{q} {len(result["checks"])}项通过')
        return
    if args.output_dir and args.command not in ('q2','q3'):
        parser.error('--output-dir仅用于q2/q3')
    paths=attachment_paths(ROOT,args.attachment_dir)
    price,load0,pv0=read_attachment1(paths['data1'])
    dates,load,pv=read_attachment2(paths['data2'])
    forecasts=read_attachment3(paths['data3'],dates)
    engine=ForecastEngine(load,pv,forecasts,load0,pv0)
    dest=ROOT/'output/experiments';configs={};summaries={}
    if args.command=='baseline':
        if args.question==2:
            from microgrid.baseline_q2 import simulate_causal,summarize
            records,checks=simulate_causal(price,load0,pv0,load,pv,'zero')
            summary=summarize(records,checks)
        else:
            from microgrid.baseline_q3 import simulate as old_sim,summarize
            from microgrid.legacy_q3_dispatch import DispatchParams
            params=DispatchParams()
            records,checks=old_sim(dates,price,load0,load,pv,forecasts,params,.8,20.,6.,.15)
            summary=summarize(records,price,params,checks)
        baseline=json.loads((ROOT/f'output/baseline/q{args.question}/summary.json').read_text(encoding='utf-8'))
        assert abs(summary['total_cost_cny']-baseline['total_cost_cny'])<.01
        print(json.dumps({'baseline_reproduced':args.question,'total_cost_cny':summary['total_cost_cny']},ensure_ascii=False))
        return
    for q in ((2,3) if args.command in ('all','experiments') else (int(args.command[-1]),)):
        if args.command=='experiments':
            summaries[q]=json.loads((ROOT/f'output/q{q}/summary.json').read_text(encoding='utf-8'))
            payload=summaries[q]['config']
        elif args.config:
            payload=json.loads(args.config.read_text(encoding='utf-8'))
            payload=payload.get('chosen',payload.get('config',payload))
        else:payload=None
        if payload:
            payload['update_hours']=tuple(payload['update_hours']);cfg=Config(**payload)
        else:cfg=select_config(engine,q,price,dest)
        configs[q]=cfg
        if args.command!='experiments':
            print(f'Q{q}锁定一月所选配置：{cfg.name}',flush=True)
            records,plans,summary=simulate(engine,q,price,cfg,snapshots=True,progress=True)
            folder=args.output_dir or ROOT/f'output/q{q}'
            write_outputs(folder,paths,q,price,records,plans,summary)
            baseline=json.loads((ROOT/f'output/baseline/q{q}/summary.json').read_text(encoding='utf-8'))
            summary['baseline_total_cost_cny']=baseline['total_cost_cny']
            summary['saving_cny']=baseline['total_cost_cny']-summary['total_cost_cny']
            summary['saving_fraction']=summary['saving_cny']/baseline['total_cost_cny']
            save_json(folder/'summary.json',summary)
            validate_output(folder,q)
            summaries[q]=summary
            print(f"Q{q}总费用{summary['total_cost_cny']:.4f}，较原基线节省{summary['saving_cny']:.4f}",flush=True)
    if args.command=='experiments' or args.experiments:
        if set(configs)!={2,3}:parser.error('完整实验请使用all --experiments或experiments命令')
        run_suite(engine,price,configs,summaries,dest)


if __name__=='__main__':main()
