"""单一执行记录生成官方工作簿、逐段审计和指定日期论文表。"""
import csv,json
from pathlib import Path
import numpy as np
from .io import write_result2,file_sha256,interval_ranges
from .timegrid import interval

KEY_DATES = ('2025-03-20','2025-06-21','2025-09-23','2025-12-21')
TRACE_KEYS = ('grid','grid0','plan_charge','plan_discharge','charge','discharge','emergency',
              'spill','energy','load','pv','load_prediction','pv_prediction','margin')


def save_json(path,content):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(content,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def csv_file(path,fields,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f); writer.writerow(fields);writer.writerows(rows)


def write_outputs(output,paths,question,price,records,plans,summary):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    selected=[r for r in records if r['day']>=31]
    summary['input_sha256']={paths[f'data{i}'].name:file_sha256(paths[f'data{i}']) for i in range(1,question+1)}
    summary['template_sha256']=file_sha256(paths[f'template{question}'])
    summary['source_sha256']={p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')}
    summary['time_convention']='end sample represents preceding 10-minute interval; Q1 unchanged'
    write_result2(paths[f'template{question}'],output/f'result{question}.xlsx',
                  [r['date'] for r in selected],price,selected,question=question)
    arrays={key:np.asarray([r[key] for r in records]) for key in TRACE_KEYS}
    np.savez_compressed(output/'trace.npz',price=price,**arrays)
    if plans:
        np.savez_compressed(output/'plans.npz',
            **{key:np.asarray([p[key] for p in plans]) for key in plans[0]})
    csv_file(output/'daily_summary.csv',
             ['日期','正常费','调整费','紧急费','总费用','紧急电量','未利用供给','日初电量','日末电量','预测方法'],
             ([str(r['date']),r['normal_cost_cny'],r['adjustment_cost_cny'],r['emergency_cost_cny'],r['total_cost_cny'],
               r['emergency'].sum(),r['spill'].sum(),r['energy'][0],r['energy'][-1],'/'.join(r['forecast_methods'])] for r in records))
    for month in range(2,13):
        rs=[r for r in selected if r['date'].month==month]
        error=np.array([(r['load']-r['pv'])*6-r['load_prediction']+r['pv_prediction'] for r in rs])
        summary.setdefault('monthly_metrics',[]).append(dict(month=month,total_cost_cny=sum(r['total_cost_cny'] for r in rs),
            net_mae_kw=float(np.abs(error).mean()),net_rmse_kw=float(np.sqrt(np.mean(error**2))),net_bias_kw=float(error.mean())))
    paper_tables(output,selected,question,price)
    save_json(output/'summary.json',summary)


def paper_tables(output,records,question,price):
    lines=[f'# 第{question}问指定日期结果','',
           '全部电量单位为kWh，费用单位为元；按统一的0:00—24:00时段映射取数。',
           '第三问调整后值为最终正常购电，不是增减差额。费用分项按完整精度计算。','']
    rows1,rows2,rows3=[],[],[]
    for r in records:
        day=str(r['date'])
        if day not in KEY_DATES:continue
        lines += [f'## {day}','','### 表1 购电量及费用','',
                  '| 时间段 | 0点计划 | 最终正常购电 |','|---|---:|---:|']
        for t in (60,72,84,96,108,120):
            row=[day,interval(t),r['grid0'][t],r['grid'][t]];rows1.append(row)
            lines.append(f'| {row[1]} | {row[2]:.4f} | {row[3]:.4f} |')
        lines += ['',f"全天正常购电：{r['grid'].sum():.4f}；正常费：{r['normal_cost_cny']:.4f}；调整费：{r['adjustment_cost_cny']:.4f}；紧急费：{r['emergency_cost_cny']:.4f}；合计：{r['total_cost_cny']:.4f}。",'',
                  '### 表2 充放电及储电量','','| 时间段 | 充电量 | 放电量 |','|---|---:|---:|']
        for b in range(6):
            row=[day,interval(24*b,24*(b+1)),r['charge'][24*b:24*(b+1)].sum(),r['discharge'][24*b:24*(b+1)].sum()]
            rows2.append(row+[r['energy'][0],r['energy'][-1]])
            lines.append(f'| {row[1]} | {row[2]:.4f} | {row[3]:.4f} |')
        lines += ['',f"0点储电量：{r['energy'][0]:.4f}；24点储电量：{r['energy'][-1]:.4f}。",'',
                  '### 表3 紧急购电','','| 时间段 | 购电量 | 费用（逐段计价） |','|---|---:|---:|']
        any_emergency=False
        for left,right,amount in interval_ranges(r['emergency']):
            fee=5*np.dot(price[left:right+1],r['emergency'][left:right+1])
            row=[day,interval(left,right+1),amount,fee];rows3.append(row)
            lines.append(f'| {row[1]} | {amount:.4f} | {fee:.4f} |');any_emergency=True
        if not any_emergency:
            rows3.append([day,'无紧急购电',0.,0.]);lines.append('| 无紧急购电 | 0 | 0 |')
        lines.append('')
    (output/'paper_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    csv_file(output/'table1.csv',['日期','时间段','0点计划kWh','最终正常购电kWh'],rows1)
    csv_file(output/'table2.csv',['日期','时间段','充电kWh','放电kWh','0点储电kWh','24点储电kWh'],rows2)
    csv_file(output/'table3.csv',['日期','时间段','紧急购电kWh','紧急费元'],rows3)
