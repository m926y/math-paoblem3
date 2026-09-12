"""Existing paper-table format with actual per-day settlement prices."""
import numpy as np
from microgrid.reporting import KEY_DATES,csv_file
from microgrid.io import interval_ranges
from microgrid.timegrid import interval

def paper_tables(output,records,question,price):
    lines=[f'# 第四问 Q4-{question} 指定日期结果','',
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
            fee=5*np.dot(r['price'][left:right+1],r['emergency'][left:right+1])
            row=[day,interval(left,right+1),amount,fee];rows3.append(row)
            lines.append(f'| {row[1]} | {amount:.4f} | {fee:.4f} |');any_emergency=True
        if not any_emergency:
            rows3.append([day,'无紧急购电',0.,0.]);lines.append('| 无紧急购电 | 0 | 0 |')
        lines.append('')
    (output/'paper_tables.md').write_text('\n'.join(lines),encoding='utf-8')
    csv_file(output/'table1.csv',['日期','时间段','0点计划kWh','最终正常购电kWh'],rows1)
    csv_file(output/'table2.csv',['日期','时间段','充电kWh','放电kWh','0点储电kWh','24点储电kWh'],rows2)
    csv_file(output/'table3.csv',['日期','时间段','紧急购电kWh','紧急费元'],rows3)
