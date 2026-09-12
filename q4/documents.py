"""Generate the Q4 appendix and a separate, evidence-grounded C evaluation."""
import json
from pathlib import Path
import numpy as np
from microgrid.reporting import save_json

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'output/q4_experiments'


def read(path):return json.loads(path.read_text(encoding='utf8'))


def daily(folder):
    with np.load(folder/'trace.npz',allow_pickle=False) as t:
        g=t['grid'][31:];p=t['price'][31:];e=t['emergency'][31:];g0=t['grid0'][31:]
        return np.sum(p*(g+5*e+.5*np.abs(g-g0)),axis=1)


def confidence(delta):
    rng=np.random.default_rng(20260913);n=len(delta)
    starts=rng.integers(0,n,(1000,int(np.ceil(n/7))))
    ids=(starts[:,:,None]+np.arange(7))%n
    values=delta[ids.reshape(1000,-1)[:,:n]].mean(1)
    return np.quantile(values,[.025,.975]).tolist()


def build():
    b={q:read(ROOT/f'output/q4-{q}/summary.json') for q in (2,3)}
    chosen={q:read(EXP/f'q{q}_C_selection.json')['chosen']['variant'] for q in (2,3)}
    deltas={q:read(EXP/f'q{q}_{chosen[q]}/summary.json')['total_cost_cny']-b[q]['total_cost_cny'] for q in (2,3)}
    leading=f"1月选定的C在Q4-2的费用变化为{deltas[2]:+.4f}元，在Q4-3为{deltas[3]:+.4f}元（相对B，正值表示更贵）。"
    if all(deltas[q]>0 for q in (2,3)):
        leading+='本次数据不支持用该C配置替换B。'
    lines=['# 方案C效果评估：与第四问方案B的实证比较','',
           '本报告位于项目根目录。方案B仍为第四问正式解答；方案C只在独立目录评估，不覆盖B及前三问结果。评价期为2025年2月1日至12月31日，共334天。','',
           leading,'',
           '## 1. 比较协议','',
           '两制度分别在1月确定B的价格预测器和净负荷分位数，再用同一价格预测器比较C。C的误差窗口56天、半衰期28天、最多5条成对场景；总费用CVaR置信水平0.9，权重仅比较0与0.15。C主比较版本在1月确定，没有利用2—12月结果重选。','',
           '| 版本 | 差异 |','|---|---|',
           '| B | 价格点预测＋净负荷经验分位裕度LP |',
           '| C_fixed | 历史净负荷场景，价格仍为同一预测器的点预测 |',
           '| C0 | 相同代表轨迹下增加价格—净负荷联合场景，最小化期望费用 |',
           '| C15 | C0加总费用CVaR，权重0.15 |','',
           'C_fixed与C0采用相同的代表轨迹和权重，区别仅为场景电价系数；因此其差值比直接比较B/C更适合分析联合价格场景的边际作用。C不再叠加B的净负荷分位裕度。所有场景共享购电与储能计划，只有即时余缺按场景变化，实际运行继续采用原储能反馈规则。','',
           '价格残差采用对数比值，与同一历史原点的净负荷误差配对。代表轨迹按“场景价格×正净需求”的累计值分层选择，概率来自原历史权重。该场景缩减可能忽略部分短时尖峰，不是可靠率保证。','',
           r'场景费用为 $C_s=\sum_t p_{s,t}g_t+\frac12\sum_{t\in A_\tau}p_{s,t}a_t+5\sum_t p_{s,t}e_{s,t}$，其中$a_t\ge|g_t-g_t^0|$；Q4-2省略调整项。目标为 $\min\sum_s\pi_sC_s+\lambda[z+(1-\beta)^{-1}\sum_s\pi_sq_s]$，约束$q_s\ge C_s-z$、$q_s\ge0$。各场景价格是已给定系数，因此仍为LP。CVaR针对总费用，区别于前三问既有代码的紧急费用CVaR；风险惩罚不计入真实账单。','',
           '## 2. 一月选择及正式期结果','']
    comparison={}
    for q in (2,3):
        sel=read(EXP/f'q{q}_C_selection.json')
        lines += [f'### Q4-{q}：1月选择{chosen[q]}','',
                  '| 版本 | 1月费用（元） |','|---|---:|']
        for row in sel['candidates']:lines.append(f"| {row['config']['variant']} | {row['total_cost_cny']:.4f} |")
        lines += ['', '| 版本 | 总费用（元） | 相对B变化（元） | 变化率 | 紧急费（元） | 最坏日费用（元） | 日费95%分位（元） |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        rows={'B':b[q],**{name:read(EXP/f'q{q}_{name}/summary.json') for name in ('C_fixed','C0','C15')}}
        for name,row in rows.items():
            delta=row['total_cost_cny']-b[q]['total_cost_cny']
            lines.append(f"| {name} | {row['total_cost_cny']:.4f} | {delta:+.4f} | {delta/b[q]['total_cost_cny']:+.3%} | {row['emergency_cost_cny']:.4f} | {row['worst_daily_cost_cny']:.4f} | {row['daily_cost_p95_cny']:.4f} |")
        main=rows[chosen[q]];diff=main['total_cost_cny']-b[q]['total_cost_cny']
        delta=daily(EXP/f'q{q}_{chosen[q]}')-daily(ROOT/f'output/q4-{q}')
        ci=confidence(delta)
        matched=read(EXP/f'q{q}_C_matched_initial/summary.json')
        comparison[str(q)]={'chosen_C':chosen[q],'C_minus_B_cny':diff,'daily_mean_difference_cny':float(delta.mean()),
                            'daily_mean_difference_block_bootstrap_95pct':ci,'C_lower_cost_days':int((delta<0).sum()),
                            'matched_initial_C_minus_B_cny':matched['total_cost_cny']-b[q]['total_cost_cny'],
                            'joint_price_increment_over_C_fixed_cny':rows['C0']['total_cost_cny']-rows['C_fixed']['total_cost_cny']}
        verb='增加' if diff>=0 else '减少'
        lines += ['',f"一月选出的{chosen[q]}在正式期相对B费用{verb}{abs(diff):.4f}元（{abs(diff)/b[q]['total_cost_cny']:.3%}），334天中有{int((delta<0).sum())}天费用低于B。C0相对C_fixed费用变化为{comparison[str(q)]['joint_price_increment_over_C_fixed_cny']:+.4f}元。",'',
                  '| 版本 | 正常费 | 调整费 | 紧急电量(kWh) | 2月初储电(kWh) | 年末储电(kWh) | 求解仿真秒数 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for name,row in rows.items():
            lines.append(f"| {name} | {row['normal_cost_cny']:.4f} | {row['adjustment_cost_cny']:.4f} | {row['emergency_energy_kwh']:.4f} | {row['february_initial_energy_kwh']:.4f} | {row['terminal_energy_kwh']:.4f} | {row['runtime_seconds']:.2f} |")
        lines += ['',f"将C的2月初电量固定为B的{b[q]['february_initial_energy_kwh']:.4f} kWh后，C正式期费用为{matched['total_cost_cny']:.4f}元，相对B变化{matched['total_cost_cny']-b[q]['total_cost_cny']:+.4f}元。这个对照仅在2月起点重置状态，不用于正式结果工作簿；自然热身版本仍保持全年连续。",'',
                  f"每日费用差(C−B)均值为{delta.mean():+.4f}元；7日循环块bootstrap（固定种子20260913，1000次）给出的均值95%区间为[{ci[0]:+.4f}, {ci[1]:+.4f}]元/日。该区间描述本年配对序列的抽样敏感性，季节非平稳和单年样本限制使其不能代表跨年份保证。",'',
                  '| 月份 | B费用（元） | 所选C费用（元） | C−B（元） |','|---|---:|---:|---:|']
        for bm,cm in zip(b[q]['monthly'],main['monthly']):
            lines.append(f"| {bm['month']} | {bm['total_cost_cny']:.4f} | {cm['total_cost_cny']:.4f} | {cm['total_cost_cny']-bm['total_cost_cny']:+.4f} |")
        lines.append('')
    c3=read(EXP/f'q3_{chosen[3]}/summary.json')
    tradeoff=(f"Q4-3的所选C确实降低了紧急购电：紧急电量从{b[3]['emergency_energy_kwh']:.4f}降至{c3['emergency_energy_kwh']:.4f} kWh，"
              f"紧急费减少{b[3]['emergency_cost_cny']-c3['emergency_cost_cny']:.4f}元；"
              f"但正常费增加{c3['normal_cost_cny']-b[3]['normal_cost_cny']:.4f}元、调整费增加{c3['adjustment_cost_cny']-b[3]['adjustment_cost_cny']:.4f}元，"
              "合计仍更贵。这说明它有降低紧急补缺的作用，但本次并未获得总成本优势。")
    lines += ['## 3. 如何解释结果','',tradeoff,'',
              '“联合考虑更多风险”不保证现金费用更低。B以分位数提高计划净需求，C按代表场景优化余缺代价；两者计划层均未精确建模实际储能反馈的未来价值。联合场景预测失准、少量代表场景及共享储能约束均可能抵消理论上的风险信息优势。CVaR本来就允许用平均费用换取尾部改善，应同时看最坏日、95%分位和紧急费用。','',
              '期末储能价值未计入题目现金费用，不能把不同期末库存完全归因于调度效率。报告列出期末电量，并提供相同2月初电量对照；尚未做统一年末状态约束，因此不能宣称已经完全隔离终端资产差异。运行秒数是本次实测，部分运行与独立复核重叠，未控制机器负载，不作为严格的硬件性能基准。','',
              '是否替换B须同时满足：因果与物理检查通过、预先确定的C在留出期有实际费用优势、最坏日与紧急购电没有不可接受恶化、对场景数及尾部参数不过度敏感。本次只考察5代表场景和两档CVaR权重，未验证场景数/年份稳定性。即使本年C较优，也不在本任务中覆盖用户指定的B。','',
              '## 4. 验证、复现及范围','',
              '新增联合价格算例验证了价格—缺口相关性可以改变最优购电量；未知未来价格与负荷扰动测试验证了此前发布计划不变。所有完整运行均执行逐段物理与真实费用审计。B另检查官方工作簿、连续紧急区间和全部发布计划，并进行独立新实例全年复算，具体见 `output/q4_experiments/reproducibility.json`。','',
              '运行：`python run_q4.py --experiments`；复核：`python -m q4.verify --rerun`；测试：`python -m unittest discover -s tests -v`；原策略新价对照：`python -m q4.reprice`；日内调整对照：`python -m q4.adjustments`；完成这些步骤后重建本报告及文档附录：`python -m q4.documents`。', '',
              '主结果见 `output/q4-2/` 和 `output/q4-3/`；C运行的摘要、逐段轨迹及日费用见 `output/q4_experiments/q2_*`、`q3_*`。事件日志保存既定试验记录，1月选择文件保留全部候选。', '',
              '方法来源：能源MPC参考[Moehle等作者页面](https://stanford.edu/~boyd/papers/dyn_ener_man.html)；CVaR参考[Rockafellar与Uryasev作者出版物目录](https://uryasev.github.io/publications/)。这些文献只支持一般方法来源，本报告中的效果判断全部来自本项目实际计算。','']
    save_json(EXP/'comparison.json',comparison)
    (ROOT/'方案C效果评估.md').write_text('\n'.join(lines),encoding='utf8')
    appendix=make_appendix(b)
    document=ROOT/'docs/前三问完整数学模型与解题思路.md'
    marker=b'\n\n<!-- Q4_APPENDIX_BEGIN -->\n'
    original=document.read_bytes().split(marker)[0]
    document.write_bytes(original+marker+appendix.encode('utf8'))


def make_appendix(b):
    text=r'''## 13. 第四问补充：方案B的完整模型与求解

本节为新增内容；第1—12节保留前三问原有模型、参数、数值与说明。第四问在独立 `q4/` 模块中实现，通过 `run_q4.py` 运行，不改变 `microgrid/` 或前三问入口。原文“附件4不进入本文前三问模型”等描述仍仅指前三问；附件4在本节用于第四问。

### 13.1 问题分解与信息假设

Q4-2为波动电价下每天0点一次提交正常购电的模型，Q4-3允许0/6/12/18点提交/修改尚未执行的购电。两者均使用附件2真实负荷/光伏及附件4真实价格执行和结算；Q4-3另外使用附件3在对应时刻发布的光伏预测，Q4-2延续第二问的历史预测信息条件。

题目未给出价格预先公告制度，本节将未来价格视为未知。在时刻τ，计划只依赖此前已经观测的价格、负荷、光伏、真实储电量及已经发布的光伏预报。实际费用按供电时段附件4价格核算，价格预测不视为成交报价。多次调整按最终正常量相对0点计划计偏差，延续第三问口径，不另引入逐次收费制度。

附件4为365×144价格矩阵，日期与附件2逐日一致，全部价格有限且大于0。使用段终采样代表前十分钟的约定，首列对应0:00—0:10，末列对应23:50—24:00；只在输出副本纠正原模板的标签错位，不更改任何原附件。

### 13.2 变量、单位和物理约束

沿用前三问符号：正常购电g、0点计划g⁰、充电c、放电r、紧急购电e、未利用供给s、储电量E，单位均为kWh；负荷与光伏功率乘Δ=1/6小时得到L、V。实际价格p及预测价格p̂的单位为元/kWh。

$$
g_{d,t}+V_{d,t}+r_{d,t}+e_{d,t}=L_{d,t}+c_{d,t}+s_{d,t},
$$

$$
E_{d,t}=E_{d,t-1}+0.9c_{d,t}-r_{d,t}/0.9,
\qquad1200\le E_{d,t}\le10800,
$$

$$
0\le c_{d,t},r_{d,t}\le5000/6,\qquad g_{d,t},e_{d,t},s_{d,t}\ge0.
$$

2025年1月1日0点储电6000 kWh，之后跨日连续，不强制每天日初日末相等。两问统一首日零正常购电冷启动，实际储能反馈和紧急购电保证供电；首日费用只计入热身。允许未利用供给，不售电、不添加未给定的外网功率上限或电池寿命费用。充放电效率各90%，往返81%。数值边界沿用原模块1e-6 kWh内缩及二级吞吐量优化；所有计划及实际轨迹检查充放电互斥。

### 13.3 因果电价预测

价格预测候选为昨日同刻与七日加权：

$$
\widehat p^{(0)}_{d,t}=p_{d-1,t},
\qquad\text{或}\qquad
\widehat p^{(0)}_{d,t}=\sum_{j=1}^{\min(7,d-1)}w_jp_{d-j,t}.
$$

七日原权重为(0.40,0.25,0.15,0.10,0.05,0.03,0.02)，历史不足时归一化。每日价格预测仅用此前完整日；跨日目标的未观测滞后值递归使用同一原点预测，不取未来实际价格。

Q4-3在6/12/18点，用之前6个十分钟段价格误差的中位数bτ修正未来预测。限幅Bτ取当日此前已经实现价格中位数的25%，h为窗口第1—144段：

$$
\widehat p_{\tau,h}=\max\left(10^{-6},\widehat p^{\rm base}_{\tau,h}
+\operatorname{clip}(b_\tau,-B_\tau,B_\tau)e^{-h/36}\right).
$$

这里36段对应6小时衰减尺度；25%和6小时是预先设定的模型参数，未用留出期优化。1e-6元/kWh仅是数值正下界，不来自全年最小值。首日无历史时声明0.7元/kWh先验，但首日正常购电固定为0，因此该先验不会驱动正常购电优化。

Q4-2只在0点产生价格计划，不在日内重购正常电量。Q4-3的预测更新只使用当前发布时刻前已实现的价格，不把未来真实价格循环进24小时窗口。

### 13.4 净负荷预测与分位安全裕度

复用原 `ForecastEngine`：因果自适应选择昨日、上周、七日加权或岭回归预测，Q4-3使用当前发布光伏预报和过去一小时负荷偏差修正。按同一发布时间已经完整实现的历史24小时预测窗口计算净负荷误差：

$$
\xi_{j,h}=(P^L_{j,h}-P^V_{j,h})-(\widehat P^L_{j,h}-\widehat P^V_{j,h}),
$$

$$
\widetilde N_{\tau,h}=\frac16\left(\widehat P^L_{\tau,h}-\widehat P^V_{\tau,h}
+Q_\alpha^{(w)}(\xi_{j,h})\right).
$$

历史窗口56天，时间权重半衰期28天。保留负净需求以表达光伏过剩。分位数为经验风险裕度，不是全天供电可靠率保证。

### 13.5 Q4-2计划优化

每天0点，对144段安全净需求求解：

$$
\min J_{4-2}=\sum_{t=1}^{144}\widehat p_{d,t}g^0_{d,t},
\qquad g^0_{d,t}+r_{d,t}-c_{d,t}-s_{d,t}=\widetilde N_{d,t},
$$

同时满足储能递推和边界。计划阶段紧急量为0，风险通过净负荷裕度体现；实际缺口仍按规则紧急补足。求得的g⁰全天固定。依次以实际净负荷执行：截取可行计划动作；缺电先取消充电、再补充放电；过剩先取消放电、再充电；剩余缺口紧急购电，剩余供给弃置。

### 13.6 Q4-3滚动优化

0点生成全天g⁰；6/12/18点以真实储电量为初值重新优化未来24小时。设Aτ为当日已提交合同且尚未执行的位置，引入a：

$$
\min J_{4-3,\tau}=\sum_{h=1}^{144}\widehat p_{\tau,h}g_{\tau,h}
+\frac12\sum_{h\in A_\tau}\widehat p_{\tau,h}a_{\tau,h},
$$

$$
a_{\tau,h}\ge g_{\tau,h}-g^0_h,\qquad
a_{\tau,h}\ge g^0_h-g_{\tau,h},\qquad a_{\tau,h}\ge0.
$$

供需与储能约束同Q4-2。跨入次日的部分只作前瞻，尚未提交次日合同，因此不收调整费。每次只执行接下来6小时，内部逐十分钟反馈；不修改过去执行量。每天固定四次优化，没有引入根据未来实际收益决定是否调整的机制。

窗口末端不设额外终端价值或回到6000的硬约束，沿用原主模型的无备用惩罚结构。12月31日跨入下一年的负荷/价格由同原点递归预测，光伏使用附件3所给未来24小时预报；超出评价年的部分只参与前瞻，不计入本年实际费用。这一有限窗口处理存在终端价值近似，结果同时列出年末储电量。

### 13.7 实际计费

设D为2月1日—12月31日。使用附件4真实p，而不是预测p̂：

$$
C_{4-2}=\sum_{d\in D,t}p_{d,t}(g^0_{d,t}+5e_{d,t}),
$$

$$
C_{4-3}=\sum_{d\in D,t}\left[p_{d,t}g_{d,t}+0.5p_{d,t}|g_{d,t}-g^0_{d,t}|+5p_{d,t}e_{d,t}\right].
$$

减购部分收50%违约价，增购部分合计按150%计费，等价于正常最终量费用加50%绝对偏差费用。不能再把0点计划总费加入一次。正常购电未使用仍收费；紧急购电不等于停电；未利用供给不能全部解释为弃光。

### 13.8 选参和计算结果

每个制度各比较2种价格预测×3个分位数(0.50/0.65/0.80)，按1月实际总费用最低选择，0.01元内优先简单预测和较低分位数。2月以后只更新历史模型与误差池，不以正式期费用重新选参。

'''
    text+='| 制度 | 价格预测 | 分位数 | 1月所选配置费用（元） |\n|---|---|---:|---:|\n'
    for q in (2,3):
        s=b[q];text+=f"| Q4-{q} | {s['config']['price_method']} | {s['config']['quantile']} | {s['january_warmup_cost_cny']:.4f} |\n"
    text+='\n下列结果均为方案B、334天、未将1月热身费用混入正式费用。\n\n| 指标 | Q4-2 | Q4-3 |\n|---|---:|---:|\n'
    fields=[('总费用（元）','total_cost_cny'),('正常购电费（元）','normal_cost_cny'),('调整费（元）','adjustment_cost_cny'),('紧急购电费（元）','emergency_cost_cny'),('正常购电量（kWh）','grid_energy_kwh'),('紧急购电量（kWh）','emergency_energy_kwh'),('未利用供给（kWh）','spill_energy_kwh'),('2月初储电（kWh）','february_initial_energy_kwh'),('年末储电（kWh）','terminal_energy_kwh'),('价格预测MAE（元/kWh）','price_mae'),('最坏日费用（元）','worst_daily_cost_cny')]
    for label,key in fields:text+=f'| {label} | {b[2][key]:.4f} | {b[3][key]:.4f} |\n'
    text+='\nQ4-2与Q4-3同时在预报信息、调整制度、所选参数及初始状态上有差异，因此二者总费差不能直接全部归因于某一次日内预报。\n'
    repriced=read(EXP/'original_policy_repriced.json')
    text+='\n原第二、三问已保存策略仅用附件4重新计费的对照如下；该对照保持旧动作，不是波动价下重优化，且与B策略并非单因素差异。\n\n| 制度 | 原策略按波动价计费 | B费用 | B费用减少额 |\n|---|---:|---:|---:|\n'
    for q in (2,3):
        old=repriced[str(q)]['total_cost_cny'];text+=f'| Q4-{q} | {old:.4f} | {b[q]["total_cost_cny"]:.4f} | {old-b[q]["total_cost_cny"]:.4f} |\n'
    control=read(EXP/'q3_B_no_updates/summary.json')
    improvement=control['total_cost_cny']-b[3]['total_cost_cny']
    text+=f'\n**日内预报与调整的价值对照：** 使用相同0点光伏预报来源、相同B参数和相同2月初储电量，若全天只保留0点计划、不在6/12/18点更新，费用为{control["total_cost_cny"]:.4f}元；四次发布时间的B费用为{b[3]["total_cost_cny"]:.4f}元，差值（无更新−有更新）为{improvement:+.4f}元（相对无更新费用{improvement/control["total_cost_cny"]:+.3%}）。这评价的是后续光伏预报、价格修正、负荷修正和计划调整的整体效果，不是单独价格更新的因果贡献。未遍历全部更新时间子集，也不宣称每一次更新都必须采用。该解释性对照不参与主模型重新选参。\n'
    text+='''
### 13.9 指定日期完整结果

以下按题目表1—3列出2025-03-20、06-21、09-23、12-21的购电、储能和紧急区间。按完整精度计算后展示四位小数。

'''
    for q in (2,3):
        tables=(ROOT/f'output/q4-{q}/paper_tables.md').read_text(encoding='utf8')
        # Keep the appended document heading hierarchy below this section.
        converted=[]
        for line in tables.splitlines():
            if line.startswith('# '):line=f'#### Q4-{q}'
            elif line.startswith('## '):line='#'*5+line[2:]
            elif line.startswith('### '):line='#'*6+line[3:]
            converted.append(line)
        text+='\n'.join(converted)+'\n\n'
    text+='''### 13.10 验证与代码索引

原测试11项和新增测试5项均通过。新增测试覆盖价格跨日原点、联合价格—缺口决策差异、总费用CVaR/调整约束、未知未来扰动与发布计划不变、Q4-2正常量冻结。全年B结果还通过365×144供需、储能、边界、互斥、跨日连续、334天官方表格及真实价格独立计费检查，并从全新预测实例复算比较逐段轨迹。数值门槛为物理/轨迹1e-6 kWh、总费用1e-5元。

| 内容 | 文件 |
|---|---|
| 独立运行 | `run_q4.py`；`python run_q4.py --experiments` |
| 价格预测、B优化、C场景与仿真 | `q4/model.py` |
| 读取、选参、输出组织 | `q4/runner.py` |
| 第四问模板与论文表 | `q4/workbooks.py`、`q4/tables.py` |
| 物理、费用、工作簿审计 | `q4/audit.py` |
| 保存结果复核及独立重算 | `q4/verify.py`；`python -m q4.verify --rerun` |
| 官方结果 | `output/q4-2/result4-2.xlsx`、`output/q4-3/result4-3.xlsx` |
| 冻结结果与逐段证据 | 两目录的`summary.json`、`frozen_numbers.json`、`trace.npz`、`plans.npz`、`validation.json` |
| 一月候选/方案C比较 | `output/q4_experiments/` |
| 方案C单独评价 | 根目录`方案C效果评估.md` |

本文对方案B给出完整实现及数值。方案C只作独立效果评估，不据其结果覆盖B，也不修改前三问任何方法。限于只有一年数据、价格点预测、经验裕度和无显式终端价值，本结果不宣称跨年份最优或概率可靠性保证。
'''
    return text


if __name__=='__main__':build()
