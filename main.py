# -*- coding: utf-8 -*-

"""
问题1：
微网购电与储能调度 MILP模型

输入：
附件1.xlsx

输出：
result1.xlsx
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pulp import *


# ===============================
# 1. 固定随机种子
# ===============================

SEED = 2026

random.seed(SEED)
np.random.seed(SEED)


# ===============================
# 2. 文件路径
# ===============================

input_file = "附件1.xlsx"

output_file = "result1.xlsx"

figure_dir = "figures"

os.makedirs(figure_dir, exist_ok=True)



# ===============================
# 3. 读取附件
# ===============================

data = pd.read_excel(
    input_file
)


print(data.head())



# ===============================
# 4. 数据预处理
# ===============================


price = data["电价"].values

load_kw = data["小区负载"].values

pv_kw = data["光伏发电预测功率"].values


# 时间间隔

dt = 10 / 60


# kW转换为kWh

load = load_kw * dt

pv = pv_kw * dt


T = len(price)



# ===============================
# 5. 建立MILP模型
# ===============================


model = LpProblem(
    "MicroGrid_MILP",
    LpMinimize
)



# --------变量--------

grid = [
    LpVariable(
        f"grid_{t}",
        lowBound=0
    )
    for t in range(T)
]


charge = [
    LpVariable(
        f"charge_{t}",
        lowBound=0
    )
    for t in range(T)
]


discharge = [
    LpVariable(
        f"discharge_{t}",
        lowBound=0
    )
    for t in range(T)
]


energy = [
    LpVariable(
        f"E_{t}",
        lowBound=1200,
        upBound=10800
    )
    for t in range(T+1)
]


binary = [
    LpVariable(
        f"u_{t}",
        cat="Binary"
    )
    for t in range(T)
]



# ===============================
# 6. 目标函数
# ===============================

model += lpSum(
    price[t]*grid[t]
    for t in range(T)
)



# ===============================
# 7. 约束条件
# ===============================


# 初始储能

model += energy[0] == 6000


# 日终储能

model += energy[T] == 6000



eta = 0.9

Qmax = 5000*dt



for t in range(T):


    # 能量平衡

    model += (
        grid[t]
        + pv[t]
        + discharge[t]
        ==
        load[t]
        + charge[t]
    )



    # 储能变化

    model += (
        energy[t+1]
        ==
        energy[t]
        + eta*charge[t]
        - discharge[t]/eta
    )



    # 充电限制

    model += (
        charge[t]
        <=
        Qmax*binary[t]
    )


    # 放电限制

    model += (
        discharge[t]
        <=
        Qmax*(1-binary[t])
    )



# ===============================
# 8. 求解
# ===============================


solver = PULP_CBC_CMD(
    msg=True
)


model.solve(solver)


print(
    "求解状态:",
    LpStatus[model.status]
)



# ===============================
# 9. 保存结果
# ===============================


result_buy = pd.DataFrame(
    {
        "时间":
        data["时间"],

        "购电量(kWh)":
        [
            value(x)
            for x in grid
        ]
    }
)



result_storage = pd.DataFrame(
    {
        "时间":
        data["时间"],

        "充电量(kWh)":
        [
            value(x)
            for x in charge
        ],

        "放电量(kWh)":
        [
            value(x)
            for x in discharge
        ]
    }
)


result_energy = pd.DataFrame(
    {
        "时刻":
        range(T+1),

        "储能量(kWh)":
        [
            value(x)
            for x in energy
        ]
    }
)



with pd.ExcelWriter(output_file) as writer:

    result_buy.to_excel(
        writer,
        sheet_name="计划购电量",
        index=False
    )

    result_storage.to_excel(
        writer,
        sheet_name="充放电量",
        index=False
    )

    result_energy.to_excel(
        writer,
        sheet_name="储能状态",
        index=False
    )



# ===============================
# 10. 绘图
# ===============================

plt.figure(figsize=(10,4))


grid_value = [
    value(x)
    for x in grid
]


charge_value = [
    value(x)
    for x in charge
]


discharge_value = [
    value(x)
    for x in discharge
]


plt.plot(
    grid_value,
    label="购电"
)


plt.plot(
    charge_value,
    label="充电"
)


plt.plot(
    discharge_value,
    label="放电"
)


plt.legend()

plt.xlabel("时间段")

plt.ylabel("kWh")




plt.savefig(
    figure_dir+"/energy_balance.png",
    dpi=300
)

plt.close()



plt.figure(figsize=(10,4))


plt.plot(
    [
        value(x)
        for x in energy
    ]
)


plt.xlabel("时间段")

plt.ylabel("储能量(kWh)")


plt.savefig(
    figure_dir+"/storage.png",
    dpi=300
)


plt.close()



print("计算完成")
print("结果已保存:", output_file)