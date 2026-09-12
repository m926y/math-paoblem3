"""第二、三问：段终采样代表其前十分钟，区间左闭右开。"""
from datetime import datetime, time
import numpy as np

T = 144
DT = 1/6


def clock(index):
    return '24:00' if index == T else f'{index//6}:{index%6*10:02d}'


def interval(index, end=None):
    return f'{clock(index)}-{clock(index+1 if end is None else end)}'


def validate_end_labels(labels):
    if len(labels) != T:
        raise ValueError('时段标签必须有144个')
    for i, value in enumerate(labels, 1):
        if isinstance(value, datetime):
            value = value.time()
        if isinstance(value, time):
            minute = value.hour*60+value.minute+value.second/60
        elif isinstance(value, (float, int)):
            minute = float(value)*1440
        elif str(value) in ('0:00+1', '24:00', '24:00:00'):
            minute = 1440
        else:
            try:
                hour, mins = str(value).split(':')[:2]
                minute = int(hour)*60+int(mins)
            except (ValueError, TypeError):
                raise ValueError(f'无效采样时标: {value!r}') from None
        if i == T and minute == 0:
            minute = 1440
        if not np.isclose(minute, i*10, atol=1e-5):
            raise ValueError(f'第{i}个时标应为段终{i*10}分钟，实际{value!r}')
