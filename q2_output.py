"""问题2各模型版本的统一输出。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_io import write_result2


def write_outputs(output_dir: Path, template: Path, dates, price: np.ndarray,
                  records: list[dict], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # 旧版本的逐日1月热身表没有统一信息边界，避免它与当前摘要并存造成误读。
    obsolete_warmup = output_dir / "january_warmup.csv"
    if obsolete_warmup.is_file():
        obsolete_warmup.unlink()
    write_result2(template, output_dir / "result2.xlsx", dates, price, records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "check.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in summary.items() if not isinstance(value, dict)),
        encoding="utf-8",
    )
    headers = [
        "日期", "正常购电成本", "紧急购电成本", "紧急购电量", "弃电量",
        "日总成本", "日初SOC", "日终SOC", "实际再调度电量",
    ]
    with (output_dir / "daily_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for date_value, record in zip(dates, records):
            writer.writerow([
                date_value.isoformat(), record["normal_cost"], record["emergency_cost"],
                float(np.sum(record["emergency"])), float(np.sum(record["spill"])),
                record["normal_cost"] + record["emergency_cost"],
                float(record["soc"][0]), float(record["soc"][-1]), record["recourse_energy"],
            ])

    normal = np.array([record["normal_cost"] for record in records])
    emergency = np.array([record["emergency_cost"] for record in records])
    end_soc = np.array([record["soc"][-1] for record in records])
    emergency_energy = np.array([np.sum(record["emergency"]) for record in records])
    for filename, ylabel, series in (
        ("daily_cost.png", "Cost (CNY)", (normal, emergency)),
        ("daily_soc.png", "End-of-day stored energy (kWh)", (end_soc,)),
        ("daily_emergency.png", "Emergency purchase (kWh)", (emergency_energy,)),
    ):
        plt.figure(figsize=(11, 4))
        for values in series:
            plt.plot(dates, values)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()
