"""问题1：使用附件1求解确定性单日调度并填写官方result1模板。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .legacy_dispatch import DT_H, solve_day_dispatch
from .io import attachment_paths, file_sha256, read_attachment1, write_result1


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment-dir", type=Path, default=ROOT / "附件")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "q1")
    args = parser.parse_args()
    paths = attachment_paths(ROOT, args.attachment_dir)
    print(f"附件目录：{paths['attachment_dir']}")

    price, load_kw, pv_kw = read_attachment1(paths["data1"])
    load_kwh = load_kw * DT_H
    pv_kwh = pv_kw * DT_H
    result = solve_day_dispatch(
        price, load_kwh, pv_kwh, initial_energy=6000.0, terminal_energy=6000.0
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "result1.xlsx"
    write_result1(
        paths["template1"], output_file,
        np.asarray(result["grid_plan"]), np.asarray(result["charge"]),
        np.asarray(result["discharge"]), np.asarray(result["soc"]),
    )

    balance = (
        np.asarray(result["grid_plan"]) + pv_kwh + np.asarray(result["discharge"])
        - load_kwh - np.asarray(result["charge"]) - np.asarray(result["spill"])
    )
    summary = {
        "model": "deterministic single-day storage dispatch",
        "objective_cost_cny": float(result["cost"]),
        "grid_energy_kwh": float(np.sum(result["grid_plan"])),
        "charge_energy_kwh": float(np.sum(result["charge"])),
        "discharge_energy_kwh": float(np.sum(result["discharge"])),
        "initial_soc_kwh": float(result["soc"][0]),
        "terminal_soc_kwh": float(result["soc"][-1]),
        "soc_min_kwh": float(np.min(result["soc"])),
        "soc_max_kwh": float(np.max(result["soc"])),
        "max_balance_residual_kwh": float(np.max(np.abs(balance))),
        "max_charge_discharge_overlap_kwh": float(result["overlap_max"]),
        "input_sha256": {"附件1.xlsx": file_sha256(paths["data1"])},
        "template_sha256": file_sha256(paths["template1"]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "check.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in summary.items() if not isinstance(value, dict)),
        encoding="utf-8",
    )

    x = np.arange(144)
    plt.figure(figsize=(11, 4))
    plt.plot(x, result["grid_plan"], label="Grid purchase")
    plt.plot(x, result["charge"], label="Charge")
    plt.plot(x, result["discharge"], label="Discharge")
    plt.xlabel("10-minute interval")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "energy_balance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 4))
    plt.plot(np.arange(145), result["soc"])
    plt.xlabel("Time point")
    plt.ylabel("Stored energy (kWh)")
    plt.tight_layout()
    plt.savefig(output_dir / "soc.png", dpi=180)
    plt.close()
    print(f"问题1完成：{output_file}")


if __name__ == "__main__":
    main()
