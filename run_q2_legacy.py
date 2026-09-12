"""问题2工程基线：保留旧1月初始化逻辑，只用于数值回归。"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from project_io import attachment_paths, file_sha256, read_attachment1, read_attachment2
from q2_core import simulate_legacy, summarize
from q2_output import write_outputs


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment-dir", type=Path, default=ROOT / "附件")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "engineering_baseline" / "q2")
    args = parser.parse_args()
    paths = attachment_paths(ROOT, args.attachment_dir)
    print(f"附件目录：{paths['attachment_dir']}")
    price, _, _ = read_attachment1(paths["data1"])
    _, load_kw, pv_kw = read_attachment2(paths["data2"])
    records, checks = simulate_legacy(price, load_kw, pv_kw)
    summary = summarize(records, checks)
    summary["input_sha256"] = {
        "附件1.xlsx": file_sha256(paths["data1"]),
        "附件2.xlsx": file_sha256(paths["data2"]),
    }
    summary["template_sha256"] = file_sha256(paths["template2"])
    dates = [date(2025, 2, 1) + timedelta(days=i) for i in range(len(records))]
    write_outputs(args.output_dir.resolve(), paths["template2"], dates, price, records, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
