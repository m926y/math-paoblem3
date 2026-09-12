#!/usr/bin/env python3
"""C题问题3：规定时刻调整购电、因果负荷修正和储能闭环运行。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from q3_dispatch import (
    DispatchParams,
    execute_causal_block,
    solve_mpc,
)
from q3_forecast import (
    ISSUE_INDEX,
    T,
    build_load_unfavourable_errors,
    build_pv_hourly_errors,
    corrected_load_horizon,
    load_profiles,
    pv_safe_horizon,
    repeat_profile,
)
from q3_xlsx_io import read_inputs, write_result3
from project_io import attachment_paths, file_sha256


# 每6小时执行一次，每小时6个十分钟时段
BLOCK = 36


def _validate_files(paths: dict[str, Path]) -> None:
    """检查所有输入文件是否存在。"""

    missing = [
        f"{name}：{path}"
        for name, path in paths.items()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "以下输入文件不存在：\n" + "\n".join(missing)
        )


def simulate(
    dates,
    price: np.ndarray,
    fallback_load_kw: np.ndarray,
    load_actual_kw: np.ndarray,
    pv_actual_kw: np.ndarray,
    pv_forecasts_kw: np.ndarray,
    params: DispatchParams,
    quantile: float,
    shrink_k: float,
    load_bias_decay_hours: float,
    load_bias_clip_fraction: float,
    update_hours: tuple[int, ...] = (0, 6, 12, 18),
    terminal_value_factor: float = 0.0,
) -> tuple[list[dict], dict]:
    """执行全年滚动优化。

    从1月1日开始运行，1月用于历史初始化和SOC传递。
    正式结果从2月1日开始输出，不在2月1日重置SOC。
    """

    if 0 not in update_hours or any(hour not in (0, 6, 12, 18) for hour in update_hours):
        raise ValueError("update_hours必须是包含0的0/6/12/18子集")
    terminal_value = terminal_value_factor * float(np.median(price))

    load_errors = build_load_unfavourable_errors(
        load_actual_kw,
        fallback_load_kw,
    )

    pv_errors = build_pv_hourly_errors(
        pv_actual_kw,
        pv_forecasts_kw,
    )

    records: list[dict] = []

    # 2025年1月1日初始储能量
    soc_now = params.e_initial

    # 数值校验指标
    max_plan_balance = 0.0
    max_plan_soc_residual = 0.0
    max_actual_balance = 0.0
    max_actual_soc_residual = 0.0

    for day, date_value in enumerate(dates):
        # ---------------------------------------------------------------
        # 1. 当日负荷预测
        # ---------------------------------------------------------------

        raw_load_kw, safe_load_kw = load_profiles(
            day=day,
            load_kw=load_actual_kw,
            fallback=fallback_load_kw,
            errors=load_errors,
            quantile=quantile,
            shrink_k=shrink_k,
        )

        # ---------------------------------------------------------------
        # 2. 0:00负荷预测
        # ---------------------------------------------------------------

        load_horizon_0, _ = corrected_load_horizon(
            day=day,
            start=0,
            raw_profile_kw=raw_load_kw,
            safe_profile_kw=safe_load_kw,
            actual_load_kw=load_actual_kw,
            decay_hours=load_bias_decay_hours,
            bias_clip_fraction=load_bias_clip_fraction,
        )

        # ---------------------------------------------------------------
        # 3. 0:00光伏预测
        # ---------------------------------------------------------------

        pv_horizon_0 = pv_safe_horizon(
            day=day,
            issue_no=0,
            pv_actual_kw=pv_actual_kw,
            forecasts_kw=pv_forecasts_kw,
            errors=pv_errors,
            quantile=quantile,
            shrink_k=shrink_k,
        )

        # ---------------------------------------------------------------
        # 4. 0:00形成全天原始购电计划
        # ---------------------------------------------------------------

        base_solution = solve_mpc(
            price=price,
            load_kwh=load_horizon_0 * params.dt_h,
            pv_kwh=pv_horizon_0 * params.dt_h,
            initial_soc=soc_now,
            params=params,
            terminal_value_per_kwh=terminal_value,
        )

        grid_plan_0 = np.asarray(
            base_solution["grid"],
            dtype=float,
        ).copy()

        # 当日最终执行结果
        grid_final = np.zeros(T)
        charge_actual = np.zeros(T)
        discharge_actual = np.zeros(T)
        soc_actual = np.zeros(T)
        emergency = np.zeros(T)
        spill_actual = np.zeros(T)
        recourse = np.zeros(T)

        # 记录0/6/12/18点负荷偏差修正
        issue_load_bias_kw = np.zeros(4)

        day_initial_soc = soc_now
        active_solution = base_solution
        active_start = 0

        # ---------------------------------------------------------------
        # 5. 0:00、6:00、12:00、18:00滚动优化
        # ---------------------------------------------------------------

        for issue_no, start in enumerate(ISSUE_INDEX):
            if issue_no == 0:
                # 0点直接使用已经形成的原始计划
                solution = base_solution
                solution_offset = 0

            elif start // 6 in update_hours:
                # -------------------------------------------------------
                # 5.1 使用当前时刻之前已经观测到的负荷修正预测
                # -------------------------------------------------------

                load_horizon_kw, bias_kw = corrected_load_horizon(
                    day=day,
                    start=start,
                    raw_profile_kw=raw_load_kw,
                    safe_profile_kw=safe_load_kw,
                    actual_load_kw=load_actual_kw,
                    decay_hours=load_bias_decay_hours,
                    bias_clip_fraction=load_bias_clip_fraction,
                )

                issue_load_bias_kw[issue_no] = bias_kw

                # -------------------------------------------------------
                # 5.2 使用当前发布时刻的未来24小时光伏预测
                # -------------------------------------------------------

                pv_horizon_kw = pv_safe_horizon(
                    day=day,
                    issue_no=issue_no,
                    pv_actual_kw=pv_actual_kw,
                    forecasts_kw=pv_forecasts_kw,
                    errors=pv_errors,
                    quantile=quantile,
                    shrink_k=shrink_k,
                )

                # 从当前时刻开始构造未来24小时电价
                horizon_price = repeat_profile(
                    price,
                    start,
                )

                # 当天剩余时段以0点计划作为调整结算基准
                # 跨入次日的部分尚未形成次日0点计划，因此设置为NaN
                reference = np.full(T, np.nan)

                reference[: T - start] = grid_plan_0[start:]

                # -------------------------------------------------------
                # 5.3 求解滚动调整模型
                # -------------------------------------------------------

                solution = solve_mpc(
                    price=horizon_price,
                    load_kwh=load_horizon_kw * params.dt_h,
                    pv_kwh=pv_horizon_kw * params.dt_h,
                    initial_soc=soc_now,
                    params=params,
                    plan_reference=reference,
                    terminal_value_per_kwh=terminal_value,
                )
                active_solution = solution
                active_start = start
                solution_offset = 0

            else:
                # 未启用当前发布时间时，继续执行最近一次滚动解中的对应部分。
                solution = active_solution
                solution_offset = start - active_start

            # -----------------------------------------------------------
            # 6. 检查计划阶段数值误差
            # -----------------------------------------------------------

            max_plan_balance = max(
                max_plan_balance,
                float(solution["balance_residual_max"]),
            )

            max_plan_soc_residual = max(
                max_plan_soc_residual,
                float(solution["soc_residual_max"]),
            )

            # 每次只执行接下来的6小时
            take = min(BLOCK, T - start)

            day_slice = slice(
                start,
                start + take,
            )

            # -----------------------------------------------------------
            # 7. 按实际负荷和实际光伏闭环执行
            # -----------------------------------------------------------

            executed = execute_causal_block(
                grid=np.asarray(solution["grid"])[solution_offset:solution_offset+take],
                charge_plan=np.asarray(
                    solution["charge_plan"]
                )[solution_offset:solution_offset+take],
                discharge_plan=np.asarray(
                    solution["discharge_plan"]
                )[solution_offset:solution_offset+take],
                load_actual_kwh=(
                    load_actual_kw[day, day_slice]
                    * params.dt_h
                ),
                pv_actual_kwh=(
                    pv_actual_kw[day, day_slice]
                    * params.dt_h
                ),
                initial_soc=soc_now,
                params=params,
            )

            # -----------------------------------------------------------
            # 8. 保存当前6小时执行结果
            # -----------------------------------------------------------

            grid_final[day_slice] = np.asarray(
                solution["grid"]
            )[solution_offset:solution_offset+take]

            charge_actual[day_slice] = np.asarray(
                executed["charge"]
            )

            discharge_actual[day_slice] = np.asarray(
                executed["discharge"]
            )

            soc_actual[day_slice] = np.asarray(
                executed["soc"]
            )[1:]

            emergency[day_slice] = np.asarray(
                executed["emergency"]
            )

            spill_actual[day_slice] = np.asarray(
                executed["spill"]
            )

            recourse[day_slice] = np.asarray(
                executed["recourse"]
            )

            # 更新实际执行阶段的数值误差
            max_actual_balance = max(
                max_actual_balance,
                float(executed["balance_residual_max"]),
            )

            max_actual_soc_residual = max(
                max_actual_soc_residual,
                float(executed["soc_residual_max"]),
            )

            # 把实际SOC传递到下一次滚动时刻
            soc_now = float(
                np.asarray(executed["soc"])[-1]
            )

        # ---------------------------------------------------------------
        # 9. 保存当天完整结果
        # ---------------------------------------------------------------

        records.append(
            {
                "date": date_value,
                "soc_initial": day_initial_soc,
                "grid_plan_0": grid_plan_0,
                "grid_final": grid_final,
                "charge_actual": charge_actual,
                "discharge_actual": discharge_actual,
                "soc_actual": soc_actual,
                "emergency": emergency,
                "spill_actual": spill_actual,
                "recourse": recourse,
                "issue_load_bias_kw": issue_load_bias_kw,
            }
        )

        if (
            (day + 1) % 30 == 0
            or day + 1 == len(dates)
        ):
            print(
                f"已完成{day + 1}/{len(dates)}天，"
                f"日末SOC={soc_now:.6f} kWh",
                flush=True,
            )

    # 记录2月1日初始SOC
    february_record = next(
        record
        for record in records
        if record["date"].month == 2
    )

    checks = {
        "january_initial_soc_kwh":
            params.e_initial,

        "february_initial_soc_kwh":
            float(february_record["soc_initial"]),

        "max_plan_balance_residual_kwh":
            max_plan_balance,

        "max_plan_soc_residual_kwh":
            max_plan_soc_residual,

        "max_actual_balance_residual_kwh":
            max_actual_balance,

        "max_actual_soc_residual_kwh":
            max_actual_soc_residual,

        "update_hours":
            list(update_hours),

        "terminal_value_factor":
            terminal_value_factor,

        "terminal_value_per_kwh":
            terminal_value,
    }

    return records, checks


def summarize(
    records: list[dict],
    price: np.ndarray,
    params: DispatchParams,
    checks: dict,
) -> dict:
    """汇总2月至12月结果。"""

    output_records = [
        record
        for record in records
        if record["date"].month >= 2
    ]

    january_records = [
        record
        for record in records
        if record["date"].month == 1
    ]

    def total_purchase_cost(
        selected_records: list[dict],
    ) -> float:
        normal_cost = sum(
            float(
                np.dot(
                    price,
                    record["grid_final"],
                )
            )
            for record in selected_records
        )

        adjustment_cost = sum(
            float(
                0.5
                * np.dot(
                    price,
                    np.abs(
                        record["grid_final"]
                        - record["grid_plan_0"]
                    ),
                )
            )
            for record in selected_records
        )

        emergency_cost = sum(
            float(
                5.0
                * np.dot(
                    price,
                    record["emergency"],
                )
            )
            for record in selected_records
        )

        return (
            normal_cost
            + adjustment_cost
            + emergency_cost
        )

    plan_energy = sum(
        float(np.sum(record["grid_plan_0"]))
        for record in output_records
    )

    final_energy = sum(
        float(np.sum(record["grid_final"]))
        for record in output_records
    )

    emergency_energy = sum(
        float(np.sum(record["emergency"]))
        for record in output_records
    )

    plan_cost = sum(
        float(
            np.dot(
                price,
                record["grid_plan_0"],
            )
        )
        for record in output_records
    )

    normal_cost = sum(
        float(
            np.dot(
                price,
                record["grid_final"],
            )
        )
        for record in output_records
    )

    adjustment_cost = sum(
        float(
            0.5
            * np.dot(
                price,
                np.abs(
                    record["grid_final"]
                    - record["grid_plan_0"]
                ),
            )
        )
        for record in output_records
    )

    emergency_cost = sum(
        float(
            5.0
            * np.dot(
                price,
                record["emergency"],
            )
        )
        for record in output_records
    )

    soc_min = min(
        float(np.min(record["soc_actual"]))
        for record in output_records
    )

    soc_max = max(
        float(np.max(record["soc_actual"]))
        for record in output_records
    )

    simultaneous_charge_discharge = max(
        float(
            np.max(
                np.minimum(
                    record["charge_actual"],
                    record["discharge_actual"],
                )
            )
        )
        for record in output_records
    )

    mean_load_bias = float(
        np.mean(
            [
                np.mean(
                    np.abs(
                        record["issue_load_bias_kw"][1:]
                    )
                )
                for record in output_records
            ]
        )
    )

    return {
        "model":
            "causal rolling-horizon MPC "
            "with real-time storage recourse",

        "output_dates": [
            str(output_records[0]["date"]),
            str(output_records[-1]["date"]),
        ],

        "output_days":
            len(output_records),

        "plan_energy_kwh":
            plan_energy,

        "final_normal_energy_kwh":
            final_energy,

        "emergency_energy_kwh":
            emergency_energy,

        "plan_cost_cny":
            plan_cost,

        "final_normal_cost_cny":
            normal_cost,

        "adjustment_cost_cny":
            adjustment_cost,

        "emergency_cost_cny":
            emergency_cost,

        "total_cost_cny":
            normal_cost
            + adjustment_cost
            + emergency_cost,

        "january_warmup_cost_cny":
            total_purchase_cost(january_records),

        "full_year_cost_cny":
            total_purchase_cost(records),

        "actual_charge_energy_kwh":
            sum(
                float(np.sum(record["charge_actual"]))
                for record in output_records
            ),

        "actual_discharge_energy_kwh":
            sum(
                float(
                    np.sum(record["discharge_actual"])
                )
                for record in output_records
            ),

        "actual_recourse_energy_kwh":
            sum(
                float(np.sum(record["recourse"]))
                for record in output_records
            ),

        "actual_spill_energy_kwh":
            sum(
                float(np.sum(record["spill_actual"]))
                for record in output_records
            ),

        "soc_min_kwh":
            soc_min,

        "soc_max_kwh":
            soc_max,

        "soc_lower_bound_kwh":
            params.e_min,

        "soc_upper_bound_kwh":
            params.e_max,

        "soc_lower_violation_kwh":
            max(
                0.0,
                params.e_min - soc_min,
            ),

        "soc_upper_violation_kwh":
            max(
                0.0,
                soc_max - params.e_max,
            ),

        "soc_numerical_guard_kwh":
            params.soc_guard,

        "max_simultaneous_charge_discharge_kwh":
            simultaneous_charge_discharge,

        "mean_absolute_load_nowcast_bias_kw":
            mean_load_bias,

        **checks,
    }


def main() -> None:
    """读取附件、执行模型并输出结果。"""

    base_dir = Path(__file__).resolve().parent

    attachment_dir = base_dir / "附件"

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--attachment-dir",
        type=Path,
        default=attachment_dir,
    )

    parser.add_argument(
        "--data1",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--data2",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--data3",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--template",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "output" / "q3" / "result3.xlsx",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=base_dir / "output" / "q3" / "summary.json",
    )

    parser.add_argument(
        "--quantile",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--shrink-k",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--load-bias-decay-hours",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--load-bias-clip-fraction",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--update-hours",
        default="0,6,12,18",
        help="逗号分隔的累计更新时间，必须包含0，例如0,6,12",
    )

    parser.add_argument(
        "--terminal-value-factor",
        type=float,
        default=0.0,
        help="窗口末端储电价值系数；0保持原模型，0.9用于软终端价值对照",
    )

    args = parser.parse_args()

    paths = attachment_paths(base_dir, args.attachment_dir)
    args.data1 = args.data1 or paths["data1"]
    args.data2 = args.data2 or paths["data2"]
    args.data3 = args.data3 or paths["data3"]
    args.template = args.template or paths["template3"]
    try:
        update_hours = tuple(int(value.strip()) for value in args.update_hours.split(",") if value.strip())
    except ValueError as exc:
        parser.error(f"--update-hours格式错误：{exc}")
    if not update_hours or update_hours[0] != 0 or tuple(sorted(set(update_hours))) != update_hours:
        parser.error("--update-hours必须是从0开始、无重复且递增的列表")
    if any(hour not in (0, 6, 12, 18) for hour in update_hours):
        parser.error("--update-hours只允许0、6、12、18")
    if args.terminal_value_factor < 0:
        parser.error("--terminal-value-factor必须非负")

    if not 0.5 <= args.quantile < 1.0:
        parser.error(
            "--quantile必须在[0.5,1)内"
        )

    if args.shrink_k < 0:
        parser.error(
            "--shrink-k必须非负"
        )

    if args.load_bias_decay_hours <= 0:
        parser.error(
            "--load-bias-decay-hours必须为正"
        )

    if not 0 <= args.load_bias_clip_fraction <= 0.5:
        parser.error(
            "负荷偏差截断比例必须位于[0,0.5]"
        )

    _validate_files(
        {
            "附件1": args.data1,
            "附件2": args.data2,
            "附件3": args.data3,
            "result3模板": args.template,
        }
    )

    params = DispatchParams()

    (
        dates,
        price,
        fallback_load_kw,
        load_actual_kw,
        pv_actual_kw,
        pv_forecasts_kw,
    ) = read_inputs(
        args.data1,
        args.data2,
        args.data3,
    )

    records, checks = simulate(
        dates=dates,
        price=price,
        fallback_load_kw=fallback_load_kw,
        load_actual_kw=load_actual_kw,
        pv_actual_kw=pv_actual_kw,
        pv_forecasts_kw=pv_forecasts_kw,
        params=params,
        quantile=args.quantile,
        shrink_k=args.shrink_k,
        load_bias_decay_hours=
            args.load_bias_decay_hours,
        load_bias_clip_fraction=
            args.load_bias_clip_fraction,
        update_hours=update_hours,
        terminal_value_factor=args.terminal_value_factor,
    )

    write_result3(
        template=args.template,
        output=args.output,
        price=price,
        records=records,
    )

    summary = summarize(
        records=records,
        price=price,
        params=params,
        checks=checks,
    )
    summary["input_sha256"] = {
        "附件1.xlsx": file_sha256(args.data1),
        "附件2.xlsx": file_sha256(args.data2),
        "附件3.xlsx": file_sha256(args.data3),
    }
    summary["template_sha256"] = file_sha256(args.template)

    args.summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
