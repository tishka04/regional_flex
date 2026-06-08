#!/usr/bin/env python3
"""Run the Regional Flexibility Optimizer and produce basic visualisations."""

import argparse
import json
import logging
import os
import time

import matplotlib
import pandas as pd
import pulp
import yaml

from rolling_utils import extract_final_states, prepare_initial_states, rolling_horizon_indices
from src.model import calculate_emissions
from src.model.optimizer_regional_flex import RegionalFlexOptimizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- presets for the paper ------------------------------------------------
PRESETS = {
    # label: (start_date, end_date)
    "winter_weekday": ("2022-01-18", "2022-01-18"),
    "autumn_weekend": ("2022-10-09", "2022-10-09"),
    "spring_weekday": ("2022-05-12", "2022-05-12"),
    "summer_holiday": ("2022-08-15", "2022-08-15"),
    "full_year": ("2022-01-01", "2022-12-31"),
}


def resolve_regional_timeseries_path(region, data_dir, data_suffix=""):
    """Resolve one regional CSV path, supporting optional year suffixes."""
    candidates = []
    if data_suffix:
        candidates.append(f"{region}{data_suffix}.csv")
        if "dAzur" in region:
            candidates.append(f"{region}{data_suffix}".replace("dAzur", "d'Azur") + ".csv")

    candidates.append(f"{region}.csv")
    if "dAzur" in region:
        candidates.append(f"{region}".replace("dAzur", "d'Azur") + ".csv")

    for filename in candidates:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Timeseries file not found for region {region} in {data_dir} "
        f"(tried suffix '{data_suffix}')"
    )


def load_regional_timeseries(regions, data_dir, data_suffix=""):
    """Load demand and RES data for each region."""
    data = {}
    for region in regions:
        path = resolve_regional_timeseries_path(region, data_dir, data_suffix=data_suffix)
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        data[region] = df
    return data


def filter_interval(data_dict, start, end):
    """Keep only the selected interval for each region."""
    return {region: df.loc[start:end].copy() for region, df in data_dict.items()}


def infer_time_step_hours(data_dict):
    """Infer the half-hourly/hourly resolution from the loaded timeseries."""
    first_df = next(iter(data_dict.values()))
    if len(first_df.index) > 1 and isinstance(first_df.index, pd.DatetimeIndex):
        delta = first_df.index[1] - first_df.index[0]
        return delta.total_seconds() / 3600.0
    return 0.5


def get_max_recovery_horizon_steps(cfg, time_step_hours):
    """Return the maximum DR recovery horizon across regions, in time steps."""
    max_steps = 0
    dr_section = cfg.get("demand_response", {})
    for params in dr_section.values():
        beta = float(params.get("recovery_beta", 0.0) or 0.0)
        horizon_hours = float(params.get("recovery_horizon_hours", 24.0) or 0.0)
        if beta <= 0.0 or horizon_hours <= 0.0:
            continue
        steps = int(round(horizon_hours / max(time_step_hours, 1e-9)))
        max_steps = max(max_steps, steps)
    return max_steps


def resolve_config_path(explicit_config, config_year):
    """Resolve the model configuration file and whether the master merge should be skipped."""
    if explicit_config and config_year is not None:
        raise ValueError("Use either --config or --config-year, not both.")

    if config_year is not None:
        config_path = os.path.join("config", f"regional_flex_config_{config_year}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Year-specific config not found: {config_path}")
        return config_path, False

    if explicit_config:
        if not os.path.exists(explicit_config):
            raise FileNotFoundError(f"Config file not found: {explicit_config}")
        return explicit_config, True

    default_config = os.path.join("config", "config_master.yaml")
    return default_config, True


def resolve_preset_dates(preset, config_year=None):
    """Resolve a preset date range, optionally shifting it to a target year."""
    start, end = PRESETS[preset]
    if config_year is None:
        return start, end

    if preset == "full_year":
        return f"{config_year}-01-01", f"{config_year}-12-31"

    start_dt = pd.Timestamp(start).replace(year=config_year)
    end_dt = pd.Timestamp(end).replace(year=config_year)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def build_solver_backend(solver_name, *, msg=True, threads=None, mip=True, time_limit=None):
    """Return a PuLP solver instance matching the requested backend."""
    solver_name = (solver_name or "highs").lower()
    if solver_name == "highs":
        return pulp.HiGHS(mip=mip, msg=msg, threads=threads, timeLimit=time_limit)
    if solver_name == "cbc":
        return pulp.PULP_CBC_CMD(mip=mip, msg=msg, threads=threads, timeLimit=time_limit)
    raise ValueError(f"Unsupported solver backend: {solver_name}")


def extract_recovery_profile(results, incoming_profile, regions, commit_len, horizon_steps):
    """Carry only already-committed recovery obligations into the next window."""
    if horizon_steps <= 0:
        return {}

    incoming_profile = incoming_profile or {}
    next_profile = {}

    for region in regions:
        carried_series = incoming_profile.get(region, [])
        profile = []
        for offset in range(horizon_steps):
            local_arrival = commit_len + offset
            carried_value = 0.0
            if local_arrival < len(carried_series):
                carried_value = float(carried_series[local_arrival] or 0.0)

            committed_recovery = 0.0
            for src_pos in range(commit_len):
                lag = local_arrival - src_pos
                if lag < 1 or lag > horizon_steps:
                    continue
                lag_key = f"recovery_commit_lag{lag}_{region}"
                lag_values = results["variables"].get(lag_key, {})
                committed_recovery += float(lag_values.get(src_pos, 0.0) or 0.0)

            profile.append(carried_value + committed_recovery)

        next_profile[region] = profile

    return next_profile


def plot_dispatch_stack(results, region, outdir):
    """Plot the stacked dispatch for one region."""
    techs = results["dispatch_techs"]
    ts = {}
    for tech in techs:
        key = f"dispatch_{tech}_{region}"
        if key in results["variables"]:
            series = pd.Series(results["variables"][key]).sort_index()
            ts[tech] = series

    if not ts:
        return

    df = pd.DataFrame(ts).clip(lower=0)
    ax = df.plot(kind="area", stacked=True)
    ax.set_xlabel("timestep")
    ax.set_ylabel("MW")
    ax.set_title(f"Dispatch stack - {region}")
    plt.tight_layout()
    fig_path = os.path.join(outdir, f"dispatch_{region}.png")
    plt.savefig(fig_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Explicit config file path; overrides the default")
    parser.add_argument(
        "--config-year",
        type=int,
        choices=[2022, 2023],
        default=None,
        help="Use config/regional_flex_config_<year>.yaml and skip the master config merge",
    )
    parser.add_argument("--data-dir", required=True, help="Folder with regional CSV files")
    parser.add_argument("--preset", choices=list(PRESETS.keys()))
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--out", default="results.pkl", help="Pickle to store raw results")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--solver",
        choices=["highs", "cbc"],
        default="highs",
        help="MILP backend to use for the scenario solves",
    )
    parser.add_argument("--data-suffix", default="", help="Optional suffix for regional CSV files, e.g. _2023")
    parser.add_argument(
        "--recovery-beta",
        type=float,
        default=None,
        help="Override the demand-response recovery beta for all regions (0.0 to 1.0)",
    )
    parser.add_argument(
        "--ignore-hydro-flex",
        action="store_true",
        help="Disable the hourly hydro_flex dispatch cap and use the aggregate hydro capacity only",
    )
    parser.add_argument(
        "--enable-curtailment",
        action="store_true",
        help="Enable curtailment constraints and variables in the optimizer",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    run_started = time.perf_counter()

    if args.recovery_beta is not None and not (0.0 <= args.recovery_beta <= 1.0):
        parser.error("--recovery-beta must be between 0.0 and 1.0")

    if args.preset:
        start, end = resolve_preset_dates(args.preset, args.config_year)
    else:
        if not (args.start and args.end):
            parser.error("Provide --start and --end or choose --preset")
        start, end = args.start, args.end

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    try:
        config_path, merge_master_config = resolve_config_path(args.config, args.config_year)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    print(f"Using config: {config_path}")
    print(f"Master config merge: {'enabled' if merge_master_config else 'disabled'}")
    print(f"Solver backend: {args.solver}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    regions = cfg["regions"]

    load_started = time.perf_counter()
    data_all = load_regional_timeseries(regions, args.data_dir, data_suffix=args.data_suffix)
    time_step_hours = infer_time_step_hours(data_all)
    recovery_lookahead_steps = get_max_recovery_horizon_steps(cfg, time_step_hours)
    recovery_lookahead = pd.Timedelta(hours=recovery_lookahead_steps * time_step_hours)

    data_commit = filter_interval(data_all, start_dt, end_dt)
    optimization_end_dt = end_dt + recovery_lookahead
    data_opt = filter_interval(data_all, start_dt, optimization_end_dt)
    load_seconds = time.perf_counter() - load_started

    nsteps = len(next(iter(data_commit.values())))
    window_size = 48 + recovery_lookahead_steps
    stride = 48
    indices = rolling_horizon_indices(nsteps, window_size, stride)

    stitched_variables = {}
    stitched_dispatch_techs = None
    stitched_regions = regions
    stitched_duals = {}

    print(f"Total time steps: {nsteps}")
    print(f"Window size: {window_size}, Stride: {stride}")
    print(f"Number of windows: {len(indices)}")

    total_objective = 0
    total_build_seconds = 0.0
    total_solver_seconds = 0.0
    total_window_seconds = 0.0
    window_timings = []
    initial_states = None
    storage_techs = ["STEP", "batteries"]
    state_vars = []
    for storage_tech in storage_techs:
        for region in regions:
            state_vars.append(f"storage_soc_{storage_tech}_{region}")

    for win_idx, (start_idx, end_idx) in enumerate(indices):
        print(f"Solving window {win_idx + 1}/{len(indices)}: steps {start_idx} to {end_idx - 1}")
        commit_len = end_idx - start_idx
        solve_end_idx = min(start_idx + window_size, len(next(iter(data_opt.values()))))
        data_win = {region: df.iloc[start_idx:solve_end_idx] for region, df in data_opt.items()}
        time_periods_local = list(range(solve_end_idx - start_idx))
        current_recovery_profile = (initial_states or {}).get("__recovery_profile__", {})
        window_started = time.perf_counter()

        opt = RegionalFlexOptimizer(
            config_path,
            enable_curtailment=args.enable_curtailment,
            merge_master_config=merge_master_config,
            recovery_beta_override=args.recovery_beta,
            ignore_hydro_flex=args.ignore_hydro_flex,
        )
        build_started = time.perf_counter()
        if initial_states is not None:
            opt.build_model(data_win, time_periods=time_periods_local, initial_states=initial_states)
        else:
            opt.build_model(data_win, time_periods=time_periods_local)
        build_seconds = time.perf_counter() - build_started
        total_build_seconds += build_seconds

        opt.model.writeLP(f"debug_window_{win_idx + 1}.lp")
        solver_obj = build_solver_backend(args.solver, msg=True, threads=args.threads, mip=True)
        dual_solver_obj = build_solver_backend(args.solver, msg=False, threads=args.threads, mip=False)
        solve_started = time.perf_counter()
        status, solve_seconds = opt.solve(solver=solver_obj)
        solve_wall_seconds = time.perf_counter() - solve_started
        total_solver_seconds += float(solve_seconds or 0.0)
        if status != 1:
            print(f"WARNING: MILP not optimal in window {win_idx + 1} (status = {status})")
            total_window_seconds += time.perf_counter() - window_started
            window_timings.append(
                {
                    "window": win_idx + 1,
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "commit_len": int(commit_len),
                    "build_seconds": float(build_seconds),
                    "solve_seconds_reported": float(solve_seconds or 0.0),
                    "solve_wall_seconds": float(solve_wall_seconds),
                    "total_window_seconds": float(time.perf_counter() - window_started),
                    "status": int(status),
                }
            )
            continue

        nodal = opt.get_nodal_prices(solver=dual_solver_obj)
        duals_dict = {region: prices.to_dict() for region, prices in nodal.items()}
        results_started = time.perf_counter()
        results = opt.get_results(dual_variables=duals_dict)
        results_seconds = time.perf_counter() - results_started

        window_obj = results.get("objective_value", 0)
        if window_obj is not None:
            total_objective += window_obj

        if stitched_dispatch_techs is None:
            stitched_dispatch_techs = results.get("dispatch_techs", [])

        for var in results["variables"]:
            stitched_variables.setdefault(var, {})

        for var, vals in results["variables"].items():
            for t_local, val in vals.items():
                t_global = t_local + start_idx
                if start_idx <= t_global < end_idx and t_local < commit_len:
                    stitched_variables[var][t_global] = val

        next_initial_states = prepare_initial_states(extract_final_states(results, state_vars, commit_len))
        if recovery_lookahead_steps > 0:
            next_initial_states["__recovery_profile__"] = extract_recovery_profile(
                results,
                current_recovery_profile,
                regions,
                commit_len,
                recovery_lookahead_steps,
            )
        initial_states = next_initial_states

        for region, dual_series in duals_dict.items():
            stitched_duals.setdefault(region, {})
            for t_local, price in dual_series.items():
                t_global = t_local + start_idx
                if start_idx <= t_global < end_idx and t_local < commit_len:
                    stitched_duals[region][t_global] = price

        window_seconds = time.perf_counter() - window_started
        total_window_seconds += window_seconds
        window_timings.append(
            {
                "window": win_idx + 1,
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "commit_len": int(commit_len),
                "build_seconds": float(build_seconds),
                "solve_seconds_reported": float(solve_seconds or 0.0),
                "solve_wall_seconds": float(solve_wall_seconds),
                "results_seconds": float(results_seconds),
                "total_window_seconds": float(window_seconds),
                "status": int(status),
                "objective_value": float(window_obj) if window_obj is not None else None,
            }
        )

    total_wall_seconds = time.perf_counter() - run_started
    solver_name = type(solver_obj).__name__ if "solver_obj" in locals() else "unknown"
    runtime_metrics = {
        "start": str(start_dt),
        "end": str(end_dt),
        "data_suffix": args.data_suffix,
        "config_path": config_path,
        "config_year": args.config_year,
        "master_config_merge": bool(merge_master_config),
        "recovery_beta_override": args.recovery_beta,
        "ignore_hydro_flex": bool(args.ignore_hydro_flex),
        "solver": args.solver,
        "solver_class": solver_name,
        "requested_threads": int(args.threads),
        "n_windows": len(indices),
        "n_steps": int(nsteps),
        "window_size": int(window_size),
        "stride": int(stride),
        "time_step_hours": float(time_step_hours),
        "recovery_lookahead_steps": int(recovery_lookahead_steps),
        "data_load_seconds": float(load_seconds),
        "build_seconds_sum": float(total_build_seconds),
        "solve_seconds_sum_reported": float(total_solver_seconds),
        "window_seconds_sum": float(total_window_seconds),
        "total_wallclock_seconds": float(total_wall_seconds),
        "window_timings": window_timings,
        "run_status": "complete",
    }

    results = {
        "variables": stitched_variables,
        "dispatch_techs": stitched_dispatch_techs,
        "regions": stitched_regions,
        "dual_variables": stitched_duals,
        "total_cost": total_objective,
        "runtime": runtime_metrics,
    }

    runtime_output_path = os.path.splitext(args.out)[0] + "_runtime.json"
    successful_windows = sum(1 for entry in window_timings if int(entry.get("status", -1)) == 1)
    all_windows_optimal = (
        stitched_dispatch_techs is not None
        and len(window_timings) == len(indices)
        and successful_windows == len(indices)
    )

    if not all_windows_optimal:
        results["run_status"] = "partial"
        runtime_metrics["run_status"] = "partial"
        pd.to_pickle(results, args.out)
        with open(runtime_output_path, "w", encoding="utf-8") as f:
            json.dump(runtime_metrics, f, indent=2)
        print(
            "WARNING: One or more optimization windows were not solved optimally. "
            "Partial results were stored, and the run exits with an error code."
        )
        raise SystemExit(1)

    results["emissions"] = calculate_emissions(results, cfg)

    print("\nValidating results...")
    tech_capacities = cfg.get("regional_capacities", {})
    validation_issues = 0
    for var, values in stitched_variables.items():
        if var.startswith("dispatch_") and "_" in var:
            tech, region = var.replace("dispatch_", "").split("_", 1)
            if tech in tech_capacities.get(region, {}):
                capacity = tech_capacities[region][tech]
                max_val = max(values.values()) if values else 0
                if max_val > capacity * 1.01:
                    print(f"WARNING: {var} exceeds capacity: max={max_val:.2f} MW, capacity={capacity} MW")
                    validation_issues += 1

    if validation_issues == 0:
        print("[OK] All technology dispatch values are within capacity limits.")
    else:
        print(f"[WARNING] Found {validation_issues} capacity constraint violations.")

    if "objective_value" in results and results["objective_value"] is not None:
        results["total_cost"] = results["objective_value"]

    pd.to_pickle(results, args.out)
    print(f"Results stored to {args.out}")

    with open(runtime_output_path, "w", encoding="utf-8") as f:
        json.dump(runtime_metrics, f, indent=2)
    print(f"Runtime metrics stored to {runtime_output_path}")
    print(
        f"Total wall-clock runtime: {total_wall_seconds:.2f} s "
        f"(build {total_build_seconds:.2f} s, solver {total_solver_seconds:.2f} s, "
        f"data load {load_seconds:.2f} s)"
    )

    full_price_index = pd.RangeIndex(start=0, stop=nsteps, step=1)
    df_price = pd.DataFrame(stitched_duals).reindex(full_price_index).sort_index().rename_axis("timestep")
    prices_output_path = os.path.join(os.path.dirname(args.out), "nodal_prices.csv")
    df_price.to_csv(prices_output_path)

    dt_h = 0.5
    demand_df = pd.DataFrame({region: data_commit[region]["demand"].to_numpy() for region in regions}, index=full_price_index)

    print("\nResults summary:")
    print(f"Total time periods: {len(df_price)}")
    print(f"Regions: {', '.join(regions)}")
    print("Price head:\n", df_price.head())
    print("Demand head:\n", demand_df.head())

    print("\nDispatch summary (MWh):")
    for tech in ["hydro", "nuclear", "biofuel", "thermal_gas", "thermal_fuel"]:
        for region in regions:
            var_key = f"dispatch_{tech}_{region}"
            if var_key in stitched_variables:
                values = list(stitched_variables[var_key].values())
                if values:
                    total = sum(values) * dt_h
                    max_val = max(values)
                    print(f"  {var_key}: total={total:.2f} MWh, max={max_val:.2f} MW")

    expense = (df_price * demand_df * dt_h).sum().sum()
    print(f"\nSimulated spot expense: {expense:.2f} EUR")

    if "emissions" in results:
        print("\nTotal emissions (tCO2):")
        for region, val in results["emissions"]["total_by_region"].items():
            print(f"  {region}: {val:.2f}")

    outdir = "plots"
    os.makedirs(outdir, exist_ok=True)
    if "dispatch_techs" in results:
        for region in regions:
            plot_dispatch_stack(results, region, outdir)
        print("Figures saved to ./plots")
    else:
        print("No plots generated: optimization did not solve successfully.")


if __name__ == "__main__":
    main()
