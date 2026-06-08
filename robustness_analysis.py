#!/usr/bin/env python3
"""Run and summarize robustness scenarios for the RegionalFlex paper."""

from __future__ import annotations

import argparse
import calendar
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import matplotlib
import pandas as pd

from run_regional_flex import resolve_regional_timeseries_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "robustness"
PLOTS_ROOT = PROJECT_ROOT / "plots" / "robustness"
LOGS_ROOT = RESULTS_ROOT / "logs"
CHECKPOINTS_ROOT = RESULTS_ROOT / "checkpoints"

REGIONS = [
    "Auvergne_Rhone_Alpes",
    "Nouvelle_Aquitaine",
    "Occitanie",
    "Provence_Alpes_Cote_dAzur",
]
DISPATCH_TECHS = ["hydro", "nuclear", "thermal_gas", "thermal_fuel", "biofuel"]
STORAGE_TECHS = ["STEP", "batteries"]
PAIR_INDEX = [
    ("Auvergne_Rhone_Alpes", "Nouvelle_Aquitaine"),
    ("Auvergne_Rhone_Alpes", "Occitanie"),
    ("Auvergne_Rhone_Alpes", "Provence_Alpes_Cote_dAzur"),
    ("Nouvelle_Aquitaine", "Occitanie"),
    ("Nouvelle_Aquitaine", "Provence_Alpes_Cote_dAzur"),
    ("Occitanie", "Provence_Alpes_Cote_dAzur"),
]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def utc_now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def load_irre_efs_reference() -> dict:
    """Load the static system-level IRRE/EFS reference values."""
    ref_path = PROJECT_ROOT / "irre_efs_timevarying_yaml_system.csv"
    if not ref_path.exists():
        return {}
    df = pd.read_csv(ref_path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {
        "irre_up_pct_ref": float(row.get("IRRE_up_%", 0.0) or 0.0),
        "irre_dn_pct_ref": float(row.get("IRRE_dn_%", 0.0) or 0.0),
        "efs_up_mw_ref": float(row.get("EFS_up_MW", 0.0) or 0.0),
        "efs_dn_mw_ref": float(row.get("EFS_dn_MW", 0.0) or 0.0),
        "irre_efs_source": ref_path.name,
    }


def series_from_variables(variables: dict, key: str) -> pd.Series:
    values = variables.get(key, {})
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values, dtype=float).sort_index()


def infer_dt_hours(runtime: dict) -> float:
    try:
        return float(runtime.get("time_step_hours", 0.5) or 0.5)
    except (TypeError, ValueError):
        return 0.5


def expected_full_year_shape(year: int) -> tuple[int, int]:
    days = 366 if calendar.isleap(year) else 365
    n_steps = days * 48
    n_windows = days
    return n_steps, n_windows


def runtime_is_full_year(runtime_path: Path, year: int) -> bool:
    runtime = load_json(runtime_path)
    if not runtime:
        return False

    expected_steps, expected_windows = expected_full_year_shape(year)
    if int(runtime.get("n_steps", -1) or -1) != expected_steps:
        return False
    if int(runtime.get("n_windows", -1) or -1) != expected_windows:
        return False

    window_timings = runtime.get("window_timings") or []
    if len(window_timings) != expected_windows:
        return False
    if any(int(window.get("status", -1) or -1) != 1 for window in window_timings):
        return False

    if runtime.get("run_status") and runtime.get("run_status") != "complete":
        return False

    return True


def compute_total_demand_mwh(regions, data_dir, data_suffix, start, end, dt_hours) -> float:
    total = 0.0
    for region in regions:
        path = resolve_regional_timeseries_path(region, data_dir, data_suffix=data_suffix)
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        window = df.loc[start:end]
        total += float(window["demand"].sum()) * dt_hours
    return total


def compute_congestion_rents(results: dict, dt_hours: float) -> tuple[float, dict]:
    """Return total congestion rent in EUR and a pair breakdown."""
    variables = results.get("variables", {})
    duals = results.get("dual_variables", {})
    regions = results.get("regions", REGIONS)

    total = 0.0
    pair_breakdown = {}
    for i, r1 in enumerate(regions):
        for r2 in regions[i + 1 :]:
            pair_total = 0.0
            for source, destination in ((r1, r2), (r2, r1)):
                flow = series_from_variables(variables, f"flow_out_{source}_{destination}")
                if flow.empty:
                    continue

                idx = flow.index
                p_source = pd.Series(duals.get(source, {}), dtype=float).reindex(idx).fillna(0.0)
                p_dest = pd.Series(duals.get(destination, {}), dtype=float).reindex(idx).fillna(0.0)
                flow = flow.reindex(idx, fill_value=0.0)

                cr = float((flow * (p_dest - p_source)).sum() * dt_hours)
                pair_total += cr
                total += cr

            pair_breakdown[f"{r1}<->{r2}"] = pair_total

    return total, pair_breakdown


def compute_exchanges_mwh(results: dict, dt_hours: float) -> float:
    variables = results.get("variables", {})
    total = 0.0
    for r1, r2 in PAIR_INDEX:
        flow_12 = series_from_variables(variables, f"flow_out_{r1}_{r2}")
        flow_21 = series_from_variables(variables, f"flow_out_{r2}_{r1}")
        idx = flow_12.index.union(flow_21.index)
        if idx.empty:
            continue
        net = flow_12.reindex(idx, fill_value=0.0) - flow_21.reindex(idx, fill_value=0.0)
        total += abs(float(net.sum()) * dt_hours)
    return total


def compute_dispatch_metrics(results: dict, dt_hours: float) -> dict:
    variables = results.get("variables", {})
    by_tech = {}
    for tech in DISPATCH_TECHS:
        total = 0.0
        for region in REGIONS:
            total += float(sum(variables.get(f"dispatch_{tech}_{region}", {}).values()) or 0.0)
        by_tech[tech] = total * dt_hours

    total_dispatchable = float(sum(by_tech.values()))
    shares = {}
    for tech, energy in by_tech.items():
        shares[f"{tech}_share_pct"] = 100.0 * energy / total_dispatchable if total_dispatchable > 0 else 0.0

    return {
        "dispatch_mwh_by_tech": by_tech,
        "total_dispatchable_mwh": total_dispatchable,
        **shares,
    }


def compute_storage_throughput(results: dict, dt_hours: float) -> dict:
    variables = results.get("variables", {})
    charge_total = 0.0
    discharge_total = 0.0
    for region in REGIONS:
        for storage_tech in STORAGE_TECHS:
            charge_total += float(sum(variables.get(f"storage_charge_{storage_tech}_{region}", {}).values()) or 0.0)
            discharge_total += float(sum(variables.get(f"storage_discharge_{storage_tech}_{region}", {}).values()) or 0.0)

    charge_mwh = charge_total * dt_hours
    discharge_mwh = discharge_total * dt_hours
    return {
        "storage_charge_mwh": charge_mwh,
        "storage_discharge_mwh": discharge_mwh,
        "storage_throughput_mwh": charge_mwh + discharge_mwh,
    }


def summarize_result(result_path: Path, runtime_path: Path, data_dir: str, data_suffix: str) -> dict:
    results = load_pickle(result_path)
    runtime = load_json(runtime_path) if runtime_path.exists() else results.get("runtime", {})
    dt_hours = infer_dt_hours(runtime)

    start = pd.Timestamp(runtime.get("start")) if runtime.get("start") else None
    end = pd.Timestamp(runtime.get("end")) if runtime.get("end") else None
    regions = results.get("regions", REGIONS)
    total_demand_mwh = compute_total_demand_mwh(regions, data_dir, data_suffix, start, end, dt_hours) if start and end else 0.0

    dispatch_metrics = compute_dispatch_metrics(results, dt_hours)
    storage_metrics = compute_storage_throughput(results, dt_hours)
    congestion_rents_eur, pair_breakdown = compute_congestion_rents(results, dt_hours)
    exchanges_mwh = compute_exchanges_mwh(results, dt_hours)

    variables = results.get("variables", {})
    dr_mwh = 0.0
    for region in regions:
        dr_mwh += float(sum(variables.get(f"demand_response_{region}", {}).values()) or 0.0) * dt_hours

    dr_share_pct = 100.0 * dr_mwh / total_demand_mwh if total_demand_mwh > 0 else 0.0
    exchange_share_pct = 100.0 * exchanges_mwh / total_demand_mwh if total_demand_mwh > 0 else 0.0

    irr = load_irre_efs_reference()
    total_cost = float(results.get("total_cost") or results.get("objective_value") or 0.0)

    summary = {
        "scenario": result_path.stem.replace(".pkl", ""),
        "result_path": str(result_path),
        "runtime_path": str(runtime_path),
        "year": int(start.year) if start is not None else None,
        "start": str(start) if start is not None else None,
        "end": str(end) if end is not None else None,
        "total_cost_meur": total_cost / 1e6,
        "hydro_mwh": dispatch_metrics["dispatch_mwh_by_tech"]["hydro"],
        "nuclear_mwh": dispatch_metrics["dispatch_mwh_by_tech"]["nuclear"],
        "gas_mwh": dispatch_metrics["dispatch_mwh_by_tech"]["thermal_gas"],
        "hydro_share_pct": dispatch_metrics["hydro_share_pct"],
        "nuclear_share_pct": dispatch_metrics["nuclear_share_pct"],
        "gas_share_pct": dispatch_metrics["thermal_gas_share_pct"],
        "total_dispatchable_mwh": dispatch_metrics["total_dispatchable_mwh"],
        "annual_demand_mwh": total_demand_mwh,
        "dr_mwh": dr_mwh,
        "dr_share_pct_demand": dr_share_pct,
        "storage_charge_mwh": storage_metrics["storage_charge_mwh"],
        "storage_discharge_mwh": storage_metrics["storage_discharge_mwh"],
        "storage_throughput_mwh": storage_metrics["storage_throughput_mwh"],
        "exchanges_mwh": exchanges_mwh,
        "exchange_share_pct_demand": exchange_share_pct,
        "congestion_rents_meur": congestion_rents_eur / 1e6,
        "runtime_total_s": float(runtime.get("total_wallclock_seconds", 0.0) or 0.0),
        "runtime_build_s": float(runtime.get("build_seconds_sum", 0.0) or 0.0),
        "runtime_solve_s": float(runtime.get("solve_seconds_sum_reported", 0.0) or 0.0),
        "runtime_data_load_s": float(runtime.get("data_load_seconds", 0.0) or 0.0),
        "runtime_windows": int(runtime.get("n_windows", 0) or 0),
        "recovery_beta_override": runtime.get("recovery_beta_override"),
        "ignore_hydro_flex": bool(runtime.get("ignore_hydro_flex", False)),
        **irr,
    }

    # Expose pair-level congestion rents in case we want to inspect them later.
    for pair, value in pair_breakdown.items():
        summary[f"congestion_{pair}_meur"] = value / 1e6

    return summary


def format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    table = df.loc[:, columns].copy()
    lines = []
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines.extend([header, divider])
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(format_value(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def build_hydro_figure(df: pd.DataFrame, outpath: Path) -> None:
    index_col = "scenario_label" if "scenario_label" in df.columns else "scenario"
    sub = df.set_index(index_col).loc[:, ["hydro_share_pct", "nuclear_share_pct", "gas_share_pct"]]
    ax = sub.plot(kind="bar", figsize=(9, 5), width=0.82)
    ax.set_ylabel("Share of dispatchable generation (%)")
    ax.set_xlabel("")
    ax.set_title("Hydropower sensitivity")
    ax.legend(["Hydro", "Nuclear", "Gas"], frameon=False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def build_dr_figure(df: pd.DataFrame, outpath: Path) -> None:
    index_col = "scenario_label" if "scenario_label" in df.columns else "scenario"
    sub = df.set_index(index_col).loc[:, ["dr_mwh", "storage_throughput_mwh"]]
    ax = sub.plot(kind="bar", figsize=(9, 5), width=0.8)
    ax.set_ylabel("Energy (MWh)")
    ax.set_xlabel("")
    ax.set_title("Demand-response sensitivity")
    ax.legend(["DR activation", "Storage throughput"], frameon=False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def build_run_command(
    year: int,
    data_dir: str,
    data_suffix: str,
    start: str | None,
    end: str | None,
    preset: str | None,
    out_path: Path,
    threads: int,
    solver: str,
    recovery_beta: float,
    ignore_hydro_flex: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "run_regional_flex.py"),
        "--config-year",
        str(year),
        "--data-dir",
        data_dir,
        "--data-suffix",
        data_suffix,
        "--out",
        str(out_path),
        "--threads",
        str(threads),
        "--solver",
        solver,
        "--recovery-beta",
        str(recovery_beta),
    ]
    if ignore_hydro_flex:
        cmd.append("--ignore-hydro-flex")
    if start and end:
        cmd.extend(["--start", start, "--end", end])
    else:
        cmd.extend(["--preset", preset or "full_year"])
    return cmd


def shift_date_to_year(date_str: str | None, year: int) -> str | None:
    if not date_str:
        return None
    ts = pd.Timestamp(date_str)
    return ts.replace(year=year).strftime("%Y-%m-%d")


def maybe_run_scenario(
    *,
    name: str,
    year: int,
    data_dir: str,
    data_suffix: str,
    start: str | None,
    end: str | None,
    preset: str | None,
    threads: int,
    solver: str,
    heartbeat_seconds: int,
    recovery_beta: float,
    ignore_hydro_flex: bool,
    force_rerun: bool,
) -> tuple[Path, Path, Path]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)

    out_path = RESULTS_ROOT / f"{name}.pkl"
    runtime_path = RESULTS_ROOT / f"{name}_runtime.json"
    log_path = LOGS_ROOT / f"{name}.log"
    checkpoint_path = CHECKPOINTS_ROOT / f"{name}.json"

    if not force_rerun and out_path.exists() and runtime_path.exists():
        if runtime_is_full_year(runtime_path, year):
            print(f"[reuse] {name}")
            write_json_atomic(
                checkpoint_path,
                {
                    "scenario": name,
                    "year": year,
                    "status": "reused",
                    "result_path": str(out_path),
                    "runtime_path": str(runtime_path),
                    "log_path": str(log_path),
                    "updated_at_utc": utc_now_iso(),
                },
            )
            return out_path, runtime_path, log_path
        print(f"[rerun] {name} cache is incomplete or not full-year; recomputing.")

    cmd = build_run_command(
        year=year,
        data_dir=data_dir,
        data_suffix=data_suffix,
        start=start,
        end=end,
        preset=preset,
        out_path=out_path,
        threads=threads,
        solver=solver,
        recovery_beta=recovery_beta,
        ignore_hydro_flex=ignore_hydro_flex,
    )

    print(f"[run] {name}")
    print("      " + " ".join(cmd))
    checkpoint_state = {
        "scenario": name,
        "year": year,
        "status": "starting",
        "result_path": str(out_path),
        "runtime_path": str(runtime_path),
        "log_path": str(log_path),
        "checkpoint_path": str(checkpoint_path),
        "command": cmd,
        "started_at_utc": utc_now_iso(),
        "updated_at_utc": utc_now_iso(),
    }
    write_json_atomic(checkpoint_path, checkpoint_state)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    checkpoint_state["status"] = "running"
    checkpoint_state["pid"] = proc.pid
    checkpoint_state["updated_at_utc"] = utc_now_iso()
    write_json_atomic(checkpoint_path, checkpoint_state)

    state_lock = threading.Lock()
    state = {
        "last_output_time": time.time(),
        "last_output_line": None,
    }

    def _reader() -> None:
        if proc.stdout is None:
            return
        with proc.stdout, log_path.open("w", encoding="utf-8") as log_f:
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                print(line, end="")
                with state_lock:
                    state["last_output_time"] = time.time()
                    state["last_output_line"] = line.rstrip("\n")

    reader = threading.Thread(target=_reader, name=f"{name}-log-reader", daemon=True)
    reader.start()

    start_wall = time.time()
    last_heartbeat = 0.0
    while True:
        return_code = proc.poll()
        now = time.time()
        if return_code is not None:
            break

        if now - last_heartbeat >= heartbeat_seconds:
            with state_lock:
                last_output_time = state["last_output_time"]
                last_output_line = state["last_output_line"]
            checkpoint_state.update(
                {
                    "status": "running",
                    "pid": proc.pid,
                    "elapsed_seconds": round(now - start_wall, 3),
                    "quiet_seconds": round(now - last_output_time, 3),
                    "last_output_line": last_output_line,
                    "updated_at_utc": utc_now_iso(),
                }
            )
            write_json_atomic(checkpoint_path, checkpoint_state)
            print(
                f"[heartbeat] {name}: running for {(now - start_wall) / 60:.1f} min, "
                f"quiet for {(now - last_output_time) / 60:.1f} min",
                flush=True,
            )
            last_heartbeat = now

        time.sleep(1.0)

    reader.join(timeout=10)

    with state_lock:
        last_output_time = state["last_output_time"]
        last_output_line = state["last_output_line"]

    checkpoint_state.update(
        {
            "status": "completed" if return_code == 0 else "failed",
            "return_code": int(return_code),
            "elapsed_seconds": round(time.time() - start_wall, 3),
            "quiet_seconds": round(time.time() - last_output_time, 3),
            "last_output_line": last_output_line,
            "finished_at_utc": utc_now_iso(),
            "updated_at_utc": utc_now_iso(),
        }
    )
    write_json_atomic(checkpoint_path, checkpoint_state)

    if return_code != 0:
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
        except Exception:
            tail = "(log unavailable)"
        raise RuntimeError(f"Scenario {name} failed with return code {return_code}\n{tail}")

    return out_path, runtime_path, log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and summarize robustness scenarios.")
    parser.add_argument("--data-dir", default="Data/processed")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Solver threads per scenario (default 1 for stability on Windows)",
    )
    parser.add_argument(
        "--solver",
        choices=["highs", "cbc"],
        default="highs",
        help="MILP backend used for each scenario",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="Seconds between batch heartbeat messages while a scenario runs",
    )
    parser.add_argument("--preset", default="full_year", help="Preset passed to the model runner when start/end are not provided")
    parser.add_argument("--start", default=None, help="Optional explicit start date (overrides preset)")
    parser.add_argument("--end", default=None, help="Optional explicit end date (overrides preset)")
    parser.add_argument("--force-rerun", action="store_true", help="Re-run scenarios even if result files already exist")
    args = parser.parse_args()

    if bool(args.start) ^ bool(args.end):
        parser.error("Provide both --start and --end, or neither.")

    if os.name == "nt" and args.threads != 1:
        print(
            "[warn] Multi-threaded solver runs can stall on Windows in this sequential batch; "
            "consider using --threads 1 if a scenario appears to hang."
        )

    # Scenario definitions aligned with the paper plan.
    scenario_specs = [
        {
            "name": "baseline_2023_beta05_realistic",
            "group": "hydro",
            "year": 2023,
            "recovery_beta": 0.5,
            "ignore_hydro_flex": False,
            "display": "Baseline 2023",
        },
        {
            "name": "hydro_simplified_2023_beta05",
            "group": "hydro",
            "year": 2023,
            "recovery_beta": 0.5,
            "ignore_hydro_flex": True,
            "display": "Simplified hydro",
        },
        {
            "name": "dr_beta00_2023",
            "group": "dr",
            "year": 2023,
            "recovery_beta": 0.0,
            "ignore_hydro_flex": False,
            "display": "beta = 0",
        },
        {
            "name": "dr_beta05_2023",
            "group": "dr",
            "year": 2023,
            "recovery_beta": 0.5,
            "ignore_hydro_flex": False,
            "display": "beta = 0.5",
            "alias_of": "baseline_2023_beta05_realistic",
        },
        {
            "name": "dr_beta10_2023",
            "group": "dr",
            "year": 2023,
            "recovery_beta": 1.0,
            "ignore_hydro_flex": False,
            "display": "beta = 1",
        },
        {
            "name": "interannual_2022_beta05_realistic",
            "group": "interannual",
            "year": 2022,
            "recovery_beta": 0.5,
            "ignore_hydro_flex": False,
            "display": "2022",
        },
        {
            "name": "interannual_2023_beta05_realistic",
            "group": "interannual",
            "year": 2023,
            "recovery_beta": 0.5,
            "ignore_hydro_flex": False,
            "display": "2023",
            "alias_of": "baseline_2023_beta05_realistic",
        },
    ]

    # Run the unique scenarios we need.
    for spec in scenario_specs:
        alias = spec.get("alias_of")
        if alias:
            # The alias will be summarized from the already-run file.
            continue
        scenario_start = shift_date_to_year(args.start, spec["year"])
        scenario_end = shift_date_to_year(args.end, spec["year"])
        maybe_run_scenario(
            name=spec["name"],
            year=spec["year"],
            data_dir=args.data_dir,
            data_suffix=f"_{spec['year']}",
            start=scenario_start,
            end=scenario_end,
            preset=args.preset,
            threads=args.threads,
            solver=args.solver,
            heartbeat_seconds=args.heartbeat_seconds,
            recovery_beta=spec["recovery_beta"],
            ignore_hydro_flex=spec["ignore_hydro_flex"],
            force_rerun=args.force_rerun,
        )

    # Summarize all scenarios, including aliases.
    summaries = []
    for spec in scenario_specs:
        source_name = spec.get("alias_of", spec["name"])
        result_path = RESULTS_ROOT / f"{source_name}.pkl"
        runtime_path = RESULTS_ROOT / f"{source_name}_runtime.json"
        checkpoint_path = CHECKPOINTS_ROOT / f"{spec['name']}.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing result file for scenario {source_name}: {result_path}")
        summary = summarize_result(result_path, runtime_path, args.data_dir, f"_{spec['year']}")
        summary["scenario"] = spec["name"]
        summary["scenario_label"] = spec["display"]
        summary["analysis_group"] = spec["group"]
        summary["year"] = spec["year"]
        summary["recovery_beta"] = spec["recovery_beta"]
        summary["ignore_hydro_flex"] = bool(spec["ignore_hydro_flex"])
        summaries.append(summary)

        if spec.get("alias_of"):
            write_json_atomic(
                checkpoint_path,
                {
                    "scenario": spec["name"],
                    "year": spec["year"],
                    "status": "alias",
                    "alias_of": spec["alias_of"],
                    "result_path": str(result_path),
                    "runtime_path": str(runtime_path),
                    "updated_at_utc": utc_now_iso(),
                },
            )

    df = pd.DataFrame(summaries)
    group_order = {"hydro": 0, "dr": 1, "interannual": 2}
    df["_group_order"] = df["analysis_group"].map(group_order).fillna(99)
    df = df.sort_values(["_group_order", "year", "scenario"]).drop(columns=["_group_order"]).reset_index(drop=True)
    summary_csv = RESULTS_ROOT / "robustness_summary_all.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Saved combined summary to {summary_csv}")

    # Hydropower sensitivity table and figure.
    hydro_df = df[df["analysis_group"] == "hydro"].copy()
    hydro_df = hydro_df.sort_values(["ignore_hydro_flex", "scenario_label"]).reset_index(drop=True)
    hydro_csv = RESULTS_ROOT / "robustness_hydro_sensitivity.csv"
    hydro_df.to_csv(hydro_csv, index=False)
    build_hydro_figure(hydro_df, PLOTS_ROOT / "hydro_sensitivity.png")

    # DR sensitivity table and figure.
    dr_df = df[df["analysis_group"] == "dr"].copy()
    dr_df = dr_df.sort_values(["recovery_beta", "scenario_label"]).reset_index(drop=True)
    dr_csv = RESULTS_ROOT / "robustness_dr_sensitivity.csv"
    dr_df.to_csv(dr_csv, index=False)
    build_dr_figure(dr_df, PLOTS_ROOT / "dr_sensitivity.png")

    # Interannual table.
    interannual_df = df[df["analysis_group"] == "interannual"].copy()
    interannual_df = interannual_df.sort_values(["year", "scenario_label"]).reset_index(drop=True)
    interannual_csv = RESULTS_ROOT / "robustness_interannual.csv"
    interannual_df.to_csv(interannual_csv, index=False)

    # Markdown summary for paper drafting.
    md_path = RESULTS_ROOT / "robustness_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Robustness Analysis Summary\n\n")
        f.write("## Hydro Sensitivity\n\n")
        f.write(
            dataframe_to_markdown(
                hydro_df,
                [
                    "scenario_label",
                    "total_cost_meur",
                    "hydro_share_pct",
                    "nuclear_share_pct",
                    "gas_share_pct",
                    "dr_mwh",
                    "storage_throughput_mwh",
                    "exchanges_mwh",
                    "congestion_rents_meur",
                    "irre_up_pct_ref",
                    "irre_dn_pct_ref",
                    "efs_up_mw_ref",
                    "efs_dn_mw_ref",
                ],
            )
        )
        f.write("\n\n## DR Sensitivity\n\n")
        f.write(
            dataframe_to_markdown(
                dr_df,
                [
                    "scenario_label",
                    "recovery_beta",
                    "total_cost_meur",
                    "dr_mwh",
                    "storage_throughput_mwh",
                    "exchanges_mwh",
                    "congestion_rents_meur",
                ],
            )
        )
        f.write("\n\n## Interannual Robustness\n\n")
        f.write(
            dataframe_to_markdown(
                interannual_df,
                [
                    "scenario_label",
                    "total_cost_meur",
                    "hydro_share_pct",
                    "nuclear_share_pct",
                    "gas_share_pct",
                    "dr_mwh",
                    "storage_throughput_mwh",
                    "exchanges_mwh",
                    "congestion_rents_meur",
                    "runtime_total_s",
                    "irre_up_pct_ref",
                    "irre_dn_pct_ref",
                    "efs_up_mw_ref",
                    "efs_dn_mw_ref",
                ],
            )
        )
        f.write("\n\nFigures:\n")
        f.write(f"- {PLOTS_ROOT / 'hydro_sensitivity.png'}\n")
        f.write(f"- {PLOTS_ROOT / 'dr_sensitivity.png'}\n")

    print(f"Saved markdown summary to {md_path}")


if __name__ == "__main__":
    main()
