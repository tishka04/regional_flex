#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation Regional Flex Optimizer
   • Dispatch, stockage, DR, etc.
   • Imports / exports / net flow agrégés par région

Corrected version – June 2025

Main fixes
-----------
1. **Python < 3.10 compatibility** – replaced PEP 604 unions (`str | None`) by `Optional[str]`.
2. **Removed duplicate imports** and grouped them logically.
3. **Guarded optional sections** – variables like `summary` are created only when requested and later calls are protected.
4. **Re‑organised emissions plotting** so it’s executed *once* after the per‑region loop, avoiding nested loops and shadowed variables.
5. **Consistent re‑indexing** (`reindex(range(len(idx))`) where Series lengths may differ.
6. **General code hygiene**: clearer variable names, docstrings, and type annotations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import Optional, Dict, List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import yaml

# --------------------------------------------------------------------------- #
# PALETTE
# --------------------------------------------------------------------------- #
DEFAULT_PALETTE: Dict[str, str] = {
    # production
    "hydro": "#0072B2",        # blue
    "nuclear": "#E69F00",      # orange
    "biofuel": "#009E73",      # green
    "thermal_gas": "#56B4E9",  # light blue
    "thermal_fuel": "#D55E00", # reddish orange
    # flexibility
    "slack_pos": "#999999",
    "slack_neg": "#CC79A7",
    "demand_response": "#F0E442",
    "curtail": "#999999",
    # storage
    "storage_charge": "#CC79A7",
    "storage_discharge": "#0072B2",
    # flow aggregates
    "imports": "#56B4E9",
    "exports": "#D55E00",
    "net": "#009E73",
}

# Will be overridden in main() if user supplies a YAML palette
PALETTE: Dict[str, str] = DEFAULT_PALETTE.copy()

# Order matters for stacked plots – lowest variable cost first, then most expensive
DISPATCH_TECHS: List[str] = [
    "hydro",
    "nuclear",
    "biofuel",
    "thermal_gas",
    "thermal_fuel",
]
STORAGE_TECHS: List[str] = ["batteries", "STEP"]

# Default emission factors in gCO2 per kWh (used only if config is missing)
DEFAULT_EMISSION_FACTORS: Dict[str, float] = {
    "hydro": 6,
    "nuclear": 12,
    "biofuel": 230,
    "thermal_gas": 400,
    "thermal_fuel": 750,
}

# Will be set in main() based on config
EMISSION_FACTORS: Dict[str, float] = DEFAULT_EMISSION_FACTORS.copy()

# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #

def load_palette(path: Optional[str] = None) -> Dict[str, str]:
    """Return palette dictionary, optionally overridden by a YAML file."""
    palette = DEFAULT_PALETTE.copy()
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                user = yaml.safe_load(f) or {}
            # allow either a top‑level mapping or a `palette:` key
            user_palette = user.get("palette", user)
            palette.update({k: str(v) for k, v in user_palette.items()})
        except Exception as exc:  # noqa: BLE001 – we only warn the user
            print(f"Warning: failed to load palette file {path}: {exc}")
    return palette


def build_df(res: dict, prefix: str) -> pd.DataFrame:
    """Assemble a DataFrame from `res['variables']` where keys start with *prefix*."""
    data = {
        k[len(prefix):]: pd.Series(v).reindex(range(len(next(iter(res["variables"].values())))), fill_value=0.0)
        for k, v in res["variables"].items()
        if k.startswith(prefix)
    }
    return pd.DataFrame(data)


def dt_index(n: int) -> pd.DatetimeIndex:
    """Half‑hourly index starting 2022‑01‑01 00:00."""
    return pd.date_range("2022-01-01", periods=n, freq="30min")


def format_title(label: str, region: str) -> str:
    """Unified title helper."""
    return f"{label} – {region}"


from matplotlib.dates import AutoDateFormatter, AutoDateLocator

def plot_df(
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    path: Path,
    colors: Optional[List[str]] = None,
    *,
    stacked: bool = True,
    area: bool = True,
    line: bool = False,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    """Generic wrapper around pandas plotting utilities, with improved x-axis readability.
    Set line=True for time series line charts.
    """
    if df.empty:
        return

    # For line plots, ensure colors is a valid list or None
    if line:
        if colors is not None:
            # Only use colors if all are valid (not None)
            safe_colors = [c if c is not None else "#888888" for c in colors]
            ax = df.plot.line(color=safe_colors, linewidth=1.3)
        else:
            ax = df.plot.line(linewidth=1.3)
        idx = df.index
    else:
        df_plot = df.clip(lower=0) if (area or stacked) else df
        ax = (
            df_plot.plot.area(color=colors, linewidth=0.2)
            if area
            else df_plot.plot.bar(stacked=stacked, linewidth=0.7, color=colors)
        )
        idx = df_plot.index

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Improve x-axis readability
    if hasattr(idx, 'dtype') and hasattr(idx, 'is_all_dates') and idx.is_all_dates:
        locator = AutoDateLocator()
        formatter = AutoDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.figure.autofmt_xdate(rotation=30, ha='right')
    else:
        # For categorical/bar/region charts
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=180)
    plt.close(ax.figure)



def aggregate_import_export(res: dict, region: str, idx: pd.DatetimeIndex):
    """Compute import, export and net flow time‑series for *region*."""
    prefix = "flow_out_"
    suffix = f"_{region}"

    imports = pd.Series(0.0, index=idx)
    exports = pd.Series(0.0, index=idx)

    for var, values in res["variables"].items():
        if not var.startswith(prefix):
            continue
        series = pd.Series(values).reindex(range(len(idx)), fill_value=0.0).set_axis(idx)

        if var.startswith(f"{prefix}{region}_"):  # exports
            exports += series
        elif var.endswith(suffix):  # imports
            imports += series

    net = imports - exports
    return (
        imports.rename("imports"),
        exports.rename("exports"),
        net.rename("net"),
    )


def compute_cumulative_metrics(res: dict, idx: pd.DatetimeIndex, config: dict) -> pd.DataFrame:
    """Return total cost (€), emissions (tCO₂) and load factors for every region."""
    dt_h = 0.5  # half‑hourly resolution

    costs_conf = config.get("costs", {})
    reg_costs = config.get("regional_costs", {})
    capacities_conf = config.get("regional_capacities", {})
    emission_factors_conf = config.get("emission_factors", {})

    # Use config emission factors if available, else fallback
    emission_factors = emission_factors_conf if emission_factors_conf else EMISSION_FACTORS

    rows = []
    n_steps = len(idx)

    for region in res["regions"]:
        cost = 0.0
        emissions = 0.0
        load_factors: Dict[str, float] = {}

        for tech in DISPATCH_TECHS:
            var_key = f"dispatch_{tech}_{region}"
            values = res["variables"].get(var_key, {})
            total_mw = sum(values.values())
            energy_mwh = total_mw * dt_h

            unit_cost = reg_costs.get(region, {}).get(tech, costs_conf.get(tech, 0.0))
            cost += energy_mwh * unit_cost
            emissions += energy_mwh * emission_factors.get(tech, 0)  # Already tCO₂/MWh if from config

            cap = capacities_conf.get(region, {}).get(tech)
            if cap:
                load_factors[tech] = energy_mwh / (cap * n_steps * dt_h)

        rows.append({
            "region": region,
            "cost": cost,
            "emissions": emissions,
            **{f"lf_{k}": v for k, v in load_factors.items()},
        })

    return pd.DataFrame(rows).set_index("region")


def animate_region(
    res: dict,
    idx: pd.DatetimeIndex,
    region: str,
    out_path: Path,
    data_dir: Optional[Path] = None,
) -> None:
    """Create a GIF showing dispatch stack and net flow over time."""

    # Assemble dispatch DataFrame
    dispatch = pd.DataFrame(index=idx)
    for tech in DISPATCH_TECHS:
        key = f"dispatch_{tech}_{region}"
        dispatch[tech] = (
            pd.Series(res["variables"].get(key, {}))
            .reindex(range(len(idx)), fill_value=0.0)
            .values
        )

    imports, exports, net = aggregate_import_export(res, region, idx)

    demand = None
    if data_dir:
        csv = data_dir / f"{region}.csv"
        if csv.exists():
            demand = (
                pd.read_csv(csv, index_col=0, parse_dates=True)["demand"].reindex(idx)
            )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    def update(frame: int):
        ax1.clear()
        ax2.clear()

        dispatch.iloc[: frame + 1].clip(lower=0).plot.area(
            ax=ax1,
            color=[PALETTE.get(t, "#888888") for t in dispatch.columns],
            stacked=True,
        )
        if demand is not None:
            ax1.plot(demand.index[: frame + 1], demand.iloc[: frame + 1], "k--", label="demand")
            ax1.legend()

        ax1.set_xlim(idx[0], idx[-1])
        ax1.set_ylim(0, max(dispatch.sum(axis=1).max(), (demand.max() if demand is not None else 0)) * 1.1)
        ax1.set_title(f"Dispatch – {region}")

        net.iloc[: frame + 1].plot(ax=ax2, color=PALETTE["net"])
        ax2.set_xlim(idx[0], idx[-1])
        ax2.set_title("Net flow")

    frames = range(0, len(idx), 4)  # speed‑up the animation
    anim = FuncAnimation(fig, update, frames=frames, interval=100)
    anim.save(out_path, writer=PillowWriter(fps=10))
    plt.close(fig)

# --------------------------------------------------------------------------- #
# MAIN SCRIPT
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RFO results")

    # --- Required I/O
    parser.add_argument("--pickle", required=True, help="Pickle produced by get_results()")
    parser.add_argument("--out", default="plots", help="Output directory")

    # --- Region filtering
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--region", help="Single region to plot (default: first in results)")
    group.add_argument("--all-regions", action="store_true", help="Plot every region available")

    # --- Temporal filtering
    parser.add_argument("--start", help="Start datetime (inclusive)")
    parser.add_argument("--end", help="End datetime (inclusive)")

    # --- Extras
    parser.add_argument("--config", help="YAML config (costs, capacities…)")
    parser.add_argument("--data-dir", type=Path, help="CSV directory for demand time‑series")
    parser.add_argument("--summary", action="store_true", help="Produce aggregated summary charts")
    parser.add_argument("--animate", action="store_true", help="Generate GIF animation per region")
    parser.add_argument("--palette-file", help="YAML file to override default colors")

    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load palette (global)
    global PALETTE
    PALETTE = load_palette(args.palette_file)

    # --- Load optimisation results
    with open(args.pickle, "rb") as f:
        res = pickle.load(f)

    # --- Optional configuration
    cfg: dict = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # --- Time index setup
    first_var = next(iter(res["variables"].values()))
    idx = dt_index(len(first_var))

    # Temporal window
    mask: slice | pd.Series | list[bool]
    if args.start or args.end:
        start = pd.to_datetime(args.start) if args.start else idx[0]
        end = pd.to_datetime(args.end) if args.end else idx[-1]
        mask = (idx >= start) & (idx <= end)
    else:
        mask = slice(None)

    # Target regions
    regions: List[str] = res["regions"]
    targets: List[str] = regions if args.all_regions else [args.region or regions[0]]

    # ------------------------------------------------------------------- #
    # Optional aggregate summary BEFORE per‑region loop (so `summary` is defined globally)
    summary: Optional[pd.DataFrame] = None

    # Load emission factors from config if present
    global EMISSION_FACTORS
    emission_factors_conf = cfg.get("emission_factors", {})
    if emission_factors_conf:
        EMISSION_FACTORS = emission_factors_conf
    else:
        EMISSION_FACTORS = DEFAULT_EMISSION_FACTORS.copy()

    if args.summary and cfg:
        summary = compute_cumulative_metrics(res, idx[mask] if mask is not slice(None) else idx, cfg)
        # Cost chart
        ax = (summary["cost"] / 1e6).plot.bar(title="Total cost by region (M€)")
        ax.set_ylabel("M€")
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')
        ax.figure.tight_layout()
        ax.figure.savefig(out_dir / "cost_by_region.png", dpi=180)
        plt.close(ax.figure)
        # Emissions chart
        ax = summary["emissions"].plot.bar(title="Total emissions by region (tCO₂)")
        ax.set_ylabel("tCO₂")
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha('right')
        ax.figure.tight_layout()
        ax.figure.savefig(out_dir / "emissions_by_region.png", dpi=180)
        plt.close(ax.figure)

        # Optional load‑factor chart
        lf_cols = [c for c in summary.columns if c.startswith("lf_")]
        if lf_cols:
            ax = summary[lf_cols].plot.bar()
            ax.set_title("Load factors by region")
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_ha('right')
            ax.figure.tight_layout()
            ax.figure.savefig(out_dir / "load_factors.png", dpi=180)
            plt.close(ax.figure)


    # ------------------------------------------------------------------- #
    # Per‑region detailed plots
    for region in targets:
        print(f"-> {region}")
        reg_out = out_dir / region
        reg_out.mkdir(exist_ok=True)

        # ======================= DISPATCH =============================== #
        dispatch = pd.DataFrame(index=idx)
        for tech in DISPATCH_TECHS:
            # Provence‑Alpes‑Côte d'Azur has no nuclear plants
            if region == "Provence_Alpes_Cote_dAzur" and tech == "nuclear":
                continue
            key = f"dispatch_{tech}_{region}"
            dispatch[tech] = (
                pd.Series(res["variables"].get(key, {}))
                .reindex(range(len(idx)), fill_value=0.0)
                .values
            )

        # Drop technologies with negligible dispatch
        dispatched_cols = [c for c in dispatch.columns if dispatch[c].sum() > 1.0]
        dispatch = dispatch.loc[mask, dispatched_cols]
        dispatch = dispatch[[c for c in DISPATCH_TECHS if c in dispatch.columns]]

        plot_df(
            dispatch,
            format_title("Dispatch", region),
            "MW",
            reg_out / f"dispatch_{region}.png",
            colors=[PALETTE.get(t) for t in dispatch.columns],
        )

        # Thermal detail (if relevant)
        thermal_cols = [c for c in ["biofuel", "thermal_gas", "thermal_fuel"] if c in dispatch.columns]
        if thermal_cols and dispatch[thermal_cols].sum().sum() > 0:
            plot_df(
                dispatch[thermal_cols],
                format_title("Thermal Dispatch Detail", region),
                "MW",
                reg_out / f"dispatch_thermal_detail_{region}.png",
                colors=[PALETTE.get(t) for t in thermal_cols],
            )

        # ======================= STORAGE SOC ============================ #
        soc_df = build_df(res, "storage_soc_").set_index(idx)
        soc_cols = [c for c in soc_df.columns if c.endswith(f"_{region}")]
        soc_plot_df = soc_df[soc_cols].loc[mask]
        soc_colors = [PALETTE.get(c.split('_')[0], "#888888") for c in soc_plot_df.columns]
        plot_df(
            soc_plot_df,
            format_title("SOC", region),
            "MWh",
            reg_out / f"soc_{region}.png",
            colors=soc_colors,
            stacked=False,
            area=False,
            line=True,
        )

        # ======================= STORAGE POWER ========================== #
        power_charge = build_df(res, "storage_charge_").set_index(idx)
        power_dis = build_df(res, "storage_discharge_").set_index(idx)

        for tech in STORAGE_TECHS:
            col = f"{tech}_{region}"
            if col not in power_charge.columns and col not in power_dis.columns:
                continue
            charge = power_charge.get(col, pd.Series(0, index=idx))
            discharge = power_dis.get(col, pd.Series(0, index=idx))
            storage_power = (discharge.fillna(0) - charge.fillna(0)).loc[mask]
            plot_df(
                storage_power.to_frame("storage_power"),
                format_title(f"{tech} – Puissance (charge‑/décharge+)", region),
                "MW",
                reg_out / f"{tech}_power_{region}.png",
                colors=[PALETTE.get(tech, "#888888")],
                stacked=False,
                area=False,
                line=True,
            )

        # ======================= SLACK ================================== #
        slack_keys = [f"slack_pos_{region}", f"slack_neg_{region}"]
        slack = pd.DataFrame({
            k: pd.Series(res["variables"].get(k, {})).reindex(range(len(idx)), fill_value=0.0)
            for k in slack_keys
        }).set_index(idx).loc[mask]
        slack_colors = [PALETTE.get(col, "#888888") for col in slack.columns]
        plot_df(
            slack,
            format_title("Slack", region),
            "MW",
            reg_out / f"slack_{region}.png",
            colors=slack_colors,
            stacked=False,
            area=False,
            line=True,
        )

        # ======================= DEMAND RESPONSE ======================== #
        dr_df = build_df(res, "demand_response_").set_index(idx)
        if region in dr_df.columns:
            dr_plot_df = dr_df[[region]].loc[mask]
            dr_colors = [PALETTE.get(region, "#888888") for region in dr_plot_df.columns]
            plot_df(
                dr_plot_df,
                format_title("Demand Response", region),
                "MW",
                reg_out / f"demand_response_{region}.png",
                colors=dr_colors,
                stacked=False,
                area=False,
                line=True,
            )

        # ======================= CURTAILMENT ============================ #
        cur_key = f"curtail_{region}"
        if cur_key in res["variables"]:
            cur_series = (
                pd.Series(res["variables"][cur_key])
                .reindex(range(len(idx)), fill_value=0.0)
                .set_axis(idx)
            ).loc[mask]
            plot_df(
                cur_series.to_frame("curtail"),
                format_title("Curtailment", region),
                "MW",
                reg_out / f"curtail_{region}.png",
                colors=[PALETTE["curtail"]],
                stacked=False,
                area=False,
            )

        # ================== IMPORT / EXPORT / NET FLOW ================= #
        imports, exports, net = aggregate_import_export(res, region, idx)
        df_ie = pd.concat([imports, exports, net], axis=1).loc[mask]
        exch_colors = [PALETTE.get(col, "#888888") for col in df_ie.columns]
        plot_df(
            df_ie,
            format_title("Imports/Exports/Net", region),
            "MW",
            reg_out / f"exchange_{region}.png",
            colors=exch_colors,
            stacked=False,
            area=False,
            line=True,
        )
        for col in ["imports", "exports", "net"]:
            plot_df(
                df_ie[[col]],
                format_title(col.capitalize(), region),
                "MW",
                reg_out / f"{col}_{region}.png",
                colors=[PALETTE.get(col, "#888888")],
                stacked=False,
                area=False,
                line=True,
            )

        # ======================= ANIMATION ============================= #
        if args.animate:
            animate_region(
                res,
                idx[mask] if mask is not slice(None) else idx,
                region,
                reg_out / f"animation_{region}.gif",
                args.data_dir,
            )

    # ------------------------------------------------------------------- #
    # =============== EMISSIONS TIME‑SERIES (ALL REGIONS) =============== #
    print("Generating emissions time‑series …")
    for region in regions:
        emissions_df = pd.DataFrame(index=idx)
        tech_cols: List[str] = []
        tech_colors: List[str] = []

        capacities_conf = cfg.get("regional_capacities", cfg.get("capacities", {}))
        region_caps = capacities_conf.get(region, {}) if capacities_conf else {}

        for tech in DISPATCH_TECHS:
            if region == "Provence_Alpes_Cote_dAzur" and tech == "nuclear":
                continue
            if region_caps.get(tech, 0) == 0:
                continue
            key = f"dispatch_{tech}_{region}"
            values = res["variables"].get(key, {})
            tech_series = (
                pd.Series(values)
                .reindex(range(len(idx)), fill_value=0.0)
                .set_axis(idx)
            )
            # Use config emission factor (tCO2/MWh) if present, else fallback to default (convert gCO2/kWh to tCO2/MWh)
            factor = EMISSION_FACTORS.get(tech, 0.0)
            if factor > 10:  # If using default in gCO2/kWh, convert to tCO2/MWh
                factor = factor / 1000.0
            emissions = tech_series * factor  # tCO₂/h
            col_name = f"emission_{tech}_{region}"
            emissions_df[col_name] = emissions
            tech_cols.append(col_name)
            tech_colors.append(PALETTE.get(tech, "#808080"))   # gris fallback

        emissions_df = emissions_df.loc[mask]
        region_dir = out_dir / region
        region_dir.mkdir(parents=True, exist_ok=True)
        if tech_cols and not emissions_df.empty:
            plot_df(
                emissions_df[tech_cols],
                format_title("Emissions", region),
                "tCO₂/h",
                region_dir / f"emissions_{region}.png",
                colors=tech_colors,
                stacked=True,
                area=True,
            )

    print(f"✅  All graphs saved in ‘{out_dir}’. Enjoy! ✨")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
