#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation Regional Flex Optimizer
   • Dispatch, stockage, DR, etc.
   • Imports / exports / net flow agrégés par région
"""

import argparse
from pathlib import Path
import pickle
import yaml

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# PALETTE
# --------------------------------------------------------------------------- #
DEFAULT_PALETTE = {
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

# Will be populated in main() and used throughout
PALETTE = DEFAULT_PALETTE.copy()
# --------------------------------------------------------------------------- #
# Palette utilities
# --------------------------------------------------------------------------- #
def load_palette(path: str | None) -> dict:
    """Return palette dictionary, optionally overridden by a YAML file."""
    palette = DEFAULT_PALETTE.copy()
    if path:
        try:
            with open(path, "r") as f:
                user = yaml.safe_load(f) or {}
            # allow 'palette:' key or direct mapping
            user_palette = user.get("palette", user)
            palette.update({k: str(v) for k, v in user_palette.items()})
        except Exception as exc:
            print(f"Warning: failed to load palette file {path}: {exc}")
    return palette
# Order matters for stacked plots - lowest variable cost first, then most expensive
# Put biofuel before thermal to match merit order
DISPATCH_TECHS = ["hydro", "nuclear", "biofuel", "thermal_gas", "thermal_fuel"]
STORAGE_TECHS  = ["batteries", "STEP"]

# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def build_df(res: dict, prefix: str) -> pd.DataFrame:
    """Assemble un DataFrame (colonnes = variables) à partir du dict res."""
    data = {
        k[len(prefix):]: pd.Series(v)
        for k, v in res["variables"].items() if k.startswith(prefix)
    }
    return pd.DataFrame(data)


def dt_index(n: int) -> pd.DatetimeIndex:
    """Index demi-horaire démarrant le 01/01/2022-00:00."""
    return pd.date_range("2022-01-01", periods=n, freq="30min")


def format_title(label: str, region: str) -> str:
    """Return unified title with descriptor and region."""
    return f"{label} – {region}"


def plot_df(
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    path: Path,
    colors=None,
    stacked: bool = True,
    area: bool = True,
    ylim=None,
):
    """Wrapper générique autour de pandas.plot."""
    if df.empty:
        return

    df_plot = df.copy()
    if area or stacked:               # pour les graphiques empilés
        df_plot = df_plot.clip(lower=0)

    ax = (
        df_plot.plot.area(color=colors, linewidth=0.2)
        if area
        else df_plot.plot(color=colors, stacked=stacked, linewidth=0.7)
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=180)
    plt.close(ax.figure)


def aggregate_import_export(res: dict, region: str, idx: pd.DatetimeIndex):
    """
    Retourne trois Series indexées par le temps :
        • imports  = Σ flow_out_<OTHER>_<REGION>
        • exports  = Σ flow_out_<REGION>_<OTHER>
        • net      = imports - exports
    """
    imports = pd.Series(0.0, index=idx)
    exports = pd.Series(0.0, index=idx)

    prefix = "flow_out_"
    suffix = f"_{region}"

    for var, values in res["variables"].items():
        if not var.startswith(prefix):
            continue

        # --- EXPORTS : la clé commence par    flow_out_<REGION>_
        if var.startswith(f"{prefix}{region}_"):
            s = pd.Series(values).reindex(range(len(idx)), fill_value=0.0).values
            exports += s

        # --- IMPORTS : la clé se termine par _<REGION>
        elif var.endswith(suffix):
            s = pd.Series(values).reindex(range(len(idx)), fill_value=0.0).values
            imports += s

    net = imports - exports
    return (
        pd.Series(imports,  index=idx, name="imports"),
        pd.Series(exports,  index=idx, name="exports"),
        pd.Series(net,      index=idx, name="net"),
    )


# --------------------------------------------------------------------------- #
def main():
    pa = argparse.ArgumentParser(description="Plot RFO results")
    pa.add_argument("--pickle", required=True, help="pickle produit par get_results()")
    pa.add_argument("--out", default="plots", help="dossier de sortie")
    pa.add_argument("--region")
    pa.add_argument("--all-regions", action="store_true")
    pa.add_argument("--start")
    pa.add_argument("--end")
    pa.add_argument("--palette-file", help="YAML file overriding default colors")
    args = pa.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load color palette (can be overridden by user file)
    global PALETTE
    PALETTE = load_palette(args.palette_file)

    with open(args.pickle, "rb") as f:
        res = pickle.load(f)

    regions = res["regions"]
    targets = regions if args.all_regions else [args.region or regions[0]]
    
    # DIAGNOSTIC - Check if biofuel is being dispatched at all
    print("\n--- DIAGNOSTIC: Biofuel Dispatch ---")
    for region in regions:
        biofuel_key = f"dispatch_biofuel_{region}"
        if biofuel_key in res["variables"]:
            biofuel_sum = sum(res["variables"][biofuel_key])
            biofuel_max = max(res["variables"][biofuel_key]) if res["variables"][biofuel_key] else 0
            if biofuel_sum > 0:
                print(f"  {region} - Total: {biofuel_sum:.2f} MWh, Max: {biofuel_max:.2f} MW")
            else:
                print(f"  {region} - No biofuel dispatched (zero values)")
        else:
            print(f"  {region} - Biofuel variable not found in results")
    print("------------------------------\n")

    # horizon temporel
    first_var = next(iter(res["variables"].values()))
    idx = dt_index(len(first_var))

    # éventuel fenêtrage
    mask = slice(None)
    if args.start or args.end:
        start = pd.to_datetime(args.start) if args.start else idx[0]
        end   = pd.to_datetime(args.end)   if args.end   else idx[-1]
        mask  = (idx >= start) & (idx <= end)

    # ------------------------------------------------------------------- #
    for region in targets:
        print(f"→ {region}")
        reg_out = out_dir / region
        reg_out.mkdir(exist_ok=True)

        # ======================= DISPATCH =============================== #
        dispatch = pd.DataFrame(index=idx)
        for tech in DISPATCH_TECHS:
            key = f"dispatch_{tech}_{region}"
            s = pd.Series(res["variables"].get(key, {})).reindex(
                range(len(idx)), fill_value=0.0
            )
            dispatch[tech] = s.values
        # Only keep technologies that are actually dispatched with meaningful values
        # Filter out columns where the sum is very small (less than 1 MWh across all time periods)
        dispatched_cols = [col for col in dispatch.columns if dispatch[col].sum() > 1.0]
        dispatch = dispatch.loc[mask, dispatched_cols]
        
        # Make sure remaining columns are in the right order for stacking
        ordered_cols = [col for col in DISPATCH_TECHS if col in dispatched_cols]
        dispatch = dispatch[ordered_cols]
        
        # Create larger figure to improve visibility
        plt.figure(figsize=(12, 8))
        
        plot_df(
            dispatch,
            format_title("Dispatch", region),
            "MW",
            reg_out / f"dispatch_{region}.png",
            colors=[PALETTE.get(t) for t in dispatch.columns],
            area=True,
            stacked=True,
        )
        
        # Also create zoomed version to highlight small dispatches
        thermal_keys = [k for k in ["biofuel", "thermal_gas", "thermal_fuel"] if k in dispatch.columns]
        if thermal_keys:
            thermal_only = dispatch[thermal_keys]
            if thermal_only.sum().sum() > 0:  # Only create plot if there's some thermal dispatch
                plt.figure(figsize=(12, 6))
                plot_df(
                    thermal_only,
                    format_title("Thermal Dispatch Detail", region),
                    "MW",
                    reg_out / f"dispatch_thermal_detail_{region}.png",
                    colors=[PALETTE.get(t) for t in thermal_only.columns],
                    area=True,
                    stacked=True,
                )

        # ======================= STORAGE SOC ============================ #
        soc = build_df(res, "storage_soc_").set_index(idx)
        cols_soc = [c for c in soc.columns if c.endswith(f"_{region}")]
        plot_df(
            soc[cols_soc].loc[mask],
            format_title("SOC", region),
            "MWh",
            reg_out / f"soc_{region}.png",
            stacked=False,
            area=False,
        )

        # ======================= STORAGE POWER (±) ====================== #
        pow_charge = build_df(res, "storage_charge_").set_index(idx)
        pow_dis    = build_df(res, "storage_discharge_").set_index(idx)

        for tech in STORAGE_TECHS:
            col = f"{tech}_{region}"
            if col not in pow_charge.columns and col not in pow_dis.columns:
                continue
            df_pow = pd.DataFrame(index=idx)
            if col in pow_charge:
                df_pow["charge"] = pow_charge[col]
            if col in pow_dis:
                df_pow["discharge"] = pow_dis[col]
            df_pow = df_pow.loc[mask].fillna(0)

            plot_df(
                df_pow,
                format_title(f"{tech} – Puissance (±)", region),
                "MW",
                reg_out / f"{tech}_power_{region}.png",
                colors=[PALETTE["storage_charge"], PALETTE["storage_discharge"]],
                stacked=False,
                area=False,
            )

        # ======================= SLACK ================================== #
        slack_keys = [f"slack_pos_{region}", f"slack_neg_{region}"]
        slack = (
            pd.DataFrame({k: pd.Series(res["variables"].get(k, {})) for k in slack_keys})
            .set_index(idx)
            .loc[mask]
        )
        plot_df(
            slack.fillna(0),
            format_title("Slack", region),
            "MW",
            reg_out / f"slack_{region}.png",
            colors=[PALETTE["slack_pos"], PALETTE["slack_neg"]],
            stacked=False,
            area=False,
        )

        # ======================= DEMAND RESPONSE ======================== #
        dr = build_df(res, "demand_response_").set_index(idx)
        if region in dr.columns:
            plot_df(
                dr[[region]].loc[mask],
                format_title("Demand Response", region),
                "MW",
                reg_out / f"demand_response_{region}.png",
                colors=[PALETTE["demand_response"]],
                stacked=False,
                area=False,
            )

        # ======================= CURTAILMENT ============================ #
        cur_key = f"curtail_{region}"
        if cur_key in res["variables"]:
            cur = (
                pd.Series(res["variables"][cur_key])
                .reindex(range(len(idx)), fill_value=0.0)
                .set_axis(idx)
            )
            plot_df(
                cur.to_frame("curtail").loc[mask],
                format_title("Curtailment", region),
                "MW",
                reg_out / f"curtail_{region}.png",
                colors=[PALETTE["curtail"]],
                stacked=False,
                area=False,
            )

        # ================== IMPORTS / EXPORTS / NET FLOW ================ #
        imports, exports, net = aggregate_import_export(res, region, idx)
        df_ie = pd.concat([imports, exports, net], axis=1).loc[mask]

        plot_df(
            df_ie[["imports"]],
            format_title("Imports", region),
            "MW",
            reg_out / f"imports_{region}.png",
            colors=[PALETTE["imports"]],
            stacked=False,
            area=False,
        )
        plot_df(
            df_ie[["exports"]],
            format_title("Exports", region),
            "MW",
            reg_out / f"exports_{region}.png",
            colors=[PALETTE["exports"]],
            stacked=False,
            area=False,
        )
        plot_df(
            df_ie[["net"]],
            format_title("Net Flow", region),
            "MW",
            reg_out / f"net_flow_{region}.png",
            colors=[PALETTE["net"]],
            stacked=False,
            area=False,
        )

    print(f"✅  Graphiques enregistrés dans « {out_dir} »")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
