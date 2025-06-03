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

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import yaml

# --------------------------------------------------------------------------- #
# PALETTE
# --------------------------------------------------------------------------- #
PALETTE = {
    # production
    "hydro": "#1f77b4",       "nuclear": "#ff7f0e",
    "biofuel": "#ff00ff",     # Changed to bright magenta to make it more visible
    "thermal_gas": "#2ca02c", "thermal_fuel": "#d62728",
    # flexibilité
    "slack_pos": "#7f7f7f",   "slack_neg": "#bcbd22",
    "demand_response": "#e377c2", "curtail": "#8e8e8e",
    # stockage
    "storage_charge": "#ff1493",   "storage_discharge": "#00ced1",
    # agrégats flux
    "imports": "#1f78b4", "exports": "#e31a1c", "net": "#17becf",
}
# Order matters for stacked plots - lowest variable cost first, then most expensive
# Put biofuel before thermal to match merit order
DISPATCH_TECHS = ["hydro", "nuclear", "biofuel", "thermal_gas", "thermal_fuel"]
STORAGE_TECHS  = ["batteries", "STEP"]

# Approximate emission factors in gCO2 per kWh
EMISSION_FACTORS = {
    "hydro": 6,
    "nuclear": 12,
    "biofuel": 230,
    "thermal_gas": 400,
    "thermal_fuel": 750,
}

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


def compute_cumulative_metrics(res: dict, idx: pd.DatetimeIndex, config: dict) -> pd.DataFrame:
    """Return total cost (\u20ac), emissions (tCO2) and load factors per region."""
    dt_h = 0.5
    costs = config.get("costs", {})
    reg_costs = config.get("regional_costs", {})
    capacities = config.get("regional_capacities", {})

    summary = []
    n_steps = len(idx)
    for region in res["regions"]:
        cost = 0.0
        emissions = 0.0
        lf = {}
        for tech in DISPATCH_TECHS:
            var_key = f"dispatch_{tech}_{region}"
            values = res["variables"].get(var_key, {})
            total_mw = sum(values.values())
            energy = total_mw * dt_h
            unit_cost = costs.get(tech, 0.0)
            if region in reg_costs and tech in reg_costs[region]:
                unit_cost = reg_costs[region][tech]
            cost += energy * unit_cost
            emissions += energy * EMISSION_FACTORS.get(tech, 0) / 1000.0

            cap = capacities.get(region, {}).get(tech)
            if cap:
                lf[tech] = energy / (cap * n_steps * dt_h)

        summary.append({"region": region, "cost": cost, "emissions": emissions, **{f"lf_{k}": v for k, v in lf.items()}})

    return pd.DataFrame(summary).set_index("region")


def animate_region(res: dict, idx: pd.DatetimeIndex, region: str, out_path: Path, data_dir: Path | None = None) -> None:
    """Create a GIF showing dispatch stack and net flow over time."""
    dispatch = pd.DataFrame(index=idx)
    for tech in DISPATCH_TECHS:
        key = f"dispatch_{tech}_{region}"
        dispatch[tech] = pd.Series(res["variables"].get(key, {})).reindex(range(len(idx)), fill_value=0.0).values

    imports, exports, net = aggregate_import_export(res, region, idx)

    demand = None
    if data_dir is not None:
        csv = Path(data_dir) / f"{region}.csv"
        if csv.exists():
            demand = pd.read_csv(csv, index_col=0, parse_dates=True)["demand"].reindex(idx)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    def update(frame):
        ax1.clear(); ax2.clear()
        dispatch.iloc[:frame+1].plot.area(ax=ax1, color=[PALETTE.get(t) for t in dispatch.columns], linewidth=0.0)
        if demand is not None:
            ax1.plot(demand.index[:frame+1], demand.iloc[:frame+1], "k--", label="demand")
            ax1.legend()
        ax1.set_xlim(idx[0], idx[-1])
        ax1.set_ylim(0, max(dispatch.sum(axis=1).max(), demand.max() if demand is not None else 0)*1.1)
        ax1.set_title(f"Dispatch – {region}")

        net.iloc[:frame+1].plot(ax=ax2, color=PALETTE["net"])
        ax2.set_xlim(idx[0], idx[-1])
        ax2.set_title("Net flow")

    frames = range(0, len(idx), 4)
    anim = FuncAnimation(fig, update, frames=frames, interval=100)
    anim.save(out_path, writer=PillowWriter(fps=10))
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    pa = argparse.ArgumentParser(description="Plot RFO results")
    pa.add_argument("--pickle", required=True, help="pickle produit par get_results()")
    pa.add_argument("--out", default="plots", help="dossier de sortie")
    pa.add_argument("--region")
    pa.add_argument("--all-regions", action="store_true")
    pa.add_argument("--start")
    pa.add_argument("--end")
    pa.add_argument("--config", help="fichier YAML de configuration (coûts, capacités)")
    pa.add_argument("--data-dir", help="répertoire CSV pour la demande")
    pa.add_argument("--summary", action="store_true", help="produit les graphiques cumulés")
    pa.add_argument("--animate", action="store_true", help="génère une animation GIF pour la région")
    args = pa.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.pickle, "rb") as f:
        res = pickle.load(f)

    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

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
            f"Dispatch – {region}",
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
                    f"Thermal Dispatch Detail – {region}",
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
            f"SOC – {region}",
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
                f"{tech} – Puissance (±) – {region}",
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
            f"Slack – {region}",
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
                f"Demand response – {region}",
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
                f"Curtailment – {region}",
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
            f"Imports – {region}",
            "MW",
            reg_out / f"imports_{region}.png",
            colors=[PALETTE["imports"]],
            stacked=False,
            area=False,
        )
        plot_df(
            df_ie[["exports"]],
            f"Exports – {region}",
            "MW",
            reg_out / f"exports_{region}.png",
            colors=[PALETTE["exports"]],
            stacked=False,
            area=False,
        )
        plot_df(
            df_ie[["net"]],
            f"Net flow (imports − exports) – {region}",
            "MW",
            reg_out / f"net_flow_{region}.png",
            colors=[PALETTE["net"]],
            stacked=False,
            area=False,
        )

        if args.animate:
            gif_path = reg_out / f"animation_{region}.gif"
            animate_region(res, idx[mask] if mask is not slice(None) else idx, region, gif_path, args.data_dir)

    if args.summary and cfg:
        summary = compute_cumulative_metrics(res, idx[mask] if mask is not slice(None) else idx, cfg)
        ax = (summary["cost"] / 1e6).plot.bar(title="Total cost by region (M€)")
        ax.set_ylabel("M€")
        ax.figure.tight_layout()
        ax.figure.savefig(out_dir / "cost_by_region.png", dpi=180)
        plt.close(ax.figure)

        ax = summary["emissions"].plot.bar(title="Total emissions by region (tCO₂)")
        ax.set_ylabel("tCO₂")
        ax.figure.tight_layout()
        ax.figure.savefig(out_dir / "emissions_by_region.png", dpi=180)
        plt.close(ax.figure)

        lf_cols = [c for c in summary.columns if c.startswith("lf_")]
        if lf_cols:
            ax = summary[lf_cols].plot.bar()
            ax.set_title("Load factors")
            ax.figure.tight_layout()
            ax.figure.savefig(out_dir / "load_factors.png", dpi=180)
            plt.close(ax.figure)

    print(f"✅  Graphiques enregistrés dans « {out_dir} »")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
