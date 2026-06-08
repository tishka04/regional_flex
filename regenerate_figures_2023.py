"""Regenerate the 2023 paper figures from the RoR-corrected + reservoir-capped
baseline pickle.

Steps:
1. Export a wide `full_year_2023.csv` (comma-separated, the format the viz
   scripts read by default) from `baseline_2023_beta05_realistic.pkl`, joining
   the optimisation variables with the per-region demand/wind/solar inputs and
   the nodal prices (duals).
2. Reuse `regionalflex_viz` to regenerate fig1-fig8 into `figs/`.
3. Rebuild the renewable weekly stack (fig2b) for 2023.

Read-only w.r.t. the model; only writes the CSV and the PNG/PDF figures.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import regionalflex_viz as viz

ROOT = Path(__file__).resolve().parent
PKL = ROOT / "results" / "robustness" / "baseline_2023_beta05_realistic.pkl"
PROCESSED = ROOT / "Data" / "processed"
FIGDIR = ROOT / "figs"
CSV_OUT = ROOT / "full_year_2023.csv"
YEAR = 2023

REGIONS = [
    "Auvergne_Rhone_Alpes",
    "Nouvelle_Aquitaine",
    "Occitanie",
    "Provence_Alpes_Cote_dAzur",
]


def build_full_year_csv() -> pd.DataFrame:
    with PKL.open("rb") as f:
        res = pickle.load(f)
    variables = res["variables"]
    duals = res.get("dual_variables", {})

    n = max(len(pd.Series(v)) for v in variables.values())
    idx = pd.RangeIndex(0, n)
    ts = pd.date_range(f"{YEAR}-01-01 00:00", periods=n, freq="30min")

    out = pd.DataFrame({"timestamp": ts})

    # optimisation variables (dispatch, storage, flows, DR, ...)
    for key, series in variables.items():
        out[key] = pd.Series(series, dtype=float).reindex(idx).to_numpy()

    # nodal prices from duals
    for region in REGIONS:
        out[f"nodal_price_{region}"] = (
            pd.Series(duals.get(region, {}), dtype=float).reindex(idx).to_numpy()
        )

    # per-region input series (demand / wind / solar) from processed data
    for region in REGIONS:
        path = PROCESSED / f"{region}_{YEAR}.csv"
        df = pd.read_csv(path, index_col=0)
        for col in ("demand", "wind", "solar", "ror"):
            vals = df[col].to_numpy() if col in df.columns else np.zeros(n)
            out[f"{col}_{region}"] = vals[:n]

    out.to_csv(CSV_OUT, index=False)
    print(f"[OK] wrote {CSV_OUT} ({out.shape[0]} rows x {out.shape[1]} cols)")
    return out


def weekly_stack_renewables(df: pd.DataFrame) -> None:
    """fig2b: weekly mean generation stack incl. solar & wind, 2023."""
    dfw = viz._resample(df, "W")

    solar = dfw[[c for c in dfw.columns if c.startswith("solar_")]].sum(axis=1)
    wind = dfw[[c for c in dfw.columns if c.startswith("wind_")]].sum(axis=1)
    ror = dfw[[c for c in dfw.columns if c.startswith("ror_")]].sum(axis=1)

    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(dfw))
    x = dfw[viz.TIME_COL]

    if solar.sum() > 0:
        ax.fill_between(x, bottom, bottom + solar, label="Solar", step="mid", color="gold")
        bottom = bottom + solar.values
    if wind.sum() > 0:
        ax.fill_between(x, bottom, bottom + wind, label="Wind", step="mid", color="lightblue")
        bottom = bottom + wind.values
    if ror.sum() > 0:
        ax.fill_between(x, bottom, bottom + ror, label="Run-of-river", step="mid", color="#2ca8a0")
        bottom = bottom + ror.values

    color_map = {
        "hydro": "C0", "nuclear": "C1", "thermal_gas": "C4",
        "thermal_fuel": "C3", "biofuel": "C2",
    }
    label_map = {"hydro": "Hydro_reservoir"}
    legend_only = []
    for label, prefix, _ in viz.TECH_PREFIXES:
        cols = [c for c in dfw.columns if c.startswith(prefix)]
        series = dfw[cols].sum(axis=1)
        name = label_map.get(label, label.replace("_", " ").capitalize())
        color = color_map.get(label, "C5")
        if series.sum() > 0 and series.mean() >= 50.0:
            ax.fill_between(x, bottom, bottom + series, label=name, step="mid", color=color)
            bottom = bottom + series.values
        else:
            legend_only.append(Patch(facecolor=color, label=name))

    ax.set_ylabel("Weekly mean dispatch [MW]")
    handles, _ = ax.get_legend_handles_labels()
    handles.extend(legend_only)
    ax.legend(handles=handles, loc="upper right", ncol=4)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig2b_weekly_stack_renewables.png", dpi=300)
    fig.savefig(FIGDIR / "fig2b_weekly_stack_renewables.pdf")
    plt.close(fig)
    print("[OK] fig2b_weekly_stack_renewables (2023)")


def write_congestion_csvs(df: pd.DataFrame) -> None:
    """Recompute the congestion-rent CSVs from the 2023 baseline (mirrors
    calculate_congestion_rents.py) so visualize_congestion_rents.py can plot."""
    dt = 0.5
    short = {
        "Auvergne_Rhone_Alpes": "ARA",
        "Nouvelle_Aquitaine": "NAQ",
        "Occitanie": "OCC",
        "Provence_Alpes_Cote_dAzur": "PAC",
    }
    cr = {}
    detailed = []
    for src in REGIONS:
        for dst in REGIONS:
            col = f"flow_out_{src}_{dst}"
            if col not in df.columns:
                continue
            f_ij = df[col]
            p_i = df[f"nodal_price_{src}"]
            p_j = df[f"nodal_price_{dst}"]
            cr_total = float((f_ij * (p_j - p_i) * dt).sum())
            name = f"{short[src]}->{short[dst]}"
            cr[name] = cr_total
            detailed.append({
                "Flow": name, "Source": short[src], "Destination": short[dst],
                "Total_Flow_MWh": float(f_ij.sum()),
                "Avg_Price_Source_€/MWh": float(p_i.mean()),
                "Avg_Price_Dest_€/MWh": float(p_j.mean()),
                "Avg_Price_Diff_€/MWh": float((p_j - p_i).mean()),
                "Congestion_Rent_€": cr_total,
                "Congestion_Rent_M€": cr_total / 1e6,
            })

    cr_all = sum(cr.values())
    region_cr = {short[r]: sum(v for k, v in cr.items() if k.startswith(short[r] + "->")) for r in REGIONS}

    pd.DataFrame(detailed).sort_values("Congestion_Rent_€", key=abs, ascending=False).to_csv(
        ROOT / "congestion_rents_detailed.csv", index=False)

    summary = [{"Region": rc, "Congestion_Rent_€": v, "Congestion_Rent_M€": v / 1e6}
               for rc, v in region_cr.items()]
    summary.append({"Region": "TOTAL", "Congestion_Rent_€": cr_all, "Congestion_Rent_M€": cr_all / 1e6})
    pd.DataFrame(summary).to_csv(ROOT / "congestion_rents_summary.csv", index=False)

    pairs = [("ARA", "NAQ"), ("ARA", "OCC"), ("ARA", "PAC"), ("NAQ", "OCC"), ("NAQ", "PAC"), ("OCC", "PAC")]
    net = []
    for a, b in pairs:
        cr_ab = cr.get(f"{a}->{b}", 0.0)
        cr_ba = cr.get(f"{b}->{a}", 0.0)
        net.append({"Pair": f"{a}<->{b}", "CR_A_to_B_M€": cr_ab / 1e6,
                    "CR_B_to_A_M€": cr_ba / 1e6, "Net_CR_M€": (cr_ab + cr_ba) / 1e6})
    pd.DataFrame(net).to_csv(ROOT / "congestion_rents_net.csv", index=False)
    print(f"[OK] congestion CSVs (total {cr_all / 1e6:,.1f} M EUR)")


def main() -> None:
    df = build_full_year_csv()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    viz.figure1_energy_mix(df, FIGDIR)
    viz.figure2_weekly_stack(df, FIGDIR)
    viz.figure3_dr_vs_storage(df, FIGDIR)
    viz.figure4_price_duration(df, FIGDIR)
    viz.figure5_flow_matrix(df, FIGDIR)
    viz.figure6_net_export(df, FIGDIR)
    viz.figure7_self_sufficiency(df, FIGDIR)
    viz.figure8_cost_breakdown(df, FIGDIR)
    print("[OK] fig1-fig8 (2023)")

    weekly_stack_renewables(df)
    print("[OK] fig1-fig8 + fig2b regenerated into", FIGDIR)

    write_congestion_csvs(df)
    print("\nAll 2023 figure inputs regenerated.")


if __name__ == "__main__":
    main()
