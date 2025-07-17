"""RegionalFlex Result Visualization Script (v2)
====================================================
Author: Théotime Coudray
Date: 2025‑07‑06
--------------------------------------------------
This **v2** reinstates *Figure 7* (regional self‑sufficiency) and
*Figure 8* (system cost breakdown) that explicitly address the
paper’s decentralisation and cost‑efficiency research questions.

Usage
-----
$ python regionalflex_viz.py --input full_year.csv --outdir figs/

Dependencies
------------
- Python ≥ 3.9
- pandas
- numpy
- matplotlib
- seaborn  (only for heat‑map & bar aesthetics)

Figure catalogue
----------------
1. Annual energy mix (bar)
2. Weekly generation stack (area)
3. Demand‑response vs. storage (bar)
4. Nodal price duration curves (line)
5. Inter‑regional exchange matrix (heat‑map)
6. Net export balance by region (bar)
7. **Regional self‑sufficiency index** (bar)
8. **System cost breakdown** (stacked bar)

All figures are exported as PNG (300 dpi) *and* PDF.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# -----------------------------------------------------------------------------
# Global configuration & constants
# -----------------------------------------------------------------------------
TIME_COL = "timestamp"          # column with half‑hourly timestamps
TIME_FREQ = "30min"            # model resolution (informative only)

# Technology dispatch column prefixes and variable costs (€/MWh)
TECH_PREFIXES = [
    ("hydro", "dispatch_hydro_", 23),
    ("nuclear", "dispatch_nuclear_", 30),
    ("thermal_gas", "dispatch_thermal_gas_", 75),
    ("thermal_fuel", "dispatch_thermal_fuel_", 85),
    ("biofuel", "dispatch_biofuel_", 45),
]

# Demand‑response settings
DR_PREFIX = "demand_response_"
DR_COST = 120          # €/MWh activated

# Storage columns and round‑trip cost approximation (€/MWh)
STOR_CH_PREFIXES = ["storage_charge_STEP_", "storage_charge_batteries_"]
STOR_DIS_PREFIXES = ["storage_discharge_STEP_", "storage_discharge_batteries_"]
STOR_COST_CH = 35      # €/MWh charged
STOR_COST_DIS = 50     # €/MWh discharged (includes losses)

# Inter‑regional power flow columns and wheeling tariff (€/MWh)
FLOW_PREFIX = "flow_out_"
FLOW_COST = 35         # €/MWh exported

# Nodal price columns
PRICE_PREFIX = "nodal_price_"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _abbr_region(name: str) -> str:
    """
    Turn 'Auvergne_Rhone_Alpes'  -> 'ARA'
         'grand_est'            -> 'GE'
         'Île-de-France'        -> 'IDF'
    Keep it automatic: take the first letter of every underscore/space-
    separated token and make it upper-case.
    """
    tokens = re.split(r"[_\\s]+", name)
    return "".join(t[0].upper() for t in tokens if t)

def _to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df


def _resample(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Down‑sample time series to *freq* (mean)."""
    df = _to_datetime(df.copy())
    return df.set_index(TIME_COL).resample(freq).mean().reset_index()


def _half_hour_to_energy(series: pd.Series) -> float:
    """Convert MW half‑hour profile to MWh."""
    return series.sum() * 0.5


def _aggregate_energy(df: pd.DataFrame, prefixes: list[str]) -> float:
    cols = []
    for p in prefixes:
        cols.extend([c for c in df.columns if c.startswith(p)])
    if not cols:
        return 0.0
    return _half_hour_to_energy(df[cols].sum(axis=1))


def _parse_flow_columns(cols: list[str]):
    pat = re.compile(r"flow_out_(.*?)_(.*)")
    parsed = []
    for c in cols:
        m = pat.match(c)
        if m:
            parsed.append((c, m.group(1), m.group(2)))
    return parsed

# -----------------------------------------------------------------------------
# Figure 1 – Annual energy mix
# -----------------------------------------------------------------------------

def figure1_energy_mix(df: pd.DataFrame, outdir: Path):
    energy = {}
    for label, prefix, _ in TECH_PREFIXES:
        cols = [c for c in df.columns if c.startswith(prefix)]
        energy[label] = _half_hour_to_energy(df[cols].sum(axis=1))
    total = sum(energy.values())
    share = {k: 100 * v / total for k, v in energy.items()}

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(share.keys()), list(share.values()))
    ax.set_ylabel("Share of annual energy [%]")
    ax.set_title("Annual energy mix – RegionalFlex")
    ax.set_ylim(0, 100)
    for i, (k, v) in enumerate(share.items()):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_energy_mix.png", dpi=300)
    fig.savefig(outdir / "fig1_energy_mix.pdf")

# -----------------------------------------------------------------------------
# Figure 2 – Weekly generation stack
# -----------------------------------------------------------------------------

def figure2_weekly_stack(df: pd.DataFrame, outdir: Path):
    df_week = _resample(df, "W")
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(df_week))
    for label, prefix, _ in TECH_PREFIXES:
        cols = [c for c in df_week.columns if c.startswith(prefix)]
        series = df_week[cols].sum(axis=1)
        ax.fill_between(df_week[TIME_COL], bottom, bottom + series,
                        label=label.replace("_", " ").capitalize(), step="mid")
        bottom += series.values
    ax.set_ylabel("Mean dispatch [MW]")
    ax.set_title("Weekly generation stack")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_weekly_stack.png", dpi=300)
    fig.savefig(outdir / "fig2_weekly_stack.pdf")

# -----------------------------------------------------------------------------
# Figure 3 – DR vs. storage energy
# -----------------------------------------------------------------------------

def figure3_dr_vs_storage(df: pd.DataFrame, outdir: Path):
    dr_energy = _aggregate_energy(df, [DR_PREFIX])
    stor_energy = _aggregate_energy(df, STOR_DIS_PREFIXES)
    data = pd.DataFrame({
        "category": ["Demand response", "Storage discharge"],
        "energy_GWh": [dr_energy / 1e3, stor_energy / 1e3]
    })
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=data, x="category", y="energy_GWh", ax=ax)
    ax.set_ylabel("Energy [GWh]")
    ax.set_title("DR eclipses storage")
    for i, r in data.iterrows():
        ax.text(i, r.energy_GWh + 5, f"{r.energy_GWh:.0f}", ha="center")
    fig.tight_layout()
    fig.savefig(outdir / "fig3_dr_vs_storage.png", dpi=300)
    fig.savefig(outdir / "fig3_dr_vs_storage.pdf")

# -----------------------------------------------------------------------------
# Figure 4 – Price duration curves
# -----------------------------------------------------------------------------

def figure4_price_duration(df: pd.DataFrame, outdir: Path):
    price_cols = [c for c in df.columns if c.startswith(PRICE_PREFIX)]
    duration = pd.DataFrame({c: np.sort(df[c].values)[::-1] for c in price_cols})
    fig, ax = plt.subplots(figsize=(6, 4))
    hours = np.arange(len(duration))
    for c in price_cols:
        region = c.replace(PRICE_PREFIX, "").replace("_", " ")
        ax.plot(hours, duration[c], label=region)
    ax.set_xlabel("Sorted hours in year [h]")
    ax.set_ylabel("Nodal price [€/MWh]")
    ax.set_title("Price duration curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig4_price_duration.png", dpi=300)
    fig.savefig(outdir / "fig4_price_duration.pdf")

# -----------------------------------------------------------------------------
# Figure 5 – Inter‑regional exchange matrix
# -----------------------------------------------------------------------------

# --- Figure 5 --------------------------------------------------------------
def figure5_flow_matrix(df: pd.DataFrame, outdir: Path):
    parsed  = _parse_flow_columns([c for c in df.columns if c.startswith(FLOW_PREFIX)])
    if not parsed:
        return
    regions = sorted({o for _, o, _ in parsed} | {d for _, _, d in parsed})
    mat     = np.zeros((len(regions), len(regions)))
    for col, o, d in parsed:
        mat[regions.index(o), regions.index(d)] += _half_hour_to_energy(df[col]) / 1e3

    labels = [_abbr_region(r) for r in regions]          # <<< here
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mat, annot=True, fmt=".0f", cmap="crest",
                cbar_kws={"label": "GWh"},
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Inter-regional annual exports (GWh)")
    fig.tight_layout()
    fig.savefig(outdir / "fig5_flow_matrix.png", dpi=300)
    fig.savefig(outdir / "fig5_flow_matrix.pdf")

# --- Figure 6 --------------------------------------------------------------
def figure6_net_export(df: pd.DataFrame, outdir: Path):
    parsed = _parse_flow_columns([c for c in df.columns if c.startswith(FLOW_PREFIX)])
    exports, imports = {}, {}
    for col, o, d in parsed:
        e            = _half_hour_to_energy(df[col])
        exports[o]   = exports.get(o, 0) + e
        imports[d]   = imports.get(d, 0) + e
    regions = sorted(set(exports) | set(imports))

    data = pd.DataFrame({
        "abbr" : [_abbr_region(r) for r in regions],     # <<< here
        "net_export_GWh": [
            (exports.get(r, 0) - imports.get(r, 0)) / 1e3 for r in regions
        ],
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=data, x="abbr", y="net_export_GWh", ax=ax)
    ax.axhline(0, linestyle="--", linewidth=0.8, color="k")
    ax.set_ylabel("Net export [GWh]")
    ax.set_title("Annual net-export balance")
    fig.tight_layout()
    fig.savefig(outdir / "fig6_net_export.png", dpi=300)
    fig.savefig(outdir / "fig6_net_export.pdf")

# --- Figure 7 --------------------------------------------------------------
def figure7_self_sufficiency(df: pd.DataFrame, outdir: Path):
    regions = [c.replace(PRICE_PREFIX, "") for c in df.columns if c.startswith(PRICE_PREFIX)]
    flows   = _parse_flow_columns([c for c in df.columns if c.startswith(FLOW_PREFIX)])

    rows = []
    for r in regions:
        gen = sum(
            _half_hour_to_energy(df[[c for c in df.columns if c.startswith(pref + r)]].sum(axis=1))
            for _, pref, _ in TECH_PREFIXES
        )
        imp = sum(_half_hour_to_energy(df[c]) for c, o, d in flows if d == r)
        exp = sum(_half_hour_to_energy(df[c]) for c, o, d in flows if o == r)
        supply = gen + max(imp - exp, 0)
        if supply:
            rows.append((_abbr_region(r), 100 * gen / supply))   # <<< here

    df_plot = pd.DataFrame(rows, columns=["abbr", "local"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df_plot, x="abbr", y="local", ax=ax)
    ax.axhline(50, linestyle="--", linewidth=0.8, color="k")
    ax.set_ylabel("Consumption met locally [%]")
    ax.set_title("Regional self-sufficiency index")
    fig.tight_layout()
    fig.savefig(outdir / "fig7_self_sufficiency.png", dpi=300)
    fig.savefig(outdir / "fig7_self_sufficiency.pdf")

# -----------------------------------------------------------------------------
# Figure 8 – System cost breakdown
# -----------------------------------------------------------------------------

def figure8_cost_breakdown(df: pd.DataFrame, outdir: Path):
    # Generation variable costs
    costs = {}
    for label, prefix, cost in TECH_PREFIXES:
        energy = _aggregate_energy(df, [prefix])
        costs[label] = energy * cost / 1e6  # → M€

        # Storage charging/discharging costs
    charge_e = _aggregate_energy(df, STOR_CH_PREFIXES)
    discharge_e = _aggregate_energy(df, STOR_DIS_PREFIXES)
    costs["storage_charge"] = charge_e * STOR_COST_CH / 1e6
    costs["storage_discharge"] = discharge_e * STOR_COST_DIS / 1e6

    # Demand-response activation cost
    dr_e = _aggregate_energy(df, [DR_PREFIX])
    costs["demand_response"] = dr_e * DR_COST / 1e6

    # Flow wheeling costs (count exports once)
    flow_e = sum(_half_hour_to_energy(df[c]) for c in df.columns
                 if c.startswith(FLOW_PREFIX))
    costs["flows"] = flow_e * FLOW_COST / 1e6

    # Pretty ordering for stacked bar
    order = ["hydro", "nuclear", "thermal_gas", "thermal_fuel", "biofuel",
             "storage_charge", "storage_discharge", "demand_response", "flows"]

    labels = [lbl.replace("_", " ").capitalize() for lbl in order]
    values = [costs.get(k, 0) for k in order]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=labels, y=values, ax=ax)
    ax.set_ylabel("Variable cost [M€]")
    ax.set_title("System cost breakdown")
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # Annotate bars
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, f"{v:.1f}", ha="center")

    fig.tight_layout()
    fig.savefig(outdir / "fig8_cost_breakdown.png", dpi=300)
    fig.savefig(outdir / "fig8_cost_breakdown.pdf")


# =============================================================================
# Main driver
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RegionalFlex visualisation v2")
    parser.add_argument("--input", required=True, help="path to full_year.csv")
    parser.add_argument("--outdir", default="figs", help="figure output folder")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    # Generate all figures
    figure1_energy_mix(df, outdir)
    figure2_weekly_stack(df, outdir)
    figure3_dr_vs_storage(df, outdir)
    figure4_price_duration(df, outdir)
    figure5_flow_matrix(df, outdir)
    figure6_net_export(df, outdir)
    figure7_self_sufficiency(df, outdir)
    figure8_cost_breakdown(df, outdir)

    print(f"Saved figures to {outdir.resolve()}")
    print("Done – v2.")


if __name__ == "__main__":
    main()
