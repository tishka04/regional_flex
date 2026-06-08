"""Build a two-panel representative-day generation-stack figure for 2023.

Reviewer 1 asked for representative-day profiles in addition to the weekly
mean stack. This script reads the merged `full_year_2023.csv` (produced by
`regenerate_figures_2023.py` from the RoR-corrected, reservoir-capped baseline
pickle), aggregates dispatch to system totals at half-hourly resolution, and
plots two representative days side by side:

  * a winter weekday  -> the highest-demand Mon-Fri in Dec/Jan/Feb 2023;
  * a summer holiday   -> 15 August 2023 (Assumption Day, a French public
    holiday in the low-demand summer period).

Colours and labels mirror `fig2b_weekly_stack_renewables` for consistency.
Read-only w.r.t. the model; only writes the PNG/PDF into figs/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "full_year_2023.csv"
FIGDIR = ROOT / "figs"
YEAR = 2023

# Dispatchable technologies, in stacking order, matching fig2b colours.
TECHS = [
    ("hydro", "dispatch_hydro_", "Hydro (reservoir)", "C0"),
    ("nuclear", "dispatch_nuclear_", "Nuclear", "C1"),
    ("thermal_gas", "dispatch_thermal_gas_", "Thermal gas", "C4"),
    ("thermal_fuel", "dispatch_thermal_fuel_", "Thermal fuel", "C3"),
    ("biofuel", "dispatch_biofuel_", "Biofuel", "C2"),
]
VRE = [
    ("solar_", "Solar", "gold"),
    ("wind_", "Wind", "lightblue"),
    ("ror_", "Run-of-river", "#2ca8a0"),
]
STORAGE_COLOR = "#555555"


def _system_total(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].sum(axis=1)


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def build_components(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of system-total components indexed by timestamp."""
    comp = pd.DataFrame({"timestamp": df["timestamp"]})
    for prefix, label, _ in VRE:
        comp[label] = _system_total(df, prefix).to_numpy()
    for _, prefix, label, _ in TECHS:
        comp[label] = _system_total(df, prefix).to_numpy()
    # storage discharge across all units / regions (charge columns excluded)
    disch = [c for c in df.columns if c.startswith("storage_discharge_")]
    comp["Storage discharge"] = df[disch].sum(axis=1).to_numpy() if disch else 0.0
    comp["Demand"] = _system_total(df, "demand_").to_numpy()
    return comp.set_index("timestamp")


def pick_days(comp: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    daily_demand = comp["Demand"].resample("D").mean()
    idx = daily_demand.index
    winter_mask = idx.month.isin([12, 1, 2]) & (idx.weekday < 5)
    winter_day = daily_demand[winter_mask].idxmax().normalize()
    summer_day = pd.Timestamp(f"{YEAR}-08-15")
    return winter_day, summer_day


def plot_panel(ax, comp: pd.DataFrame, day: pd.Timestamp, title: str) -> None:
    sl = comp.loc[day : day + pd.Timedelta(hours=23, minutes=30)]
    x = sl.index
    bottom = np.zeros(len(sl))

    stack_order = (
        [(lbl, col) for _, lbl, col in VRE]
        + [(lbl, col) for _, _, lbl, col in TECHS]
    )
    for label, color in stack_order:
        series = sl[label].to_numpy()
        if series.sum() <= 0:
            continue
        ax.fill_between(x, bottom, bottom + series, step="mid",
                        label=label, color=color, alpha=0.9)
        bottom = bottom + series

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Hour of day")
    ax.xaxis.set_major_formatter(DateFormatter("%H"))
    ax.margins(x=0)
    ax.grid(True, alpha=0.3)


def main() -> None:
    df = load()
    comp = build_components(df)
    winter_day, summer_day = pick_days(comp)
    print(f"[INFO] winter weekday : {winter_day.date()} ({winter_day.day_name()})")
    print(f"[INFO] summer holiday : {summer_day.date()} ({summer_day.day_name()})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    plot_panel(axes[0], comp,
               winter_day, f"(a) Winter weekday — {winter_day.strftime('%a %d %b %Y')}")
    plot_panel(axes[1], comp,
               summer_day, f"(b) Summer holiday — {summer_day.strftime('%a %d %b %Y')}")

    axes[0].set_ylabel("System dispatch [MW]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.04), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    FIGDIR.mkdir(parents=True, exist_ok=True)
    png = FIGDIR / "fig_representative_days_2023.png"
    pdf = FIGDIR / "fig_representative_days_2023.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {png}")
    print(f"[OK] wrote {pdf}")


if __name__ == "__main__":
    main()
