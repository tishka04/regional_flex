#!/usr/bin/env python3
"""Extract paper-ready numbers for the Applied Energy revision.

Read-only: loads the robustness pickles + 2023 config/data and prints
(a) storage volume-weighted charge/discharge prices,
(b) flexible-energy share of inter-regional exchanges,
(c) slack activation totals (feasibility),
(d) directed congestion-rent rows,
(e) cost-component decomposition for the beta=0 vs beta=0.5 anomaly,
(f) 2023 config values for Appendix C and the hydro-cap appendix.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
ROB = ROOT / "results" / "robustness"
DT = 0.5  # hours per half-hourly step

REGIONS = [
    "Auvergne_Rhone_Alpes",
    "Nouvelle_Aquitaine",
    "Occitanie",
    "Provence_Alpes_Cote_dAzur",
]
SHORT = {
    "Auvergne_Rhone_Alpes": "ARA",
    "Nouvelle_Aquitaine": "NAQ",
    "Occitanie": "OCC",
    "Provence_Alpes_Cote_dAzur": "PACA",
}
DISPATCH = ["hydro", "nuclear", "thermal_gas", "thermal_fuel", "biofuel"]
STORAGE = ["STEP", "batteries"]


def load_pkl(name: str) -> dict:
    with (ROB / name).open("rb") as f:
        return pickle.load(f)


def svar(results: dict, key: str) -> pd.Series:
    d = results.get("variables", {}).get(key, {})
    if not d:
        return pd.Series(dtype=float)
    return pd.Series(d, dtype=float).sort_index()


def region_price(results: dict, region: str, idx) -> pd.Series:
    duals = results.get("dual_variables", {})
    return pd.Series(duals.get(region, {}), dtype=float).reindex(idx).fillna(0.0)


# ---------------------------------------------------------------------------
def storage_prices(results: dict) -> dict:
    """Volume-weighted average nodal price faced when charging / discharging."""
    num_ch = den_ch = num_dis = den_dis = 0.0
    for region in REGIONS:
        for st in STORAGE:
            ch = svar(results, f"storage_charge_{st}_{region}")
            dis = svar(results, f"storage_discharge_{st}_{region}")
            idx = ch.index.union(dis.index)
            if idx.empty:
                continue
            p = region_price(results, region, idx)
            ch = ch.reindex(idx, fill_value=0.0)
            dis = dis.reindex(idx, fill_value=0.0)
            num_ch += float((ch * p).sum())
            den_ch += float(ch.sum())
            num_dis += float((dis * p).sum())
            den_dis += float(dis.sum())
    return {
        "avg_charge_price": num_ch / den_ch if den_ch else float("nan"),
        "avg_discharge_price": num_dis / den_dis if den_dis else float("nan"),
        "charge_volume_mwh": den_ch * DT,
        "discharge_volume_mwh": den_dis * DT,
    }


# ---------------------------------------------------------------------------
def flexible_energy_shares(results: dict) -> dict:
    """Replicate analyze_flexibility.py denominators (dt-invariant shares)."""
    # exchanges = sum of |net flow| over directed pairs (raw MW-step sums)
    exch = 0.0
    for i, r1 in enumerate(REGIONS):
        for r2 in REGIONS[i + 1:]:
            f12 = svar(results, f"flow_out_{r1}_{r2}").sum()
            f21 = svar(results, f"flow_out_{r2}_{r1}").sum()
            exch += abs(float(f12) - float(f21))

    dr = sum(svar(results, f"demand_response_{r}").sum() for r in REGIONS)
    st_dis = sum(
        svar(results, f"storage_discharge_{s}_{r}").sum()
        for r in REGIONS for s in STORAGE
    )
    st_ch = sum(
        svar(results, f"storage_charge_{s}_{r}").sum()
        for r in REGIONS for s in STORAGE
    )
    disp = sum(
        svar(results, f"dispatch_{t}_{r}").sum()
        for r in REGIONS for t in DISPATCH
    )
    dr = float(dr); st_dis = float(st_dis); st_ch = float(st_ch); disp = float(disp)

    # (a) old-paper definition: includes dispatchable generation + both storage dirs
    tot_a = exch + dr + st_dis + st_ch + disp
    # (b) directional flexibility only: exchanges + DR + storage discharge
    tot_b = exch + dr + st_dis
    return {
        "exchanges_abs_net": exch,
        "dr": dr,
        "storage_discharge": st_dis,
        "storage_charge": st_ch,
        "dispatchable": disp,
        "share_exchanges_old_def_pct": 100.0 * exch / tot_a if tot_a else float("nan"),
        "share_exchanges_directional_pct": 100.0 * exch / tot_b if tot_b else float("nan"),
    }


# ---------------------------------------------------------------------------
def slack_totals(results: dict) -> dict:
    pos = neg = 0.0
    active = 0
    mx = 0.0
    for region in REGIONS:
        sp = svar(results, f"slack_pos_{region}")
        sn = svar(results, f"slack_neg_{region}")
        pos += float(sp.sum())
        neg += float(sn.sum())
        combined = sp.reindex(sp.index.union(sn.index), fill_value=0.0) \
            .add(sn.reindex(sp.index.union(sn.index), fill_value=0.0), fill_value=0.0)
        active += int((combined > 1e-3).sum())
        if not combined.empty:
            mx = max(mx, float(combined.max()))
    return {
        "slack_pos_mwh": pos * DT,
        "slack_neg_mwh": neg * DT,
        "slack_total_mwh": (pos + neg) * DT,
        "active_timesteps": active,
        "max_slack_mw": mx,
    }


# ---------------------------------------------------------------------------
def directed_congestion(results: dict) -> pd.DataFrame:
    rows = []
    for i in REGIONS:
        for j in REGIONS:
            if i == j:
                continue
            flow = svar(results, f"flow_out_{i}_{j}")
            if flow.empty:
                continue
            idx = flow.index
            pi = region_price(results, i, idx)
            pj = region_price(results, j, idx)
            cr = float((flow * (pj - pi)).sum() * DT)
            rows.append({
                "from": SHORT[i], "to": SHORT[j],
                "flow_mwh": float(flow.sum()) * DT,
                "avg_dP_eur_mwh": float((pj - pi).mean()),
                "cr_meur": cr / 1e6,
            })
    df = pd.DataFrame(rows).sort_values("cr_meur", key=abs, ascending=False)
    return df


# ---------------------------------------------------------------------------
def merged_config() -> dict:
    with (ROOT / "config" / "regional_flex_config_2023.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def cost_decomposition(results: dict, cfg: dict) -> dict:
    costs = cfg.get("costs", {})
    regional_costs = cfg.get("regional_costs", {})
    uc = cfg.get("uc_params", {})
    distances = cfg.get("regional_distances", {})
    flow_cost = costs.get("exchange", costs.get("flow", 35.0))
    flow_km = costs.get("flow_km_coeff", 0.0)
    slack_pen = costs.get("slack_penalty", 50000.0)

    comp = {k: 0.0 for k in
            ["dispatch_var", "fixed", "startup", "storage", "dr", "flow", "slack"]}

    for region in REGIONS:
        for tech in DISPATCH:
            c = regional_costs.get(region, {}).get(tech, costs.get(tech, 0.0))
            comp["dispatch_var"] += float(svar(results, f"dispatch_{tech}_{region}").sum()) * c
            fc = uc.get(region, {}).get(tech, {}).get("fixed_cost", 0.0)
            sc = uc.get(region, {}).get(tech, {}).get("startup_cost", 0.0)
            comp["fixed"] += float(svar(results, f"uc_{tech}_{region}").sum()) * fc
            comp["startup"] += float(svar(results, f"startup_{tech}_{region}").sum()) * sc
        for st in STORAGE:
            comp["storage"] += float(svar(results, f"storage_charge_{st}_{region}").sum()) * costs.get("storage_charge", 35.0)
            comp["storage"] += float(svar(results, f"storage_discharge_{st}_{region}").sum()) * costs.get("storage_discharge", 50.0)
        comp["dr"] += float(svar(results, f"demand_response_{region}").sum()) * costs.get("demand_response", 120.0)
        comp["slack"] += (float(svar(results, f"slack_pos_{region}").sum())
                          + float(svar(results, f"slack_neg_{region}").sum())) * slack_pen

    for i in REGIONS:
        for j in REGIONS:
            if i == j:
                continue
            d = distances.get(i, {}).get(j, 0.0)
            unit = flow_cost + flow_km * d
            comp["flow"] += float(svar(results, f"flow_out_{i}_{j}").sum()) * unit

    comp["TOTAL_reconstructed"] = sum(v for k, v in comp.items())
    comp["objective_in_pkl"] = float(results.get("total_cost") or results.get("objective_value") or 0.0)
    return comp


# ---------------------------------------------------------------------------
def hydro_budgets(cfg: dict) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        path = ROOT / "Data" / "processed" / f"{region}_2023.csv"
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        hf = df["hydro_flex"] if "hydro_flex" in df.columns else pd.Series(dtype=float)
        ror = df["ror"] if "ror" in df.columns else pd.Series(dtype=float)
        hydro = df["hydro"] if "hydro" in df.columns else pd.Series(dtype=float)
        rows.append({
            "region": SHORT[region],
            "hydro_total_twh": float(hydro.sum()) * DT / 1e6 if not hydro.empty else float("nan"),
            "ror_twh": float(ror.sum()) * DT / 1e6 if not ror.empty else float("nan"),
            "hydro_flex_budget_twh": float(hf.sum()) * DT / 1e6 if not hf.empty else float("nan"),
            "hydro_flex_max_mw": float(hf.max()) if not hf.empty else float("nan"),
            "hydro_flex_mean_mw": float(hf.mean()) if not hf.empty else float("nan"),
        })
    return pd.DataFrame(rows)


def appendix_capacities(cfg: dict) -> pd.DataFrame:
    caps = cfg.get("regional_capacities", {})
    stor = cfg.get("regional_storage", {})
    rcost = cfg.get("regional_costs", {})
    rows = []
    for region in REGIONS:
        c = caps.get(region, {})
        s = stor.get(region, {})
        rc = rcost.get(region, {})
        storage_power = float(s.get("STEP_puissance_MW", 0.0)) + float(s.get("batteries_puissance_MW", 0.0))
        storage_energy = float(s.get("STEP_stockage_MWh", 0.0)) + float(s.get("batteries_stockage_MWh", 0.0))
        rows.append({
            "region": SHORT[region],
            "hydro_MW": c.get("hydro"), "hydro_cost": rc.get("hydro"),
            "nuclear_MW": c.get("nuclear"), "nuclear_cost": rc.get("nuclear"),
            "gas_MW": c.get("thermal_gas"), "fuel_MW": c.get("thermal_fuel"),
            "biofuel_MW": c.get("biofuel"), "biofuel_cost": rc.get("biofuel"),
            "storage_power_MW": round(storage_power, 1),
            "storage_energy_MWh": round(storage_energy, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)

    base = load_pkl("baseline_2023_beta05_realistic.pkl")
    beta0 = load_pkl("dr_beta00_2023.pkl")
    cfg = merged_config()

    print("\n================ (a) STORAGE PRICES — baseline 2023 ================")
    print(storage_prices(base))

    print("\n================ (b) FLEXIBLE-ENERGY SHARES — baseline 2023 ========")
    for k, v in flexible_energy_shares(base).items():
        print(f"  {k}: {v:,.4f}")

    print("\n================ (c) SLACK TOTALS ================")
    print("baseline beta=0.5:", slack_totals(base))
    print("dr beta=0       :", slack_totals(beta0))

    print("\n================ (d) DIRECTED CONGESTION RENTS — baseline 2023 ====")
    dcr = directed_congestion(base)
    print(dcr.to_string(index=False))
    print(f"  TOTAL directed CR: {dcr['cr_meur'].sum():,.2f} M EUR")

    print("\n================ (e) COST DECOMPOSITION (anomaly check) ==========")
    print("-- baseline beta=0.5 --")
    for k, v in cost_decomposition(base, cfg).items():
        print(f"  {k}: {v/1e6:,.2f} M (units of objective)")
    print("-- dr beta=0 --")
    for k, v in cost_decomposition(beta0, cfg).items():
        print(f"  {k}: {v/1e6:,.2f} M (units of objective)")

    print("\n================ (f) 2023 HYDRO-CAP BUDGETS (Appendix) ============")
    print(hydro_budgets(cfg).to_string(index=False))

    print("\n================ (f) 2023 CAPACITIES (Appendix C) =================")
    print(appendix_capacities(cfg).to_string(index=False))
    print(f"\n  min_nuclear_capacity_fraction = {cfg.get('min_nuclear_capacity_fraction')}")


if __name__ == "__main__":
    main()
