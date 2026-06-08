"""Reconstruct IRRE / EFS ramping-adequacy metrics from a RegionalFlex result.

Implements the definitions in the paper (Sec. 3.4): for each region and
half-hourly step, the residual-load ramp dRL_r(t) = RL_r(t) - RL_r(t-1) is
compared against the *local* upward / downward ramping headroom R_r^+/-(t)
derived from generation ramp limits, storage power, and demand response.
Imports/exchanges are deliberately excluded so the metric measures local
ramping adequacy (hence PACA, with no nuclear, shows the most stress).

    R_r^+(t) = sum_k min(rho_k G_k, G_k - g_k(t))         (generation ramp-up)
             + sum_s (Pmax_s - discharge_s(t))            (spare discharge power)
             + dr_shift_frac * RL_r(t)                     (sheddable DR)
    R_r^-(t) = sum_k min(rho_k G_k, g_k(t) - gmin_k)       (generation ramp-down)
             + sum_s (Pmax_s - charge_s(t))                (spare charge power)

    IRRE^+_r = mean_t  1[ +dRL_r(t) > R_r^+(t) ]
    IRRE^-_r = mean_t  1[ -dRL_r(t) > R_r^-(t) ]
    EFS^+_r  = mean_t ( dRL_r(t) - R_r^+(t) | +dRL > R^+ )
    EFS^-_r  = mean_t (-dRL_r(t) - R_r^-(t) | -dRL > R^-)

Usage:
    python compute_irre_efs.py --result results/robustness/baseline_2023_beta05_realistic.pkl \
                               --config config/regional_flex_config_2023.yaml \
                               --data-dir Data/processed --data-suffix _2023
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import yaml

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
DISPATCH = ["hydro", "nuclear", "biofuel", "thermal_gas", "thermal_fuel"]
STORAGE = ["STEP", "batteries"]
DR_SHIFT_FRAC = 0.05  # max_shift 5% * participation 1.0


def series_to_array(d, n):
    arr = np.zeros(n)
    for t, v in d.items():
        if isinstance(t, int) and 0 <= t < n:
            arr[t] = float(v or 0.0)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default="Data/processed")
    ap.add_argument("--data-suffix", default="_2023")
    ap.add_argument("--out-prefix", default="irre_efs_timevarying_yaml")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    results = pickle.load(open(args.result, "rb"))
    v = results["variables"]

    caps = cfg.get("regional_capacities", {})
    ramp = {k: float(p.get("ramp_rate", 0.0) or 0.0) for k, p in (cfg.get("tech_params") or {}).items()}
    gmin_frac = float(cfg.get("min_nuclear_capacity_fraction", 0.0) or 0.0)
    storage_cfg = cfg.get("regional_storage", {})
    sparams = cfg.get("storage_params", {})
    mpr = {s: float(sparams.get(s, {}).get("max_power_ratio", 1.0) or 1.0) for s in STORAGE}

    # determine number of steps from residual_load length of first region
    rl = {}
    for reg in REGIONS:
        df = pd.read_csv(f"{args.data_dir}/{reg}{args.data_suffix}.csv", parse_dates=[0], index_col=0)
        rl[reg] = df["residual_load"].to_numpy(dtype=float)
    n = min(len(rl[reg]) for reg in REGIONS)
    # align with the committed result horizon
    n_res = max((max(d.keys()) for d in v.values() if d), default=n - 1) + 1
    n = min(n, n_res)
    for reg in REGIONS:
        rl[reg] = rl[reg][:n]

    rows = []
    agg_dRL = np.zeros(n - 1)
    agg_Rp = np.zeros(n - 1)
    agg_Rm = np.zeros(n - 1)

    for reg in REGIONS:
        RL = rl[reg]
        dRL = np.diff(RL)  # length n-1, ramp into step t (t=1..n-1)

        Rp = np.zeros(n)
        Rm = np.zeros(n)
        for tech in DISPATCH:
            G = float(caps.get(reg, {}).get(tech, 0.0) or 0.0)
            if G <= 0:
                continue
            rho = ramp.get(tech, 0.0)
            max_ramp = rho * G  # corrected per-step ramp (MW)
            g = series_to_array(v.get(f"dispatch_{tech}_{reg}", {}), n)
            gmin = gmin_frac * G if tech == "nuclear" else 0.0
            Rp += np.minimum(max_ramp, np.maximum(G - g, 0.0))
            Rm += np.minimum(max_ramp, np.maximum(g - gmin, 0.0))

        for s in STORAGE:
            pmax = float(storage_cfg.get(reg, {}).get(f"{s}_puissance_MW", 0.0) or 0.0) * mpr[s]
            dis = series_to_array(v.get(f"storage_discharge_{s}_{reg}", {}), n)
            chg = series_to_array(v.get(f"storage_charge_{s}_{reg}", {}), n)
            Rp += np.maximum(pmax - dis, 0.0)
            Rm += np.maximum(pmax - chg, 0.0)

        # DR sheddable headroom (upward only)
        Rp += DR_SHIFT_FRAC * np.maximum(RL, 0.0)

        # align headroom to the ramp target step t (1..n-1): use headroom at t
        Rp_t = Rp[1:]
        Rm_t = Rm[1:]

        short_up = dRL > Rp_t
        short_dn = (-dRL) > Rm_t
        irre_up = 100.0 * short_up.mean()
        irre_dn = 100.0 * short_dn.mean()
        efs_up = float((dRL[short_up] - Rp_t[short_up]).mean()) if short_up.any() else 0.0
        efs_dn = float(((-dRL[short_dn]) - Rm_t[short_dn]).mean()) if short_dn.any() else 0.0

        rows.append({
            "Region": SHORT[reg],
            "IRRE_up_%": round(irre_up, 3),
            "IRRE_dn_%": round(irre_dn, 3),
            "EFS_up_MW": round(efs_up, 3),
            "EFS_dn_MW": round(efs_dn, 3),
        })

        agg_dRL += dRL
        agg_Rp += Rp_t
        agg_Rm += Rm_t

    # system aggregate: single-node (summed residual-load ramp vs summed headroom)
    s_up = agg_dRL > agg_Rp
    s_dn = (-agg_dRL) > agg_Rm
    sys_row = {
        "Region": "SYSTEM_AGGREGATE",
        "IRRE_up_%": round(100.0 * s_up.mean(), 3),
        "IRRE_dn_%": round(100.0 * s_dn.mean(), 3),
        "EFS_up_MW": round(float((agg_dRL[s_up] - agg_Rp[s_up]).mean()) if s_up.any() else 0.0, 3),
        "EFS_dn_MW": round(float(((-agg_dRL[s_dn]) - agg_Rm[s_dn]).mean()) if s_dn.any() else 0.0, 3),
    }

    reg_df = pd.DataFrame(rows)
    sys_df = pd.DataFrame([sys_row])
    print(reg_df.to_string(index=False))
    print(sys_df.to_string(index=False))

    reg_df.to_csv(f"{args.out_prefix}_regional.csv", index=False)
    sys_df.to_csv(f"{args.out_prefix}_system.csv", index=False)
    print(f"\n[OK] wrote {args.out_prefix}_regional.csv and {args.out_prefix}_system.csv")


if __name__ == "__main__":
    main()
