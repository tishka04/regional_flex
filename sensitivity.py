import copy
import yaml
from src.model.optimizer_regional_flex import RegionalFlexOptimizer

# Load baseline config
with open('config/config_master.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

# Max DR shift values and battery capacity multipliers
max_shift_values = [0.0, 2.0, 5.0, 10.0]  # Test multiple values to see variation
# Battery multipliers apply to both batteries_puissance_MW and batteries_stockage_MWh in each region
battery_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0]
results = {}

import subprocess
import pickle
import os

# Ensure directories
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
temp_cfg_dir = 'temp_configs'
os.makedirs(temp_cfg_dir, exist_ok=True)

for max_shift in max_shift_values:
    for batt_mult in battery_multipliers:
        temp_config = copy.deepcopy(base_config)
        # Modify demand response max_shift for each region
        print(f"\n=== Creating config for max_shift={max_shift}, battery_mult={batt_mult} ===")
        if 'demand_response' not in temp_config:
            temp_config['demand_response'] = {}
        for region in temp_config['regions']:
            if region not in temp_config['demand_response']:
                temp_config['demand_response'][region] = {}
            temp_config['demand_response'][region]['max_shift'] = max_shift
            # Keep existing values or set defaults
            temp_config['demand_response'][region]['participation_rate'] = temp_config['demand_response'][region].get('participation_rate', 1.0)
            temp_config['demand_response'][region]['max_total'] = temp_config['demand_response'][region].get('max_total',  
                                                                                                               10000.0 if region == 'Auvergne_Rhone_Alpes' else
                                                                                                               8000.0 if region == 'Nouvelle_Aquitaine' else
                                                                                                               6000.0 if region == 'Occitanie' else 4000.0)
        # Scale battery storage capacities per region
        if 'regional_storage' not in temp_config:
            temp_config['regional_storage'] = {}
        for region in temp_config['regions']:
            rs = temp_config['regional_storage'].get(region, {})
            base_p = rs.get('batteries_puissance_MW', 0.0)
            base_e = rs.get('batteries_stockage_MWh', 0.0)
            # If base config somehow lacks entries, keep zeros scaled
            rs['batteries_puissance_MW'] = float(base_p) * batt_mult
            rs['batteries_stockage_MWh'] = float(base_e) * batt_mult
            temp_config['regional_storage'][region] = rs
            print(f"  {region}: batteries MW={rs['batteries_puissance_MW']}, MWh={rs['batteries_stockage_MWh']}")

        # Save temporary config outside 'config' so master override does not apply
        safe_shift = str(max_shift).replace('.', 'p')
        safe_bmult = str(batt_mult).replace('.', 'p')
        temp_config_path = os.path.join(temp_cfg_dir, f'temp_config_shift_{safe_shift}_bmult_{safe_bmult}.yaml')
        with open(temp_config_path, 'w') as f:
            yaml.dump(temp_config, f)

        # Define output pickle file for this scenario
        out_pickle = os.path.join(results_dir, f'sensitivity_winter_weekday_shift_{safe_shift}_bmult_{safe_bmult}.pkl')
        # Call run_regional_flex.py with desired arguments
        cmd = [
            'python', 'run_regional_flex.py',
            '--config', temp_config_path,
            '--data-dir', 'data/processed',
            '--preset', 'winter_weekday',
            '--out', out_pickle
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(out_pickle):
            with open(out_pickle, 'rb') as pf:
                results[(max_shift, batt_mult)] = pickle.load(pf)
        else:
            print(f"Run failed for max_shift={max_shift:.1f}%, batt_mult={batt_mult}. Output:\n{result.stdout}\n{result.stderr}")
            results[(max_shift, batt_mult)] = None
        # Do NOT delete temp_config_path or out_pickle (keep for inspection)

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: total demand energy for the study horizon (used to normalise DR)
# ---------------------------------------------------------------------------

def compute_total_demand_kwh(regions, data_dir, start_date, end_date):
    """Load regional CSVs and compute total demand (kWh) in half-hourly data."""
    total_energy = 0.0
    for r in regions:
        path = os.path.join(data_dir, f"{r}.csv")
        if not os.path.exists(path):
            # Provence file name edge-case with apostrophe
            if "dAzur" in r:
                alt = path.replace("dAzur.csv", "d'Azur.csv")
                path = alt if os.path.exists(alt) else path
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        df = df.loc[start_date:end_date]
        # Half-hour timesteps → kWh = MW * 0.5 h * 1000
        total_energy += df['demand'].sum() * 0.5 * 1000
    return total_energy  # in kWh

# --- Visualization ---
# Extract metrics for plotting
max_shifts = []
total_costs = []
total_drs = []
dr_ratio = []  # % of demand covered by DR

def extract_total_dr_utilization(res):
    # Sum all DR variables (keys containing 'dr_' or 'demand_response') – MW at 0.5-h timesteps
    total_dr_mwh = 0.0
    if 'variables' in res:
        for var_name, vals in res['variables'].items():
            if ('dr_' in var_name) or ('demand_response' in var_name):
                total_dr_mwh += sum(v for v in vals.values() if v is not None) * 0.5  # MW→MWh
    return total_dr_mwh if total_dr_mwh > 0 else float('nan')

# Pre-compute total demand energy (kWh) for normalisation
winter_date = ('2022-01-18', '2022-01-18')  # matches winter_weekday preset
regions = base_config['regions']
TOTAL_DEMAND_KWH = compute_total_demand_kwh(regions, 'data/processed', *winter_date)

# Detailed diagnostic: examine all DR variables
print("\n=== Results Summary ===")
for (max_shift, batt_mult), res in results.items():
    if res is not None:
        print(f"\n--- max_shift={max_shift}, battery_mult={batt_mult} ---")
        # Find all DR-related variables
        dr_variables = {k: v for k, v in res.get('variables', {}).items() 
                        if 'dr_' in k or 'demand_response' in k}
        print(f"DR variables found: {list(dr_variables.keys())}")
        total_dr_by_var = {}
        for var_name, var_values in dr_variables.items():
            if var_values:
                total_dr = sum(v for v in var_values.values() if v is not None and v > 0) * 0.5
                total_dr_by_var[var_name] = total_dr
                non_zero_count = sum(1 for v in var_values.values() if v is not None and v > 0)
                print(f"  {var_name}: {total_dr:.1f} MWh ({non_zero_count} non-zero timesteps)")
            else:
                total_dr_by_var[var_name] = 0.0
                print(f"  {var_name}: 0.0 MWh (no values)")
        total_dr = sum(total_dr_by_var.values())
        dr_percentage = (total_dr / TOTAL_DEMAND_KWH * 1000) * 100
        print(f"Total DR = {total_dr:.1f} MWh ({dr_percentage:.2f}% of demand)")
    else:
        print(f"max_shift={max_shift}, battery_mult={batt_mult}: Failed")

# --- Actual plotting code ---
# 1) Line plots per battery multiplier vs. DR max_shift
os.makedirs('plots', exist_ok=True)
for batt_mult in battery_multipliers:
    xs, costs, drs, ratios = [], [], [], []
    for max_shift in max_shift_values:
        res = results.get((max_shift, batt_mult))
        if res is None:
            continue
        xs.append(max_shift)
        costs.append(res.get('total_cost', float('nan')))
        dr_mwh = extract_total_dr_utilization(res)
        drs.append(dr_mwh)
        if TOTAL_DEMAND_KWH > 0 and pd.notnull(dr_mwh):
            ratios.append(100 * (dr_mwh * 1000) / TOTAL_DEMAND_KWH)
        else:
            ratios.append(float('nan'))
    if xs:
        plt.figure(figsize=(8, 5))
        plt.plot(xs, costs, marker='o')
        plt.xlabel('Max DR Shift (%)')
        plt.ylabel('Total System Cost')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'plots/cost_vs_dr_shift_batt_{str(batt_mult).replace(".", "p")}.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(xs, drs, marker='o', color='green')
        plt.xlabel('Max DR Shift (%)')
        plt.ylabel('Total DR Utilisation (MWh)')
        plt.title(f'DR Utilisation vs. DR Shift (battery_mult={batt_mult})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'plots/dr_utilisation_vs_shift_mwh_batt_{str(batt_mult).replace(".", "p")}.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(xs, ratios, marker='s', color='purple')
        plt.xlabel('Max DR Shift (%)')
        plt.ylabel('DR Utilisation (% of demand)')
        plt.title(f'DR Share vs. DR Shift (battery_mult={batt_mult})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'plots/dr_utilisation_percent_vs_shift_batt_{str(batt_mult).replace(".", "p")}.png')
        plt.close()

# 2) Heatmaps across both dimensions
def collect_metric_as_matrix(metric_fn):
    mat = []
    for batt_mult in battery_multipliers:
        row = []
        for max_shift in max_shift_values:
            res = results.get((max_shift, batt_mult))
            if res is None:
                row.append(float('nan'))
            else:
                row.append(metric_fn(res))
        mat.append(row)
    return pd.DataFrame(mat, index=battery_multipliers, columns=max_shift_values)

cost_df = collect_metric_as_matrix(lambda r: r.get('total_cost', float('nan')))
dr_mwh_df = collect_metric_as_matrix(lambda r: extract_total_dr_utilization(r))
dr_share_df = collect_metric_as_matrix(lambda r: (100 * (extract_total_dr_utilization(r) * 1000) / TOTAL_DEMAND_KWH) if TOTAL_DEMAND_KWH > 0 else float('nan'))

def save_heatmap(df, title, fname, cmap='viridis'):
    plt.figure(figsize=(8, 5))
    plt.imshow(df.values, aspect='auto', cmap=cmap, origin='lower')
    plt.colorbar(label=title)
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df.index)), df.index)
    plt.xlabel('Max DR Shift (%)')
    plt.ylabel('Battery capacity multiplier')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

save_heatmap(cost_df, 'Total Cost', 'plots/heatmap_total_cost.png')
save_heatmap(dr_mwh_df, 'DR Utilisation (MWh)', 'plots/heatmap_dr_mwh.png')
save_heatmap(dr_share_df, 'DR Utilisation (% of demand)', 'plots/heatmap_dr_share.png')

# 3) 3D scatter: Total Cost vs DR Share vs Battery Capacity Multiplier
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed to enable 3D
    from matplotlib import cm
    xs_batt, ys_dr_share, zs_cost, colors = [], [], [], []
    for batt_mult in battery_multipliers:
        for max_shift in max_shift_values:
            res = results.get((max_shift, batt_mult))
            if res is None:
                continue
            dr_mwh = extract_total_dr_utilization(res)
            if pd.isna(dr_mwh) or TOTAL_DEMAND_KWH <= 0:
                continue
            dr_share = 100 * (dr_mwh * 1000) / TOTAL_DEMAND_KWH
            cost_val = res.get('total_cost', float('nan'))
            if pd.isna(cost_val):
                continue
            xs_batt.append(batt_mult)
            # Store max_shift on Y axis (was DR share previously)
            ys_dr_share.append(max_shift)
            zs_cost.append(cost_val)
            # Color points by DR share to retain that information
            colors.append(dr_share)

    if xs_batt:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(xs_batt, ys_dr_share, zs_cost, c=colors, cmap=cm.viridis, s=40, depthshade=True)
        cbar = plt.colorbar(sc, pad=0.1)
        cbar.set_label('DR utilisation (% of demand)')
        ax.set_xlabel('Battery capacity multiplier')
        ax.set_ylabel('Max DR Shift (%)')
        ax.set_zlabel('Total system cost')
        ax.set_title('Total Cost vs DR Share vs Battery Capacity')
        plt.tight_layout()
        plt.savefig('plots/3d_cost_vs_drshare_vs_batt.png')
        # Quadratic surface fit: cost ~ a0 + a1*x + a2*y + a3*x^2 + a4*x*y + a5*y^2
        try:
            import numpy as np
            x = np.asarray(xs_batt)
            y = np.asarray(ys_dr_share)
            z = np.asarray(zs_cost)
            if x.size >= 6:
                A = np.column_stack([
                    np.ones_like(x), x, y, x**2, x*y, y**2
                ])
                coef, *_ = np.linalg.lstsq(A, z, rcond=None)
                # Create grid and evaluate surface
                xg = np.linspace(x.min(), x.max(), 40)
                yg = np.linspace(y.min(), y.max(), 40)
                Xg, Yg = np.meshgrid(xg, yg)
                Zg = (
                    coef[0]
                    + coef[1]*Xg
                    + coef[2]*Yg
                    + coef[3]*Xg**2
                    + coef[4]*Xg*Yg
                    + coef[5]*Yg**2
                )
                ax.plot_surface(Xg, Yg, Zg, alpha=0.35, cmap=cm.viridis, linewidth=0)
                plt.tight_layout()
                plt.savefig('plots/3d_cost_vs_drshare_vs_batt_surface.png')
        except Exception as ee:
            print(f"Surface fit failed: {ee}")
        plt.close(fig)
except Exception as e:
    print(f"3D plot generation failed: {e}")

# 4) Interactive 3D (Plotly): scatter + quadratic surface fit
try:
    import numpy as np
    import plotly.graph_objects as go
    xs_batt_i, ys_dr_share_i, zs_cost_i, colors_i = [], [], [], []
    for batt_mult in battery_multipliers:
        for max_shift in max_shift_values:
            res = results.get((max_shift, batt_mult))
            if res is None:
                continue
            dr_mwh = extract_total_dr_utilization(res)
            if pd.isna(dr_mwh) or TOTAL_DEMAND_KWH <= 0:
                continue
            dr_share = 100 * (dr_mwh * 1000) / TOTAL_DEMAND_KWH
            cost_val = res.get('total_cost', float('nan'))
            if pd.isna(cost_val):
                continue
            xs_batt_i.append(batt_mult)
            # Use Y axis for Max DR Shift instead of DR share
            ys_dr_share_i.append(max_shift)
            zs_cost_i.append(cost_val)
            # Color by DR share to keep that info
            colors_i.append(dr_share)

    if xs_batt_i:
        fig_i = go.Figure()
        fig_i.add_trace(go.Scatter3d(
            x=xs_batt_i,
            y=ys_dr_share_i,
            z=zs_cost_i,
            mode='markers',
            marker=dict(size=4, color=colors_i, colorscale='Viridis', colorbar=dict(title='DR utilisation (% of demand)')),
            name='Scenarios'
        ))

        x = np.asarray(xs_batt_i)
        y = np.asarray(ys_dr_share_i)
        z = np.asarray(zs_cost_i)
        if x.size >= 6:
            A = np.column_stack([np.ones_like(x), x, y, x**2, x*y, y**2])
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
            xg = np.linspace(x.min(), x.max(), 50)
            yg = np.linspace(y.min(), y.max(), 50)
            Xg, Yg = np.meshgrid(xg, yg)
            Zg = (coef[0] + coef[1]*Xg + coef[2]*Yg + coef[3]*Xg**2 + coef[4]*Xg*Yg + coef[5]*Yg**2)
            fig_i.add_trace(go.Surface(x=Xg, y=Yg, z=Zg, colorscale='Viridis', opacity=0.5, showscale=False, name='Quadratic fit'))

        fig_i.update_layout(
            title='Total Cost vs DR Share vs Battery Capacity (Interactive)',
            scene=dict(
                xaxis_title='Battery capacity multiplier',
                yaxis_title='Max DR Shift (%)',
                zaxis_title='Total system cost',
            ),
            template='plotly_white',
            margin=dict(l=0, r=0, t=50, b=0)
        )
        os.makedirs('plots', exist_ok=True)
        fig_i.write_html('plots/interactive_3d_cost_drshare_batt.html', include_plotlyjs='cdn')
except Exception as e:
    print(f"Interactive 3D plot generation failed: {e}")

print('Sensitivity analysis complete. Plots saved as PNG files.')