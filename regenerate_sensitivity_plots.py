"""
Regenerate sensitivity analysis plots from existing pickle files
without rerunning the optimization.
"""
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Configuration matching original sensitivity.py
max_shift_values = [0.0, 2.0, 5.0, 10.0]
battery_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0]
results_dir = 'results'
plots_dir = 'plots'

# Load existing results from pickle files
results = {}
print("Loading existing sensitivity results...")
for max_shift in max_shift_values:
    for batt_mult in battery_multipliers:
        safe_shift = str(max_shift).replace('.', 'p')
        safe_bmult = str(batt_mult).replace('.', 'p')
        pickle_file = os.path.join(results_dir, f'sensitivity_winter_weekday_shift_{safe_shift}_bmult_{safe_bmult}.pkl')
        
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as pf:
                results[(max_shift, batt_mult)] = pickle.load(pf)
            print(f"  Loaded: max_shift={max_shift}, batt_mult={batt_mult}")
        else:
            print(f"  Missing: {pickle_file}")
            results[(max_shift, batt_mult)] = None

# Helper functions
def extract_total_dr_utilization(res):
    total_dr_mwh = 0.0
    if 'variables' in res:
        for var_name, vals in res['variables'].items():
            if ('dr_' in var_name) or ('demand_response' in var_name):
                total_dr_mwh += sum(v for v in vals.values() if v is not None) * 0.5
    return total_dr_mwh if total_dr_mwh > 0 else float('nan')

def compute_total_demand_kwh(regions, data_dir, start_date, end_date):
    total_energy = 0.0
    for r in regions:
        path = os.path.join(data_dir, f"{r}.csv")
        if not os.path.exists(path):
            if "dAzur" in r:
                alt = path.replace("dAzur.csv", "d'Azur.csv")
                path = alt if os.path.exists(alt) else path
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        df = df.loc[start_date:end_date]
        total_energy += df['demand'].sum() * 0.5 * 1000
    return total_energy

# Compute total demand
regions = ['Auvergne_Rhone_Alpes', 'Nouvelle_Aquitaine', 'Occitanie', 'Provence_Alpes_Cote_dAzur']
winter_date = ('2022-01-18', '2022-01-18')
TOTAL_DEMAND_KWH = compute_total_demand_kwh(regions, 'data/processed', *winter_date)

# Create plots directory
os.makedirs(plots_dir, exist_ok=True)

print("\nRegenerating plots...")

# 1) Line plots per battery multiplier vs. DR max_shift
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
        # Cost vs DR Shift plot (NO TITLE)
        plt.figure(figsize=(8, 5))
        plt.plot(xs, costs, marker='o')
        plt.xlabel('Max DR Shift (%)')
        plt.ylabel('Total System Cost')
        plt.grid(True)
        plt.tight_layout()
        output_path = f'{plots_dir}/cost_vs_dr_shift_batt_{str(batt_mult).replace(".", "p")}.png'
        plt.savefig(output_path)
        plt.close()
        print(f"  Created: {output_path}")

        # DR Utilisation plot
        plt.figure(figsize=(8, 5))
        plt.plot(xs, drs, marker='o', color='green')
        plt.xlabel('Max DR Shift (%)')
        plt.ylabel('Total DR Utilisation (MWh)')
        plt.title(f'DR Utilisation vs. DR Shift (battery_mult={batt_mult})')
        plt.grid(True)
        plt.tight_layout()
        output_path = f'{plots_dir}/dr_utilisation_vs_shift_mwh_batt_{str(batt_mult).replace(".", "p")}.png'
        plt.savefig(output_path)
        plt.close()
        print(f"  Created: {output_path}")

        # DR Share plot
        plt.figure(figsize=(8, 5))
        plt.plot(xs, ratios, marker='s', color='purple')
        plt.xlabel('Max DR Shift (%)')
        plt.ylabel('DR Utilisation (% of demand)')
        plt.title(f'DR Share vs. DR Shift (battery_mult={batt_mult})')
        plt.grid(True)
        plt.tight_layout()
        output_path = f'{plots_dir}/dr_utilisation_percent_vs_shift_batt_{str(batt_mult).replace(".", "p")}.png'
        plt.savefig(output_path)
        plt.close()
        print(f"  Created: {output_path}")

# 2) Heatmaps
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
    print(f"  Created: {fname}")

save_heatmap(cost_df, 'Total Cost', f'{plots_dir}/heatmap_total_cost.png')
save_heatmap(dr_mwh_df, 'DR Utilisation (MWh)', f'{plots_dir}/heatmap_dr_mwh.png')
save_heatmap(dr_share_df, 'DR Utilisation (% of demand)', f'{plots_dir}/heatmap_dr_share.png')

print("\n✓ All plots regenerated successfully!")
