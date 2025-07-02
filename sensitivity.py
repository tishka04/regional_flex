import copy
import yaml
from src.model.optimizer_regional_flex import RegionalFlexOptimizer

# Load baseline config
with open('config/config_master.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

# Max DR shift values
max_shift_values = [0.0, 2.0, 5.0, 10.0]  # Test multiple values to see variation
results = {}

import subprocess
import pickle
import os

for max_shift in max_shift_values:
    temp_config = copy.deepcopy(base_config)
    # Modify demand response max_shift for each region
    print(f"\n=== Creating config for max_shift={max_shift} ===")
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
        print(f"  {region}: max_shift={max_shift}, participation_rate={temp_config['demand_response'][region]['participation_rate']}, max_total={temp_config['demand_response'][region]['max_total']}")
    
    # Save temporary config
    # Safe filename (e.g. 0p5 -> 0p5)
    safe_val = str(max_shift).replace('.', 'p')
    temp_config_path = os.path.join('config', f'temp_config_{safe_val}.yaml')
    with open(temp_config_path, 'w') as f:
        yaml.dump(temp_config, f)
    # Ensure results directory exists
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Define output pickle file for this scenario
    out_pickle = os.path.join(results_dir, f'sensitivity_winter_weekday_{safe_val}.pkl')
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
            results[max_shift] = pickle.load(pf)
    else:
        print(f"Run failed for max_shift={max_shift*100:.1f}%. Output:\n{result.stdout}\n{result.stderr}")
        results[max_shift] = None
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
for max_shift, res in results.items():
    if res is not None:
        print(f"\n--- max_shift={max_shift} ---")
        
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
        print(f"max_shift={max_shift}: Failed")

# --- Actual plotting code ---
for max_shift, res in results.items():
    if res is not None:
        max_shifts.append(max_shift)
        total_costs.append(res.get('total_cost', float('nan')))
        dr_mwh = extract_total_dr_utilization(res)
        total_drs.append(dr_mwh)
        # Normalised utilisation (% of total demand)
        if TOTAL_DEMAND_KWH > 0:
            dr_ratio.append( 100 * (dr_mwh * 1000) / TOTAL_DEMAND_KWH )  # convert MWh→kWh
        else:
            dr_ratio.append(float('nan'))

# Plot Total System Cost vs. DR max_shift
plt.figure(figsize=(8, 5))
plt.plot(max_shifts, total_costs, marker='o')
plt.xlabel('Max DR Shift (%)')
plt.ylabel('Total System Cost')
plt.title('Total System Cost vs. Max DR Shift')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/cost_vs_dr_shift.png')
plt.show()

# Plot DR Utilisation (MWh)
plt.figure(figsize=(8, 5))
plt.plot(max_shifts, total_drs, marker='o', color='green')
plt.xlabel('Max DR Shift (%)')
plt.ylabel('Total DR Utilisation (MWh)')
plt.title('Total DR Utilisation vs. Max DR Shift')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/dr_utilisation_vs_shift_mwh.png')
plt.show()

# Plot DR Utilisation as % of demand
plt.figure(figsize=(8,5))
plt.plot(max_shifts, dr_ratio, marker='s', color='purple')
plt.xlabel('Max DR Shift (%)')
plt.ylabel('DR Utilisation (% of demand)')
plt.title('DR Utilisation Share vs. Max DR Shift')
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/dr_utilisation_percent_vs_shift.png')
plt.show()

print('Sensitivity analysis complete. Plots saved as PNG files.')