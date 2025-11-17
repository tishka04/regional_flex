"""
Regenerate the simple cost_vs_dr_shift.png plot without title
"""
import os
import pickle
import matplotlib.pyplot as plt

results_dir = 'results'
plots_dir = 'plots'

# Use battery_mult = 1.0 for the main plot
batt_mult = 1.0
max_shift_values = [0.0, 2.0, 5.0, 10.0]

# Load results
xs, costs = [], []
for max_shift in max_shift_values:
    safe_shift = str(max_shift).replace('.', 'p')
    safe_bmult = str(batt_mult).replace('.', 'p')
    pickle_file = os.path.join(results_dir, f'sensitivity_winter_weekday_shift_{safe_shift}_bmult_{safe_bmult}.pkl')
    
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as pf:
            res = pickle.load(pf)
            xs.append(max_shift)
            costs.append(res.get('total_cost', float('nan')))

if xs:
    plt.figure(figsize=(8, 5))
    plt.plot(xs, costs, marker='o')
    plt.xlabel('Max DR Shift (%)')
    plt.ylabel('Total System Cost')
    # NO TITLE
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(plots_dir, 'cost_vs_dr_shift.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Created: {output_path}")
    print("Title removed successfully!")
else:
    print("Error: No data found")
