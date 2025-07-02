#!/usr/bin/env python3
"""Run the Regional Flexibility Optimizer and produce basic visualisations.

Usage examples
--------------
Full‑year run (2022):
    python run_regional_flex.py --config config_master.yaml --data-dir data/processed --out results/full_year.pkl

Specific calendar day (built‑in shortcuts):
    python run_regional_flex.py --config config_master.yaml --data-dir data/processed --preset winter_weekday

Custom interval:
    python run_regional_flex.py --config config_master.yaml --data-dir data/processed \
        --start 2022-03-15 --end 2022-03-15 --out results/2022-03-15.pkl
"""

import argparse, logging, os, sys
import yaml
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import pulp
from rolling_utils import rolling_horizon_indices
from src.model.optimizer_regional_flex import RegionalFlexOptimizer
from src.model import calculate_emissions

# ----- presets for the paper ------------------------------------------------
PRESETS = {
    # label: (start_date, end_date)
    'winter_weekday':   ('2022-01-18', '2022-01-18'),     # mardi
    'autumn_weekend':   ('2022-10-09', '2022-10-09'),     # dimanche
    'spring_weekday':   ('2022-05-12', '2022-05-12'),     # jeudi
    'summer_holiday':   ('2022-08-15', '2022-08-15'),     # Assomption
    'full_year':        ('2022-01-01', '2022-12-31')
}

# ---------------------------------------------------------------------------
def load_regional_timeseries(regions, data_dir):
    """Load demand & RES data for each region.
    Expects <region>.csv with half‑hourly index and at least columns: demand.
    Adjust to your actual data schema.
    """
    data = {}
    for r in regions:
        filename = f"{r}.csv"
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            # Fallback: try apostrophe variant for Provence
            if "dAzur" in r:
                alt_filename = filename.replace("dAzur.csv", "d'Azur.csv")
                alt_path = os.path.join(data_dir, alt_filename)
                if os.path.exists(alt_path):
                    path = alt_path
                else:
                    raise FileNotFoundError(f"Timeseries file not found: {path}")
            else:
                raise FileNotFoundError(f"Timeseries file not found: {path}")
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        data[r] = df
    return data

def filter_interval(data_dict, start, end):
    new = {}
    for r, df in data_dict.items():
        new[r] = df.loc[start:end].copy()
    return new

def plot_dispatch_stack(results, region, outdir):
    techs = results['dispatch_techs']
    # build dataframe
    ts = {}
    for tech in techs:
        key = f"dispatch_{tech}_{region}"
        if key in results['variables']:
            series = pd.Series(results['variables'][key]).sort_index()
            ts[tech] = series
    if not ts:
        return
    df = pd.DataFrame(ts)
    # force any tiny negatives to zero so stacked=True works
    df = df.clip(lower=0)
    ax = df.plot(kind='area', stacked=True)
    ax.set_xlabel('timestep')
    ax.set_ylabel('MW')
    ax.set_title(f'Dispatch stack – {region}')
    plt.tight_layout()
    fig_path = os.path.join(outdir, f"dispatch_{region}.png")
    plt.savefig(fig_path)
    plt.close()

def main():
    # Add CLI flag for curtailment
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config_master.yaml')
    parser.add_argument('--data-dir', required=True, help='Folder with regional CSV files')
    parser.add_argument('--preset', choices=list(PRESETS.keys()))
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--out', default='results.pkl', help='Pickle to store raw results')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--enable-curtailment', action='store_true', help='Enable curtailment constraints and variables in the optimizer')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s – %(message)s')

    # decide interval
    import pandas as pd 

    if args.preset:
        start, end = PRESETS[args.preset]
    else:
        if not (args.start and args.end):
            parser.error('Provide --start and --end or choose --preset')
        start, end = args.start, args.end

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # inclusive end

    # --- load config & data -------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    regions = cfg['regions']
    data_all = load_regional_timeseries(regions, args.data_dir)
    data_int = filter_interval(data_all, start_dt, end_dt)

    # --- ROLLING HORIZON LOGIC ---
    nsteps = len(next(iter(data_int.values())))
    window_size = 336  # 2 weeks of half-hours
    stride = 336       # non-overlapping windows
    indices = rolling_horizon_indices(nsteps, window_size, stride)

    # Prepare containers for stitched results
    stitched_variables = {}
    stitched_dispatch_techs = None
    stitched_regions = regions
    stitched_duals = {}
    
    # Debug info
    print(f"Total time steps: {nsteps}")
    print(f"Window size: {window_size}, Stride: {stride}")
    print(f"Number of windows: {len(indices)}")

    total_objective = 0  # Sum of all window objective values
    for win_idx, (start_idx, end_idx) in enumerate(indices):
        print(f"Solving window {win_idx+1}/{len(indices)}: steps {start_idx} to {end_idx-1}")
        # Slice data for this window
        data_win = {r: df.iloc[start_idx:end_idx] for r, df in data_int.items()}
        time_periods = list(range(start_idx, end_idx))
        # Re-index time_periods to local window
        time_periods_local = list(range(end_idx - start_idx))
        # Build and solve model for this window
        opt = RegionalFlexOptimizer(args.config, enable_curtailment=args.enable_curtailment)
        opt.build_model(data_win, time_periods=time_periods_local)
        opt.model.writeLP(f"debug_window_{win_idx+1}.lp")
        highs_solver = pulp.HiGHS_CMD(msg=True)
        status, _ = opt.solve(solver=highs_solver)
        if status != 1:
            print(f"⚠️  MILP non optimal in window {win_idx+1} (status = {status})")
            continue
        # Get results for this window
        nodal = opt.get_nodal_prices()
        duals_dict = {region: prices.to_dict() for region, prices in nodal.items()}
        results = opt.get_results(dual_variables=duals_dict)
        # Accumulate objective value
        window_obj = results.get('objective_value', 0)
        if window_obj is not None:
            total_objective += window_obj
        # Stitch variables
        if stitched_dispatch_techs is None:
            stitched_dispatch_techs = results.get('dispatch_techs', [])
        
        # Initialize variables if they don't exist
        for var in results['variables']:
            if var not in stitched_variables:
                stitched_variables[var] = {}
        
        # Map local window indices to global indices
        # IMPORTANT: Each time step should appear ONLY ONCE in the final result
        for var, vals in results['variables'].items():
            for t_local, val in vals.items():
                t_global = t_local + start_idx  
                # Only add if in current window's range (avoid duplicates)
                if start_idx <= t_global < end_idx:
                    # Check for capacity constraints in debug mode
                    if var.startswith('dispatch_') and '_' in var:
                        tech, region = var.replace('dispatch_', '').split('_', 1)
                        if tech in ['biofuel', 'thermal_gas', 'thermal_fuel']:
                            pass  # Debug placeholder if needed
                    
                    # Store the value for this time step
                    stitched_variables[var][t_global] = val
        
        # Stitch duals
        for region, dual_series in duals_dict.items():
            if region not in stitched_duals:
                stitched_duals[region] = {}
            for t_local, price in dual_series.items():
                t_global = t_local + start_idx
                # Only add if in current window's range
                if start_idx <= t_global < end_idx:
                    stitched_duals[region][t_global] = price

    # Assemble final stitched results
    results = {
        'variables': stitched_variables,
        'dispatch_techs': stitched_dispatch_techs,
        'regions': stitched_regions,
        'dual_variables': stitched_duals,
        'total_cost': total_objective  # Add the sum of all window objectives
    }

    # Calculate environmental indicators
    results['emissions'] = calculate_emissions(results, cfg)
    # Validate results before saving
    print("\nValidating results...")
    tech_capacities = cfg.get('regional_capacities', {})
    validation_issues = 0
    
    # Check if any technology exceeds capacity
    for var, values in stitched_variables.items():
        if var.startswith('dispatch_') and '_' in var:
            tech, region = var.replace('dispatch_', '').split('_', 1)
            if tech in tech_capacities.get(region, {}):
                capacity = tech_capacities[region][tech]
                max_val = max(values.values()) if values else 0
                if max_val > capacity * 1.01:  # Allow 1% tolerance for numerical issues
                    print(f"WARNING: {var} exceeds capacity: max={max_val:.2f} MW, capacity={capacity} MW")
                    validation_issues += 1
    
    if validation_issues == 0:
        print("[OK] All technology dispatch values are within capacity limits.")
    else:
        print(f"[WARNING] Found {validation_issues} capacity constraint violations.")
    
    # Add total system cost to results for compatibility with sensitivity.py
    if 'objective_value' in results and results['objective_value'] is not None:
        results['total_cost'] = results['objective_value']
    # Save results
    pd.to_pickle(results, args.out)
    print(f'Results stored to {args.out}')

    # --- Save nodal prices CSV and print expense summary (optional) ---
    df_price = pd.DataFrame(stitched_duals).sort_index().rename_axis('timestep')
    prices_output_path = os.path.join(os.path.dirname(args.out), "nodal_prices.csv")
    df_price.to_csv(prices_output_path)
    
    # Calculate statistics
    dt_h = 0.5  # half-hour intervals
    demand_df = pd.DataFrame({r: data_int[r]['demand'].to_numpy() for r in regions}, index=df_price.index)
    
    # Print summary info
    print("\nResults summary:")
    print(f"Total time periods: {len(df_price)}")
    print(f"Regions: {', '.join(regions)}")
    print("Price head:\n", df_price.head())
    print("Demand head:\n", demand_df.head())
    
    # Print dispatch summaries for key technologies
    print("\nDispatch summary (MWh):")
    for tech in ['hydro', 'nuclear', 'biofuel', 'thermal_gas', 'thermal_fuel']:
        for region in regions:
            var_key = f"dispatch_{tech}_{region}"
            if var_key in stitched_variables:
                values = list(stitched_variables[var_key].values())
                if values:
                    total = sum(values) * dt_h  # Convert MW to MWh
                    max_val = max(values)
                    print(f"  {var_key}: total={total:.2f} MWh, max={max_val:.2f} MW")
    
    # Calculate total expense
    expense = (df_price * demand_df * dt_h).sum().sum()
    print(f"\nDépense spot simulée : {expense:.2f}€")

    # Print emission summary
    if 'emissions' in results:
        print("\nÉmissions totales (tCO2) :")
        for region, val in results['emissions']['total_by_region'].items():
            print(f"  {region}: {val:.2f}")

    # --- basic plots --------------------------------------------------------
    outdir = 'plots'
    os.makedirs(outdir, exist_ok=True)
    if 'dispatch_techs' in results:
        for r in regions:
            plot_dispatch_stack(results, r, outdir)
        print('Figures saved to ./plots')
    else:
        print('No plots generated: optimization did not solve successfully and dispatch_techs is missing from results.')


if __name__ == '__main__':
    main()
