
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
        path = os.path.join(data_dir, f"{r}.csv")
        if not os.path.exists(path):
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
    ax = df.plot(kind='area', stacked=True)
    ax.set_xlabel('timestep')
    ax.set_ylabel('MW')
    ax.set_title(f'Dispatch stack – {region}')
    plt.tight_layout()
    fig_path = os.path.join(outdir, f"dispatch_{region}.png")
    plt.savefig(fig_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config_master.yaml')
    parser.add_argument('--data-dir', required=True, help='Folder with regional CSV files')
    parser.add_argument('--preset', choices=list(PRESETS.keys()))
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--out', default='results.pkl', help='Pickle to store raw results')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s – %(message)s')

    # decide interval
    if args.preset:
        start, end = PRESETS[args.preset]
    else:
        if not (args.start and args.end):
            parser.error('Provide --start and --end or choose --preset')
        start, end = args.start, args.end

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) + pd.Timedelta(minutes=30)  # inclusive end

    # --- load config & data -------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    regions = cfg['regions']
    data_all = load_regional_timeseries(regions, args.data_dir)
    data_int = filter_interval(data_all, start_dt, end_dt)

    # build list of continuous integer timesteps for optimizer
    # map datetime index to integer positions
    nsteps = len(next(iter(data_int.values())))
    time_periods = list(range(nsteps))

    from src.model.optimizer_regional_flex import RegionalFlexOptimizer
    opt = RegionalFlexOptimizer(args.config)
    opt.build_model(data_int, time_periods=time_periods)
    status, tsolve = opt.solve(threads=args.threads)
    if status != 1:
        print('WARNING – model not optimal')

    results = opt.get_results()
    pd.to_pickle(results, args.out)
    print(f'Results stored to {args.out}')

    # --- basic plots --------------------------------------------------------
    outdir = 'plots'
    os.makedirs(outdir, exist_ok=True)
    for r in regions:
        plot_dispatch_stack(results, r, outdir)

    print('Figures saved to ./plots')

if __name__ == '__main__':
    main()
