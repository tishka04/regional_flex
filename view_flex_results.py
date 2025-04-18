
#!/usr/bin/env python3
"""Visualiser les résultats du Regional Flex Optimizer.

Usage:
    python view_flex_results.py --pickle path/to/full_year.pkl --out plots --region Nouvelle_Aquitaine
    python view_flex_results.py --pickle path/to/full_year.pkl --all-regions

Génère des PNG dans le dossier --out pour :
  * Dispatch empilé
  * SOC des stockages
  * Slack positif / négatif
  * Curtailment
  * Échanges nets
"""

import argparse, os, pickle, pandas as pd
import matplotlib.pyplot as plt

PALETTE = {
    'hydro': '#1f77b4', 'nuclear': '#ff7f0e', 'thermal_gas': '#2ca02c',
    'thermal_coal': '#d62728', 'biofuel': '#9467bd',
    'slack_pos': '#7f7f7f', 'slack_neg': '#bcbd22',
    'abs_exchange': '#17becf', 'abs_transport': '#8c564b'
}
DISPATCH_TECHS = ['hydro','nuclear','thermal_gas','thermal_coal','biofuel']

def build_df(res, prefix):
    return pd.DataFrame({k[len(prefix):]: pd.Series(v)
                         for k,v in res['variables'].items() if k.startswith(prefix)})

def dt_index(length):
    return pd.date_range('2022-01-01', periods=length, freq='30min')

def plot_df(df, mask, title, ylabel, path, colors=None, area=False):
    if df.empty: return
    df = df.loc[mask]
    ax = df.plot.area(stacked=True, color=colors) if area else df.plot()
    ax.set_title(title); ax.set_ylabel(ylabel); ax.figure.tight_layout()
    ax.figure.savefig(path); plt.close(ax.figure)

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--pickle', required=True)
    pa.add_argument('--out', default='plots')
    pa.add_argument('--region')
    pa.add_argument('--all-regions', action='store_true')
    pa.add_argument('--start'); pa.add_argument('--end')
    args = pa.parse_args()

    os.makedirs(args.out, exist_ok=True)
    res = pickle.load(open(args.pickle, 'rb'))
    regions = res['regions']
    targets = regions if args.all_regions else [args.region or regions[0]]

    first_var = next(iter(res['variables'].values()))
    idx = dt_index(len(first_var))
    mask = slice(None)
    if args.start or args.end:
        start = pd.to_datetime(args.start or idx[0])
        end   = pd.to_datetime(args.end   or idx[-1])
        mask = (idx >= start) & (idx <= end)

    for region in targets:
        print(f'→ {region}')
        
        # ---------------- DISPATCH empilé ----------------
        df_disp = build_df(res, 'dispatch_')
    
        # 1. colonnes à conserver : techno dispatchables + bonne région
        cols = [c for c in df_disp.columns
                if c.endswith(f'_{region}') and c.split('_')[0] in DISPATCH_TECHS]
    
        # 2. matrice (index temporel déjà dans idx)
        dispatch = df_disp[cols].set_index(idx).fillna(0).loc[mask]
    
        # 3. supprimer les séries totalement nulles
        dispatch = dispatch.loc[:, dispatch.sum() != 0]
    
        # 4. couleurs cohérentes
        colors = [PALETTE[c.split('_')[0]] for c in dispatch.columns]
    
        # 5. traçage + sauvegarde
        plot_df(dispatch, mask, f'Dispatch – {region}', 'MW',
                os.path.join(args.out, f'dispatch_{region}.png'),
                colors=colors, area=True)

        # SOC
        soc = build_df(res, 'storage_soc_')
        cols_soc = [c for c in soc.columns if c.endswith(f'_{region}')]
        plot_df(soc[cols_soc].set_index(idx).fillna(0), mask,
                f'SOC – {region}', 'MWh', os.path.join(args.out, f'soc_{region}.png'))

        # Slack
        slack_keys = [f'slack_pos_{region}', f'slack_neg_{region}']
        slack = pd.DataFrame({k: pd.Series(res['variables'][k]) for k in slack_keys}).set_index(idx)
        plot_df(slack.fillna(0), mask, f'Slack – {region}', 'MW',
                os.path.join(args.out, f'slack_{region}.png'))

        # Curtailment
        cur_key = f'curtail_{region}'
        if cur_key in res['variables']:
            cur = pd.Series(res['variables'][cur_key]).set_axis(idx)
            plot_df(cur.to_frame('curtail'), mask, f'Curtailment – {region}',
                    'MW', os.path.join(args.out, f'curtail_{region}.png'))

        # Exchanges
        ex_cols = [k for k in res['variables']
                   if k.startswith('exchange_') and k.endswith(f'_{region}')]
        if ex_cols:
            exch = pd.DataFrame({k: pd.Series(res['variables'][k]) for k in ex_cols}).set_index(idx)
            plot_df(exch.fillna(0), mask, f'Exchanges – {region}', 'MW',
                    os.path.join(args.out, f'exchanges_{region}.png'))

    print(f'PNG enregistrés dans {args.out}')

if __name__ == '__main__':
    main()
