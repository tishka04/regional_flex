import pandas as pd

REG = ["Auvergne_Rhone_Alpes", "Nouvelle_Aquitaine", "Occitanie", "Provence_Alpes_Cote_dAzur"]
for year in (2023, 2022):
    print(f"\n==== {year} ====")
    print(f"{'region':>26} {'hydroTWh':>9} {'rorTWh':>8} {'flexBudgetTWh':>14} {'RLmean_MW':>10} {'demandTWh':>10}")
    for r in REG:
        n = pd.read_csv(f"Data/processed/{r}_{year}.csv", index_col=0)
        print(f"{r:>26} {n['hydro'].sum()*0.5/1e6:9.2f} {n['ror'].sum()*0.5/1e6:8.2f} "
              f"{n['hydro_flex'].sum()*0.5/1e6:14.2f} {n['residual_load'].mean():10.0f} "
              f"{n['demand'].sum()*0.5/1e6:10.2f}")
