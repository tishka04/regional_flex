# Regional Flex Optimizer

A lightweight dispatch model to analyse cross-region flexibility on the French power system. It computes the half-hourly flexibility dispatch that minimises the total cost. Hydro, nuclear, gas, fuel and biofuel units are handled with demand response and storage dynamics. The model also optimises inter-regional exchanges with transmission losses.

The repository contains scripts to run the MILP optimisation, visualise results, and explore them through a Jupyter or Streamlit dashboard.

## Model overview

The solver is a mixed integer linear program written with [PuLP](https://pypi.org/project/PuLP/). It models hourly dispatch decisions for each region with the following elements:

- generation technologies (nuclear, gas, fuel, biofuel, hydro)
- optional curtailment of renewables
- battery and pumped-hydro storage with state of charge dynamics
- demand response
- slack variables for unmet demand
- cross‑regional flows with transmission losses

All parameters are defined in `config/config_master.yaml` and half‑hourly input data is read from `data/processed/<REGION>.csv`.

## 1. Installation

```bash
# clone and install dependencies
git clone https://github.com/tishka04/regional_flex.git
cd regional_flex
pip install -r requirements.txt        # or use a conda environment
```

The requirements also install Plotly, Kaleido and Folium which are needed for the geographic visualisations.

The solver relies on the [PuLP](https://pypi.org/project/PuLP/) package which ships with the HiGHS backend.

## 2. Data preparation

Data for each region must be provided in `data/processed/<REGION>.csv` (or the directory you pass via `--data-dir`). Files are expected to contain half-hourly time series (`48 * 365 = 17,520` rows for one year) with at least the following columns:

- `demand` (or `consumption`, `load`)
- optional generation series such as `hydro`, `nuclear`, `thermal_gas`, `thermal_fuel`, `biofuel`, etc.

A master configuration file `config/config_master.yaml` defines available regions, capacities, costs and other constraints. Adjust it to match your dataset.

## 3. Running an optimisation

The main entry point is `run_regional_flex.py`. You can run a full year or specific periods using built-in presets or custom dates. Results are stored as a pickle file containing all decision variables.

### Examples

```bash
# full year simulation
python run_regional_flex.py --config config/config_master.yaml --data-dir data/processed --preset full_year --out results/full_year.pkl

# winter weekday (preset)
python run_regional_flex.py --config config/config_master.yaml --data-dir data/processed --preset winter_weekday --out results/winter_weekday.pkl

# custom interval
python run_regional_flex.py --config config/config_master.yaml --data-dir data/processed --start 2022-03-01 --end 2022-03-07 --out results/march.pkl
```

Passing `--enable-curtailment` allows the solver to curtail renewable generation. Without it, curtailment variables are omitted and the optimisation enforces full use of available generation.

A rolling-horizon scheme is implemented internally (two‑week windows) so even long horizons remain tractable.

## 4. Visualising results

Use the companion script `view_flex_results.py` to generate PNG plots from a result pickle:

```bash
python view_flex_results.py --pickle results/full_year.pkl --all-regions --out plots
```

It produces stacked dispatch graphs, state of charge of storages, slack values, curtailment and import/export flows for each region. You can restrict the output to a single region or a date range using `--region`, `--start` and `--end`. The script uses a colorblind‑friendly palette defined in `config/colors.yaml`. You may override these colors with `--palette-file my_colors.yaml`.


Additional options produce cumulative summaries and animations:

```bash
python view_flex_results.py --pickle results/full_year.pkl --all-regions --config config/config_master.yaml --out plots

python view_flex_results.py --pickle results/winter_weekday.pkl --all-regions --config config/config_master.yaml  --out plots_winter --summary --animate
```

`--summary` creates bar charts of total cost, emissions and load factors by region, while `--animate` generates a GIF illustrating dispatch and flows over time.

For an interactive exploration of the output you can open `interactive_flex_dashboard.ipynb` in Jupyter:

For additional charts and analysis, explore `check_results.ipynb` in Jupyter.

Finally, to generate charts for the flows, use :

```bash
python geo_flows.py --pickle results/full_year.pkl --out plots
```

## 5. Streamlit application

To explore the results in a web interface you can run the Streamlit app. It reads scenario CSV files such as `full_year.csv` or `winter_weekday.csv` provided in this repository (or generated from your own simulations):

```bash
python -m streamlit run flex_app.py
```

Select a scenario, region and date range in the sidebar to display production, storage and price plots.

## 6. Customising scenarios

Most parameters are defined in `config/config_master.yaml`:

- `regional_capacities`: generation capacities by region and technology
- `regional_storage`: power and energy ratings for batteries and pumped hydro
- `costs` and `regional_costs`: variable and fixed costs
- `emission_factors`: parameters for the environmental cost
- `uc_params`: unit commitment constraints (start‑up costs, minimum up/down time...)
- `regional_distances` and `loss_factor_per_km`: transmission distances and losses

To explore new scenarios, modify these sections or add new regions/time series in `data/processed`. Presets for common dates are defined at the top of `run_regional_flex.py`; you can add your own or simply use `--start`/`--end`.

## Repository structure

```
├── config/                     # configuration files
├── data/processed/             # input time series (CSV)
├── results/                    # output pickles and CSVs
├── plots/                      # figures generated by view_flex_results.py
├── src/                        # optimisation code
└── run_regional_flex.py        # command-line runner
```

## License

MIT License © 2025 Théotime Coudray

