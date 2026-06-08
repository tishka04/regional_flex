# Regional Flex Optimizer

A mixed-integer linear programming (MILP) dispatch model for analysing cross-region flexibility in the French power system. The model computes half-hourly optimal dispatch that minimizes total system cost, accounting for generation technologies (hydro, nuclear, gas, fuel, biofuel), demand response, battery and pumped-hydro storage, and inter-regional power flows with transmission losses.

## Table of Contents

1. [Installation](#1-installation)
2. [Data Structure](#2-data-structure)
3. [Running the Baseline 2023 Model](#3-running-the-baseline-2023-model)
4. [Sensitivity Studies](#4-sensitivity-studies)
5. [Generating Plots and Results](#5-generating-plots-and-results)
6. [Advanced Analysis](#6-advanced-analysis)
7. [Configuration](#7-configuration)
8. [Repository Structure](#8-repository-structure)

---

## 1. Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/tishka04/regional_flex.git
cd regional_flex

# Create and activate a virtual environment (recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

The model relies on:
- **PuLP**: MILP solver interface (ships with HiGHS backend)
- **pandas, numpy**: Data manipulation
- **matplotlib**: Plotting
- **pyyaml**: Configuration management
- **plotly, kaleido, folium**: Geographic visualisations (optional)

---

## 2. Data Structure

### Input Data Location

Half-hourly time series data must be placed in `Data/processed/` with the following naming convention:
```
Data/processed/<REGION>_<YEAR>.csv
```

### Available Datasets

The repository includes 2022 and 2023 data for four French regions:
- `Auvergne_Rhone_Alpes_2023.csv`
- `Nouvelle_Aquitaine_2023.csv`
- `Occitanie_2023.csv`
- `Provence_Alpes_Cote_dAzur_2023.csv`

### Required CSV Columns

Each file must contain half-hourly data (17,520 rows for a non-leap year) with:

| Column | Description | Unit |
|--------|-------------|------|
| `timestamp` | Datetime index (YYYY-MM-DD HH:MM:SS) | - |
| `demand` | Electricity demand | MW |
| `wind` | Wind generation (optional) | MW |
| `solar` | Solar generation (optional) | MW |
| `ror` | Run-of-river hydro (optional) | MW |
| `hydro` | Reservoir hydro availability (optional) | MW |

### Configuration File

Model parameters are defined in `config/config_master.yaml`:

| Section | Description |
|---------|-------------|
| `regions` | List of modelled regions |
| `regional_capacities` | Generation capacities by region and technology (MW) |
| `regional_storage` | Battery and pumped-hydro capacities (MW/MWh) |
| `costs` | Variable costs by technology (€/MWh) |
| `emission_factors` | CO₂ emission factors (tCO₂/MWh) |
| `demand_response` | DR parameters per region (max shift %, participation rate) |
| `constraints` | Transmission capacities and operational limits |
| `uc_params` | Unit commitment parameters (startup costs, min up/down times) |

---

## 3. Running the Baseline 2023 Model

### Step 3.1: Full-Year Baseline Run

The baseline 2023 scenario uses realistic hydro flexibility (`recovery_beta = 0.5`) with all flexibility options enabled.

```bash
python run_regional_flex.py \
    --config config/config_master.yaml \
    --data-dir Data/processed \
    --data-suffix _2023 \
    --preset full_year \
    --out results/baseline_2023.pkl \
    --config-year 2023
```

**Parameters explained:**
- `--data-suffix _2023`: Appends `_2023` to region names when loading CSV files
- `--preset full_year`: Runs the complete year (January 1 - December 31)
- `--config-year 2023`: Uses year-specific configuration resolution

### Step 3.2: Representative Day Runs

Run specific representative days for detailed analysis:

```bash
# Winter weekday (highest demand Mon-Fri in Dec/Jan/Feb)
python run_regional_flex.py \
    --config config/config_master.yaml \
    --data-dir Data/processed \
    --data-suffix _2023 \
    --preset winter_weekday \
    --out results/winter_weekday_2023.pkl

# Summer holiday (15 August - Assumption Day)
python run_regional_flex.py \
    --config config/config_master.yaml \
    --data-dir Data/processed \
    --data-suffix _2023 \
    --preset summer_holiday \
    --out results/summer_holiday_2023.pkl

# Spring weekday
python run_regional_flex.py \
    --config config/config_master.yaml \
    --data-dir Data/processed \
    --data-suffix _2023 \
    --preset spring_weekday \
    --out results/spring_weekday_2023.pkl

# Autumn weekend
python run_regional_flex.py \
    --config config/config_master.yaml \
    --data-dir Data/processed \
    --data-suffix _2023 \
    --preset autumn_weekend \
    --out results/autumn_weekend_2023.pkl
```

### Step 3.3: Custom Date Range

For a specific period:

```bash
python run_regional_flex.py \
    --config config/config_master.yaml \
    --data-dir Data/processed \
    --data-suffix _2023 \
    --start 2023-07-01 \
    --end 2023-07-31 \
    --out results/july_2023.pkl
```

### Step 3.4: Export Results to CSV

Convert pickle results to CSV for analysis:

```bash
python -c "
import pickle
import pandas as pd

with open('results/baseline_2023.pkl', 'rb') as f:
    res = pickle.load(f)

# Extract variables to DataFrame
variables = res['variables']
n = max(len(pd.Series(v)) for v in variables.values())
idx = pd.RangeIndex(0, n)
ts = pd.date_range('2023-01-01', periods=n, freq='30min')

df = pd.DataFrame({'timestamp': ts})
for key, series in variables.items():
    df[key] = pd.Series(series, dtype=float).reindex(idx).to_numpy()

df.to_csv('full_year_2023.csv', index=False)
print(f'Exported {df.shape[0]} rows x {df.shape[1]} columns')
"
```

---

## 4. Sensitivity Studies

### 4.1 Automated Robustness Analysis (Recommended)

Run all sensitivity scenarios with a single command:

```bash
python robustness_analysis.py \
    --data-dir Data/processed \
    --threads 1 \
    --solver highs
```

This executes the following scenarios:

| Scenario | Description | Purpose |
|----------|-------------|---------|
| `baseline_2023_beta05_realistic` | Full hydro flexibility (β=0.5) | Main baseline |
| `hydro_simplified_2023_beta05` | Simplified hydro (no within-day flexibility) | Hydro sensitivity |
| `dr_beta00_2023` | DR with no energy recovery (β=0) | DR shedding scenario |
| `dr_beta05_2023` | DR with partial recovery (β=0.5) | DR balanced scenario |
| `dr_beta10_2023` | DR with full recovery (β=1.0) | DR shift scenario |
| `interannual_2022_beta05` | 2022 data comparison | Inter-annual robustness |
| `interannual_2023_beta05` | 2023 baseline | Inter-annual robustness |

**Outputs generated:**
- `results/robustness/*.pkl` - Result pickles for each scenario
- `results/robustness/robustness_summary_all.csv` - Combined metrics
- `results/robustness/robustness_hydro_sensitivity.csv` - Hydro comparison
- `results/robustness/robustness_dr_sensitivity.csv` - DR comparison
- `results/robustness/robustness_interannual.csv` - Year comparison
- `results/robustness/robustness_summary.md` - Markdown summary table

### 4.2 DR and Battery Sensitivity (Grid Search)

Run a grid search over demand response and battery storage parameters:

```bash
# Set environment variable for 2023 data
set REGIONAL_DATA_SUFFIX=_2023  # Windows
export REGIONAL_DATA_SUFFIX=_2023  # Linux/Mac

python sensitivity.py
```

This tests combinations of:
- **Max DR shift**: 0%, 2%, 5%, 10% of instantaneous demand
- **Battery multiplier**: 0×, 0.5×, 1×, 1.5×, 2× baseline capacity

**Outputs:**
- `results/sensitivity_winter_weekday_shift_*_bmult_*.pkl` - Individual results
- `plots/cost_vs_dr_shift_batt_*.png` - Cost sensitivity curves
- `plots/dr_utilisation_vs_shift_mwh_batt_*.png` - DR utilization (MWh)
- `plots/dr_utilisation_percent_vs_shift_batt_*.png` - DR share of demand

### 4.3 Manual Sensitivity Run

To manually run a custom sensitivity scenario:

```bash
# Example: Test doubled battery capacity
python run_regional_flex.py \
    --config config/config_master.yaml \
    --data-dir Data/processed \
    --data-suffix _2023 \
    --preset winter_weekday \
    --out results/sensitivity_double_battery.pkl
```

Modify the config temporarily by creating `temp_config_double_battery.yaml`:

```yaml
# Copy from config_master.yaml and modify:
regional_storage:
  Auvergne_Rhone_Alpes:
    batteries_puissance_MW: 200.0      # doubled from 100
    batteries_stockage_MWh: 400.0      # doubled from 200
    STEP_puissance_MW: 3610.0
    STEP_stockage_MWh: 61100.0
  # ... (modify other regions similarly)
```

Then run with the modified config.

---

## 5. Generating Plots and Results

### 5.1 Standard Visualization (Per-Region Dispatch)

Generate dispatch plots for all regions:

```bash
python view_flex_results.py \
    --pickle results/baseline_2023.pkl \
    --all-regions \
    --config config/config_master.yaml \
    --out plots/baseline_2023
```

**Generated plots per region:**
- `dispatch_<region>.png` - Stacked generation dispatch
- `storage_soc_<region>.png` - State of charge (batteries and pumped hydro)
- `slack_<region>.png` - Slack variables (unmet demand)
- `curtailment_<region>.png` - Renewable curtailment (if enabled)
- `flows_<region>.png` - Import/export flows

### 5.2 Summary Statistics and Emissions

Add summary charts with emissions analysis:

```bash
python view_flex_results.py \
    --pickle results/baseline_2023.pkl \
    --all-regions \
    --config config/config_master.yaml \
    --out plots/baseline_2023_summary \
    --summary
```

**Additional outputs:**
- `total_cost_by_region.png` - Bar chart of costs
- `emissions_by_region.png` - CO₂ emissions by technology
- `load_factors.png` - Generator capacity factors

### 5.3 Animated Dispatch Visualization

Generate a GIF showing dispatch evolution:

```bash
python view_flex_results.py \
    --pickle results/winter_weekday_2023.pkl \
    --all-regions \
    --config config/config_master.yaml \
    --out plots/animation_winter \
    --animate
```

Output: `animation_dispatch.gif` showing half-hourly dispatch evolution.

### 5.4 Paper Figures (2023 Baseline)

Generate all paper-quality figures from the 2023 baseline:

```bash
# Step 1: Ensure you have the baseline pickle
# results/robustness/baseline_2023_beta05_realistic.pkl

# Step 2: Generate CSV and figures
python regenerate_figures_2023.py
```

**Outputs in `figs/`:**
- `fig1_energy_mix.png/pdf` - Annual energy mix by region
- `fig2_weekly_stack.png/pdf` - Weekly dispatch stacks
- `fig2b_weekly_stack_renewables.png/pdf` - Weekly stacks with VRE
- `fig3_dr_vs_storage.png/pdf` - Flexibility sources comparison
- `fig4_price_duration.png/pdf` - Price duration curves
- `fig5_flow_matrix.png/pdf` - Inter-regional flow matrix
- `fig6_net_export.png/pdf` - Net export duration curves
- `fig7_self_sufficiency.png/pdf` - Self-sufficiency analysis
- `fig8_cost_breakdown.png/pdf` - Cost breakdown by category

### 5.5 Representative Days Plot

Generate the two-panel representative days figure:

```bash
python make_representative_days_2023.py
```

Outputs: `figs/fig_representative_days_2023.png/pdf`

### 5.6 Geographic Flow Visualization

Create geographic flow maps:

```bash
python geo_flows.py \
    --pickle results/baseline_2023.pkl \
    --out plots/geo_flows
```

### 5.7 Congestion Rent Analysis

Calculate and visualize congestion rents:

```bash
# Compute congestion rents
python calculate_congestion_rents.py --pickle results/baseline_2023.pkl

# Visualize
python visualize_congestion_rents.py
```

Outputs:
- `congestion_rents_detailed.csv` - Flow-by-flow rents
- `congestion_rents_summary.csv` - Regional aggregates
- `congestion_rents_net.csv` - Net rents by region pair
- `congestion_rents_analysis.png` - Visualization

### 5.8 Flexibility Shares Analysis

Compute flexibility contribution by source:

```bash
python analyze_flexibility.py
```

Output: `flexibility_shares_summary.csv` with categories:
- Interregional exchanges
- Demand response + storage
- Dispatchable generation

---

## 6. Advanced Analysis

### 6.1 Jupyter Notebook Analysis

Launch interactive analysis notebooks:

```bash
jupyter notebook check_results.ipynb
jupyter notebook interactive_flex_dashboard.ipynb
jupyter notebook analyze_results.ipynb
```

### 6.2 Streamlit Dashboard

Run the interactive web dashboard:

```bash
python -m streamlit run flex_app.py
```

Select scenarios, regions, and date ranges in the sidebar to explore results.

### 6.3 Compute IRRE/EFS Metrics

Calculate Integrated Renewable Resource Efficiency and Energy Flexibility Sufficiency metrics:

```bash
python compute_irre_efs.py
```

---

## 7. Configuration

### Key Parameters in `config_master.yaml`

**Demand Response Settings:**
```yaml
demand_response:
  Auvergne_Rhone_Alpes:
    max_shift: 5              # % of instantaneous demand
    max_total: 10000.0        # MWh over horizon
    participation_rate: 1.0   # 0-1 fraction
    recovery_beta: 0.5        # 0=shedding, 1=full recovery
    recovery_horizon_hours: 24.0
```

**Storage Capacities:**
```yaml
regional_storage:
  Auvergne_Rhone_Alpes:
    batteries_puissance_MW: 100.0    # Battery power rating
    batteries_stockage_MWh: 200.0  # Battery energy capacity
    STEP_puissance_MW: 3610.0      # Pumped hydro power
    STEP_stockage_MWh: 61100.0     # Pumped hydro energy
```

**Transmission Capacities:**
```yaml
constraints:
  regional_transport_capacities:
    Auvergne_Rhone_Alpes:
      Nouvelle_Aquitaine: 500
      Occitanie: 2750
      Provence_Alpes_Cote_dAzur: 5000
```

### Solver Options

Control solver behavior with additional flags:

```bash
python run_regional_flex.py \
    --config config/config_master.yaml \
    --data-dir Data/processed \
    --preset full_year \
    --out results/test.pkl \
    --solver highs \
    --threads 4 \
    --time-limit 3600
```

| Option | Description |
|--------|-------------|
| `--solver highs` | Use HiGHS solver (default) |
| `--solver cbc` | Use CBC solver |
| `--threads N` | Parallel threads (default: 1) |
| `--time-limit SECONDS` | Maximum solve time per window |

---

## 8. Repository Structure

```
regional_flex/
├── config/
│   ├── config_master.yaml          # Main configuration
│   └── colors.yaml                 # Plot color palette
├── Data/
│   ├── processed/                  # Input time series
│   │   ├── Auvergne_Rhone_Alpes_2023.csv
│   │   ├── Nouvelle_Aquitaine_2023.csv
│   │   ├── Occitanie_2023.csv
│   │   └── Provence_Alpes_Cote_dAzur_2023.csv
│   └── Raw/                        # Raw data sources
├── src/
│   └── model/
│       └── optimizer_regional_flex.py  # Core MILP model
├── results/                        # Output pickles
│   ├── baseline_2023.pkl
│   └── robustness/                 # Sensitivity results
├── plots/                          # Generated figures
├── figs/                           # Paper-quality figures
├── run_regional_flex.py            # Main entry point
├── robustness_analysis.py          # Automated sensitivity suite
├── sensitivity.py                  # DR/battery grid search
├── view_flex_results.py            # Result visualization
├── regenerate_figures_2023.py      # Paper figure generation
├── analyze_flexibility.py           # Flexibility share analysis
├── calculate_congestion_rents.py   # Congestion rent computation
├── visualize_congestion_rents.py    # Congestion rent plots
├── compute_irre_efs.py             # IRRE/EFS metrics
├── flex_app.py                     # Streamlit dashboard
├── check_results.ipynb             # Analysis notebook
└── requirements.txt                # Python dependencies
```

---

## Quick Reference: Complete Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run baseline 2023
python run_regional_flex.py --config config/config_master.yaml --data-dir Data/processed --data-suffix _2023 --preset full_year --out results/baseline_2023.pkl

# 3. Run sensitivity analysis
python robustness_analysis.py --data-dir Data/processed

# 4. Generate all paper figures
python regenerate_figures_2023.py
python make_representative_days_2023.py

# 5. Generate congestion rents
python calculate_congestion_rents.py --pickle results/robustness/baseline_2023_beta05_realistic.pkl
python visualize_congestion_rents.py

# 6. View standard plots
python view_flex_results.py --pickle results/baseline_2023.pkl --all-regions --config config/config_master.yaml --out plots/baseline_2023 --summary

# 7. Analyze flexibility shares
python analyze_flexibility.py  # Uses full_year_2023.csv
```

---

## License

MIT License © 2025 Théotime Coudray

