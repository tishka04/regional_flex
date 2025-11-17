"""
Generate Weekly Generation Stack with Renewables (Solar & Wind)
================================================================
This script creates a new visualization (fig2b) that extends the weekly 
generation stack to include renewable energy sources (solar and wind) 
from the multi_region_data.csv file.

Usage:
------
python generate_renewable_stack.py

The script will:
1. Load the optimization results from full_year.csv
2. Load renewable data from Data/multi_region_data.csv
3. Merge both datasets
4. Create a weekly stacked area chart showing all generation sources
5. Save the figure as fig2b_weekly_stack_renewables.png and .pdf in figs/

Author: Generated for RegionalFlex project
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
TIME_COL = "timestamp"
TECH_PREFIXES = [
    ("hydro", "dispatch_hydro_", 23),
    ("nuclear", "dispatch_nuclear_", 30),
    ("thermal_gas", "dispatch_thermal_gas_", 75),
    ("thermal_fuel", "dispatch_thermal_fuel_", 85),
    ("biofuel", "dispatch_biofuel_", 45),
]

# File paths
BASE_DIR = Path(__file__).parent
RESULTS_FILE = BASE_DIR / "full_year.csv"
PROCESSED_DATA_DIR = BASE_DIR / "Data" / "processed"
OUTPUT_DIR = BASE_DIR / "figs"

# Regional files
REGIONAL_FILES = [
    "Auvergne_Rhone_Alpes.csv",
    "Nouvelle_Aquitaine.csv",
    "Occitanie.csv",
    "Provence_Alpes_Cote_dAzur.csv"
]

def _to_datetime(df: pd.DataFrame, col: str = TIME_COL) -> pd.DataFrame:
    """Convert timestamp column to datetime."""
    df_copy = df.copy()
    df_copy[col] = pd.to_datetime(df_copy[col])
    return df_copy

def _resample(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Down-sample time series to *freq* (mean)."""
    return df.set_index(TIME_COL).resample(freq).mean().reset_index()

def generate_renewable_stack():
    """Main function to generate the renewable generation stack visualization."""
    
    print("=" * 70)
    print("Weekly Generation Stack with Renewables (Solar & Wind)")
    print("=" * 70)
    
    # Check if files exist
    if not RESULTS_FILE.exists():
        print(f"[ERROR] Results file not found at {RESULTS_FILE}")
        print("   Please ensure full_year.csv exists in the regional_flex directory.")
        return
    
    if not PROCESSED_DATA_DIR.exists():
        print(f"[ERROR] Processed data directory not found at {PROCESSED_DATA_DIR}")
        print("   Please ensure Data/processed folder exists.")
        return
    
    print(f"\n[LOADING] Data files...")
    print(f"   Results: {RESULTS_FILE}")
    
    # Load results data
    try:
        df_results = pd.read_csv(RESULTS_FILE)
        df_results = _to_datetime(df_results)
        print(f"   [OK] Loaded {len(df_results)} rows from results")
    except Exception as e:
        print(f"[ERROR] Error loading results: {e}")
        return
    
    # Load renewable data from regional files
    print(f"\n[LOADING] Regional renewable data from {len(REGIONAL_FILES)} files...")
    renewable_dfs = []
    
    for region_file in REGIONAL_FILES:
        file_path = PROCESSED_DATA_DIR / region_file
        if not file_path.exists():
            print(f"   [WARNING] File not found: {region_file}")
            continue
        
        try:
            df_region = pd.read_csv(file_path, index_col=0)
            df_region.index = pd.to_datetime(df_region.index)
            df_region = df_region.reset_index()
            df_region.rename(columns={'index': TIME_COL}, inplace=True)
            
            # Extract region name
            region_name = region_file.replace('.csv', '').replace('_', ' ')
            
            # Keep only timestamp, solar, and wind columns, rename them
            df_region_renewable = df_region[[TIME_COL, 'solar', 'wind']].copy()
            df_region_renewable.rename(columns={
                'solar': f'{region_name}_solar_MW',
                'wind': f'{region_name}_wind_MW'
            }, inplace=True)
            
            renewable_dfs.append(df_region_renewable)
            print(f"   [OK] {region_file}: {len(df_region)} rows")
            
        except Exception as e:
            print(f"   [ERROR] Failed to load {region_file}: {e}")
            continue
    
    if not renewable_dfs:
        print(f"[ERROR] No regional data could be loaded")
        return
    
    # Merge all regional renewable data
    print(f"\n[MERGE] Combining regional renewable data...")
    df_renewable = renewable_dfs[0]
    for df in renewable_dfs[1:]:
        df_renewable = pd.merge(df_renewable, df, on=TIME_COL, how='outer')
    
    print(f"   [OK] Combined renewable data: {len(df_renewable)} rows")
    
    # Merge with results data
    print(f"\n[MERGE] Merging with dispatch results...")
    df_merged = pd.merge(df_results, df_renewable, on=TIME_COL, how='inner')
    print(f"   [OK] Merged dataset has {len(df_merged)} rows")
    
    # Resample to weekly
    print(f"\n[RESAMPLE] Resampling to weekly averages...")
    df_week = _resample(df_merged, "W")
    print(f"   [OK] Resampled to {len(df_week)} weeks")
    
    # Calculate total solar and wind across all regions
    solar_cols = [c for c in df_week.columns if '_solar_MW' in c]
    wind_cols = [c for c in df_week.columns if '_wind_MW' in c]
    
    print(f"\n[SOLAR] Solar columns found: {len(solar_cols)}")
    print(f"[WIND] Wind columns found: {len(wind_cols)}")
    
    total_solar = df_week[solar_cols].sum(axis=1) if solar_cols else pd.Series(0, index=df_week.index)
    total_wind = df_week[wind_cols].sum(axis=1) if wind_cols else pd.Series(0, index=df_week.index)
    
    print(f"   Solar avg: {total_solar.mean():.1f} MW")
    print(f"   Wind avg: {total_wind.mean():.1f} MW")
    
    # Create figure (same size as original fig2_weekly_stack)
    print(f"\n[PLOT] Creating visualization...")
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(df_week))
    
    # Plot solar in gold (lighter than goldenrod, darker than yellow)
    if total_solar.sum() > 0:
        ax.fill_between(df_week[TIME_COL], bottom, bottom + total_solar,
                        label='Solar', step="mid", color='gold')
        bottom += total_solar.values
    
    # Plot wind in light blue
    if total_wind.sum() > 0:
        ax.fill_between(df_week[TIME_COL], bottom, bottom + total_wind,
                        label='Wind', step="mid", color='lightblue')
        bottom += total_wind.values
    
    # Plot conventional generation sources with explicit colors
    # Original: hydro(blue), nuclear(orange), thermal_gas(green), thermal_fuel(red), biofuel(purple)
    # Swapped: thermal_gas -> purple, biofuel -> green
    color_map = {
        'hydro': 'C0',          # Blue
        'nuclear': 'C1',        # Orange
        'thermal_gas': 'C4',    # Purple (swapped with biofuel)
        'thermal_fuel': 'C3',   # Red
        'biofuel': 'C2'         # Green (swapped with thermal_gas)
    }
    
    # Track which technologies to add to legend only (not plotted due to low contribution)
    from matplotlib.patches import Patch
    legend_elements = []
    
    # Threshold for plotting: technologies with average < 50 MW are legend-only
    plot_threshold_mw = 50.0
    
    for tech_label, prefix, _ in TECH_PREFIXES:
        cols = [c for c in df_week.columns if c.startswith(prefix)]
        series = df_week[cols].sum(axis=1)
        label_name = tech_label.replace("_", " ").capitalize()
        color = color_map.get(tech_label, 'C5')
        
        if series.sum() > 0:
            avg_mw = series.mean()
            if avg_mw < plot_threshold_mw:
                # Too small to plot - add to legend only
                legend_elements.append(Patch(facecolor=color, label=label_name))
                print(f"   {tech_label}: {avg_mw:.1f} MW (avg) [legend only, < {plot_threshold_mw} MW]")
            else:
                # Plot normally
                ax.fill_between(df_week[TIME_COL], bottom, bottom + series,
                                label=label_name, 
                                step="mid", color=color)
                bottom += series.values
                print(f"   {tech_label}: {avg_mw:.1f} MW (avg)")
        else:
            # Add to legend even with zero production
            legend_elements.append(Patch(facecolor=color, label=label_name))
    
    # Format the plot (same style as original)
    ax.set_ylabel("Weekly mean dispatch [MW]")
    # No title
    
    # Combine existing legend with zero-production items
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(legend_elements)
    ax.legend(handles=handles, loc="upper right", ncol=4)
    
    fig.tight_layout()
    
    # Save figure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "fig2b_weekly_stack_renewables.png"
    pdf_path = OUTPUT_DIR / "fig2b_weekly_stack_renewables.pdf"
    
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    
    print(f"\n[SUCCESS] Figure saved successfully!")
    print(f"   PNG: {png_path}")
    print(f"   PDF: {pdf_path}")
    
    # Calculate and display energy statistics
    print(f"\n[STATS] Energy Statistics:")
    print(f"   {'Source':<20} {'Weekly Avg (MW)':<20} {'Share (%)'}")
    print(f"   {'-'*60}")
    
    total_gen = bottom[-1] if len(bottom) > 0 else 1
    if total_solar.sum() > 0:
        share = 100 * total_solar.mean() / (total_gen if total_gen > 0 else 1)
        print(f"   {'Solar':<20} {total_solar.mean():>15.1f} {share:>15.1f}")
    if total_wind.sum() > 0:
        share = 100 * total_wind.mean() / (total_gen if total_gen > 0 else 1)
        print(f"   {'Wind':<20} {total_wind.mean():>15.1f} {share:>15.1f}")
    
    for tech_label, prefix, _ in TECH_PREFIXES:
        cols = [c for c in df_week.columns if c.startswith(prefix)]
        series = df_week[cols].sum(axis=1)
        if series.sum() > 0:
            share = 100 * series.mean() / (total_gen if total_gen > 0 else 1)
            print(f"   {tech_label.capitalize():<20} {series.mean():>15.1f} {share:>15.1f}")
    
    print(f"\n{'='*70}")
    print("[DONE] Visualization complete!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    generate_renewable_stack()
