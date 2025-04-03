"""
Advanced visualization functions for regional flexibility analysis.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from datetime import datetime, timedelta
import os
import calendar

def plot_yearly_overview(results: Dict, regions: List[str], save_dir: str = None):
    """
    Generate yearly overview visualization showing trends across all regions.
    
    Args:
        results: Dictionary of optimization results
        regions: List of region names
        save_dir: Optional directory to save plots
    """
    if 'regional_results' not in results:
        raise KeyError(f"Expected 'regional_results' in results dictionary. Found keys: {list(results.keys())}")
    
    # Create mapping for region names
    region_keys = {}
    for region in regions:
        for key in results['regional_results'].keys():
            normalized_key = key.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
            normalized_region = region.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
            if normalized_key == normalized_region:
                region_keys[region] = key
                break
    
    # Create synthetic dates for a full year (assuming 24 time periods = 1 day)
    start_date = datetime(2024, 1, 1)
    if any(results['regional_results']):
        first_region = next(iter(results['regional_results'].values()))
        time_periods = len(first_region.get('dispatch', []))
        days = time_periods // 24  # Assuming 24 time periods per day
        if days < 1:
            days = 1  # Ensure at least one day
    else:
        days = 365  # Default to full year if no data
    
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Aggregate daily dispatch data for each region
    daily_dispatch = {}
    daily_storage = {}
    
    for region in regions:
        region_key = region_keys.get(region, region)
        if region_key not in results['regional_results']:
            continue
            
        region_data = results['regional_results'][region_key]
        
        # Get dispatch data and aggregate to daily values
        if 'dispatch' in region_data:
            dispatch_data = region_data['dispatch']
            # Reshape into days if we have hourly data
            if len(dispatch_data) >= 24:
                daily_values = []
                for i in range(0, len(dispatch_data), 24):
                    day_slice = dispatch_data[i:i+24]
                    daily_values.append(sum(day_slice))
                daily_dispatch[region] = daily_values
            else:
                daily_dispatch[region] = dispatch_data
        
        # Get storage data
        if 'storage_charge' in region_data and 'storage_discharge' in region_data:
            storage_charge = region_data['storage_charge']
            storage_discharge = region_data['storage_discharge']
            
            # Calculate net storage operations
            if len(storage_charge) == len(storage_discharge):
                net_storage = []
                for i in range(len(storage_charge)):
                    net_storage.append(storage_charge[i] - storage_discharge[i])
                
                # Reshape into days if we have hourly data
                if len(net_storage) >= 24:
                    daily_values = []
                    for i in range(0, len(net_storage), 24):
                        day_slice = net_storage[i:i+24]
                        daily_values.append(sum(day_slice))
                    daily_storage[region] = daily_values
                else:
                    daily_storage[region] = net_storage
    
    # Plot dispatch by region over time - Yearly view
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for region, data in daily_dispatch.items():
        # Ensure data length matches dates
        plot_data = data[:len(dates)] if len(data) > len(dates) else data + [0] * (len(dates) - len(data))
        ax.plot(dates, plot_data, label=region)
    
    # Format the axis
    ax.set_title('Yearly Energy Dispatch by Region', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Daily Dispatch (MWh)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format the date x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    plt.tight_layout()
    plt.legend()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'yearly_dispatch_overview.png'))
    
    plt.close()
    
    # Plot storage by region over time - Yearly view
    if daily_storage:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for region, data in daily_storage.items():
            # Ensure data length matches dates
            plot_data = data[:len(dates)] if len(data) > len(dates) else data + [0] * (len(dates) - len(data))
            ax.plot(dates, plot_data, label=region)
        
        # Format the axis
        ax.set_title('Yearly Net Storage Operations by Region', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Daily Net Storage (MWh, positive = charging)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format the date x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        plt.tight_layout()
        plt.legend()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'yearly_storage_overview.png'))
        
        plt.close()

def plot_regional_comparison(results: Dict, regions: List[str], save_dir: str = None):
    """
    Generate visualizations comparing metrics across regions.
    
    Args:
        results: Dictionary of optimization results
        regions: List of region names
        save_dir: Optional directory to save plots
    """
    if 'regional_results' not in results:
        raise KeyError(f"Expected 'regional_results' in results dictionary. Found keys: {list(results.keys())}")
    
    # Create mapping for region names
    region_keys = {}
    for region in regions:
        for key in results['regional_results'].keys():
            normalized_key = key.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
            normalized_region = region.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
            if normalized_key == normalized_region:
                region_keys[region] = key
                break
    
    # Collect metrics for comparison
    total_dispatch = []
    total_storage_charge = []
    total_storage_discharge = []
    max_dispatch = []
    region_names = []
    
    for region in regions:
        region_key = region_keys.get(region, region)
        if region_key not in results['regional_results']:
            continue
            
        region_data = results['regional_results'][region_key]
        
        if 'dispatch' in region_data:
            region_names.append(region)
            total_dispatch.append(sum(region_data['dispatch']))
            max_dispatch.append(max(region_data['dispatch']))
            
            if 'storage_charge' in region_data:
                total_storage_charge.append(sum(region_data['storage_charge']))
            else:
                total_storage_charge.append(0)
                
            if 'storage_discharge' in region_data:
                total_storage_discharge.append(sum(region_data['storage_discharge']))
            else:
                total_storage_discharge.append(0)
    
    if not region_names:
        print("No valid data found for regional comparison")
        return
    
    # Set up bar chart colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(region_names)))
    
    # Plot total dispatch comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(region_names, total_dispatch, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height):,}',
                ha='center', va='bottom', rotation=0)
    
    ax.set_title('Total Energy Dispatch by Region', fontsize=16)
    ax.set_ylabel('Total Dispatch (MWh)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'regional_dispatch_comparison.png'))
    
    plt.close()
    
    # Plot total storage operations comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(region_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, total_storage_charge, width, label='Storage Charge', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, total_storage_discharge, width, label='Storage Discharge', color='red', alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height):,}',
                    ha='center', va='bottom', rotation=0)
    
    ax.set_title('Storage Operations by Region', fontsize=16)
    ax.set_ylabel('Energy (MWh)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(region_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'regional_storage_comparison.png'))
    
    plt.close()

def plot_energy_balance(results: Dict, regions: List[str], save_dir: str = None):
    """
    Generate energy balance visualizations showing how dispatch, storage, 
    and demand response contribute to meeting demand.
    
    Args:
        results: Dictionary of optimization results
        regions: List of region names
        save_dir: Optional directory to save plots
    """
    if 'regional_results' not in results:
        raise KeyError(f"Expected 'regional_results' in results dictionary. Found keys: {list(results.keys())}")
    
    # Create mapping for region names
    region_keys = {}
    for region in regions:
        for key in results['regional_results'].keys():
            normalized_key = key.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
            normalized_region = region.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
            if normalized_key == normalized_region:
                region_keys[region] = key
                break
    
    for region in regions:
        region_key = region_keys.get(region, region)
        if region_key not in results['regional_results']:
            continue
            
        region_data = results['regional_results'][region_key]
        
        if 'dispatch' not in region_data:
            continue
            
        # Get the data for the region
        dispatch = np.array(region_data['dispatch'])
        storage_charge = np.array(region_data.get('storage_charge', np.zeros_like(dispatch)))
        storage_discharge = np.array(region_data.get('storage_discharge', np.zeros_like(dispatch)))
        
        # Create time periods
        time_periods = range(len(dispatch))
        
        # Calculate total energy flows
        net_storage = storage_discharge - storage_charge
        
        # Plot stacked area chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the base dispatch
        ax.plot(time_periods, dispatch, label='Base Dispatch', color='blue', linewidth=2)
        
        # Plot the storage discharge as positive contribution
        ax.fill_between(time_periods, dispatch, dispatch + storage_discharge, 
                        label='Storage Discharge', color='green', alpha=0.5)
        
        # Plot the storage charge as negative contribution
        ax.fill_between(time_periods, dispatch, dispatch - storage_charge, 
                        label='Storage Charge', color='red', alpha=0.5)
        
        ax.set_title(f'Energy Balance for {region}', fontsize=16)
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Energy (MWh)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'energy_balance_{region.lower().replace(" ", "_")}.png'))
        
        plt.close()

def generate_comprehensive_report(results: Dict, regions: List[str], config: Dict, save_dir: str = None):
    """
    Generate a comprehensive analysis report of the optimization results.
    
    Args:
        results: Dictionary of optimization results
        regions: List of region names
        config: Configuration dictionary
        save_dir: Optional directory to save the report
        
    Returns:
        DataFrame with analysis metrics and saves HTML report if save_dir provided
    """
    from datetime import datetime
    
    if 'regional_results' not in results:
        raise KeyError(f"Expected 'regional_results' in results dictionary. Found keys: {list(results.keys())}")
    
    # Get analysis dataframe
    from utils.visualization import analyze_results
    analysis_df = analyze_results(results, regions)
    
    # Create a more detailed table with percentages and comparisons
    if len(analysis_df) > 0:
        # Calculate regional percentages
        total_dispatch = analysis_df['Total Dispatch (MWh)'].sum()
        total_storage_charge = analysis_df['Total Storage Charge (MWh)'].sum()
        total_storage_discharge = analysis_df['Total Storage Discharge (MWh)'].sum()
        
        analysis_df['Dispatch Share (%)'] = (analysis_df['Total Dispatch (MWh)'] / total_dispatch * 100).round(2)
        analysis_df['Storage Charge Share (%)'] = (analysis_df['Total Storage Charge (MWh)'] / total_storage_charge * 100).round(2) if total_storage_charge > 0 else 0
        analysis_df['Storage Discharge Share (%)'] = (analysis_df['Total Storage Discharge (MWh)'] / total_storage_discharge * 100).round(2) if total_storage_discharge > 0 else 0
        
        # Calculate storage efficiency
        analysis_df['Storage Efficiency (%)'] = (analysis_df['Total Storage Discharge (MWh)'] / analysis_df['Total Storage Charge (MWh)'] * 100).round(2)
        analysis_df.loc[analysis_df['Total Storage Charge (MWh)'] == 0, 'Storage Efficiency (%)'] = 0
        
        # Generate HTML report if save_dir provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Create HTML report
            html_content = f"""
            <html>
            <head>
                <title>Regional Flexibility Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #305090; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .summary {{ background-color: #e6f0ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .timestamp {{ color: #666; font-style: italic; }}
                </style>
            </head>
            <body>
                <h1>Regional Flexibility Analysis Report</h1>
                <p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="summary">
                    <h2>Executive Summary</h2>
                    <p>This report analyzes the energy flexibility across {len(regions)} French regions: {", ".join(regions)}.</p>
                    <p>Total energy dispatched: {total_dispatch:,.2f} MWh</p>
                    <p>Total storage operations: {total_storage_charge:,.2f} MWh charged, {total_storage_discharge:,.2f} MWh discharged</p>
                    <p>Overall storage efficiency: {(total_storage_discharge/total_storage_charge*100 if total_storage_charge>0 else 0):,.2f}%</p>
                </div>
                
                <h2>Regional Analysis</h2>
            """
            
            # Add the analysis table
            html_content += analysis_df.to_html(index=False, float_format=lambda x: f"{x:,.2f}")
            
            # Add interpretation
            html_content += f"""
                <h2>Interpretation</h2>
                <p>The simulation results show the following key insights:</p>
                <ul>
                    <li><strong>Regional Dispatch Distribution:</strong> {analysis_df.iloc[analysis_df['Total Dispatch (MWh)'].argmax()]['Region']} has the highest energy dispatch at {analysis_df['Total Dispatch (MWh)'].max():,.2f} MWh ({analysis_df['Dispatch Share (%)'].max():.2f}% of total).</li>
                    <li><strong>Storage Utilization:</strong> {analysis_df.iloc[analysis_df['Total Storage Charge (MWh)'].argmax()]['Region']} utilizes storage charging the most at {analysis_df['Total Storage Charge (MWh)'].max():,.2f} MWh.</li>
                    <li><strong>Storage Efficiency:</strong> Overall storage efficiency is {(total_storage_discharge/total_storage_charge*100 if total_storage_charge>0 else 0):,.2f}%, indicating {'good' if total_storage_discharge/total_storage_charge>0.7 else 'moderate' if total_storage_discharge/total_storage_charge>0.5 else 'poor'} conversion efficiency.</li>
                </ul>
                
                <h2>Recommendations</h2>
                <ul>
                    <li>Optimize storage operations in regions with lower efficiency rates.</li>
                    <li>Investigate opportunities for increased inter-regional exchanges to balance load.</li>
                    <li>Consider expanding storage capacity in regions with high dispatch needs.</li>
                </ul>
                
                <h3>Configuration Parameters</h3>
                <ul>
            """
            
            # Add configuration parameters
            for key, value in config.items():
                if key != "regions" and not isinstance(value, dict):
                    html_content += f"<li><strong>{key}:</strong> {value}</li>"
            
            html_content += """
                </ul>
            </body>
            </html>
            """
            
            # Save HTML report
            report_path = os.path.join(save_dir, f"flexibility_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            print(f"Comprehensive report saved to {report_path}")
    
    return analysis_df

def plot_seasonal_patterns(results: Dict, regions: List[str], save_dir: str = None):
    """
    Generate visualizations showing seasonal patterns in energy flexibility.
    
    Args:
        results: Dictionary of optimization results
        regions: List of region names
        save_dir: Optional directory to save plots
    """
    if 'regional_results' not in results:
        raise KeyError(f"Expected 'regional_results' in results dictionary. Found keys: {list(results.keys())}")
    
    # Create mapping for region names
    region_keys = {}
    for region in regions:
        for key in results['regional_results'].keys():
            normalized_key = key.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
            normalized_region = region.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
            if normalized_key == normalized_region:
                region_keys[region] = key
                break
    
    # Create synthetic dates for a full year (assuming 24 time periods = 1 day)
    start_date = datetime(2024, 1, 1)
    if any(results['regional_results']):
        first_region = next(iter(results['regional_results'].values()))
        time_periods = len(first_region.get('dispatch', []))
        days = time_periods // 24  # Assuming 24 time periods per day
        if days < 1:
            days = 1  # Ensure at least one day
    else:
        days = 365  # Default to full year if no data
    
    # Create seasonal data
    # For simplicity, we'll define seasons as:
    # Winter: Dec-Feb, Spring: Mar-May, Summer: Jun-Aug, Fall: Sep-Nov
    seasonal_data = {
        "Winter": [],
        "Spring": [],
        "Summer": [],
        "Fall": []
    }
    
    # Map time periods to seasons
    for i in range(days):
        date = start_date + timedelta(days=i)
        month = date.month
        
        if month in [12, 1, 2]:
            season = "Winter"
        elif month in [3, 4, 5]:
            season = "Spring"
        elif month in [6, 7, 8]:
            season = "Summer"
        else:  # month in [9, 10, 11]
            season = "Fall"
            
        seasonal_data[season].append(i)
    
    # Create seasonal aggregates for each region
    seasonal_metrics = {region: {season: {"dispatch": 0, "storage_charge": 0, "storage_discharge": 0} 
                                for season in seasonal_data.keys()} 
                        for region in regions}
    
    for region in regions:
        region_key = region_keys.get(region, region)
        if region_key not in results['regional_results']:
            continue
            
        region_data = results['regional_results'][region_key]
        
        if 'dispatch' not in region_data:
            continue
            
        dispatch = region_data['dispatch']
        storage_charge = region_data.get('storage_charge', [0] * len(dispatch))
        storage_discharge = region_data.get('storage_discharge', [0] * len(dispatch))
        
        for season, days_indices in seasonal_data.items():
            # Get all time periods for these days
            time_indices = []
            for day in days_indices:
                start_idx = day * 24
                end_idx = start_idx + 24
                time_indices.extend(range(start_idx, end_idx))
                
            # Filter to indices that are within our data range
            valid_indices = [idx for idx in time_indices if idx < len(dispatch)]
            
            if valid_indices:
                # Calculate seasonal metrics
                seasonal_metrics[region][season]["dispatch"] = sum([dispatch[idx] for idx in valid_indices])
                seasonal_metrics[region][season]["storage_charge"] = sum([storage_charge[idx] for idx in valid_indices if idx < len(storage_charge)])
                seasonal_metrics[region][season]["storage_discharge"] = sum([storage_discharge[idx] for idx in valid_indices if idx < len(storage_discharge)])
    
    # Plot seasonal dispatch patterns
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data for plotting
    seasons = list(seasonal_data.keys())
    x = np.arange(len(seasons))
    width = 0.8 / len(regions)  # Bar width
    
    for i, region in enumerate(regions):
        if region not in seasonal_metrics:
            continue
            
        dispatch_values = [seasonal_metrics[region][season]["dispatch"] for season in seasons]
        ax.bar(x + i*width - 0.4 + width/2, dispatch_values, width, label=region)
    
    ax.set_title('Seasonal Energy Dispatch Patterns', fontsize=16)
    ax.set_ylabel('Total Dispatch (MWh)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'seasonal_dispatch_patterns.png'))
    
    plt.close()
    
    # Plot seasonal storage operations
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, region in enumerate(regions):
        if region not in seasonal_metrics:
            continue
            
        charge_values = [seasonal_metrics[region][season]["storage_charge"] for season in seasons]
        discharge_values = [seasonal_metrics[region][season]["storage_discharge"] for season in seasons]
        
        # Plot net storage (discharge - charge)
        net_storage = [d - c for d, c in zip(discharge_values, charge_values)]
        ax.bar(x + i*width - 0.4 + width/2, net_storage, width, label=region)
    
    ax.set_title('Seasonal Net Storage Operations', fontsize=16)
    ax.set_ylabel('Net Storage (MWh, positive = net discharge)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'seasonal_storage_patterns.png'))
    
    plt.close()
