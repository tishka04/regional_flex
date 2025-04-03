import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
from datetime import datetime
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def plot_regional_dispatch(results: Dict, region: str, save_path: str = None):
    """Plot dispatch results for a specific region.
    
    Args:
        results: Dictionary of optimization results
        region: Region name to plot
        save_path: Optional path to save the plot
    """
    if 'regional_results' not in results:
        raise KeyError(f"Expected 'regional_results' in results dictionary. Found keys: {list(results.keys())}")
        
    # Normalize region names to match the keys in regional_results dictionary
    region_key = None
    for key in results['regional_results'].keys():
        # Convert both the key and the region name to lowercase and remove special chars for comparison
        normalized_key = key.lower().replace('-', ' ').replace('ô', 'o').replace('ó', 'o').replace('ò', 'o').replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('à', 'a')
        normalized_region = region.lower().replace('-', ' ').replace('ô', 'o').replace('ó', 'o').replace('ò', 'o').replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('à', 'a')
        
        if normalized_key == normalized_region:
            region_key = key
            break
    
    if not region_key:
        raise KeyError(f"Region '{region}' not found in results. Available regions: {list(results['regional_results'].keys())}")
    
    # Create DataFrame from regional results
    region_data = results['regional_results'][region_key]
    time_periods = range(results['time_periods'])
    df = pd.DataFrame(region_data, index=time_periods)
    
    plt.figure(figsize=(12, 6))
    # Only plot data that exists in the results
    available_fields = list(df.columns)
    print(f"Available fields for {region_key}: {available_fields}")
    
    # Always plot dispatch if available
    if 'dispatch' in available_fields:
        plt.plot(df.index, df['dispatch'], label='Dispatch')
    
    # Plot storage operations if available
    if 'storage_charge' in available_fields:
        plt.plot(df.index, df['storage_charge'], label='Storage Charge')
    if 'storage_discharge' in available_fields:
        plt.plot(df.index, df['storage_discharge'], label='Storage Discharge')
    
    # Plot slack variables if available
    if 'slack_pos' in available_fields:
        plt.plot(df.index, df['slack_pos'], label='Positive Slack')
    if 'slack_neg' in available_fields:
        plt.plot(df.index, df['slack_neg'], label='Negative Slack')
    
    plt.title(f'Energy Dispatch Results - {region}')
    plt.xlabel('Time Period')
    plt.ylabel('Power (MW)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_inter_regional_exchanges(results: Dict, regions: List[str], save_path: str = None):
    """Plot inter-regional power exchanges.
    
    Args:
        results: Dictionary of optimization results
        regions: List of region names
        save_path: Optional path to save the plot
    """
    if 'exchange' not in results:
        raise KeyError(f"Expected 'exchange' in results dictionary. Found keys: {list(results.keys())}")
        
    plt.figure(figsize=(15, 10))
    n_regions = len(regions)
    time_periods = range(results['time_periods'])
    
    # Get the actual region keys from the results dictionary
    region_keys = {}
    for region in regions:
        for key in results['regional_results'].keys():
            normalized_key = key.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
            normalized_region = region.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
            if normalized_key == normalized_region:
                region_keys[region] = key
                break
    
    for i, r1 in enumerate(regions):
        for j, r2 in enumerate(regions):
            if i < j:
                # Try finding exchange data using both original and normalized names
                exchange_found = False
                exchange_data = None
                
                # Try all possible combinations of region names
                for key in results['exchange'].keys():
                    # Check if this key matches the current region pair in any combination
                    if (r1 in key and r2 in key) or \
                       (region_keys.get(r1, '') in key and region_keys.get(r2, '') in key):
                        exchange_data = results['exchange'][key]
                        exchange_found = True
                        break
                
                if exchange_found:
                    plt.subplot(n_regions-1, n_regions-1, i*(n_regions-1) + j)
                    plt.plot(time_periods, exchange_data)
                    plt.title(f'{r1} → {r2}')
                    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_storage_levels(results: Dict, regions: List[str], save_path: str = None):
    """Plot storage levels for all regions.
    
    Args:
        results: Dictionary of optimization results
        regions: List of region names
        save_path: Optional path to save the plot
    """
    if 'regional_results' not in results:
        raise KeyError(f"Expected 'regional_results' in results dictionary. Found keys: {list(results.keys())}")
        
    plt.figure(figsize=(12, 8))
    time_periods = range(results['time_periods'])
    
    # Find matching region keys in results
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
        if region_key in results['regional_results']:
            region_data = results['regional_results'][region_key]
            
            # Calculate storage level from charge and discharge if not available
            if 'soc' in region_data and any(val > 0 for val in region_data['soc']):
                plt.plot(time_periods, region_data['soc'], label=region)
            elif 'storage_charge' in region_data and 'storage_discharge' in region_data:
                storage_level = np.array(region_data['storage_charge']).cumsum() - np.array(region_data['storage_discharge']).cumsum()
                plt.plot(time_periods, storage_level, label=region)
            else:
                print(f"Warning: Storage data not available for {region}")
        else:
            print(f"Warning: No data available for {region} (available regions: {list(results['regional_results'].keys())})")
    plt.title('Storage Levels by Region')
    plt.xlabel('Time Period')
    plt.ylabel('Energy (MWh)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def analyze_results(results: Dict, regions: List[str]):
    """Analyze optimization results.
    
    Args:
        results: Dictionary of optimization results
        regions: List of region names
        
    Returns:
        DataFrame with analysis metrics
    """
    if 'regional_results' not in results:
        raise KeyError(f"Expected 'regional_results' in results dictionary. Found keys: {list(results.keys())}")
        
    # Find matching region keys in results
    region_keys = {}
    for region in regions:
        for key in results['regional_results'].keys():
            normalized_key = key.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
            normalized_region = region.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
            if normalized_key == normalized_region:
                region_keys[region] = key
                break
                
    metrics = []
    
    for region in regions:
        region_key = region_keys.get(region, region)
        if region_key not in results['regional_results']:
            print(f"Warning: No data available for {region} (available regions: {list(results['regional_results'].keys())})")
            continue
            
        region_data = results['regional_results'][region_key]
        
        # Calculate dispatch metrics
        total_dispatch = sum(region_data['dispatch'])
        mean_dispatch = sum(region_data['dispatch']) / len(region_data['dispatch']) if region_data['dispatch'] else 0
        max_dispatch = max(region_data['dispatch']) if region_data['dispatch'] else 0
        
        # Calculate storage metrics
        total_storage_charge = sum(region_data['storage_charge']) if 'storage_charge' in region_data else 0
        total_storage_discharge = sum(region_data['storage_discharge']) if 'storage_discharge' in region_data else 0
        net_storage = total_storage_charge - total_storage_discharge
        
        # Calculate storage level if soc is available, otherwise derive it from charge/discharge
        if 'soc' in region_data and any(val > 0 for val in region_data['soc']):
            max_storage_level = max(region_data['soc'])
        elif 'storage_charge' in region_data and 'storage_discharge' in region_data:
            storage_level = np.array(region_data['storage_charge']).cumsum() - np.array(region_data['storage_discharge']).cumsum()
            max_storage_level = max(storage_level) if len(storage_level) > 0 else 0
        else:
            max_storage_level = None
        
        # Calculate demand response metrics if available
        total_demand_response = 0  # Default value
        
        metrics.append({
            'Region': region,
            'Total Dispatch (MWh)': total_dispatch,
            'Total Storage Charge (MWh)': total_storage_charge,
            'Total Storage Discharge (MWh)': total_storage_discharge,
            'Net Storage Use (MWh)': net_storage,
            'Total Demand Response (MWh)': total_demand_response,
            'Max Dispatch (MW)': max_dispatch,
            'Max Storage Level (MWh)': max_storage_level
        })
    
    return pd.DataFrame(metrics)

def plot_seasonal_patterns(results: Dict, regions: List[str], save_dir: str = None) -> None:
    """Plot seasonal patterns of dispatch, storage, and demand response.
    
    Args:
        results: Optimization results
        regions: List of region names
        save_dir: Directory to save plots
    """
    print("Plotting seasonal patterns...")
    
    # Calculate seasonal metrics
    seasonal_metrics = {}
    
    # Find matching region keys in results
    region_keys = {}
    if 'regional_results' in results:
        for region in regions:
            for key in results['regional_results'].keys():
                normalized_key = key.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
                normalized_region = region.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
                if normalized_key == normalized_region:
                    region_keys[region] = key
                    break
    
    time_periods = range(results.get('time_periods', 0))
    for region in regions:
        # Try to find the region in results
        region_key = region_keys.get(region, region)
        if 'regional_results' in results and region_key in results['regional_results']:
            # Get region-specific data
            region_results = results['regional_results'][region_key]
            
            # Create a frequency string based on estimated time periods for one year
            # For half-hourly data, we expect ~17520 periods (365*24*2)
            freq = '30min' if len(time_periods) > 8000 else 'H'
            
            # Convert time indices to datetime for 2022
            time_indices = pd.date_range(
                start='2022-01-01',
                periods=len(time_periods),
                freq=freq
            )
        
            # Create DataFrame for easier manipulation
            data_dict = {
                'time': time_indices,
            }
            
            # Add data columns that exist in the results
            for field in ['dispatch', 'storage_charge', 'storage_discharge']:
                if field in region_results and len(region_results[field]) == len(time_indices):
                    data_dict[field] = region_results[field]
                else:
                    data_dict[field] = [0] * len(time_indices)
            
            # Add demand response if available
            if 'demand_response' in region_results and len(region_results['demand_response']) == len(time_indices):
                data_dict['demand_response'] = region_results['demand_response']
            else:
                data_dict['demand_response'] = [0] * len(time_indices)
                
            df = pd.DataFrame(data_dict)
        
        # Add seasonal information
        df['season'] = df['time'].dt.quarter
        df['month'] = df['time'].dt.month
        df['hour'] = df['time'].dt.hour
        
        # Calculate seasonal statistics
        seasonal_stats = df.groupby(['season', 'hour']).agg({
            'dispatch': ['mean', 'sum'],
            'storage_charge': ['mean', 'sum'],
            'storage_discharge': ['mean', 'sum'],
            'demand_response': ['mean', 'sum']
        }).reset_index()
        
        # Map quarter numbers to season names
        season_map = {
            1: 'Winter',
            2: 'Spring',
            3: 'Summer',
            4: 'Fall'
        }
        seasonal_stats['season'] = seasonal_stats['season'].map(season_map)
        
        seasonal_metrics[region] = seasonal_stats
        
    # Plot seasonal patterns
    for region in tqdm(regions, desc="Plotting seasonal patterns"):
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f"Seasonal Patterns - {region}", fontsize=16)
        
        # Dispatch patterns
        ax = axes[0, 0]
        seasonal_data = seasonal_metrics[region]
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            season_data = seasonal_data[seasonal_data['season'] == season]
            ax.plot(season_data['hour'], season_data['dispatch']['mean'], label=season)
        ax.set_title('Hourly Dispatch Patterns')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Dispatch (MW)')
        ax.legend()
        
        # Storage patterns
        ax = axes[0, 1]
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            season_data = seasonal_data[seasonal_data['season'] == season]
            ax.plot(season_data['hour'], season_data['storage_charge']['mean'], label=f'{season} Charge')
            ax.plot(season_data['hour'], -season_data['storage_discharge']['mean'], label=f'{season} Discharge')
        ax.set_title('Hourly Storage Patterns')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Storage (MW)')
        ax.legend()
        
        # Demand response patterns
        ax = axes[1, 0]
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            season_data = seasonal_data[seasonal_data['season'] == season]
            ax.plot(season_data['hour'], season_data['demand_response']['mean'], label=season)
        ax.set_title('Hourly Demand Response Patterns')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Demand Response (MW)')
        ax.legend()
        
        # Monthly patterns
        ax = axes[1, 1]
        monthly_data = seasonal_data.groupby('month').agg({
            'dispatch': ['mean', 'sum'],
            'storage_charge': ['mean', 'sum'],
            'storage_discharge': ['mean', 'sum'],
            'demand_response': ['mean', 'sum']
        }).reset_index()
        
        ax.plot(monthly_data['month'], monthly_data['dispatch']['mean'], label='Dispatch')
        ax.plot(monthly_data['month'], monthly_data['storage_charge']['mean'], label='Storage Charge')
        ax.plot(monthly_data['month'], -monthly_data['storage_discharge']['mean'], label='Storage Discharge')
        ax.plot(monthly_data['month'], monthly_data['demand_response']['mean'], label='Demand Response')
        ax.set_title('Monthly Patterns')
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Value (MW)')
        ax.legend()
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'seasonal_patterns_{region}.png'))
        else:
            plt.show()
        plt.close()

def plot_yearly_overview(results: Dict, regions: List[str], save_dir: str = None) -> None:
    """Plot yearly overview of flexibility metrics.
    
    Args:
        results: Optimization results
        regions: List of region names
        save_dir: Directory to save plots
    """
    print("Plotting yearly overview...")
    
    # Find matching region keys in results
    region_keys = {}
    if 'regional_results' in results:
        for region in regions:
            for key in results['regional_results'].keys():
                normalized_key = key.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
                normalized_region = region.lower().replace('-', ' ').replace('ô', 'o').replace('é', 'e').replace('è', 'e').replace('à', 'a')
                if normalized_key == normalized_region:
                    region_keys[region] = key
                    break
    
    # Calculate yearly metrics
    yearly_metrics = {}
    
    time_periods = range(results.get('time_periods', 0))
    if len(time_periods) == 0:
        print("Warning: No time periods found in results")
        return
        
    for region in tqdm(regions, desc="Calculating yearly metrics"):
        # Try to find the region in results
        region_key = region_keys.get(region, region)
        if 'regional_results' in results and region_key in results['regional_results']:
            # Get region-specific data
            region_results = results['regional_results'][region_key]
            
            # Create a frequency string based on estimated time periods for one year
            # For half-hourly data, we expect ~17520 periods (365*24*2)
            freq = '30min' if len(time_periods) > 8000 else 'H'
            
            # Convert time indices to datetime for 2022
            time_indices = pd.date_range(
                start='2022-01-01',
                periods=len(time_periods),
                freq=freq
            )
            
            # Create DataFrame for easier manipulation
            data_dict = {
                'time': time_indices,
            }
            
            # Add data columns that exist in the results
            for field in ['dispatch', 'storage_charge', 'storage_discharge', 'demand_response']:
                if field in region_results and len(region_results[field]) == len(time_indices):
                    data_dict[field] = region_results[field]
                else:
                    data_dict[field] = [0] * len(time_indices)
                    
            # Add SOC if available
            if 'soc' in region_results and len(region_results['soc']) == len(time_indices):
                data_dict['soc'] = region_results['soc']
            else:
                # Try to calculate SOC if not directly available
                if 'storage_charge' in data_dict and 'storage_discharge' in data_dict:
                    data_dict['soc'] = np.cumsum(data_dict['storage_charge']) - np.cumsum(data_dict['storage_discharge'])
                else:
                    data_dict['soc'] = [0] * len(time_indices)
            
            df = pd.DataFrame(data_dict)
        
        # Calculate yearly statistics
        yearly_stats = df.groupby(df['time'].dt.date).agg({
            'dispatch': ['mean', 'sum', 'max', 'min'],
            'storage_charge': ['mean', 'sum', 'max', 'min'],
            'storage_discharge': ['mean', 'sum', 'max', 'min'],
            'demand_response': ['mean', 'sum', 'max', 'min'],
            'soc': ['mean', 'max', 'min']
        })
        
        yearly_metrics[region] = yearly_stats
        
    # Plot yearly overview
    for region in regions:
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle(f"Yearly Overview - {region}", fontsize=16)
        
        # Dispatch metrics
        ax = axes[0, 0]
        yearly_data = yearly_metrics[region]
        ax.plot(yearly_data.index, yearly_data['dispatch']['mean'], label='Average')
        ax.plot(yearly_data.index, yearly_data['dispatch']['max'], label='Max')
        ax.plot(yearly_data.index, yearly_data['dispatch']['min'], label='Min')
        ax.set_title('Daily Dispatch Metrics')
        ax.set_xlabel('Date')
        ax.set_ylabel('Dispatch (MW)')
        ax.legend()
        
        # Storage metrics
        ax = axes[0, 1]
        ax.plot(yearly_data.index, yearly_data['storage_charge']['mean'], label='Charge Average')
        ax.plot(yearly_data.index, yearly_data['storage_discharge']['mean'], label='Discharge Average')
        ax.plot(yearly_data.index, yearly_data['soc']['mean'], label='SOC Average')
        ax.set_title('Daily Storage Metrics')
        ax.set_xlabel('Date')
        ax.set_ylabel('Storage (MW)')
        ax.legend()
        
        # Demand response metrics
        ax = axes[1, 0]
        ax.plot(yearly_data.index, yearly_data['demand_response']['mean'], label='Average')
        ax.plot(yearly_data.index, yearly_data['demand_response']['max'], label='Max')
        ax.plot(yearly_data.index, yearly_data['demand_response']['min'], label='Min')
        ax.set_title('Daily Demand Response Metrics')
        ax.set_xlabel('Date')
        ax.set_ylabel('Demand Response (MW)')
        ax.legend()
        
        # Monthly totals
        ax = axes[1, 1]
        monthly_data = yearly_data.groupby(yearly_data.index.month).agg({
            'dispatch': ['sum'],
            'storage_charge': ['sum'],
            'storage_discharge': ['sum'],
            'demand_response': ['sum']
        })
        
        ax.bar(monthly_data.index, monthly_data['dispatch']['sum'], label='Dispatch')
        ax.bar(monthly_data.index, monthly_data['storage_charge']['sum'], bottom=monthly_data['dispatch']['sum'], label='Storage Charge')
        ax.bar(monthly_data.index, -monthly_data['storage_discharge']['sum'], bottom=monthly_data['dispatch']['sum'] + monthly_data['storage_charge']['sum'], label='Storage Discharge')
        ax.bar(monthly_data.index, monthly_data['demand_response']['sum'], bottom=monthly_data['dispatch']['sum'] + monthly_data['storage_charge']['sum'] + monthly_data['storage_discharge']['sum'], label='Demand Response')
        ax.set_title('Monthly Totals')
        ax.set_xlabel('Month')
        ax.set_ylabel('Total Energy (MWh)')
        ax.legend()
        
        # Storage capacity utilization
        ax = axes[2, 0]
        ax.plot(yearly_data.index, yearly_data['soc']['max'], label='Max SOC')
        ax.plot(yearly_data.index, yearly_data['soc']['min'], label='Min SOC')
        ax.plot(yearly_data.index, yearly_data['soc']['mean'], label='Average SOC')
        ax.set_title('Storage Capacity Utilization')
        ax.set_xlabel('Date')
        ax.set_ylabel('Storage Level (MW)')
        ax.legend()
        
        # Correlation analysis
        ax = axes[2, 1]
        correlation_matrix = yearly_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'yearly_overview_{region}.png'))
        else:
            plt.show()
        plt.close()
