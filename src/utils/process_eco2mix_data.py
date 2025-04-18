#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process eco2mix data for 2022 and prepare it for the technology-specific optimizer.
This script focuses on the four French regions of interest:
- Auvergne Rhone Alpes
- Nouvelle Aquitaine
- Occitanie
- Provence Alpes Cote d'Azur
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger('process_eco2mix')

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_eco2mix_data(eco2mix_file, output_dir, config, year=2022):
    """Process eco2mix data for specified year and regions.
    
    Args:
        eco2mix_file (str): Path to eco2mix CSV file
        output_dir (str): Directory where processed data will be saved
        config (dict): Configuration dictionary
        year (int): Year to filter data for
    
    Returns:
        dict: Dictionary of processed regional data
    """
    logger.info(f"Processing eco2mix data from {eco2mix_file} for year {year}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the list of regions from config
    regions = config.get("regions", [])
    if not regions:
        logger.error("No regions defined in config")
        return False
    
    # Define region mappings (for potential different naming)
    region_mappings = {
        "Auvergne Rhone Alpes": ["Auvergne-Rhône-Alpes", "Auvergne Rhone Alpes", "Auvergne-Rhône-Alpes"],
        "Nouvelle Aquitaine": ["Nouvelle-Aquitaine", "Nouvelle Aquitaine"],
        "Occitanie": ["Occitanie"],
        "Provence Alpes Cote d'Azur": ["Provence-Alpes-Côte d'Azur", "Provence Alpes Cote d'Azur", "PACA"]
    }
    
    # Load the eco2mix data
    logger.info(f"Loading eco2mix data from {eco2mix_file}")
    
    try:
        # Try with different encodings
        for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'ISO-8859-1']:
            try:
                df = pd.read_csv(eco2mix_file, sep=';', encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        logger.info(f"Loaded data with {len(df)} rows")
        
        # Check if required columns exist
        required_columns = [
            'Date', 'Heure', 'Région', 
            'Consommation (MW)', 'Thermique (MW)', 'Nucléaire (MW)',
            'Eolien (MW)', 'Solaire (MW)', 'Hydraulique (MW)', 
            'Bioénergies (MW)'
        ]
        
        # Handle column name variations
        column_mappings = {
            'Date': ['Date'],
            'Heure': ['Heure'],
            'Région': ['Région', 'Region'],
            'Consommation (MW)': ['Consommation (MW)', 'Consommation'],
            'Thermique (MW)': ['Thermique (MW)', 'Thermique'],
            'Nucléaire (MW)': ['Nucléaire (MW)', 'Nucleaire (MW)', 'Nucléaire', 'Nucleaire'],
            'Eolien (MW)': ['Eolien (MW)', 'Eolien'],
            'Solaire (MW)': ['Solaire (MW)', 'Solaire'],
            'Hydraulique (MW)': ['Hydraulique (MW)', 'Hydraulique'],
            'Bioénergies (MW)': ['Bioénergies (MW)', 'Bioenergies (MW)', 'Bioénergies', 'Bioenergies']
        }
        
        # Create a mapping from actual column names to standardized names
        actual_to_standard = {}
        for standard_name, variations in column_mappings.items():
            for var in variations:
                if var in df.columns:
                    actual_to_standard[var] = standard_name
                    break
        
        # Rename columns to standardized names
        df.rename(columns=actual_to_standard, inplace=True)
        
        # Check for missing columns after renaming
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            logger.info(f"Available columns: {df.columns.tolist()}")
        
        # Convert date and time columns to datetime
        if 'Date' in df.columns and 'Heure' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Heure'], format='%Y-%m-%d %H:%M')
            except ValueError:
                logger.warning("Error parsing date/time with default format, trying alternative format")
                df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Heure'])
        elif 'Date - Heure' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Date - Heure'])
        else:
            logger.error("No date/time columns found to create timestamp")
            return False
        
        # Filter for the target year
        df['year'] = df['timestamp'].dt.year
        df_year = df[df['year'] == year].copy()
        
        if len(df_year) == 0:
            logger.warning(f"No data found for year {year}")
            # Try to find what years are available
            available_years = sorted(df['year'].unique())
            logger.info(f"Available years: {available_years}")
            
            if len(available_years) > 0:
                logger.info(f"Using the most recent available year: {available_years[-1]}")
                df_year = df[df['year'] == available_years[-1]].copy()
            else:
                return False
        
        logger.info(f"Filtered to {len(df_year)} rows for year {df_year['year'].iloc[0]}")
        
        # Process each region
        regional_data = {}
        
        for target_region in regions:
            logger.info(f"Processing data for region: {target_region}")
            
            # Find matches in the data for this region
            region_matches = []
            for name_var in region_mappings.get(target_region, [target_region]):
                region_df = df_year[df_year['Région'].str.contains(name_var, case=False, na=False)]
                if len(region_df) > 0:
                    region_matches.append(region_df)
            
            if not region_matches:
                logger.warning(f"No data found for region {target_region}")
                continue
            
            # Combine all matches for this region
            region_df = pd.concat(region_matches)
            logger.info(f"Found {len(region_df)} rows for {target_region}")
            
            # Create a clean DataFrame with required columns
            clean_data = pd.DataFrame({
                'timestamp': region_df['timestamp'],
                'demand': region_df.get('Consommation (MW)', np.nan),
                'thermal': region_df.get('Thermique (MW)', np.nan),
                'nuclear': region_df.get('Nucléaire (MW)', np.nan),
                'wind': region_df.get('Eolien (MW)', np.nan),
                'solar': region_df.get('Solaire (MW)', np.nan),
                'hydro': region_df.get('Hydraulique (MW)', np.nan),
                'biofuel': region_df.get('Bioénergies (MW)', np.nan)
            })
            
            # Replace missing values
            clean_data.fillna(0, inplace=True)
            
            # Remove duplicates (if any)
            clean_data.drop_duplicates(subset=['timestamp'], inplace=True)
            
            # Set timestamp as index
            clean_data.set_index('timestamp', inplace=True)
            
            # Resample to the desired time resolution
            resolution = config['time_settings'].get('resolution', '30min')
            if resolution:
                clean_data = clean_data.resample(resolution).mean()
                logger.info(f"Resampled data to {resolution} resolution")
            
            regional_data[target_region] = clean_data
            
            # Plot some basic statistics for verification
            plt.figure(figsize=(15, 10))
            
            # Plot daily load profiles
            clean_data.reset_index(inplace=True)
            clean_data['hour'] = clean_data['timestamp'].dt.hour
            hourly_avg = clean_data.groupby('hour').mean()
            
            plt.subplot(2, 2, 1)
            hourly_avg['demand'].plot(label='Demand')
            plt.title(f'Average Daily Load Profile - {target_region}')
            plt.xlabel('Hour of Day')
            plt.ylabel('Power (MW)')
            plt.grid(True)
            
            # Plot technology mix
            plt.subplot(2, 2, 2)
            tech_cols = ['thermal', 'nuclear', 'wind', 'solar', 'hydro', 'biofuel']
            hourly_avg[tech_cols].plot.area(stacked=True)
            plt.title(f'Technology Mix by Hour - {target_region}')
            plt.xlabel('Hour of Day')
            plt.ylabel('Power (MW)')
            plt.grid(True)
            
            # Monthly variations
            clean_data['month'] = clean_data['timestamp'].dt.month
            monthly_avg = clean_data.groupby('month').mean()
            
            plt.subplot(2, 2, 3)
            monthly_avg['demand'].plot(marker='o')
            plt.title(f'Monthly Average Demand - {target_region}')
            plt.xlabel('Month')
            plt.ylabel('Power (MW)')
            plt.grid(True)
            plt.xticks(range(1, 13))
            
            # Technology contribution
            plt.subplot(2, 2, 4)
            tech_total = clean_data[tech_cols].sum()
            tech_total.plot.pie(autopct='%1.1f%%', startangle=90)
            plt.title(f'Technology Contribution - {target_region}')
            plt.axis('equal')
            
            # Save plot
            plot_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plot_file = os.path.join(plot_dir, f"{target_region.replace(' ', '_')}_stats.png")
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.close()
            logger.info(f"Saved statistics plot to {plot_file}")
            
            # Reset the index for further processing
            clean_data.set_index('timestamp', inplace=True)
        
        # Create time index
        start_date = config['time_settings'].get('start_date')
        end_date = config['time_settings'].get('end_date')
        
        if start_date and end_date:
            # Convert string dates to datetime if needed
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
                
            # Create time index
            time_index = pd.date_range(start=start_date, end=end_date, freq=resolution)
            logger.info(f"Created time index with {len(time_index)} periods from {start_date} to {end_date}")
        else:
            # Use minimum and maximum timestamps from data
            min_ts = min(df['timestamp'].min() for df in regional_data.values())
            max_ts = max(df['timestamp'].max() for df in regional_data.values())
            time_index = pd.date_range(start=min_ts, end=max_ts, freq=resolution)
            logger.info(f"Created time index with {len(time_index)} periods from {min_ts} to {max_ts}")
        
        # Ensure all regions have data for all time periods
        for region, data in regional_data.items():
            # Reindex to the common time index
            reindexed = data.reindex(time_index)
            
            # Fill missing values
            reindexed = reindexed.interpolate(method='time', limit=24)  # Interpolate gaps up to 24 time periods
            reindexed = reindexed.fillna(method='ffill').fillna(method='bfill')  # Fill any remaining NaNs
            
            # Store back
            regional_data[region] = reindexed
            logger.info(f"Standardized time index for {region}, now has {len(reindexed)} time periods")
        
        # Save each region's data to separate CSV files
        for region, data in regional_data.items():
            region_file = os.path.join(output_dir, f"{region.replace(' ', '_')}.csv")
            data.to_csv(region_file)
            logger.info(f"Saved {region} data to {region_file}")
        
        # Create a combined data structure for the optimizer
        combined_data = {
            'time_index': time_index
        }
        
        # Add each region's data
        for region, data in regional_data.items():
            combined_data[region] = data
        
        # Save time index to CSV
        time_index_file = os.path.join(output_dir, "time_index.csv")
        pd.DataFrame({'timestamp': time_index}).to_csv(time_index_file, index=False)
        logger.info(f"Saved time index to {time_index_file}")
        
        return combined_data
        
    except Exception as e:
        logger.error(f"Error processing eco2mix data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_simulation(data, config_path, results_dir, time_period=None):
    """Run the technology-specific optimizer simulation.
    
    Args:
        data (dict): Processed regional data
        config_path (str): Path to configuration YAML file
        results_dir (str): Directory to save results
        time_period (tuple, optional): Start and end times for the simulation
    """
    logger.info("Running technology-specific optimizer simulation")
    
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Import the optimizer here to avoid circular imports
    from src.model.optimizer_tech import RegionalFlexOptimizerTech
    
    try:
        # Initialize the optimizer
        optimizer = RegionalFlexOptimizerTech(config_path)
        
        # Process the time periods
        if time_period:
            start_time, end_time = time_period
            time_periods = [t for t in data['time_index'] if start_time <= t <= end_time]
            logger.info(f"Using {len(time_periods)} time periods from {start_time} to {end_time}")
        else:
            time_periods = data['time_index']
            logger.info(f"Using all {len(time_periods)} time periods")
        
        # Build the model
        logger.info("Building optimization model")
        optimizer.build_model(data, time_periods)
        
        # Solve the model
        logger.info("Solving optimization model")
        results = optimizer.solve_model()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"tech_results_{timestamp}.json")
        
        import json
        
        # Convert all timestamps in dictionary keys to strings
        def convert_timestamps_in_dict(obj):
            if isinstance(obj, dict):
                return {str(k) if isinstance(k, (pd.Timestamp, datetime)) else k: convert_timestamps_in_dict(v) 
                        for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_timestamps_in_dict(item) for item in obj]
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return str(obj)
            elif isinstance(obj, (np.int64, np.float64)):
                return float(obj)
            else:
                return obj
        
        # Convert to serializable objects
        serializable_results = convert_timestamps_in_dict(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Results saved to {results_file}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main execution function."""
    logger.info("Starting eco2mix data processing and simulation")
    
    # Define paths
    eco2mix_file = 'data/Raw/eco2mix-regional-cons-def.csv'
    output_dir = 'data/processed'
    results_dir = 'results/tech'
    config_path = 'config/config.yaml'
    
    # Load configuration
    config = load_config(config_path)
    if not config:
        logger.error("Failed to load configuration")
        return False
    
    # Process eco2mix data
    data = process_eco2mix_data(eco2mix_file, output_dir, config, year=2022)
    
    if not data:
        logger.error("Failed to process eco2mix data")
        return False
    
    # Run simulation
    # Use a limited time period for faster testing (1 week)
    time_period = (
        data['time_index'][0],  # Start with first timestamp
        data['time_index'][min(len(data['time_index'])-1, 336)]  # Use first week (48 periods per day * 7 days)
    )
    
    success = run_simulation(data, config_path, results_dir, time_period)
    
    if success:
        logger.info("Simulation completed successfully")
    else:
        logger.error("Simulation failed")
    
    return success

if __name__ == "__main__":
    main()
