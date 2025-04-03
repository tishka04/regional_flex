import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import os
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and preprocessing regional energy data."""
    
    def __init__(self, data_dir: str = 'data', config: Dict = None):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing data files
            config: Configuration settings
        """
        self.data_dir = Path(data_dir)
        self.config = config or {}
        
        # Define region patterns for column matching
        self.region_patterns = {
            # French regions with special patterns to handle spaces and accents
            "Auvergne Rhone Alpes": "Auvergne Rhone Alpes",
            "Nouvelle Aquitaine": "Nouvelle Aquitaine",
            "Occitanie": "Occitanie",
            "Provence Alpes Cote d'Azur": "Provence Alpes Cote d'Azur"
        }
        
        logger.info(f"Initialized region patterns: {self.region_patterns}")
        
        # Initialize regions from config if available
        if config and "regions" in config:
            self.regions = [r for r in config["regions"] if r != 'timestamp' and r.strip()]
        else:
            # Use all regions from the pattern dictionary
            self.regions = [r for r in self.region_patterns.keys() if r != 'timestamp' and r.strip()]
        
        # Set time settings from config
        self.time_settings = {
            'resolution': '30min',
            'start_date': '2022-01-01',
            'end_date': '2022-12-31'
        }
        
        if config and "time_settings" in config:
            self.time_settings.update(config["time_settings"])
            
        # Convert date strings to datetime objects
        if isinstance(self.time_settings['start_date'], str):
            self.time_settings['start_date'] = pd.to_datetime(self.time_settings['start_date'])
        if isinstance(self.time_settings['end_date'], str):
            self.time_settings['end_date'] = pd.to_datetime(self.time_settings['end_date'])
            
        print(f"Using regions from config: {self.regions}")
        print(f"Time settings: {self.time_settings}")
        
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw data files.
        
        Returns:
            Dictionary of DataFrames by region
        """
        # Load multi-region data
        data_path = self.data_dir / 'multi_region_data.csv'
        print(f"Loading data from: {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        data = pd.read_csv(data_path, sep=';', encoding='utf-8')
        
        # Convert timestamp to datetime
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d/%m/%Y %H:%M')
        except ValueError:
            # Try alternative formats if the first one fails
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            except ValueError as e:
                print(f"Error parsing timestamps: {e}")
                raise
                
        print(f"Loaded data with {len(data)} rows from {data['timestamp'].min()} to {data['timestamp'].max()}")
        
        # Filter data to the desired time range if specified
        if self.time_settings.get('start_date') and self.time_settings.get('end_date'):
            start_date = self.time_settings['start_date']
            end_date = self.time_settings['end_date']
            data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
            print(f"Filtered data to {len(data)} rows from {start_date} to {end_date}")
        
        # Split data by region
        regional_data = {}
        
        for region, pattern in self.region_patterns.items():
            # Skip regions not in the configured list
            if region not in self.regions:
                continue
                
            # Check if required columns exist
            required_columns = [f'{pattern}_demand_MW', f'{pattern}_solar_MW', f'{pattern}_wind_MW']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                print(f"Warning: Missing columns for region {region}: {missing_columns}")
                print(f"Available columns: {data.columns.tolist()}")
                continue
            
            # Create DataFrame for each region
            try:
                region_df = pd.DataFrame({
                    'timestamp': data['timestamp'],
                    'demand': data[f'{pattern}_demand_MW'],
                    'solar': data[f'{pattern}_solar_MW'],
                    'wind': data[f'{pattern}_wind_MW']
                })
                
                # Set timestamp as index
                region_df.set_index('timestamp', inplace=True)
                
                # Resample to the desired time resolution
                resolution = self.time_settings.get('resolution', '30min')
                if resolution and region_df.index.to_series().diff().min() != pd.Timedelta(resolution):
                    print(f"Resampling {region} data to {resolution} resolution")
                    region_df = region_df.resample(resolution).mean()
                
                regional_data[region] = region_df
                print(f"Prepared data for {region} with {len(region_df)} time periods")
                
            except Exception as e:
                print(f"Error preparing data for region {region}: {e}")
                continue
            
        return regional_data

    def _find_data_file(self) -> Path:
        """Find the most appropriate data file based on configuration.
        
        Returns:
            Path to the data file
        """
        # Default data file path
        data_path = self.data_dir / 'multi_region_data.csv'
        
        # Check if the default path exists
        if not data_path.exists():
            # Try alternative file names
            alternatives = [
                'regional_data.csv',
                'region_data.csv',
                'energy_data.csv',
                'flex_data.csv'
            ]
            
            for alt_file in alternatives:
                alt_path = self.data_dir / alt_file
                if alt_path.exists():
                    logger.info(f"Using alternative data file: {alt_path}")
                    return alt_path
            
            # If we reach this point, no suitable file was found
            raise FileNotFoundError(f"No suitable data file found in {self.data_dir}")
        
        logger.info(f"Using data file: {data_path}")
        return data_path
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load multi-regional data from files.
        
        Returns:
            Dictionary of DataFrames by region
        """
        logger.info("Starting data loading process...")
        
        # Create a progress bar for data loading steps
        steps = ['Finding data file', 'Loading data file', 'Updating regions', 'Processing regional data']
        
        with tqdm(total=len(steps), desc="Loading data", position=0, leave=False) as pbar:
            # Find appropriate data file
            pbar.set_description("Finding data file")
            data_file = self._find_data_file()
            pbar.update(1)
            
            # Load data file
            pbar.set_description("Loading data file")
            data_df = self._load_data_file(data_file)
            pbar.update(1)
            
            # Update available regions based on the data
            pbar.set_description("Updating regions")
            self._update_available_regions(data_df)
            pbar.update(1)
            
            # Process the data for each region
            pbar.set_description("Processing regional data")
            regional_data = self._process_regional_data(data_df)
            pbar.update(1)
        
        # Validate that we have at least one valid region with data
        if not isinstance(regional_data, dict) or len(regional_data) == 0:
            logger.warning("No valid regional data was extracted. Will retry with direct processing.")
            
            # Try to explicitly process each configured region
            regional_data = {}
            for region in self.regions:
                if region != 'timestamp' and region.strip():
                    region_df = self._extract_region_data(data_df, region)
                    if isinstance(region_df, pd.DataFrame) and not region_df.empty and len(region_df.columns) > 0:
                        regional_data[region] = region_df
                        logger.info(f"Successfully extracted data for region {region}")
        
        logger.info(f"Data loading completed for {len(regional_data)} regions: {list(regional_data.keys())}")
        return regional_data

    def _load_data_file(self, file_path: Path) -> pd.DataFrame:
        """Load data from file with appropriate parsing.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from: {file_path}")
        
        try:
            # Try to load CSV with various delimiters
            for delimiter in [',', ';', '\t']:
                try:
                    data = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
                    # If we successfully loaded data with columns, use this delimiter
                    if len(data.columns) > 1:
                        logger.info(f"Successfully loaded data with delimiter: '{delimiter}'")
                        break
                except Exception:
                    continue
            else:
                # If no delimiter worked, try default reading
                data = pd.read_csv(file_path, encoding='utf-8')
                
            # Try to identify and convert timestamp column
            timestamp_columns = [col for col in data.columns if 'time' in col.lower() 
                              or 'date' in col.lower() or 'timestamp' in col.lower()]
            
            if timestamp_columns:
                time_col = timestamp_columns[0]
                logger.info(f"Using '{time_col}' as timestamp column")
                
                # Try various datetime formats
                for fmt in ['%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M', None]:
                    try:
                        if fmt:
                            data[time_col] = pd.to_datetime(data[time_col], format=fmt)
                        else:
                            data[time_col] = pd.to_datetime(data[time_col])
                        break
                    except Exception:
                        continue
                        
                # Set timestamp as index
                data.set_index(time_col, inplace=True)
                
                # Filter to the time range in settings
                if self.time_settings.get('start_date') and self.time_settings.get('end_date'):
                    start_date = self.time_settings['start_date']
                    end_date = self.time_settings['end_date']
                    data = data[(data.index >= start_date) & (data.index <= end_date)]
                    logger.info(f"Filtered data to {len(data)} rows from {start_date} to {end_date}")
            else:
                logger.warning("No timestamp column found. Using row numbers as index.")
                
            return data
            
        except Exception as e:
            logger.error(f"Error loading data file: {e}")
            raise
            
    def _update_available_regions(self, data_df: pd.DataFrame) -> None:
        """Update available regions based on columns in the data.
        
        Args:
            data_df: DataFrame containing data for all regions
        """
        # Start with a clean regions list from the configuration
        configured_regions = self.config.get('regions', [])
        
        # Ensure 'timestamp' is not in the regions list
        if 'timestamp' in configured_regions:
            configured_regions.remove('timestamp')
            logger.warning("Removed 'timestamp' from configured regions list as it's a special column")
        
        # Also ensure it's not in the self.regions list
        if 'timestamp' in self.regions:
            self.regions.remove('timestamp')
            logger.warning("Removed 'timestamp' from regions list as it's a special column")
            
        available_regions = []
        
        # Iterate through configured regions
        for region, pattern in self.region_patterns.items():
            # Skip timestamp or empty regions
            if region.lower() == 'timestamp' or not region.strip():
                continue
                
            # Check for region-specific columns with more explicit matching
            # Check that the pattern is exactly at the start of the column name
            # followed by an underscore or that it's part of the column name
            region_cols = [col for col in data_df.columns 
                           if (pattern.lower() in col.lower() and 
                               (col.lower().startswith(pattern.lower() + '_') or 
                                f"_{pattern.lower()}_" in f"_{col.lower()}_"))]
            
            if region_cols:
                logger.info(f"Found data columns for region {region}: {region_cols}")
                available_regions.append(region)
            else:
                logger.warning(f"No data columns found for region {region}")
                
        # Update regions list if we found any
        if available_regions:
            self.regions = available_regions
            logger.info(f"Updated regions list: {self.regions}")
        else:
            # If no regions found in data, use configured regions but ensure they're valid
            self.regions = [r for r in configured_regions if r != 'timestamp' and r.strip()]
            logger.warning(f"No regions found in data. Using configured regions: {self.regions}")
            
    def _extract_region_data(self, data_df: pd.DataFrame, region: str) -> pd.DataFrame:
        """Extract region-specific data columns.
        
        Args:
            data_df: DataFrame containing data for all regions
            region: Region name to extract data for
            
        Returns:
            DataFrame with region-specific data
        """
        if region == 'timestamp' or not region.strip():
            logger.warning(f"Skipping invalid region: {region}")
            return pd.DataFrame()
            
        pattern = self.region_patterns.get(region, region)
        logger.info(f"Looking for columns matching pattern '{pattern}' for region '{region}'")
        
        # Find columns for this region with more explicit matching
        # Check that the pattern is exactly at the start of the column name
        # followed by an underscore or that it's part of the column name
        region_cols = [col for col in data_df.columns 
                      if (pattern.lower() in col.lower() and 
                          (col.lower().startswith(pattern.lower() + '_') or 
                           f"_{pattern.lower()}_" in f"_{col.lower()}_"))]
        region_data = pd.DataFrame(index=data_df.index)
        
        # Check for region-specific columns with different energy types
        # Try to find demand, solar, and wind columns
        demand_cols = [col for col in data_df.columns if isinstance(col, str) and col.startswith(f"{pattern}_demand")]
        solar_cols = [col for col in data_df.columns if isinstance(col, str) and col.startswith(f"{pattern}_solar")]
        wind_cols = [col for col in data_df.columns if isinstance(col, str) and col.startswith(f"{pattern}_wind")]
        
        if len(demand_cols) > 0:
            region_data['demand'] = data_df[demand_cols[0]]
        else:
            logger.warning(f"No demand column found for region {region}")
            
        if len(solar_cols) > 0:
            region_data['solar'] = data_df[solar_cols[0]]
        else:
            logger.warning(f"No solar column found for region {region}")
            region_data['solar'] = 0
            
        if len(wind_cols) > 0:
            region_data['wind'] = data_df[wind_cols[0]]
        else:
            logger.warning(f"No wind column found for region {region}")
            region_data['wind'] = 0
        
        # Check if we have at least demand data
        if 'demand' not in region_data.columns:
            logger.warning(f"No demand data for region {region}, cannot proceed")
            return pd.DataFrame()
            
        # Add other columns as needed
        region_data['net_load'] = region_data['demand'] - region_data['solar'] - region_data['wind']
                               
        logger.info(f"Extracted data for {region} with columns: {region_data.columns.tolist()}")
        return region_data
        
    def _resample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample data to the specified resolution.
        
        Args:
            data: DataFrame to resample
            
        Returns:
            Resampled DataFrame
        """
        resolution = self.time_settings.get('resolution', '30min')
        
        try:
            resampled = data.resample(resolution).mean()
            logger.info(f"Resampled data to {resolution} resolution")
            return resampled
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return data
            
    def _apply_data_quality_improvements(self, data: pd.DataFrame, region: str) -> pd.DataFrame:
        """Apply data quality improvements.
        
        Args:
            data: DataFrame to improve
            region: Region name for region-specific improvements
            
        Returns:
            Improved DataFrame
        """
        # Make a copy to avoid modifying the original
        improved = data.copy()
        
        # Get data quality settings
        quality_settings = self.config.get('data_quality', {})
        
        # Handle missing values
        if improved.isna().any().any():
            replace_value = quality_settings.get('replace_nan_value', 0)
            logger.info(f"Replacing NaN values with {replace_value}")
            improved.fillna(replace_value, inplace=True)
            
        # Apply smoothing if configured
        smoothing_window = quality_settings.get('smoothing_window')
        if smoothing_window:
            logger.info(f"Applying smoothing with window size {smoothing_window}")
            for col in improved.columns:
                improved[col] = improved[col].rolling(window=smoothing_window, 
                                                  min_periods=1).mean()
                                                  
        # Apply bounds if configured
        max_net_load = quality_settings.get('max_net_load')
        min_net_load = quality_settings.get('min_net_load')
        
        if max_net_load is not None and 'net_load' in improved.columns:
            over_max = improved['net_load'] > max_net_load
            if over_max.any():
                logger.info(f"Capping {over_max.sum()} high net_load values to {max_net_load}")
                improved.loc[over_max, 'net_load'] = max_net_load
                
        if min_net_load is not None and 'net_load' in improved.columns:
            under_min = improved['net_load'] < min_net_load
            if under_min.any():
                logger.info(f"Capping {under_min.sum()} low net_load values to {min_net_load}")
                improved.loc[under_min, 'net_load'] = min_net_load
                
        logger.info(f"Applied data quality improvements for {region}")
        return improved
        
    def _process_regional_data(self, data_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Process data for each region with progress tracking.
        
        Args:
            data_df: DataFrame containing data for all regions
            
        Returns:
            Dictionary of processed DataFrames by region
        """
        # First, update available regions based on the data
        self._update_available_regions(data_df)
        
        regional_data = {}
        
        # Ensure timestamp is not in self.regions
        if 'timestamp' in self.regions:
            self.regions.remove('timestamp')
            logger.warning("Removed 'timestamp' from regions list before processing")
        
        # Verify that regions list is not empty and doesn't contain invalid entries
        valid_regions = [r for r in self.regions if r != 'timestamp' and r.strip()]
        
        if len(valid_regions) == 0:
            logger.warning("No valid regions to process. Check configuration.")
            return {}
        
        # Debug info to help identify region data    
        logger.info(f"Processing these valid regions: {valid_regions}")
        
        # Create a progress bar for processing each region
        with tqdm(total=len(valid_regions), desc="Processing regional data", position=1, leave=False) as region_pbar:
            for region in valid_regions:
                region_pbar.set_description(f"Processing {region}")
                logger.info(f"Processing data for region: {region}")
                
                # Extract region-specific columns
                region_df = self._extract_region_data(data_df, region)
                
                if isinstance(region_df, pd.DataFrame) and region_df.empty:
                    logger.warning(f"Warning: Missing columns ['demand', 'solar', 'wind'] for region {region}")
                    region_pbar.update(1)
                    continue
                
                # Show processing steps for this region
                process_steps = ['Resampling', 'Data quality', 'Final preparation']
                with tqdm(total=len(process_steps), desc=f"  {region} steps", position=2, leave=False) as steps_pbar:
                    # Resample data to desired frequency if needed
                    steps_pbar.set_description("Resampling data")
                    target_resolution = self.time_settings.get('resolution')
                    current_freq = getattr(data_df.index, 'freq', None)
                    
                    if target_resolution is not None and str(target_resolution) != str(current_freq):
                        logger.info(f"Resampling {region} data to {target_resolution} resolution (current: {current_freq})")
                        region_df = self._resample_data(region_df)
                    steps_pbar.update(1)
                    
                    # Apply any data quality improvements
                    steps_pbar.set_description("Improving data quality")
                    region_df = self._apply_data_quality_improvements(region_df, region)
                    steps_pbar.update(1)
                    
                    # Final preparation
                    steps_pbar.set_description("Final preparation")
                    # Additional processing could go here
                    steps_pbar.update(1)
                
                # Add processed data to output if we have valid data    
                if isinstance(region_df, pd.DataFrame) and not region_df.empty and len(region_df.columns) > 0:
                    regional_data[region] = region_df
                    logger.info(f"Added {region} to output with {len(region_df)} time periods and columns: {region_df.columns.tolist()}")
                else:
                    logger.warning(f"No usable data for {region}")
                region_pbar.update(1)
                
        logger.info(f"Processed data for {len(regional_data)} regions")
        if len(regional_data) == 0:
            logger.warning("No regions were successfully processed")
            
        return regional_data

    def _apply_data_quality_improvements(self, df: pd.DataFrame, region: str) -> pd.DataFrame:
        """Apply data quality improvements to the regional data with progress tracking.
        
        Args:
            df: Regional DataFrame
            region: Region name
            
        Returns:
            Improved DataFrame
        """
        if df.empty:
            logger.warning(f"Warning: Empty dataset for region {region}")
            return df
            
        # Make a copy to avoid modifying the original
        df_improved = df.copy()
        
        # Create a progress bar for data quality improvement steps
        dq_steps = ['Handling NaN values', 'Outlier detection', 'Smoothing', 'Value constraints']
        with tqdm(total=len(dq_steps), desc=f"    {region} data quality", position=2, leave=False) as dq_pbar:
            # Handle NaN values
            dq_pbar.set_description("Handling NaN values")
            for col in df_improved.columns:
                if df_improved[col].isna().any():
                    nan_count = df_improved[col].isna().sum()
                    # Get replacement value from config or use column median
                    replace_val = self.config.get("data_quality", {}).get("replace_nan_value", df_improved[col].median())
                    df_improved[col].fillna(replace_val, inplace=True)
                    logger.info(f"Filled {nan_count} NaN values in {region}.{col}")
            dq_pbar.update(1)
            
            # Handle outliers
            dq_pbar.set_description("Outlier detection")
            # For simplicity, capping values at min/max thresholds from config
            if "data_quality" in self.config:
                max_net_load = self.config["data_quality"].get("max_net_load", 30000)
                min_net_load = self.config["data_quality"].get("min_net_load", -10000)
                
                outlier_counts = {}
                if "demand" in df_improved.columns:
                    outliers = ((df_improved["demand"] < 0) | (df_improved["demand"] > max_net_load)).sum()
                    df_improved["demand"] = df_improved["demand"].clip(lower=0, upper=max_net_load)
                    outlier_counts["demand"] = outliers
                    
                if "solar" in df_improved.columns:
                    outliers = ((df_improved["solar"] < 0) | (df_improved["solar"] > max_net_load)).sum()
                    df_improved["solar"] = df_improved["solar"].clip(lower=0, upper=max_net_load)
                    outlier_counts["solar"] = outliers
                    
                if "wind" in df_improved.columns:
                    outliers = ((df_improved["wind"] < 0) | (df_improved["wind"] > max_net_load)).sum()
                    df_improved["wind"] = df_improved["wind"].clip(lower=0, upper=max_net_load)
                    outlier_counts["wind"] = outliers
                
                for col, count in outlier_counts.items():
                    if count > 0:
                        logger.info(f"Fixed {count} outliers in {region}.{col}")
            dq_pbar.update(1)
            
            # Apply smoothing if configured
            dq_pbar.set_description("Smoothing")
            if "data_quality" in self.config and "smoothing_window" in self.config["data_quality"]:
                window = self.config["data_quality"]["smoothing_window"]
                if window > 1:
                    for col in ["demand", "solar", "wind"]:
                        if col in df_improved.columns:
                            orig_values = df_improved[col].copy()
                            df_improved[col] = df_improved[col].rolling(window=window, center=True).mean().fillna(df_improved[col])
                            # Calculate how many values were significantly changed by smoothing
                            threshold = 0.05  # 5% difference threshold
                            diff_pct = abs(df_improved[col] - orig_values) / (orig_values + 1e-10)  # Avoid division by zero
                            changed = (diff_pct > threshold).sum()
                            if changed > 0:
                                logger.info(f"Smoothed {changed} values in {region}.{col} using window={window}")
            dq_pbar.update(1)
            
            # Value constraints
            dq_pbar.set_description("Value constraints")
            # Additional value constraints could go here
            dq_pbar.update(1)
        
        return df_improved

    def preprocess_data(self, config=None) -> Dict[str, pd.DataFrame]:
        """Preprocess raw data for each region with data quality handling.
        
        Args:
            config: Configuration settings
            
        Returns:
            Dictionary with region names as keys and preprocessed DataFrames as values
        """
        print("Preprocessing data...")
        start_time = datetime.now()
        print(f"Started at: {start_time}")
        
        # Update config if provided
        if config:
            self.config = config
            # Update time settings if they've changed
            if "time_settings" in config:
                self.time_settings.update(config["time_settings"])
                # Convert date strings to datetime objects
                if isinstance(self.time_settings['start_date'], str):
                    self.time_settings['start_date'] = pd.to_datetime(self.time_settings['start_date'])
                if isinstance(self.time_settings['end_date'], str):
                    self.time_settings['end_date'] = pd.to_datetime(self.time_settings['end_date'])
                print(f"Updated time settings: {self.time_settings}")
            
            # Update regions if they've changed
            if "regions" in config:
                self.regions = config["regions"]
                print(f"Updated regions: {self.regions}")
        
        # Load and pre-filter the data
        regional_data = self.load_raw_data()
        
        # Log regions with available data
        available_regions = list(regional_data.keys())
        print(f"Available regions in data: {available_regions}")
        print(f"Configured regions to process: {self.regions}")
        
        # Check if we have all required regions
        missing_regions = [region for region in self.regions if region not in regional_data]
        if missing_regions:
            print(f"Warning: The following configured regions have no data: {missing_regions}")
        
        processed_data = {}
        
        # Process each region's data
        for region in self.regions:
            if region not in regional_data:
                print(f"Warning: No data found for region: {region}")
                continue
            
            region_start_time = datetime.now()
            print(f"Processing {region}, started at {region_start_time}")
            
            # Process the region's data
            df = self._process_regional_data(regional_data[region], self.config)
            
            if not df.empty:  # Only add non-empty dataframes
                processed_data[region] = df
                
                # Store the actual region name in the DataFrame attributes for reference
                df.attrs['actual_region_name'] = region
                
                # Store metadata about the processing
                df.attrs['processing_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df.attrs['time_periods'] = len(df)
                df.attrs['start_time'] = df['timestamp'].min() if 'timestamp' in df.columns else None
                df.attrs['end_time'] = df['timestamp'].max() if 'timestamp' in df.columns else None
                
                region_end_time = datetime.now()
                processing_duration = region_end_time - region_start_time
                print(f"Finished processing {region} in {processing_duration}")
                print(f"Processed {len(df)} time periods from {df.attrs.get('start_time')} to {df.attrs.get('end_time')}")
            else:
                print(f"Warning: No valid data after processing for region: {region}")
                
                # Store the time series data in DataFrame attributes for reference
                if 'time_series' not in df.attrs:
                    df.attrs['time_series'] = {}
        
        return processed_data
    
    def _process_regional_data(self, data: pd.DataFrame, config=None) -> pd.DataFrame:
        """Process data for a specific region with improved data quality handling.
        
        Args:
            data: Regional data
            config: Configuration settings including data quality parameters
            
        Returns:
            Processed data for the region with clean, validated data
        """
        region_name = getattr(data.index, 'name', 'Unknown')
        print(f"Processing data for region: {region_name}")
        
        # Set default data quality parameters if config is not provided
        data_quality_params = {
            'replace_nan_value': 1000.0,
            'max_net_load': 30000.0,
            'min_net_load': -10000.0,
            'smoothing_window': 3
        }
        
        # Override with config values if provided
        if config and 'data_quality' in config:
            data_quality_params.update(config['data_quality'])
            
        # Check if we have empty data
        if data.empty:
            print(f"Warning: Empty dataset for region {region_name}")
            return pd.DataFrame()
        
        # Ensure data has the expected columns
        expected_columns = ['demand', 'solar', 'wind']
        if not all(col in data.columns for col in expected_columns):
            missing = [col for col in expected_columns if col not in data.columns]
            print(f"Warning: Missing columns {missing} for region {region_name}")
            return pd.DataFrame()
        
        # Examine the original dataset
        original_len = len(data)
        print(f"Original dataset for {region_name}: {original_len} time periods from {data.index[0]} to {data.index[-1]}")
        
        # Check for missing time periods or gaps
        expected_freq = self.time_settings.get('resolution', '30min')
        if not data.index.is_monotonic_increasing:
            print(f"Warning: Index is not monotonic increasing for {region_name}, sorting")
            data = data.sort_index()
        
        # Check for gaps and fill them if needed
        if not data.index.inferred_freq:
            full_index = pd.date_range(
                start=data.index.min(),
                end=data.index.max(),
                freq=expected_freq
            )
            
            # Count missing periods
            missing_periods = len(full_index) - len(data.index)
            if missing_periods > 0:
                print(f"Found {missing_periods} missing time periods for {region_name}, reindexing")
                data = data.reindex(full_index)
        
        # Process generation data with improved data quality handling
        demand_data = data['demand'].astype(float)
        solar_data = data['solar'].astype(float)
        wind_data = data['wind'].astype(float)
        
        # Count NaN values before processing
        nan_counts = {
            'demand': demand_data.isna().sum(),
            'solar': solar_data.isna().sum(),
            'wind': wind_data.isna().sum()
        }
        
        if sum(nan_counts.values()) > 0:
            print(f"NaN values before processing: {nan_counts}")
        
        # Replace NaN/Inf values with more sophisticated methods
        for series_name, series in [('demand', demand_data), ('solar', solar_data), ('wind', wind_data)]:
            # Replace infinities with NaN first
            series = series.replace([np.inf, -np.inf], np.nan)
            
            # Calculate daily and weekly patterns for more accurate filling
            # This is especially important for demand data which follows patterns
            if series_name == 'demand' and len(series) > 48*7:  # At least a week of data
                # Create a copy with timestamp index for pattern extraction
                temp_series = series.copy()
                if not hasattr(temp_series, 'index') or not hasattr(temp_series.index, 'hour'):
                    temp_series.index = data.index
                
                # Group by hour of day for pattern
                hourly_pattern = temp_series.groupby([temp_series.index.dayofweek, temp_series.index.hour]).mean()
                
                # Fill NaN with the pattern where possible
                for idx, val in series[series.isna()].items():
                    if hasattr(idx, 'dayofweek') and hasattr(idx, 'hour'):
                        pattern_val = hourly_pattern.get((idx.dayofweek, idx.hour))
                        if not pd.isna(pattern_val):
                            series.loc[idx] = pattern_val
            
            # Fill remaining NaNs with standard methods
            series = series.ffill()  # Forward fill first (use previous valid value)
            series = series.bfill()  # Backward fill if still NaN (use next valid value)
            
            # If any NaNs still remain, use the replace_nan_value from config
            remaining_nans = series.isna().sum()
            if remaining_nans > 0:
                print(f"Filling {remaining_nans} remaining NaNs in {series_name} with default value")
                series = series.fillna(data_quality_params['replace_nan_value'])
            
            # Update the original series
            if series_name == 'demand':
                demand_data = series
            elif series_name == 'solar':
                solar_data = series
            elif series_name == 'wind':
                wind_data = series
        
        # Apply smoothing to reduce data spikes with more sophisticated methods
        if data_quality_params['smoothing_window'] > 1:
            # Detect outliers before smoothing (values far from moving median)
            outlier_detection_window = min(49, len(demand_data) // 10)  # Use reasonable window size
            
            for series_name, series in [('demand', demand_data), ('solar', solar_data), ('wind', wind_data)]:
                # Calculate rolling median and standard deviation
                rolling_median = series.rolling(window=outlier_detection_window, center=True, min_periods=1).median()
                rolling_std = series.rolling(window=outlier_detection_window, center=True, min_periods=1).std()
                
                # Identify outliers (more than 3 standard deviations from the median)
                outliers = (series - rolling_median).abs() > 3 * rolling_std
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    print(f"Detected {outlier_count} outliers in {series_name} data")
                    # Replace outliers with the median value
                    series[outliers] = rolling_median[outliers]
                
                # Apply moving average smoothing
                series = series.rolling(
                    window=data_quality_params['smoothing_window'],
                    min_periods=1,
                    center=True
                ).mean()
                
                # Update the original series
                if series_name == 'demand':
                    demand_data = series
                elif series_name == 'solar':
                    solar_data = series
                elif series_name == 'wind':
                    wind_data = series
        
        # Ensure values are within reasonable bounds
        # Solar and wind should be >= 0
        solar_data = solar_data.clip(lower=0)
        wind_data = wind_data.clip(lower=0)
        
        # Create result DataFrame
        df = pd.DataFrame({
            'demand': demand_data,
            'solar': solar_data,
            'wind': wind_data
        }, index=data.index)
        
        # Calculate net load (what must be met by dispatchable resources)
        df['net_load'] = df['demand'] - df['solar'] - df['wind']
        
        # Clip net load to reasonable bounds from config
        too_high = (df['net_load'] > data_quality_params['max_net_load']).sum()
        too_low = (df['net_load'] < data_quality_params['min_net_load']).sum()
        
        if too_high > 0 or too_low > 0:
            print(f"Clipping net load: {too_high} values too high, {too_low} values too low")
            df['net_load'] = df['net_load'].clip(
                lower=data_quality_params['min_net_load'],
                upper=data_quality_params['max_net_load']
            )
        
        # Create numeric time indices that the optimizer will use
        df['time_index'] = range(len(df))
        
        # Create a dictionary mapping time periods to net load values for the optimizer
        time_series_data = {}
        for t in range(len(df)):
            # For each time point, store the relevant data
            row = df.iloc[t]
            time_series_data[t] = row['net_load']
        
        # Reset the index to time_index for the optimizer
        df = df.reset_index().rename(columns={'index': 'timestamp'}).set_index('time_index')
        
        # Store the time series data in DataFrame attributes
        df.attrs['time_series'] = time_series_data
        df.attrs['region_name'] = region_name
        
        print(f"Processed {len(df)} time periods for {region_name}")
        return df
    
    def save_processed_data(self, processed_data: Dict[str, pd.DataFrame]) -> None:
        """Save processed data to CSV files.
        
        Args:
            processed_data: Dictionary of processed DataFrames by region
        """
        output_dir = self.data_dir / 'processed'
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual region files
        for region, df in processed_data.items():
            # Reset the index to include the timestamp column in the CSV
            if 'timestamp' in df.columns:
                output_df = df
            else:
                output_df = df.reset_index()
                
            # Normalize the region name for the filename
            normalized_region = region.lower().replace(' ', '_').replace('\'', '')
            output_file = output_dir / f"{normalized_region}_processed_{timestamp}.csv"
            
            try:
                output_df.to_csv(output_file)
                print(f"Saved processed data for {region} to {output_file}")
            except Exception as e:
                print(f"Error saving data for {region}: {str(e)}")
        
        # Save a combined file with all regions
        try:
            combined_data = {}
            for region, df in processed_data.items():
                # Create a multi-index DataFrame with region and time
                region_data = df.copy()
                if 'timestamp' not in region_data.columns and hasattr(region_data, 'index'):
                    region_data = region_data.reset_index()
                
                # Add region column
                region_data['region'] = region
                combined_data[region] = region_data
            
            if combined_data:
                # Concatenate all region DataFrames
                combined_df = pd.concat(combined_data.values())
                combined_file = output_dir / f"all_regions_processed_{timestamp}.csv"
                combined_df.to_csv(combined_file, index=False)
                print(f"Saved combined processed data to {combined_file}")
        except Exception as e:
            print(f"Error saving combined data: {str(e)}")
