import yaml
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import time
import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from data.data_loader import DataLoader
from model.optimizer import RegionalFlexOptimizer
from utils.visualization import (
    plot_regional_dispatch,
    plot_inter_regional_exchanges,
    plot_storage_levels,
    analyze_results,
    plot_seasonal_patterns,
    plot_yearly_overview
)

# Check if advanced visualization module exists, import if available
try:
    from utils.advanced_visualization import (
        plot_regional_comparison,
        plot_energy_balance,
        generate_comprehensive_report
    )
    ADVANCED_VIZ_AVAILABLE = True
except ImportError:
    ADVANCED_VIZ_AVAILABLE = False
    logging.warning("Advanced visualization module not available.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_memory_usage() -> float:
    """Get current memory usage of the process in MB.
    
    Returns:
        Memory usage in megabytes
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / 1024 / 1024  # Convert bytes to MB
    except ImportError:
        # If psutil is not available, return a placeholder value
        return 0.0

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Record start time for performance tracking
    total_start_time = time.time()
    memory_check_start = get_memory_usage()
    
    # Initialize timing variables
    data_start_time = time.time()
    data_end_time = time.time()
    viz_start = time.time()
    viz_end_time = time.time()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = load_config(config_path)
    
    # Set up paths
    data_dir = Path(__file__).parent.parent / config["paths"]["data_dir"]
    output_dir = Path(__file__).parent.parent / config["paths"]["output_dir"]
    output_dir.mkdir(exist_ok=True)
    
    # Use timestamp for unique run identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a specific results directory for this run
    results_dir = output_dir / f"results_{timestamp}"
    results_dir.mkdir(exist_ok=True)
    
    # Configure logging with timestamp and multiprocessing info
    log_filename = results_dir / f"simulation_{timestamp}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log system information 
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Initial memory usage: {memory_check_start:.2f} MB")
    
    # Log configuration for reproducibility
    logger.info(f"Starting regional flexibility simulation run {timestamp}")
    logger.info(f"Configuration: {config}")
    
    # Extract time settings for simulation
    time_settings = config.get("time_settings", {
        "resolution": "30min",
        "start_date": "2022-01-01", 
        "end_date": "2022-12-31"
    })
    
    logger.info(f"Time settings: {time_settings}")
    logger.info(f"Simulating from {time_settings['start_date']} to {time_settings['end_date']} with {time_settings['resolution']} resolution")
    
    # Save configuration for this run
    with open(results_dir / "run_config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Define main simulation steps for overall progress tracking
    simulation_steps = [
        'Loading data',
        'Building optimization model',
        'Solving optimization',
        'Processing results',
        'Generating visualizations',
        'Saving results'
    ]
    
    # Create overall progress bar for the simulation
    with tqdm(total=len(simulation_steps), desc="Regional Flexibility Simulation", position=0) as main_pbar:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data_start_time = time.time()
        
        try:
            # Initialize data loader
            logger.info("Initializing data loader...")
            data_loader = DataLoader(data_dir, config)
            
            # Load and process data
            main_pbar.set_description("Loading data")
            logger.info("Loading data...")
            data_load_start = time.time()
            regional_data = data_loader.load_data()
            data_load_time = time.time() - data_load_start
            
            logger.info(f"Data loading completed in {data_load_time:.2f} seconds")
            memory_after_data = get_memory_usage()
            logger.info(f"Memory usage after data loading: {memory_after_data:.2f} MB")
            main_pbar.update(1)
            logger.info(f"Memory increase during data loading: {memory_after_data - memory_check_start:.2f} MB")
            data_end_time = time.time()
        
        except Exception as e:
            logger.error(f"Error during data loading and preprocessing: {str(e)}")
            raise
        
        # Initialize and build optimization model
        logger.info("Building optimization model...")
        
        # Check if regional_data is a dictionary or a DataFrame
        if isinstance(regional_data, pd.DataFrame):
            logger.warning("Received a DataFrame instead of regional data dictionary. Reprocessing data.")
            # Process regional data directly from the loader
            data_loader._update_available_regions(regional_data)
            regional_data = data_loader._process_regional_data(regional_data)
            
        # Ensure we have valid regions
        if isinstance(regional_data, dict):
            if len(regional_data) == 0:
                logger.error("No valid regional data available for optimization")
                raise ValueError("No valid regional data to process")
        else:
            logger.error("Regional data is not a dictionary")
            raise ValueError("Regional data must be a dictionary of DataFrames by region")
            
        # Ensure timestamp is not in regions list
        if 'timestamp' in regional_data:
            logger.warning("Removing 'timestamp' from regions as it is not a valid region")
            regional_data.pop('timestamp')
            
        logger.info(f"Processing the following regions: {list(regional_data.keys())}")
        
        # Initialize the optimizer with the config
        optimizer = RegionalFlexOptimizer(config)
        
        # Ensure regions from config match those in the data
        if set(optimizer.regions) != set(regional_data.keys()):
            logger.warning(f"Mismatch between configured regions {optimizer.regions} and data regions {list(regional_data.keys())}. Using data regions.")
            optimizer.regions = list(regional_data.keys())
        
        # Build and solve model
        try:
            optimizer.build_model(regional_data)
            # Solve the optimization model
            main_pbar.set_description("Solving optimization model")
            logger.info("Solving optimization model...")
            solve_start = time.time()
            results = optimizer.solve_model()
            solve_time = time.time() - solve_start
            
            logger.info(f"Optimization completed in {solve_time:.2f} seconds")
            logger.info(f"Memory usage after optimization: {get_memory_usage():.2f} MB")
            main_pbar.update(1)

            # Check if optimization was successful
            if results is None:
                logger.error("Optimization failed. No results to analyze.")
                main_pbar.close()
                return results_dir
                
        except Exception as e:
            import traceback
            logger.error(f"Error during model building or solving: {e}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            main_pbar.close()
            return results_dir
            
            # Process results
            main_pbar.set_description("Processing results")
            logger.info("Processing results...")
            results_start = time.time()
            
            # Add regional_data to results for visualization
            results['input_data'] = {}
            for region, data in regional_data.items():
                results['input_data'][region] = data.to_dict('list')
                
            results_time = time.time() - results_start
            logger.info(f"Results processing completed in {results_time:.2f} seconds")
            main_pbar.update(1)
            
            # Generate visualizations
            main_pbar.set_description("Generating visualizations")
            logger.info("Generating visualizations...")
            viz_start = time.time()  # Update the viz_start time
            
            # Create visualization directories with more detailed structure
            viz_dir = results_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Regional directories
            regional_viz_dir = viz_dir / "regional"
            regional_viz_dir.mkdir(exist_ok=True)
            
            # Seasonal analysis directories
            seasonal_viz_dir = viz_dir / "seasonal"
            seasonal_viz_dir.mkdir(exist_ok=True)
            
            # Create season-specific subdirectories
            for season in ["winter", "spring", "summer", "autumn"]:
                (seasonal_viz_dir / season).mkdir(exist_ok=True)
            
            # Exchanges directory
            exchange_viz_dir = viz_dir / "exchanges"
            exchange_viz_dir.mkdir(exist_ok=True)
            
            # Monthly analysis directory
            monthly_viz_dir = viz_dir / "monthly"
            monthly_viz_dir.mkdir(exist_ok=True)
            
            # Storage analysis directory
            storage_viz_dir = viz_dir / "storage"
            storage_viz_dir.mkdir(exist_ok=True)
            
            # Advanced analysis directory (if available)
            advanced_viz_dir = viz_dir / "advanced"
            advanced_viz_dir.mkdir(exist_ok=True)
            
            # Define visualization steps
            viz_steps = [
                'Regional Dispatch', 
                'Inter-regional Exchanges', 
                'Storage Levels', 
                'Seasonal Patterns', 
                'Yearly Overview'
            ]
            
            # Add advanced visualization steps if available
            if ADVANCED_VIZ_AVAILABLE:
                viz_steps.extend(['Regional Comparison', 'Energy Balance', 'Comprehensive Report'])
            
            # Create progress bar for visualization steps
            with tqdm(total=len(viz_steps), desc="Visualization Progress", position=1, leave=False) as viz_pbar:
                # Generate regional dispatch visualizations
                viz_pbar.set_description("Regional Dispatch")
                logger.info("Generating regional dispatch visualizations...")
                
                try:
                    for region in optimizer.regions:
                        normalized_region = region.lower().replace(' ', '_')
                        logger.info(f"Generating dispatch visualization for {region}")
                        plot_regional_dispatch(
                            results,
                            region,
                            save_path=regional_viz_dir / f"dispatch_{normalized_region}.png"
                        )
                    viz_pbar.update(1)
                except Exception as e:
                    logger.error(f"Error generating dispatch visualizations: {str(e)}")
                
                # Generate inter-regional exchange visualization
                viz_pbar.set_description("Inter-regional Exchanges")
                logger.info("Generating inter-regional exchange visualization")
                
                try:
                    plot_inter_regional_exchanges(
                        results,
                        optimizer.regions,
                        save_path=exchange_viz_dir / "inter_regional_exchanges.png"
                    )
                    viz_pbar.update(1)
                except Exception as e:
                    logger.error(f"Error generating exchange visualization: {str(e)}")
                
                # Generate storage visualization
                viz_pbar.set_description("Storage Levels")
                logger.info("Generating storage visualization")
                
                try:
                    plot_storage_levels(
                        results,
                        optimizer.regions,
                        save_path=storage_viz_dir / "storage_levels.png"
                    )
                    viz_pbar.update(1)
                except Exception as e:
                    logger.error(f"Error generating storage visualization: {str(e)}")
                
                # Generate seasonal patterns
                viz_pbar.set_description("Seasonal Patterns")
                logger.info("Generating seasonal patterns...")
                
                try:
                    plot_seasonal_patterns(
                        results,
                        optimizer.regions,
                        save_dir=str(seasonal_viz_dir)
                    )
                    viz_pbar.update(1)
                except Exception as e:
                    logger.error(f"Error generating seasonal patterns: {str(e)}")
                
                # Generate yearly overview
                viz_pbar.set_description("Yearly Overview")
                logger.info("Generating yearly overview...")
                
                try:
                    plot_yearly_overview(
                        results,
                        optimizer.regions,
                        save_dir=str(viz_dir)
                    )
                    viz_pbar.update(1)
                except Exception as e:
                    logger.error(f"Error generating yearly overview: {str(e)}")
                
                # Generate result analysis
                viz_pbar.set_description("Result Analysis")
                logger.info("Analyzing results...")
                
                try:
                    analysis = analyze_results(results, optimizer.regions)
                    analysis_file = results_dir / "analysis_summary.json"
                    with open(analysis_file, 'w') as f:
                        json.dump(analysis, f, indent=2)
                    viz_pbar.update(1)
                except Exception as e:
                    logger.error(f"Error analyzing results: {str(e)}")
                
                # Advanced visualizations if module is available
                if ADVANCED_VIZ_AVAILABLE:
                    # Regional comparison
                    viz_pbar.set_description("Regional Comparison")
                    logger.info("Generating regional comparison...")
                    
                    try:
                        plot_regional_comparison(
                            results,
                            optimizer.regions,
                            save_dir=str(regional_viz_dir)
                        )
                        viz_pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error generating regional comparison: {str(e)}")
                    
                    # Energy balance
                    viz_pbar.set_description("Energy Balance")
                    logger.info("Generating energy balance charts...")
                    
                    try:
                        plot_energy_balance(
                            results,
                            optimizer.regions,
                            save_dir=str(advanced_viz_dir)
                        )
                        viz_pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error generating energy balance charts: {str(e)}")
                    
                    # Comprehensive report
                    viz_pbar.set_description("Comprehensive Report")
                    logger.info("Generating comprehensive report...")
                    
                    try:
                        report_path = results_dir / "comprehensive_report.html"
                        generate_comprehensive_report(
                            results, 
                            optimizer.regions,
                            analysis,
                            str(report_path)
                        )
                        viz_pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error generating comprehensive report: {str(e)}")
            
            viz_time = time.time() - viz_start
            logger.info(f"Visualization generation completed in {viz_time:.2f} seconds")
            main_pbar.update(1)
    
    # Performance tracking for visualization
    viz_end_time = time.time()
    logger.info(f"Visualization generation completed in {viz_end_time - viz_start:.2f} seconds")
    
    # Record final memory usage
    memory_final = get_memory_usage()
    logger.info(f"Final memory usage: {memory_final:.2f} MB")
    logger.info(f"Total memory increase: {memory_final - memory_check_start:.2f} MB")
    
    # Create performance summary
    performance_data = {
        "timing": {
            "total_execution_seconds": time.time() - total_start_time,
            "data_processing_seconds": data_end_time - data_start_time,
            "visualization_seconds": viz_end_time - viz_start
        },
        "memory": {
            "initial_mb": memory_check_start,
            "after_data_mb": memory_after_data,
            "final_mb": memory_final
        },
        "data_size": {
            "regions": len(optimizer.regions),
            "time_periods": len(results.get("time_periods", []) or results.get("timesteps", []))
        }
    }
    
    # Save performance data
    with open(results_dir / "performance_metrics.json", "w") as f:
        json.dump(performance_data, f, indent=2)
    
    # End of script timing
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logger.info(f"Total execution time: {total_duration:.2f} seconds")
    logger.info("Regional flexibility simulation completed successfully")
    
    # Create a summary file with key results
    with open(results_dir / "simulation_summary.txt", "w") as f:
        f.write(f"Regional Flexibility Simulation Summary\n")
        f.write(f"Run timestamp: {timestamp}\n")
        f.write(f"Total execution time: {total_duration:.2f} seconds\n")
        f.write(f"Number of regions: {len(optimizer.regions)}\n")
        f.write(f"Regions: {', '.join(optimizer.regions)}\n")
        f.write(f"Time periods: {len(results.get('time_periods', []) or results.get('timesteps', []))}\n")
        f.write(f"Regions analyzed: {', '.join(optimizer.regions)}\n")
        f.write(f"Time period: {time_settings['start_date']} to {time_settings['end_date']}\n")
        f.write(f"Resolution: {time_settings['resolution']}\n")
        f.write(f"Optimization status: {results.get('status', 'Unknown')}\n")
        objective_value = results.get('objective_value')
        if objective_value is not None:
            f.write(f"Objective value: {objective_value:.2f} EUR\n")
        else:
            f.write(f"Objective value: 0.00 EUR\n")
        f.write(f"Execution time: {total_duration:.2f} seconds\n")
        f.write(f"\nResults directory: {results_dir}\n")
    
    return results_dir

def save_raw_results(results, output_dir):
    """Save raw optimization results to CSV files with improved handling for full-year half-hourly data.
    
    Args:
        results: Dictionary containing optimization results
        output_dir: Directory to save results
    """
    logger.info("Saving raw optimization results...")
    save_start_time = time.time()
    
    # Create raw results directory with subdirectories
    raw_dir = output_dir / "raw_data"
    raw_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different result types
    regional_dir = raw_dir / "regional"
    regional_dir.mkdir(exist_ok=True)
    
    exchange_dir = raw_dir / "exchanges"
    exchange_dir.mkdir(exist_ok=True)
    
    storage_dir = raw_dir / "storage"
    storage_dir.mkdir(exist_ok=True)
    
    demand_response_dir = raw_dir / "demand_response"
    demand_response_dir.mkdir(exist_ok=True)
    
    # Save timestep information if available
    if 'timesteps' in results:
        try:
            # Convert timestep data to a dataframe
            timesteps_df = pd.DataFrame({
                'time_index': range(len(results['timesteps'])),
                'timestamp': results['timesteps']
            })
            timesteps_file = raw_dir / "timesteps.csv"
            timesteps_df.to_csv(timesteps_file, index=False)
            logger.info(f"Saved {len(timesteps_df)} timesteps to {timesteps_file}")
        except Exception as e:
            logger.error(f"Error saving timestep data: {str(e)}")
    
    # Save regional results with chunking for large datasets
    if 'regional_results' in results:
        regions = list(results['regional_results'].keys())
        logger.info(f"Saving results for {len(regions)} regions: {', '.join(regions)}")
        
        for region, region_data in results['regional_results'].items():
            try:
                # Normalize region name for filenames
                normalized_region = region.lower().replace(' ', '_').replace("'", "").replace("-", "_")
                
                # Create directory for this region
                region_subdir = regional_dir / normalized_region
                region_subdir.mkdir(exist_ok=True)
                
                # Convert to DataFrame for easier manipulation
                if isinstance(region_data, dict):
                    region_df = pd.DataFrame(region_data)
                else:
                    region_df = pd.DataFrame(region_data)
                
                # Handle potential memory issues with large dataframes
                num_rows = len(region_df)
                
                if num_rows > 10000:  # For very large datasets, split into multiple files
                    chunk_size = 4380  # Approximately 3 months of half-hourly data
                    num_chunks = (num_rows + chunk_size - 1) // chunk_size
                    
                    logger.info(f"Splitting {region} data into {num_chunks} files of {chunk_size} rows each")
                    
                    for i in range(num_chunks):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, num_rows)
                        
                        chunk_df = region_df.iloc[start_idx:end_idx]
                        chunk_file = region_subdir / f"{normalized_region}_chunk_{i+1}_of_{num_chunks}.csv"
                        chunk_df.to_csv(chunk_file)
                        
                    # Also save a metadata file
                    with open(region_subdir / f"{normalized_region}_metadata.json", 'w') as f:
                        json.dump({
                            'region': region,
                            'total_rows': num_rows,
                            'chunks': num_chunks,
                            'chunk_size': chunk_size,
                            'columns': list(region_df.columns)
                        }, f, indent=2)
                else:
                    # Save as a single file
                    region_file = region_subdir / f"{normalized_region}_complete.csv"
                    region_df.to_csv(region_file)
                
                logger.info(f"Saved {num_rows} rows of data for region {region}")
            except Exception as e:
                logger.error(f"Error saving results for region {region}: {str(e)}")
    
    # Save exchange results with improved handling
    if 'exchange' in results:
        try:
            # Check if exchange data is nested or flat
            if isinstance(results['exchange'], dict):
                # Handle different possible structures
                if all(isinstance(v, dict) for v in results['exchange'].values()):
                    # Nested structure - exchanges by region pairs
                    for from_region, to_regions in results['exchange'].items():
                        for to_region, data in to_regions.items():
                            if from_region != to_region:  # Skip self-exchanges
                                exchange_name = f"{from_region.lower().replace(' ', '_')}_to_{to_region.lower().replace(' ', '_')}"
                                exchange_df = pd.DataFrame(data)
                                exchange_file = exchange_dir / f"{exchange_name}.csv"
                                exchange_df.to_csv(exchange_file)
                                logger.info(f"Saved exchange data from {from_region} to {to_region}")
                else:
                    # Flat structure
                    exchange_df = pd.DataFrame(results['exchange'])
                    exchange_file = exchange_dir / "all_exchanges.csv"
                    exchange_df.to_csv(exchange_file)
                    logger.info(f"Saved combined exchange data with {len(exchange_df)} rows")
            else:
                # Handle any other format
                exchange_df = pd.DataFrame(results['exchange'])
                exchange_file = exchange_dir / "exchanges.csv"
                exchange_df.to_csv(exchange_file)
                logger.info(f"Saved exchange data with {len(exchange_df)} rows")
        except Exception as e:
            logger.error(f"Error saving exchange results: {str(e)}")
    
    # Save storage results if available
    if 'storage' in results:
        try:
            if isinstance(results['storage'], dict):
                for region, storage_data in results['storage'].items():
                    normalized_region = region.lower().replace(' ', '_').replace("'", "").replace("-", "_")
                    storage_df = pd.DataFrame(storage_data)
                    storage_file = storage_dir / f"{normalized_region}_storage.csv"
                    storage_df.to_csv(storage_file)
                    logger.info(f"Saved storage data for {region} with {len(storage_df)} rows")
            else:
                storage_df = pd.DataFrame(results['storage'])
                storage_file = storage_dir / "all_storage.csv"
                storage_df.to_csv(storage_file)
                logger.info(f"Saved combined storage data with {len(storage_df)} rows")
        except Exception as e:
            logger.error(f"Error saving storage results: {str(e)}")
    
    # Save demand response results if available
    if 'demand_response' in results:
        try:
            if isinstance(results['demand_response'], dict):
                for region, dr_data in results['demand_response'].items():
                    normalized_region = region.lower().replace(' ', '_').replace("'", "").replace("-", "_")
                    dr_df = pd.DataFrame(dr_data)
                    dr_file = demand_response_dir / f"{normalized_region}_demand_response.csv"
                    dr_df.to_csv(dr_file)
                    logger.info(f"Saved demand response data for {region} with {len(dr_df)} rows")
            else:
                dr_df = pd.DataFrame(results['demand_response'])
                dr_file = demand_response_dir / "all_demand_response.csv"
                dr_df.to_csv(dr_file)
                logger.info(f"Saved combined demand response data with {len(dr_df)} rows")
        except Exception as e:
            logger.error(f"Error saving demand response results: {str(e)}")
    
    # Save any additional result components
    for key, data in results.items():
        if key not in ['regional_results', 'exchange', 'storage', 'demand_response', 'timesteps']:
            try:
                # Try to convert to DataFrame if possible
                if isinstance(data, dict) or isinstance(data, list):
                    extra_df = pd.DataFrame(data)
                    extra_file = raw_dir / f"{key}.csv"
                    extra_df.to_csv(extra_file)
                    logger.info(f"Saved additional data for {key}")
            except Exception as e:
                logger.error(f"Error saving additional results for {key}: {str(e)}")
    
    # Create a summary file of all saved data
    with open(raw_dir / "data_summary.txt", "w") as f:
        f.write(f"Regional Flexibility Optimization Results Summary\n")
        f.write(f"===========================================\n\n")
        f.write(f"Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Regional data:\n")
        if 'regional_results' in results:
            for region in results['regional_results'].keys():
                f.write(f"  - {region}\n")
        else:
            f.write(f"  No regional data available\n")
        
        f.write(f"\nExchange data:\n")
        if 'exchange' in results:
            f.write(f"  Exchange data available\n")
        else:
            f.write(f"  No exchange data available\n")
        
        # Add other components
        for component in ['storage', 'demand_response']:
            f.write(f"\n{component.replace('_', ' ').title()} data:\n")
            if component in results:
                f.write(f"  {component.replace('_', ' ').title()} data available\n")
            else:
                f.write(f"  No {component.replace('_', ' ').title()} data available\n")
    
    save_end_time = time.time()
    logger.info(f"Raw results saved to {raw_dir} in {save_end_time - save_start_time:.2f} seconds")

if __name__ == "__main__":
    try:
        results_dir = main()
        print(f"\nSimulation completed successfully!")
        print(f"Results available at: {results_dir}")
    except Exception as e:
        logger.error(f"Simulation failed with error: {str(e)}")
        print(f"\nSimulation failed with error: {str(e)}")
