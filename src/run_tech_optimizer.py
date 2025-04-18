#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run script for the technology-specific regional flexibility optimizer.

This script loads data, initializes the technology-specific optimizer,
and runs the optimization model to analyze regional flexibility with 
separate dispatch variables for each technology type and separate storage
variables for different storage technologies.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
import argparse
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader
from src.model.optimizer_tech import RegionalFlexOptimizerTech
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger('run_tech_optimizer')

def main(config_path, data_dir, results_dir, time_period=None):
    """
    Run the technology-specific regional flexibility optimizer.
    
    Args:
        config_path (str): Path to configuration file
        data_dir (str): Directory containing input data
        results_dir (str): Directory where results will be saved
        time_period (tuple, optional): Start and end times for optimization
    """
    try:
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Load data
        logger.info(f"Loading data from {data_dir}")
        data_loader = DataLoader(data_dir, config_path)
        data = data_loader.load_data()
        
        # Process the time periods
        if time_period:
            start_time, end_time = time_period
            time_periods = [t for t in data['time_index'] if start_time <= t <= end_time]
        else:
            time_periods = data['time_index']
        
        logger.info(f"Optimizing for {len(time_periods)} time periods")
        
        # Initialize the optimizer
        optimizer = RegionalFlexOptimizerTech(config_path)
        
        # Build the model
        logger.info("Building optimization model")
        optimizer.build_model(data, time_periods)
        
        # Solve the model
        logger.info("Solving optimization model")
        results = optimizer.solve_model()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"tech_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            # Convert non-serializable objects (like time periods) to strings
            serializable_results = results.copy()
            
            # Convert variables section
            for var_type, var_dict in serializable_results.get('variables', {}).items():
                for var_name, time_dict in var_dict.items():
                    serializable_results['variables'][var_type][var_name] = {
                        str(t): v for t, v in time_dict.items()
                    }
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in optimization process: {e}")
        raise
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run technology-specific regional flexibility optimizer')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data/processed', 
                        help='Directory containing input data')
    parser.add_argument('--results_dir', type=str, default='results', 
                        help='Directory where results will be saved')
    parser.add_argument('--start_time', type=str, default=None, 
                        help='Start time for optimization (format: YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_time', type=str, default=None, 
                        help='End time for optimization (format: YYYY-MM-DD HH:MM:SS)')
    
    args = parser.parse_args()
    
    # Process time period if provided
    time_period = None
    if args.start_time and args.end_time:
        time_period = (
            pd.to_datetime(args.start_time),
            pd.to_datetime(args.end_time)
        )
    
    # Run the optimizer
    main(args.config, args.data_dir, args.results_dir, time_period)
