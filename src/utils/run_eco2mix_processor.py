#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run script for processing eco2mix data for the four French regions.

This script processes eco2mix data for the year 2022 and prepares it for 
the technology-specific optimizer, focusing on the four French regions:
- Auvergne Rhone Alpes
- Nouvelle Aquitaine
- Occitanie
- Provence Alpes Cote d'Azur
"""

import os
import sys
import argparse
import yaml

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """Run the eco2mix data processing workflow."""
    parser = argparse.ArgumentParser(description='Process eco2mix data for regional flexibility analysis')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--eco2mix', type=str, default='data/Raw/eco2mix-regional-cons-def.csv',
                        help='Path to eco2mix CSV file')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Directory where processed data will be saved')
    parser.add_argument('--year', type=int, default=2022,
                        help='Year to filter data for')
    parser.add_argument('--run-simulation', action='store_true',
                        help='Run a simulation after processing data')
    parser.add_argument('--time-period', type=str, default=None,
                        help='Time period for simulation in format "YYYY-MM-DD,YYYY-MM-DD"')
    parser.add_argument('--palette-file', type=str, default=None,
                        help='YAML file with custom plotting colors')
    args = parser.parse_args()
    
    # Import the eco2mix processor module functions
    from src.utils.process_eco2mix_data import load_config, process_eco2mix_data, run_simulation
    
    print(f"Processing eco2mix data for {args.year} from {args.eco2mix}")
    print(f"Output directory: {args.output}")
    print(f"Configuration file: {args.config}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Failed to load configuration")
        return False
    
    # Process eco2mix data
    data = process_eco2mix_data(args.eco2mix, args.output, config,
                                year=args.year, palette_file=args.palette_file)
    
    if not data:
        print("Failed to process eco2mix data")
        return False
    
    print("Eco2mix data processed successfully")
    
    # Run simulation if requested
    if args.run_simulation:
        results_dir = os.path.join('results', 'tech')
        os.makedirs(results_dir, exist_ok=True)
        
        # Parse time period if provided
        time_period = None
        if args.time_period:
            start_date, end_date = args.time_period.split(',')
            time_period = (start_date, end_date)
        else:
            # Use a limited time period for faster testing (1 week)
            time_period = (
                data['time_index'][0],  # Start with first timestamp
                data['time_index'][min(len(data['time_index'])-1, 336)]  # Use first week (48 periods per day * 7 days)
            )
        
        print(f"Running simulation for time period: {time_period}")
        success = run_simulation(data, args.config, results_dir, time_period)
        
        if success:
            print("Simulation completed successfully")
        else:
            print("Simulation failed")
    
    return True

if __name__ == "__main__":
    main()
