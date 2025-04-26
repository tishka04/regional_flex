#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regional Flexibility Optimizer

This module implements a regional flexibility optimizer that builds upon the technology-specific
optimizer but incorporates additional parameters from config_master.yaml for more comprehensive
optimization of regional energy systems.
"""

import os
import time
import uuid
import yaml
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pulp import (
    LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, 
    LpStatusOptimal, LpConstraintEQ, LpConstraintLE, LpConstraintGE,
    PULP_CBC_CMD, value
)
from tqdm import tqdm
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)

class RegionalFlexOptimizer:
    """Regional Flexibility Optimizer for multiple regions with various technologies.
    
    This optimizer builds upon the RegionalFlexOptimizerTech class but incorporates additional
    parameters from config_master.yaml for more comprehensive optimization.
    
    Attributes:
        config (dict): Configuration parameters loaded from the YAML file
        model (LpProblem): PuLP optimization model
        regions (list): List of regions to optimize
        variables (dict): Dictionary of optimization variables
        tech_capacities (dict): Dictionary of technology capacities by region
        storage_techs (list): List of storage technologies
        dispatch_techs (list): List of dispatchable technologies
        renewable_techs (list): List of renewable technologies
        region_name_map (dict): Mapping of region names
        tech_name_map (dict): Mapping of technology names
    """
    
    def __init__(self, config_path: str):
        """Initialize the regional flexibility optimizer.
        
        Args:
            config_path (str): Path to config file
        """
        # Store config_path as an attribute
        self.config_path = config_path

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Model simplification flags
        self.use_simplified_model = False
        self.simplification_options = {
            "skip_ramping": False,
            "aggregate_storage": False,
            "relaxed_exchange": False,
            "skip_cyclical_storage": False,
            "lp_relaxation": False  # Relax binary/integer variables to continuous (MILP → LP)
        }
        
        # Load master configuration if available
        master_config_path = os.path.join(os.path.dirname(config_path), 'config_master.yaml')
        if os.path.exists(master_config_path):
            with open(master_config_path, 'r') as f:
                self.master_config = yaml.safe_load(f)
                # Merge master config with regular config, with master config taking precedence
                for key, value in self.master_config.items():
                    if key not in self.config:
                        self.config[key] = value
                    elif isinstance(value, dict) and isinstance(self.config.get(key), dict):
                        # Deep merge dictionaries
                        self.config[key].update(value)
        
        # Initialize model
        self.model = LpProblem("RegionalFlexModel", LpMinimize)
        self.variables = {}
        
        # Get regions from config
        self.regions = self.config.get('regions', [
            'Auvergne Rhone Alpes',
            'Nouvelle Aquitaine',
            'Occitanie',
            "Provence Alpes Cote d'Azur"
        ])
        
        logger.info(f"Initialized optimizer with {len(self.regions)} regions: {', '.join(self.regions)}")
        
        # Region name mapping (from config if available)
        self.region_name_map = self.config.get('region_name_map', {
            "Auvergne-Rhone-Alpes": "Auvergne Rhone Alpes",
            "Nouvelle-Aquitaine": "Nouvelle Aquitaine", 
            "Occitanie": "Occitanie",
            "Provence-Alpes-Cote-dAzur": "Provence Alpes Cote d'Azur"
        })
        
        # Technology name mapping
        self.tech_name_map = self.config.get('tech_name_map', {
            "hydraulique": "hydro",
            "nucleaire": "nuclear",
            "thermique_fossile": "thermal",
            "thermique_gaz": "thermal_gas",
            "thermique_charbon": "thermal_coal",
            "bioenergie": "biofuel"
        })
        
        # Define technology types
        self.dispatch_techs = [
            "hydro", "nuclear", "thermal_gas", "thermal_coal", "biofuel"
        ]
        
        self.storage_techs = ["STEP", "batteries"]
        self.renewable_techs = ["hydro"]
        
        # Load regional capacities from config
        self.tech_capacities = {}
        if 'regional_capacities' in self.config:
            self.tech_capacities = self.config['regional_capacities']
            logger.info(f"Loaded technology capacities for {len(self.tech_capacities)} regions")
        
        # Load regional storage capacities
        self.storage_capacities = {}
        if 'regional_storage' in self.config:
            self.storage_capacities = self.config['regional_storage']
            logger.info(f"Loaded storage capacities for {len(self.storage_capacities)} regions")
        
        # Regional cost multipliers
        self.regional_cost_multipliers = {}
        if 'regional_constraints' in self.config:
            self.regional_cost_multipliers = self.config['regional_constraints']
            logger.info(f"Loaded regional cost multipliers for {len(self.regional_cost_multipliers)} regions")

        
        
        # Initialize dictionaries for regional parameters
        self.regional_multipliers = {}
        self.tech_params = {}
        
        # Track which variables should be binary in standard model
        # This allows us to relax them to continuous when using LP relaxation
        self.binary_vars = []
        
        # Initialize the regional cost multipliers if not already set
        if not hasattr(self, 'regional_cost_multipliers') or self.regional_cost_multipliers is None:
            self.regional_cost_multipliers = {}
        
        if 'regional_costs' in self.config:
            self.regional_costs = self.config['regional_costs']
            logger.info(f"Loaded regional costs for {len(self.regional_costs)} regions")

        # --- harmonise TOUS les noms de région (espace -> underscore) -------------
        def _norm(r): return r.replace(" ", "_")

        self.regions           = [_norm(r) for r in self.regions]
        self.tech_capacities   = {_norm(r): caps for r, caps in self.tech_capacities.items()}
        self.regional_costs    = {_norm(r): c    for r, c    in self.regional_costs.items()}
        self.storage_capacities= {_norm(r): s    for r, s    in self.storage_capacities.items()}
        self.regional_constraints = {_norm(r): d for r, d in self.regional_cost_multipliers.items()}
            
    def build_model(self, regional_data: Dict[str, pd.DataFrame], time_periods=None):
        """Build the optimization model with variables, objective, and constraints.
        
        Args:
            regional_data (Dict[str, pd.DataFrame]): Dictionary mapping region names to data frames
            time_periods (Optional): Time periods to optimize for
        """
        logger.info("Building regional flexibility optimization model")
        
        # Process time periods
        if time_periods is None:
            # Use all time periods from the first region's data
            first_region = next(iter(regional_data.values()))
            time_periods = list(range(len(first_region)))
        
        # Initialize progress bar for model building
        total_steps = 7  # Variables, objective, and 5 constraint types
        with tqdm(total=total_steps, desc="Building optimization model") as pbar:
            # Initialize variables first
            self._init_variables(regional_data, time_periods)
            pbar.update(1)
            
            # Add objective function
            self._add_objective(regional_data, time_periods)
            pbar.update(1)
            
            # Add constraints
            self._add_constraints(regional_data, time_periods)
            pbar.update(5)  # Multiple constraint types
        
        logger.info(f"Model built with {len(self.model.variables())} variables and {len(self.model.constraints)} constraints")
    
    def _init_variables(self, data: Dict[str, pd.DataFrame], time_periods):
        """Initialize optimization variables for all regions and technologies.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary mapping region names to data frames
            time_periods: Time periods to optimize for
        """
        logger.info("Initializing optimization variables")
        
        # Check if we're using LP relaxation (converting binary/integer vars to continuous)
        using_lp_relaxation = self.use_simplified_model and self.simplification_options.get("lp_relaxation", False)
        if using_lp_relaxation:
            logger.info("Using LP relaxation: converting all binary/integer variables to continuous")
        
        # Determine the number of time steps
        if isinstance(time_periods, list):
            T = time_periods
        else:
            # Use all time periods from the first region's data
            first_region = next(iter(data.values()))
            T = list(range(len(first_region)))
        
        # Process each region
        for region in self.regions:
            regional_data = data.get(region)
            if regional_data is None:
                logger.warning(f"No data provided for region {region}")
                continue
            
            # 1. Dispatch variables for each technology
            for tech in self.dispatch_techs:
                self.variables[f"dispatch_{tech}_{region}"] = {}
                
                for t in T:
                    # Create dispatch variable with lower bound of 0 (can't dispatch negative power)
                    self.variables[f"dispatch_{tech}_{region}"][t] = LpVariable(
                        f"dispatch_{tech}_{region}_{t}", 
                        lowBound=0
                    )
            
            # 2. Storage variables for each storage technology
            for storage_tech in self.storage_techs:
                # Storage level (state of charge)
                self.variables[f"storage_soc_{storage_tech}_{region}"] = {}
                # Charging power
                self.variables[f"storage_charge_{storage_tech}_{region}"] = {}
                # Discharging power
                self.variables[f"storage_discharge_{storage_tech}_{region}"] = {}
                
                for t in T:
                    # Storage level (can't go negative)
                    self.variables[f"storage_soc_{storage_tech}_{region}"][t] = LpVariable(
                        f"storage_soc_{storage_tech}_{region}_{t}",
                        lowBound=0
                    )
                    
                    # Charging (can't go negative)
                    self.variables[f"storage_charge_{storage_tech}_{region}"][t] = LpVariable(
                        f"storage_charge_{storage_tech}_{region}_{t}",
                        lowBound=0
                    )
                    
                    # Discharging (can't go negative)
                    self.variables[f"storage_discharge_{storage_tech}_{region}"][t] = LpVariable(
                        f"storage_discharge_{storage_tech}_{region}_{t}",
                        lowBound=0
                    )
            
            # 3. Demand response variables (can be positive or negative)
            self.variables[f"demand_response_{region}"] = {}
            for t in T:
                # No bounds on DR - can be positive (increase) or negative (decrease)
                self.variables[f"demand_response_{region}"][t] = LpVariable(
                    f"demand_response_{region}_{t}"
                )
            
            # 4. Slack variables for balance equation
            self.variables[f"slack_pos_{region}"] = {}
            self.variables[f"slack_neg_{region}"] = {}
            for t in T:
                # Positive slack
                self.variables[f"slack_pos_{region}"][t] = LpVariable(
                    f"slack_pos_{region}_{t}",
                    lowBound=0
                )
                
                # Negative slack
                self.variables[f"slack_neg_{region}"][t] = LpVariable(
                    f"slack_neg_{region}_{t}", 
                    lowBound=0
                )

            # 5. Curtailment (MWh renonçant à être injectés)
            self.variables[f"curtail_{region}"] = {}
            for t in T:
                self.variables[f"curtail_{region}"][t] = LpVariable(
                    f"curtail_{region}_{t}", lowBound=0  # ≥0, pas de bornes supérieures
                )
        
        # 5. Exchange and transport variables between regions
        for i, r1 in enumerate(self.regions):
            for r2 in self.regions[i+1:]:
                # Exchange variables - can be positive or negative
                self.variables[f"exchange_{r1}_{r2}"] = {}
                # Transport variables - can be positive or negative
                self.variables[f"transport_{r1}_{r2}"] = {}
                # Absolute value variables for exchange (for objective function)
                self.variables[f"abs_exchange_{r1}_{r2}"] = {}
                # Absolute value variables for transport
                self.variables[f"abs_transport_{r1}_{r2}"] = {}
                
                for t in T:
                    # Exchange between regions (can be positive or negative)
                    self.variables[f"exchange_{r1}_{r2}"][t] = LpVariable(
                        f"exchange_{r1}_{r2}_{t}"
                    )
                    
                    # Transport between regions (can be positive or negative)
                    self.variables[f"transport_{r1}_{r2}"][t] = LpVariable(
                        f"transport_{r1}_{r2}_{t}"
                    )
                    
                    # Absolute value variables for exchange (for objective function)
                    self.variables[f"abs_exchange_{r1}_{r2}"][t] = LpVariable(
                        f"abs_exchange_{r1}_{r2}_{t}",
                        lowBound=0
                    )
                    
                    # Absolute value variables for transport
                    self.variables[f"abs_transport_{r1}_{r2}"][t] = LpVariable(
                        f"abs_transport_{r1}_{r2}_{t}",
                        lowBound=0
                    )
        
        # Add time-dependent variables for demand response activation
        for region in self.regions:
            # 1. Demand response activation (binary or continuous with LP relaxation)
            dr_var_name = f"dr_active_{region}"
            self.variables[dr_var_name] = {}
            
            for t in T:
                if using_lp_relaxation:
                    self.variables[dr_var_name][t] = LpVariable(
                        f"{dr_var_name}_{t}",
                        lowBound=0,
                        upBound=1,
                        cat='Continuous'
                    )
                else:
                    self.variables[dr_var_name][t] = LpVariable(
                        f"{dr_var_name}_{t}",
                        cat='Binary'
                    )
                    # Track which variables should be binary
                    self.binary_vars.append(self.variables[dr_var_name][t])
    
    def _add_objective(self, data: Dict[str, pd.DataFrame], time_periods):
        """Define the objective function for minimizing costs.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary mapping region names to data frames
            time_periods: Time periods to optimize for
        """
        logger.info("Adding objective function")
        
        # Determine time periods
        if isinstance(time_periods, list):
            T = time_periods
        else:
            first_region = next(iter(data.values()))
            T = list(range(len(first_region)))
        
        # Get costs from config
        costs = self.config.get('costs', {})
        
        # Default costs (fallback values)
        default_costs = {
            'hydro': 30.0,
            'nuclear': 40.0,
            'thermal': 85.0,
            'thermal_gas': 80.0,
            'thermal_coal': 90.0,
            'biofuel': 70.0,
            'dispatch': 50.0,
            'storage': 5.0,
            'demand_response': 120.0,
            'exchange': 25.0,
            'storage_charge': 4.0,
            'storage_discharge': 3.0,
            'transport_exchange': 30.0,
            'slack_penalty': 50000.0
        }
        
        # Merge default costs with config costs
        for key, value in default_costs.items():
            if key not in costs:
                costs[key] = value
        
        # Initialize objective function
        objective = 0
        
        # 1. Dispatch costs (by technology and region)
        for region in self.regions:
            for tech in self.dispatch_techs:
                # Determine the technology cost for this region
                tech_cost = costs[tech]  # Default from global costs
                
                # Check if there's a regional specific cost
                if region in self.regional_costs and tech in self.regional_costs[region]:
                    tech_cost = self.regional_costs[region][tech]
                    logger.debug(f"Using regional cost for {tech} in {region}: {tech_cost}")
                
                # Add to objective: sum of dispatch * cost for all time periods
                if f"dispatch_{tech}_{region}" in self.variables:
                    for t in T:
                        if t in self.variables[f"dispatch_{tech}_{region}"]:
                            objective += self.variables[f"dispatch_{tech}_{region}"][t] * tech_cost
        
        # 2. Storage costs
        for region in self.regions:
            for storage_tech in self.storage_techs:
                # Get technology-specific storage costs if available
                charge_cost_key = f"storage_{storage_tech}_charge"
                discharge_cost_key = f"storage_{storage_tech}_discharge"
                
                charge_cost = costs.get(charge_cost_key, costs.get('storage_charge', 4.0))
                discharge_cost = costs.get(discharge_cost_key, costs.get('storage_discharge', 3.0))
                
                # Add regional cost multipliers if available
                if (region in self.regional_cost_multipliers and 
                    'storage_cost_multiplier' in self.regional_cost_multipliers[region]):
                    multiplier = self.regional_cost_multipliers[region]['storage_cost_multiplier']
                    charge_cost *= multiplier
                    discharge_cost *= multiplier
                
                # Add to objective: sum of charge/discharge * cost for all time periods
                for t in T:
                    # Add charging cost
                    if (f"storage_charge_{storage_tech}_{region}" in self.variables and
                        t in self.variables[f"storage_charge_{storage_tech}_{region}"]):
                        objective += self.variables[f"storage_charge_{storage_tech}_{region}"][t] * charge_cost
                    
                    # Add discharging cost
                    if (f"storage_discharge_{storage_tech}_{region}" in self.variables and
                        t in self.variables[f"storage_discharge_{storage_tech}_{region}"]):
                        objective += self.variables[f"storage_discharge_{storage_tech}_{region}"][t] * discharge_cost
        
        # 3. Demand response costs
        dr_cost = costs.get('demand_response', 120.0)
        for region in self.regions:
            # Apply regional cost multiplier if available
            regional_dr_cost = dr_cost
            if (region in self.regional_cost_multipliers and 
                'dr_cost_multiplier' in self.regional_cost_multipliers[region]):
                regional_dr_cost *= self.regional_cost_multipliers[region]['dr_cost_multiplier']
            
            # Add to objective: sum of absolute value of demand response * cost
            if f"demand_response_{region}" in self.variables:
                for t in T:
                    if t in self.variables[f"demand_response_{region}"]:
                        # Create absolute value variables for demand response
                        pos_dr = LpVariable(f"pos_dr_{region}_{t}", lowBound=0)
                        neg_dr = LpVariable(f"neg_dr_{region}_{t}", lowBound=0)
                        
                        # Add constraints to model absolute value
                        self.model += (self.variables[f"demand_response_{region}"][t] == pos_dr - neg_dr, 
                                      f"dr_abs_value_{region}_{t}")
                        
                        # Add cost of demand response (both positive and negative) to objective
                        objective += (pos_dr + neg_dr) * regional_dr_cost
        
        # 4. Exchange costs
        exchange_cost = costs.get('exchange', 25.0)
        for i, r1 in enumerate(self.regions):
            for r2 in self.regions[i+1:]:
                # Calculate the exchange cost between these regions
                # Check for regional cost modifiers
                r1_multiplier = 1.0
                r2_multiplier = 1.0
                
                if r1 in self.regional_cost_multipliers and 'exchange_cost_multiplier' in self.regional_cost_multipliers[r1]:
                    r1_multiplier = self.regional_cost_multipliers[r1]['exchange_cost_multiplier']
                
                if r2 in self.regional_cost_multipliers and 'exchange_cost_multiplier' in self.regional_cost_multipliers[r2]:
                    r2_multiplier = self.regional_cost_multipliers[r2]['exchange_cost_multiplier']
                
                # Average the multipliers for exchanges
                regional_exchange_cost = exchange_cost * (r1_multiplier + r2_multiplier) / 2
                
                # Add to objective: sum of absolute exchange * cost for all time periods
                for t in T:
                    if (f"abs_exchange_{r1}_{r2}" in self.variables and
                        t in self.variables[f"abs_exchange_{r1}_{r2}"]):
                        objective += self.variables[f"abs_exchange_{r1}_{r2}"][t] * regional_exchange_cost
        
        # 5. Transport costs
        transport_cost = costs.get('transport_exchange', 30.0)
        for i, r1 in enumerate(self.regions):
            for r2 in self.regions[i+1:]:
                # Calculate the transport cost between these regions
                # Check for regional cost modifiers
                r1_multiplier = 1.0
                r2_multiplier = 1.0
                
                if r1 in self.regional_cost_multipliers and 'transport_cost_multiplier' in self.regional_cost_multipliers[r1]:
                    r1_multiplier = self.regional_cost_multipliers[r1]['transport_cost_multiplier']
                
                if r2 in self.regional_cost_multipliers and 'transport_cost_multiplier' in self.regional_cost_multipliers[r2]:
                    r2_multiplier = self.regional_cost_multipliers[r2]['transport_cost_multiplier']
                
                # Average the multipliers for transport
                regional_transport_cost = transport_cost * (r1_multiplier + r2_multiplier) / 2
                
                # Add to objective: sum of absolute transport * cost for all time periods
                for t in T:
                    if (f"abs_transport_{r1}_{r2}" in self.variables and
                        t in self.variables[f"abs_transport_{r1}_{r2}"]):
                        objective += self.variables[f"abs_transport_{r1}_{r2}"][t] * regional_transport_cost
        
        # 6. Slack penalties (highest cost to minimize slack usage)
        slack_penalty = costs.get('slack_penalty', 50000.0)
        for region in self.regions:
            # Apply regional slack penalty multiplier if available
            regional_slack_penalty = slack_penalty
            if (region in self.regional_cost_multipliers and 
                'slack_penalty_multiplier' in self.regional_cost_multipliers[region]):
                regional_slack_penalty *= self.regional_cost_multipliers[region]['slack_penalty_multiplier']
            
            # Add to objective: sum of slack variables * penalty for all time periods
            for t in T:
                if (f"slack_pos_{region}" in self.variables and
                    t in self.variables[f"slack_pos_{region}"]):
                    objective += self.variables[f"slack_pos_{region}"][t] * regional_slack_penalty
                
                if (f"slack_neg_{region}" in self.variables and
                    t in self.variables[f"slack_neg_{region}"]):
                    objective += self.variables[f"slack_neg_{region}"][t] * regional_slack_penalty

        # 6 bis. Curtailment penalty
        curtail_pen = costs.get('curtailment_penalty', 10_000.0)
        for region in self.regions:
            if f"curtail_{region}" in self.variables:
                for t in T:
                    if t in self.variables[f"curtail_{region}"]:
                        objective += self.variables[f"curtail_{region}"][t] * curtail_pen

        
        # Set model objective
        self.model += objective
        
    def _add_constraints(self, data: Dict[str, pd.DataFrame], time_periods):
        """Add constraints to the optimization model.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary mapping region names to data frames
            time_periods: Time periods to optimize for
        """
        logger.info("Adding optimization constraints")
        
        # Determine time periods
        if isinstance(time_periods, list):
            T = time_periods
        else:
            first_region = next(iter(data.values()))
            T = list(range(len(first_region)))
        
        # Set up progress bar for constraints
        with tqdm(total=6, desc="Adding constraints") as pbar:
            # 1. Energy balance constraints
            pbar.set_description("Energy balance constraints")
            self._add_energy_balance_constraints(data, T)
            pbar.update(1)
            
            # 2. Capacity constraints
            pbar.set_description("Capacity constraints")
            self._add_capacity_constraints(data, T)
            pbar.update(1)
            
            # 3. Storage constraints
            pbar.set_description("Storage constraints")
            if self.use_simplified_model and self.simplification_options.get("aggregate_storage", False):
                logger.info("Skipping storage constraints due to aggregate_storage simplification")
            else:
                self._add_storage_constraints(data, T)
            pbar.update(1)
            
            # 4. Exchange network constraints
            pbar.set_description("Exchange network constraints")
            self._add_exchange_constraints(data, T)
            pbar.update(1)
            
            # 5. Demand response and ramping constraints
            pbar.set_description("DR and ramping constraints")
            self._add_dr_and_ramping_constraints(data, T)
            pbar.update(1)
            
            # 6. Flexibility diversity constraints to ensure balanced use
            pbar.set_description("Flexibility diversity constraints")
            self._add_flexibility_diversity_constraints(data, T)
            pbar.update(1)
    
    def _add_energy_balance_constraints(self, data: Dict[str, pd.DataFrame], T):
        """Add energy balance constraints for each region and time period.
        
        Balance: sum(dispatch) + sum(discharge) - sum(charge) + exchanges = demand ± DR ± slack
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary mapping region names to data frames
            T (List): List of time periods
        """
        # Process each region separately
        for region in self.regions:
            regional_data = data.get(region)
            if regional_data is None:
                logger.warning(f"No data for region {region}, skipping energy balance constraints")
                continue
                
            # Process each time period for this region
            for t in T:
                # Skip if the time index is out of range
                if t >= len(regional_data):
                    logger.debug(f"Time index {t} out of range for {region} with data length {len(regional_data)}")
                    continue
                    
                # Get demand for this time period (column name may vary, try several options)
                demand = None
                for col in ['consumption', 'demand', 'load']:
                    if col in regional_data.columns:
                        demand = float(regional_data.iloc[t][col])
                        break
                
                if demand is None:
                    logger.warning(f"No demand data found for {region} at time {t}, using 0")
                    demand = 0.0
                
                # Create the left side of the balance equation
                balance_expr = 0
                
                # 1. Add all dispatch variables for technologies
                for tech in self.dispatch_techs:
                    if f"dispatch_{tech}_{region}" in self.variables and t in self.variables[f"dispatch_{tech}_{region}"]:
                        balance_expr += self.variables[f"dispatch_{tech}_{region}"][t]
                
                # 2. Add storage discharge (positive contribution)
                for storage_tech in self.storage_techs:
                    if (f"storage_discharge_{storage_tech}_{region}" in self.variables and 
                        t in self.variables[f"storage_discharge_{storage_tech}_{region}"]):
                        balance_expr += self.variables[f"storage_discharge_{storage_tech}_{region}"][t]
                
                # 3. Subtract storage charge (negative contribution)
                for storage_tech in self.storage_techs:
                    if (f"storage_charge_{storage_tech}_{region}" in self.variables and 
                        t in self.variables[f"storage_charge_{storage_tech}_{region}"]):
                        balance_expr -= self.variables[f"storage_charge_{storage_tech}_{region}"][t]
                
                # 4. Add exchange with other regions
                for other_region in self.regions:
                    if other_region == region:
                        continue
                    
                    # Arrange regions alphabetically to match variable naming
                    r1, r2 = sorted([region, other_region])
                    
                    # If this region is r1, a positive exchange means exporting energy (negative contribution)
                    # If this region is r2, a positive exchange means importing energy (positive contribution)
                    if f"exchange_{r1}_{r2}" in self.variables and t in self.variables[f"exchange_{r1}_{r2}"]:
                        if region == r1:
                            balance_expr -= self.variables[f"exchange_{r1}_{r2}"][t]
                        else:  # region == r2
                            balance_expr += self.variables[f"exchange_{r1}_{r2}"][t]
                    
                    # Transport works the same way
                    if f"transport_{r1}_{r2}" in self.variables and t in self.variables[f"transport_{r1}_{r2}"]:
                        if region == r1:
                            balance_expr -= self.variables[f"transport_{r1}_{r2}"][t]
                        else:  # region == r2
                            balance_expr += self.variables[f"transport_{r1}_{r2}"][t]
                
                # 5. Add demand response (can be positive or negative)
                if f"demand_response_{region}" in self.variables and t in self.variables[f"demand_response_{region}"]:
                    balance_expr += self.variables[f"demand_response_{region}"][t]

                # 5 bis. Curtailment (énergie disponible mais non injectée)
                if f"curtail_{region}" in self.variables and t in self.variables[f"curtail_{region}"]:
                    balance_expr -= self.variables[f"curtail_{region}"][t]
                
                # 6. Add slack variables (to ensure feasibility)
                slack_pos_term = 0
                slack_neg_term = 0
                
                if f"slack_pos_{region}" in self.variables and t in self.variables[f"slack_pos_{region}"]:
                    slack_pos_term = self.variables[f"slack_pos_{region}"][t]
                
                if f"slack_neg_{region}" in self.variables and t in self.variables[f"slack_neg_{region}"]:
                    slack_neg_term = self.variables[f"slack_neg_{region}"][t]
                
                # Balance equation: generation + import - export + DR + slack = demand
                try:
                    self.model += (
                        balance_expr + slack_pos_term - slack_neg_term == demand,
                        f"balance_{region}_{t}_{uuid.uuid4().hex[:8]}"
                    )
                except Exception as e:
                    logger.error(f"Error adding balance constraint for {region} at time {t}: {e}")
    
    def _add_storage_constraints(self, data: Dict[str, pd.DataFrame], T):
        """Add constraints for storage technologies (batteries, pumped hydro, etc.)
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary mapping region names to data frames
            T (List): List of time periods
        """
        logger.info("Adding storage constraints")
        
        # Skip if no storage technologies defined
        if not hasattr(self, 'storage_techs') or not self.storage_techs:
            logger.warning("No storage technologies defined, skipping storage constraints")
            return
        
        # Process each region and storage technology
        for region in self.regions:
            for tech in self.storage_techs:
                # Get storage parameters from config
                # Get separate charge and discharge efficiencies
                charge_efficiency = self.config.get('storage_params', {}).get(tech, {}).get('charge_efficiency', 0.95)  # Default 95%
                discharge_efficiency = self.config.get('storage_params', {}).get(tech, {}).get('discharge_efficiency', 0.95)  # Default 95%
                # For backward compatibility, also check for combined efficiency
                if 'efficiency' in self.config.get('storage_params', {}).get(tech, {}):
                    combined_efficiency = self.config.get('storage_params', {}).get(tech, {}).get('efficiency')
                    # Split the combined efficiency into charge and discharge components
                    charge_efficiency = discharge_efficiency = combined_efficiency ** 0.5
                max_capacity = 0
                
                # Get max capacity if available in regional capacities
                if region in self.tech_capacities and tech in self.tech_capacities[region]:
                    max_capacity = float(self.tech_capacities[region][tech])
                
                # Skip if no storage capacity for this tech/region
                if max_capacity <= 0:
                    continue
                
                # Get max power for charge/discharge if specified, otherwise use capacity
                max_power = self.config.get('storage_params', {}).get(tech, {}).get('max_power_ratio', 1.0) * max_capacity
                
                # Apply regional multiplier if available
                if region in self.regional_multipliers:
                    if f'{tech}_capacity_multiplier' in self.regional_multipliers[region]:
                        max_capacity *= self.regional_multipliers[region][f'{tech}_capacity_multiplier']
                    if f'{tech}_power_multiplier' in self.regional_multipliers[region]:
                        max_power *= self.regional_multipliers[region][f'{tech}_power_multiplier']
                
                # Check if storage variables exist for this tech/region
                storage_vars_exist = (
                    f"storage_charge_{tech}_{region}" in self.variables and
                    f"storage_discharge_{tech}_{region}" in self.variables and
                    f"storage_soc_{tech}_{region}" in self.variables
                )
                
                if not storage_vars_exist:
                    continue
                
                # 1. Add power capacity constraints for each time period
                for t in T:
                    # Skip if variables don't exist for this time period
                    if (t not in self.variables[f"storage_charge_{tech}_{region}"] or
                        t not in self.variables[f"storage_discharge_{tech}_{region}"]):
                        continue
                    
                    # Max charge power constraint
                    self.model += (
                        self.variables[f"storage_charge_{tech}_{region}"][t] <= max_power,
                        f"storage_max_charge_{tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                    )
                    
                    # Max discharge power constraint
                    self.model += (
                        self.variables[f"storage_discharge_{tech}_{region}"][t] <= max_power,
                        f"storage_max_discharge_{tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                    )
                    
                    # Ensure charge and discharge are not both positive in the same period
                    # This is handled by binary variables already created in _init_variables
                    if f"storage_mode_{tech}_{region}" in self.variables and t in self.variables[f"storage_mode_{tech}_{region}"]:
                        binary_var = self.variables[f"storage_mode_{tech}_{region}"][t]
                        
                        # When binary=1, discharge must be 0
                        self.model += (
                            self.variables[f"storage_discharge_{tech}_{region}"][t] <= max_power * (1 - binary_var),
                            f"storage_mode_discharge_{tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                        )
                        
                        # When binary=0, charge must be 0
                        self.model += (
                            self.variables[f"storage_charge_{tech}_{region}"][t] <= max_power * binary_var,
                            f"storage_mode_charge_{tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                        )
                
                # 2. Add state of charge (SOC) constraints and dynamics
                for i in range(len(T)):
                    t = T[i]
                    
                    # Skip if SOC variable doesn't exist for this time period
                    if t not in self.variables[f"storage_soc_{tech}_{region}"]:
                        continue
                    
                    # Maximum energy storage capacity constraint
                    self.model += (
                        self.variables[f"storage_soc_{tech}_{region}"][t] <= max_capacity,
                        f"storage_max_soc_{tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                    )
                    
                    # Minimum energy storage capacity constraint (usually 0)
                    self.model += (
                        self.variables[f"storage_soc_{tech}_{region}"][t] >= 0,
                        f"storage_min_soc_{tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                    )
                    
                    # SOC dynamics: SOC[t+1] = SOC[t] + charge[t]*efficiency - discharge[t]/efficiency
                    if i < len(T) - 1:
                        t_next = T[i+1]
                        
                        # Skip if next time period variables don't exist
                        if (t_next not in self.variables[f"storage_soc_{tech}_{region}"] or
                            t not in self.variables[f"storage_charge_{tech}_{region}"] or
                            t not in self.variables[f"storage_discharge_{tech}_{region}"]):
                            continue
                        
                        # Check if times are consecutive for proper SOC tracking
                        is_consecutive = (isinstance(t, int) and isinstance(t_next, int) and t_next == t + 1) or \
                                        (not isinstance(t, int) and not isinstance(t_next, int) and \
                                         (t_next - t).total_seconds() / 3600 <= 1.01)  # Within 1.01 hours (allowing for rounding)
                        
                        if not is_consecutive:
                            continue
                        
                        # SOC dynamics constraint with separate charge and discharge efficiencies
                        self.model += (
                            self.variables[f"storage_soc_{tech}_{region}"][t_next] == \
                            self.variables[f"storage_soc_{tech}_{region}"][t] + \
                            self.variables[f"storage_charge_{tech}_{region}"][t] * charge_efficiency - \
                            self.variables[f"storage_discharge_{tech}_{region}"][t] / discharge_efficiency,
                            f"storage_soc_dynamics_{tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                        )

    def _add_exchange_constraints(self, data: Dict[str, pd.DataFrame], T):
        """Add network exchange constraints between regions.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary mapping region names to data frames
            T (List): List of time periods
        """
        logger.info("Adding exchange network constraints")
        
        # Skip if only one region
        if len(self.regions) <= 1:
            logger.info("Only one region, skipping exchange constraints")
            return
            
        # Get constraints from config
        constraints = self.config.get('constraints', {})
        
        # Default constraints if not specified
        max_exchange = constraints.get('max_exchange', 20000.0)  # MW
        exchange_capacity = constraints.get('exchange_capacity', 20000.0)  # MW
        transport_capacity = constraints.get('transport_capacity', 20000.0)  # MW
        
        # Relaxed exchange constraints (if enabled)
        relaxed_exchange = self.use_simplified_model and self.simplification_options.get("relaxed_exchange", False)
        
        # Process each region pair
        for i, r1 in enumerate(self.regions):
            for j, r2 in enumerate(self.regions[i+1:], i+1):
                # Region pairs are stored alphabetically for consistency
                region1, region2 = sorted([r1, r2])
                
                # 1. Exchange capacity constraints
                exchange_var_name = f"exchange_{region1}_{region2}"
                if exchange_var_name in self.variables:
                    for t in T:
                        if t not in self.variables[exchange_var_name]:
                            continue
                            
                        # Apply exchange capacity constraints (bidirectional)
                        self.model += (
                            self.variables[exchange_var_name][t] <= exchange_capacity,
                            f"max_exchange_pos_{region1}_{region2}_{t}_{uuid.uuid4().hex[:8]}"
                        )
                        
                        # Allow negative flow (from region2 to region1)
                        self.model += (
                            self.variables[exchange_var_name][t] >= -exchange_capacity,
                            f"max_exchange_neg_{region1}_{region2}_{t}_{uuid.uuid4().hex[:8]}"
                        )
                
                # 2. Transport capacity constraints (if separate from exchange)
                transport_var_name = f"transport_{region1}_{region2}"
                if transport_var_name in self.variables:
                    for t in T:
                        if t not in self.variables[transport_var_name]:
                            continue
                            
                        # Apply transport capacity constraints (bidirectional)
                        self.model += (
                            self.variables[transport_var_name][t] <= transport_capacity,
                            f"max_transport_pos_{region1}_{region2}_{t}_{uuid.uuid4().hex[:8]}"
                        )
                        
                        # Allow negative flow (from region2 to region1)
                        self.model += (
                            self.variables[transport_var_name][t] >= -transport_capacity,
                            f"max_transport_neg_{region1}_{region2}_{t}_{uuid.uuid4().hex[:8]}"
                        )
                
                # 3. Combined exchange and transport limit (if applicable)
                if exchange_var_name in self.variables and transport_var_name in self.variables:
                    for t in T:
                        if (t not in self.variables[exchange_var_name] or 
                            t not in self.variables[transport_var_name]):
                            continue
                            
                        # Apply combined limit if relaxed_exchange is disabled
                        if not relaxed_exchange:
                            self.model += (
                                self.variables[exchange_var_name][t] + self.variables[transport_var_name][t] <= max_exchange,
                                f"max_combined_exchange_{region1}_{region2}_{t}_{uuid.uuid4().hex[:8]}"
                            )
                            
                        # Add constraints for absolute value variables for exchange
                        abs_exchange_var = f"abs_exchange_{region1}_{region2}"
                        if abs_exchange_var in self.variables and t in self.variables[abs_exchange_var]:
                            # abs_exchange >= exchange (handles positive values)
                            self.model += (
                                self.variables[abs_exchange_var][t] >= self.variables[exchange_var_name][t],
                                f"abs_exchange_pos_{region1}_{region2}_{t}_{uuid.uuid4().hex[:8]}"
                            )
                            # abs_exchange >= -exchange (handles negative values)
                            self.model += (
                                self.variables[abs_exchange_var][t] >= -self.variables[exchange_var_name][t],
                                f"abs_exchange_neg_{region1}_{region2}_{t}_{uuid.uuid4().hex[:8]}"
                            )
                            
                        # Add constraints for absolute value variables for transport
                        abs_transport_var = f"abs_transport_{region1}_{region2}"
                        if abs_transport_var in self.variables and t in self.variables[abs_transport_var]:
                            # abs_transport >= transport (handles positive values)
                            self.model += (
                                self.variables[abs_transport_var][t] >= self.variables[transport_var_name][t],
                                f"abs_transport_pos_{region1}_{region2}_{t}_{uuid.uuid4().hex[:8]}"
                            )
                            # abs_transport >= -transport (handles negative values)
                            self.model += (
                                self.variables[abs_transport_var][t] >= -self.variables[transport_var_name][t],
                                f"abs_transport_neg_{region1}_{region2}_{t}_{uuid.uuid4().hex[:8]}"
                            )
                            
        # Special constraints for grid stability (if applicable)
        if 'grid_stability' in constraints:
            stability_params = constraints['grid_stability']
            
            # Maximum total exchange per region
            if 'max_total_exchange_per_region' in stability_params:
                max_total = float(stability_params['max_total_exchange_per_region'])
                
                for region in self.regions:
                    for t in T:
                        # Calculate total exchange for this region
                        total_exchange_expr = 0
                        
                        # Add all exchanges involving this region
                        for other_region in self.regions:
                            if other_region == region:
                                continue
                                
                            # Sort regions alphabetically to match variable naming
                            r1, r2 = sorted([region, other_region])
                            
                            # Add exchange variable if it exists
                            exchange_var_name = f"exchange_{r1}_{r2}"
                            if exchange_var_name in self.variables and t in self.variables[exchange_var_name]:
                                total_exchange_expr += self.variables[exchange_var_name][t]
                            
                            # Add transport variable if it exists
                            transport_var_name = f"transport_{r1}_{r2}"
                            if transport_var_name in self.variables and t in self.variables[transport_var_name]:
                                total_exchange_expr += self.variables[transport_var_name][t]
                        
                        # Apply total exchange limit
                        if total_exchange_expr != 0:  # Only add constraint if there are exchange variables
                            self.model += (
                                total_exchange_expr <= max_total,
                                f"max_total_exchange_{region}_{t}_{uuid.uuid4().hex[:8]}"
                            )

    def _add_dr_and_ramping_constraints(self, data: Dict[str, pd.DataFrame], T):
        """Add demand response and ramping constraints for each region.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary mapping region names to data frames
            T (List): List of time periods
        """
        logger.info("Adding demand response and ramping constraints")
        
        # Get constraints from config
        constraints = self.config.get('constraints', {})
        
        # Skip ramping constraints if enabled in simplification options
        skip_ramping = self.use_simplified_model and self.simplification_options.get("skip_ramping", False)
        if skip_ramping:
            logger.info("Skipping ramping constraints due to skip_ramping simplification")
        
        # 1. Add demand response constraints
        for region in self.regions:
            # Get demand response parameters from config for this region
            dr_params = self.config.get('demand_response', {}).get(region, {})
            
            # Default DR parameters if not specified
            max_dr_shift = dr_params.get('max_shift', 0.0)  # % of demand
            max_dr_total = dr_params.get('max_total', 0.0)  # MWh
            dr_participation_rate = dr_params.get('participation_rate', 0.0)  # % of consumers
            
            # Skip if no DR capability for this region
            if max_dr_shift <= 0 or max_dr_total <= 0 or dr_participation_rate <= 0:
                continue
            
            # Check if DR variables exist for this region
            dr_var_name = f"demand_response_{region}"
            if dr_var_name not in self.variables:
                continue
            
            # Add demand response constraints for each time period
            for t in T:
                if t not in self.variables[dr_var_name]:
                    continue
                
                # Get demand for this time period
                demand = None
                for col in ['consumption', 'demand', 'load']:
                    if col in data[region].columns:
                        demand = float(data[region].iloc[t][col])
                        break
                
                if demand is None:
                    logger.warning(f"No demand data found for {region} at time {t}, skipping DR constraint")
                    continue
                
                # Calculate maximum DR shift for this time period based on demand
                max_shift = min(demand * max_dr_shift * dr_participation_rate / 100.0, max_dr_total)
                
                # Add DR bounds constraints
                self.model += (
                    self.variables[dr_var_name][t] <= max_shift,
                    f"dr_max_{region}_{t}_{uuid.uuid4().hex[:8]}"
                )
                
                self.model += (
                    self.variables[dr_var_name][t] >= -max_shift,
                    f"dr_min_{region}_{t}_{uuid.uuid4().hex[:8]}"
                )
            
            # Add DR balance constraint (net zero over time horizon)
            if len(T) > 1:
                dr_sum = 0
                valid_times = [t for t in T if t in self.variables[dr_var_name]]
                
                if valid_times:
                    for t in valid_times:
                        dr_sum += self.variables[dr_var_name][t]
                    
                    self.model += (
                        dr_sum == 0,
                        f"dr_balance_{region}_{uuid.uuid4().hex[:8]}"
                    )
        
        # 2. Add ramping constraints (if not skipped)
        if not skip_ramping:
            for region in self.regions:
                for tech in self.dispatch_techs:
                    # Get technology parameters
                    ramp_rate = self.config.get('tech_params', {}).get(tech, {}).get('ramp_rate', 0.0)  # % of capacity per hour
                    
                    # Skip if no ramping constraint for this tech
                    if ramp_rate <= 0:
                        continue
                    
                    # Get tech capacity for this region
                    capacity = 0
                    if region in self.tech_capacities and tech in self.tech_capacities[region]:
                        capacity = float(self.tech_capacities[region][tech])
                    
                    # Skip if no capacity
                    if capacity <= 0:
                        continue
                    
                    # Calculate maximum ramp in MW per time step
                    # Determine time step dynamically if possible
                    time_step_hours = 1.0  # Default 1-hour resolution
                    
                    # Try to determine time resolution from data if available
                    if len(T) > 1 and isinstance(T[0], int) and isinstance(T[1], int):
                        # For integer time steps, assume hourly by default
                        time_step_hours = 1.0
                    elif len(T) > 1 and not isinstance(T[0], int) and not isinstance(T[1], int):
                        # For datetime time steps, calculate the difference
                        time_diff_seconds = (T[1] - T[0]).total_seconds()
                        time_step_hours = time_diff_seconds / 3600.0
                    
                    # Calculate maximum ramp based on the determined time step
                    max_ramp = capacity * ramp_rate / 100.0 * time_step_hours
                    
                    # Check if dispatch variables exist for this tech/region
                    dispatch_var_name = f"dispatch_{tech}_{region}"
                    if dispatch_var_name not in self.variables:
                        continue
                        
                    # Add ramping constraints between consecutive time periods
                    for i in range(len(T) - 1):
                        t = T[i]
                        t_next = T[i+1]
                        
                        # Skip if variables don't exist for these time periods
                        if (t not in self.variables[dispatch_var_name] or 
                            t_next not in self.variables[dispatch_var_name]):
                            continue
                        
                        # Check if times are consecutive for proper ramping constraint
                        is_consecutive = (isinstance(t, int) and isinstance(t_next, int) and t_next == t + 1) or \
                                        (not isinstance(t, int) and not isinstance(t_next, int) and \
                                         (t_next - t).total_seconds() / 3600 <= time_step_hours * 1.01)  # Within expected time step
                        
                        if not is_consecutive:
                            continue
                        
                        # Add up-ramping constraint (t_next - t_curr <= max_ramp)
                        self.model += (
                            self.variables[dispatch_var_name][t_next] - self.variables[dispatch_var_name][t] <= max_ramp,
                            f"ramp_up_{tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                        )
                        
                        # Add down-ramping constraint (t_curr - t_next <= max_ramp)
                        self.model += (
                            self.variables[dispatch_var_name][t] - self.variables[dispatch_var_name][t_next] <= max_ramp,
                            f"ramp_down_{tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                        )

    def _add_capacity_constraints(self, data: Dict[str, pd.DataFrame], T):
        """Add capacity constraints for dispatch, exchange, and transport.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary mapping region names to data frames
            T (List): List of time periods
        """
        logger.info("Adding capacity constraints")
        
        # Get global constraints
        constraints = self.config.get('constraints', {})
        
        # Add capacity constraints for each region and technology
        for region in self.regions:
            # Get regional capacities
            regional_caps = self.tech_capacities.get(region, {})
            
            # Add dispatch capacity constraints
            for tech in self.dispatch_techs:
                if f"dispatch_{tech}_{region}" in self.variables:
                    for t in T:
                        # Get regional capacity for this technology
                        max_capacity = regional_caps.get(tech)
                        
                        if max_capacity is not None:
                            # Ensure dispatch does not exceed capacity
                            self.model += (
                                self.variables[f"dispatch_{tech}_{region}"][t] <= max_capacity,
                                f"max_dispatch_{tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                            )
            
            # Add storage capacity constraints
            for storage_tech in self.storage_techs:
                if f"storage_charge_{storage_tech}_{region}" in self.variables:
                    for t in T:
                        # Get storage capacity for this technology
                        storage_key = f"{storage_tech}_puissance_MW"
                        max_capacity = self.storage_capacities.get(region, {}).get(storage_key, constraints.get('max_storage', 5000.0))
                        
                        # Charge rate constraint
                        self.model += (
                            self.variables[f"storage_charge_{storage_tech}_{region}"][t] <= max_capacity,
                            f"max_charge_{storage_tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                        )
                        
                        # Discharge rate constraint
                        self.model += (
                            self.variables[f"storage_discharge_{storage_tech}_{region}"][t] <= max_capacity,
                            f"max_discharge_{storage_tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                        )
            
            # Add exchange capacity constraints
            for other_region in self.regions:
                if other_region != region:
                    # Get regional exchange capacity
                    max_exchange = constraints.get('max_exchange', 2000.0)
                    
                    # Add constraints for both directions
                    if f"exchange_{region}_{other_region}" in self.variables:
                        for t in T:
                            self.model += (
                                self.variables[f"exchange_{region}_{other_region}"][t] <= max_exchange,
                                f"max_exchange_{region}_{other_region}_{t}_{uuid.uuid4().hex[:8]}"
                            )
                    
                    if f"exchange_{other_region}_{region}" in self.variables:
                        for t in T:
                            self.model += (
                                self.variables[f"exchange_{other_region}_{region}"][t] <= max_exchange,
                                f"max_exchange_{other_region}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                            )
        
        
            # Get storage-related constraints from config
            constraints = self.config.get('constraints', {})
            
            # Default storage constraints
            default_storage_constraints = {
                'storage_efficiency': 0.85,  # Round-trip efficiency
                'max_storage_capacity': 10000,  # MWh
                'max_storage_charge_rate': 1000,  # MW
                'max_storage_discharge_rate': 1000,  # MW
                'storage_charge_efficiency': 0.95,  # Charge efficiency
                'storage_discharge_efficiency': 0.95  # Discharge efficiency
            }
            
            # Use config values or defaults
            for key, value in default_storage_constraints.items():
                if key not in constraints:
                    constraints[key] = value
            
            # Storage constraints for each region and storage technology
            for region in self.regions:
                for storage_tech in self.storage_techs:
                    # Get storage capacities from config
                    max_energy_capacity = constraints['max_storage_capacity']
                    max_charge_rate = constraints['max_storage_charge_rate']
                    max_discharge_rate = constraints['max_storage_discharge_rate']
                    
                    # Check for technology and region specific capacity values
                    storage_capacity_key = f"{storage_tech}_capacity"
                    if region in self.tech_capacities and storage_capacity_key in self.tech_capacities[region]:
                        max_energy_capacity = float(self.tech_capacities[region][storage_capacity_key])
                        logger.debug(f"Using regional {storage_tech} capacity for {region}: {max_energy_capacity} MWh")
                        
                        # Also adjust charge/discharge rates based on capacity
                        # Typical charge/discharge rates are about 1/4 to 1/6 of total capacity for most storage techs
                        max_charge_rate = min(max_energy_capacity / 4, constraints['max_storage_charge_rate'])
                        max_discharge_rate = min(max_energy_capacity / 4, constraints['max_storage_discharge_rate'])
                    
                    # Create variable for storage level (state of charge)
                    if f"storage_level_{storage_tech}_{region}" not in self.variables:
                        self.variables[f"storage_level_{storage_tech}_{region}"] = {}
                        
                        for t in T:
                            self.variables[f"storage_level_{storage_tech}_{region}"][t] = LpVariable(
                                f"storage_level_{storage_tech}_{region}_{t}",
                                lowBound=0,
                                upBound=max_energy_capacity
                            )
                    
                    # Get efficiencies
                    charge_efficiency = constraints.get(f"{storage_tech}_charge_efficiency", 
                                                       constraints['storage_charge_efficiency'])
                    discharge_efficiency = constraints.get(f"{storage_tech}_discharge_efficiency", 
                                                          constraints['storage_discharge_efficiency'])
                    
                    # 1. Charging and discharging rate constraints
                    for t in T:
                        # Charge rate constraint
                        if f"storage_charge_{storage_tech}_{region}" in self.variables and t in self.variables[f"storage_charge_{storage_tech}_{region}"]:
                            self.model += (
                                self.variables[f"storage_charge_{storage_tech}_{region}"][t] <= max_charge_rate,
                                f"max_charge_{storage_tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                            )
                        
                        # Discharge rate constraint
                        if f"storage_discharge_{storage_tech}_{region}" in self.variables and t in self.variables[f"storage_discharge_{storage_tech}_{region}"]:
                            self.model += (
                                self.variables[f"storage_discharge_{storage_tech}_{region}"][t] <= max_discharge_rate,
                                f"max_discharge_{storage_tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                            )
                    
                    # 2. Storage level dynamics - initial condition
                    if T and 0 in T:
                        # Start with storage at 50% unless otherwise specified
                        initial_storage_level = max_energy_capacity * 0.5
                        
                        # Check if storage initialization parameters exist
                        storage_initial_key = f"{storage_tech}_initial"
                        
                        # Try to find initial storage level in various config sections
                        if 'storage_initial' in self.config:
                            if region in self.config['storage_initial'] and storage_tech in self.config['storage_initial'][region]:
                                initial_storage_level = float(self.config['storage_initial'][region][storage_tech])
                        elif region in self.tech_params and storage_initial_key in self.tech_params[region]:
                            initial_storage_level = float(self.tech_params[region][storage_initial_key])
                        # If we have region in regional_params, try that too
                        elif hasattr(self, 'regional_params') and region in self.regional_params and storage_initial_key in self.regional_params[region]:
                            initial_storage_level = float(self.regional_params[region][storage_initial_key])
                        
                        # Set initial storage level
                        self.model += (
                            self.variables[f"storage_level_{storage_tech}_{region}"][0] == initial_storage_level,
                            f"initial_storage_{storage_tech}_{region}_{uuid.uuid4().hex[:8]}"
                        )
                    
                    # 3. Storage level dynamics - time evolution
                    for i in range(len(T) - 1):
                        t = T[i]
                        t_next = T[i+1]
                        
                        # If the times are consecutive, use standard evolution
                        is_consecutive = (isinstance(t, int) and isinstance(t_next, int) and t_next == t + 1) or \
                                        (not isinstance(t, int) and not isinstance(t_next, int) and \
                                         (t_next - t).total_seconds() / 3600 <= 1.01)  # Within 1.01 hours (allowing for rounding)
                        
                        if not is_consecutive:
                            continue
                        
                        # Storage evolution: level[t+1] = level[t] + charge[t]*efficiency - discharge[t]/efficiency
                        if all(k in self.variables and t in self.variables[k] and t_next in self.variables[k] for k in [
                            f"storage_level_{storage_tech}_{region}",
                            f"storage_charge_{storage_tech}_{region}", 
                            f"storage_discharge_{storage_tech}_{region}"
                        ]):
                            self.model += (
                                self.variables[f"storage_level_{storage_tech}_{region}"][t_next] == \
                                self.variables[f"storage_level_{storage_tech}_{region}"][t] + \
                                self.variables[f"storage_charge_{storage_tech}_{region}"][t] * charge_efficiency - \
                                self.variables[f"storage_discharge_{storage_tech}_{region}"][t] * (1.0 / discharge_efficiency),
                                f"storage_evolution_{storage_tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                            )
                    
                    # 4. Optional: cyclical constraint (end level = start level)
                    skip_cyclical = self.use_simplified_model and self.simplification_options.get("skip_cyclical_storage", False)
                    if not skip_cyclical and self.config.get('use_cyclical_storage', False) and T and T[0] in self.variables[f"storage_level_{storage_tech}_{region}"] and T[-1] in self.variables[f"storage_level_{storage_tech}_{region}"]:
                        # Add the constraint that the final storage level should be close to the initial
                        self.model += (
                            self.variables[f"storage_level_{storage_tech}_{region}"][T[-1]] >= \
                            self.variables[f"storage_level_{storage_tech}_{region}"][T[0]] * 0.95,
                            f"cyclical_storage_min_{storage_tech}_{region}_{uuid.uuid4().hex[:8]}"
                        )
                        
                        self.model += (
                            self.variables[f"storage_level_{storage_tech}_{region}"][T[-1]] <= \
                            self.variables[f"storage_level_{storage_tech}_{region}"][T[0]] * 1.05,
                            f"cyclical_storage_max_{storage_tech}_{region}_{uuid.uuid4().hex[:8]}"
                        )

    def _add_cyclical_storage_constraints(self, T):
        """Add constraints for cyclical storage (end SOC = start SOC)
        
        Args:
            T (List): List of time periods
        """
        # Skip if no storage technologies defined
        if not hasattr(self, 'storage_techs') or not self.storage_techs:
            return
        
        # Only add if we have at least a day's worth of data
        if len(T) < 24:  # Assuming hourly data
            logger.info("Time period too short, skipping cyclical storage constraint")
            return
            
        # Get first and last time step
        first_t = T[0]
        last_t = T[-1]
        
        # Process each region and storage technology
        for region in self.regions:
            for tech in self.storage_techs:
                # Skip if storage variables not defined for this region/tech
                if (f"storage_soc_{tech}_{region}" not in self.variables or
                    first_t not in self.variables[f"storage_soc_{tech}_{region}"] or
                    last_t not in self.variables[f"storage_soc_{tech}_{region}"]):
                    continue
                
                # Add constraint: SOC[last_t] = SOC[first_t]
                self.model += (
                    self.variables[f"storage_soc_{tech}_{region}"][last_t] == 
                    self.variables[f"storage_soc_{tech}_{region}"][first_t],
                    f"storage_cyclical_{tech}_{region}"
                )

    def _add_flexibility_diversity_constraints(self, data: Dict[str, pd.DataFrame], T):
        """Add constraints to ensure a balanced utilization of different flexibility options.
        
        This method implements constraints that enforce minimum utilization thresholds for each 
        flexibility option (dispatch, storage, demand response, exchange) to ensure a balanced 
        and diverse flexibility portfolio.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary mapping region names to data frames
            T (List): List of time periods
        """
        logger.info("Adding flexibility diversity constraints")
        
        # Get minimum utilization percentages from config or use defaults
        min_storage_utilization = self.config.get('constraints', {}).get('min_storage_utilization', 0.15)  # 15% minimum storage contribution
        min_dr_utilization = self.config.get('constraints', {}).get('min_dr_utilization', 0.10)  # 10% minimum demand response contribution
        min_exchange_utilization = self.config.get('constraints', {}).get('min_exchange_utilization', 0.15)  # 15% minimum exchange contribution
        
        # 1. Calculate total flexibility potential for each region
        for region in self.regions:
            # Calculate total demand to determine flexibility needs
            total_demand = {}
            regional_data = data.get(region)
            if regional_data is None:
                logger.warning(f"No data for region {region} in diversity constraints")
                continue
            
            # Get demand values for this region
            for t in T:
                if t < len(regional_data):
                    for col in ['consumption', 'demand', 'load']:
                        if col in regional_data.columns:
                            demand_value = float(regional_data.iloc[t][col])
                            if demand_value > 0:  # Only consider positive demand
                                total_demand[t] = demand_value
                                break
            
            # Skip if no demand data
            if not total_demand:
                logger.warning(f"No demand data for region {region}, skipping diversity constraints")
                continue
            
            # Add minimum storage utilization constraint for each time period
            for t in T:
                if t not in total_demand:
                    continue
                    
                # Calculate total storage activity (charge + discharge)
                storage_activity = 0
                for storage_tech in self.storage_techs:
                    if f"storage_charge_{storage_tech}_{region}" in self.variables and t in self.variables[f"storage_charge_{storage_tech}_{region}"]:
                        storage_activity += self.variables[f"storage_charge_{storage_tech}_{region}"][t]
                    
                    if f"storage_discharge_{storage_tech}_{region}" in self.variables and t in self.variables[f"storage_discharge_{storage_tech}_{region}"]:
                        storage_activity += self.variables[f"storage_discharge_{storage_tech}_{region}"][t]
                
                # Add constraint: storage activity >= min_percentage * total_demand
                min_storage_req = min_storage_utilization * total_demand[t]
                self.model += (storage_activity >= min_storage_req, 
                               f"min_storage_utilization_{region}_{t}")
                
            # Add minimum demand response utilization constraint
            for t in T:
                if t not in total_demand:
                    continue
                    
                # Get demand response variables
                if f"demand_response_{region}" in self.variables and t in self.variables[f"demand_response_{region}"]:
                    dr_var = self.variables[f"demand_response_{region}"][t]
                    
                    # Create absolute value for demand response
                    dr_abs = LpVariable(f"dr_abs_{region}_{t}", lowBound=0)
                    self.model += (dr_abs >= dr_var, f"dr_abs_pos_{region}_{t}")
                    self.model += (dr_abs >= -dr_var, f"dr_abs_neg_{region}_{t}")
                    
                    # Add constraint: |DR| >= min_percentage * total_demand
                    min_dr_req = min_dr_utilization * total_demand[t]
                    self.model += (dr_abs >= min_dr_req, 
                                  f"min_dr_utilization_{region}_{t}")
        
        # 2. Add minimum exchange utilization constraints
        for i, r1 in enumerate(self.regions):
            for r2 in self.regions[i+1:]:
                for t in T:
                    # Skip if either region has no demand data
                    r1_demand = None
                    r2_demand = None
                    
                    for region, var_name in [(r1, 'r1_demand'), (r2, 'r2_demand')]:
                        regional_data = data.get(region)
                        if regional_data is not None and t < len(regional_data):
                            for col in ['consumption', 'demand', 'load']:
                                if col in regional_data.columns:
                                    demand_value = float(regional_data.iloc[t][col])
                                    if demand_value > 0:
                                        if var_name == 'r1_demand':
                                            r1_demand = demand_value
                                        else:
                                            r2_demand = demand_value
                                        break
                    
                    if r1_demand is None or r2_demand is None:
                        continue
                    
                    # Get exchange variable
                    if f"abs_exchange_{r1}_{r2}" in self.variables and t in self.variables[f"abs_exchange_{r1}_{r2}"]:
                        exchange_var = self.variables[f"abs_exchange_{r1}_{r2}"][t]
                        
                        # Calculate average demand of both regions
                        avg_demand = (r1_demand + r2_demand) / 2
                        
                        # Add constraint: |exchange| >= min_percentage * avg_demand
                        min_exchange_req = min_exchange_utilization * avg_demand
                        self.model += (exchange_var >= min_exchange_req, 
                                      f"min_exchange_utilization_{r1}_{r2}_{t}")
                    
                    # Similarly for transport variables
                    if f"abs_transport_{r1}_{r2}" in self.variables and t in self.variables[f"abs_transport_{r1}_{r2}"]:
                        transport_var = self.variables[f"abs_transport_{r1}_{r2}"][t]
                        
                        # Calculate average demand of both regions
                        avg_demand = (r1_demand + r2_demand) / 2
                        
                        # Add constraint: |transport| >= min_percentage * avg_demand
                        min_transport_req = min_exchange_utilization * avg_demand / 2  # Divide by 2 to avoid overconstraining
                        self.model += (transport_var >= min_transport_req, 
                                      f"min_transport_utilization_{r1}_{r2}_{t}")
        
        logger.info(f"Added flexibility diversity constraints with minimum utilization thresholds: "
                   f"Storage={min_storage_utilization*100:.1f}%, "
                   f"DR={min_dr_utilization*100:.1f}%, "
                   f"Exchange={min_exchange_utilization*100:.1f}%")

    def solve(self, time_limit=None, solver=None, threads=None):
        """Solve the optimization model.
        
        Args:
            time_limit (int, optional): Time limit in seconds. Defaults to None.
            solver (str, optional): Solver to use (e.g., 'PULP_CBC_CMD', 'GUROBI_CMD'). Defaults to None.
            threads (int, optional): Number of threads to use. Defaults to None.
        
        Returns:
            tuple: (status, solve_time_seconds)
        """
        logger.info("Running optimization model")
        
        # Apply LP relaxation by forcing binary variables to be continuous if enabled
        if self.use_simplified_model and self.simplification_options.get("lp_relaxation", False):
            logger.info("Converting MILP to LP by relaxing integer constraints")
            for var in self.binary_vars:
                # Change binary variables to continuous with bounds [0,1]
                var.cat = pl.LpContinuous
                var.lowBound = 0
                var.upBound = 1
        
        # Save model to file if debugging is enabled
        if logger.level <= logging.DEBUG:
            self.model.writeLP("regional_flex_model.lp")
            logger.debug("Model written to regional_flex_model.lp")
        
        # Set up solver options
        solver_options = []
        if time_limit is not None:
            solver_options.append(("timeLimit", time_limit))
            
        if threads is not None:
            solver_options.append(("threads", threads))
        
        # Choose solver
        if solver == "GUROBI_CMD":
            # Check if GUROBI_CMD is available in PuLP
            try:
                from pulp import GUROBI_CMD
                solver_to_use = GUROBI_CMD(options=solver_options)
            except ImportError:
                logger.warning("GUROBI_CMD not available, falling back to CBC")
                solver_options_str = [f"{name}={value}" for name, value in solver_options]
                solver_to_use = PULP_CBC_CMD(options=solver_options_str)
        elif solver == "CPLEX_CMD":
            # Check if CPLEX_CMD is available in PuLP
            try:
                from pulp import CPLEX_CMD
                solver_to_use = CPLEX_CMD(options=solver_options)
            except ImportError:
                logger.warning("CPLEX_CMD not available, falling back to CBC")
                solver_options_str = [f"{name}={value}" for name, value in solver_options]
                solver_to_use = PULP_CBC_CMD(options=solver_options_str)
        else:
            # Default to CBC with options
            solver_options_str = [f"{name}={value}" for name, value in solver_options]
            solver_to_use = PULP_CBC_CMD(options=solver_options_str)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Solve the model with the selected solver
            status = self.model.solve(solver=solver_to_use)
            logger.info(f"Optimization completed with status: {LpStatus[status]}")
            
            # Calculate solve time
            solve_time = time.time() - start_time
            logger.info(f"Solve time: {solve_time:.2f} seconds")
            
            # Get objective value
            if status == LpStatusOptimal:
                objective_value = value(self.model.objective)
                logger.info(f"Objective value: {objective_value}")
            
            return status, solve_time
            
        except Exception as e:
            logger.error(f"Error solving model: {e}")
            return pl.LpStatusNotSolved, time.time() - start_time

    def run_model(self, time_limit=None, threads=None):
        """Run the optimization model - wrapper for the solve method.
        
        Args:
            time_limit (int, optional): Maximum solver time in seconds
            threads (int, optional): Number of parallel threads to use for solving
            
        Returns:
            Tuple[str, float]: Optimization status and solve time in seconds
        """
        return self.solve(time_limit=time_limit, threads=threads)
    
    def get_results(self) -> Dict:
        """Extract results from the optimized model.
        
        Returns:
            Dict: Dictionary with optimization results including variable values and metadata
        """
        logger.info("Extracting optimization results")
        
        # Check if model was solved
        if self.model.status != LpStatusOptimal:
            logger.warning(f"Model not optimally solved (status: {LpStatus[self.model.status]}), results may be suboptimal")
        
        # Initialize results dictionary
        results = {
            'status': LpStatus[self.model.status],
            'objective_value': self.model.objective.value,
            'regions': self.regions,
            'dispatch_techs': self.dispatch_techs,
            'storage_techs': self.storage_techs,
            'variables': {}
        }
        
        # Extract variable values
        for var_name, var_dict in self.variables.items():
            # Initialize the variable result dictionary
            results['variables'][var_name] = {}
            
            # Extract each time period value
            for t, var in var_dict.items():
                results['variables'][var_name][t] = var.value()
        
        # Add metadata about the optimization
        results['metadata'] = {
            'model_name': self.model.name,
            'num_variables': len(self.model.variables()),
            'num_constraints': len(self.model.constraints),
            'timestamp': datetime.now().isoformat(),
            'config_path': self.config_path
        }
        
        # Calculate total generation, imports, exports by region
        results['regional_summary'] = {}
        for region in self.regions:
            results['regional_summary'][region] = {
                'total_dispatch': {},
                'total_storage_charge': {},
                'total_storage_discharge': {},
                'total_imports': 0,
                'total_exports': 0,
                'demand_response': {}
            }
            
            # Sum up technology dispatch
            for tech in self.dispatch_techs:
                var_key = f"dispatch_{tech}_{region}"
                if var_key in results['variables']:
                    total_dispatch = sum(results['variables'][var_key].values())
                    results['regional_summary'][region]['total_dispatch'][tech] = total_dispatch
            
            # Sum up storage operations
            for storage_tech in self.storage_techs:
                # Charging
                var_key = f"storage_charge_{storage_tech}_{region}"
                if var_key in results['variables']:
                    total_charge = sum(results['variables'][var_key].values())
                    results['regional_summary'][region]['total_storage_charge'][storage_tech] = total_charge
                
                # Discharging
                var_key = f"storage_discharge_{storage_tech}_{region}"
                if var_key in results['variables']:
                    total_discharge = sum(results['variables'][var_key].values())
                    results['regional_summary'][region]['total_storage_discharge'][storage_tech] = total_discharge
            
            # Sum up demand response
            var_key = f"demand_response_{region}"
            if var_key in results['variables']:
                total_dr = sum(results['variables'][var_key].values())
                results['regional_summary'][region]['demand_response']['total'] = total_dr
                
                # Count positive and negative DR hours
                pos_dr = sum(1 for v in results['variables'][var_key].values() if v > 0)
                neg_dr = sum(1 for v in results['variables'][var_key].values() if v < 0)
                results['regional_summary'][region]['demand_response']['positive_hours'] = pos_dr
                results['regional_summary'][region]['demand_response']['negative_hours'] = neg_dr
        
        # Calculate exchanges between regions
        results['exchanges'] = {}
        for i, r1 in enumerate(self.regions):
            for r2 in self.regions[i+1:]:
                var_key = f"exchange_{r1}_{r2}"
                if var_key in results['variables']:
                    # Net exchange between regions
                    exchange_values = results['variables'][var_key]
                    net_exchange = sum(exchange_values.values())
                    
                    # Calculate imports and exports for each region
                    for t, value in exchange_values.items():
                        if value > 0:  # Positive means r1 exports to r2
                            results['regional_summary'][r1]['total_exports'] += value
                            results['regional_summary'][r2]['total_imports'] += value
                        else:  # Negative means r2 exports to r1
                            results['regional_summary'][r1]['total_imports'] += -value
                            results['regional_summary'][r2]['total_exports'] += -value
                    
                    # Record exchange information
                    results['exchanges'][f"{r1}_to_{r2}"] = {
                        'net_exchange': net_exchange,
                        'time_series': exchange_values
                    }
        
        # Calculate slack usage
        results['slack_usage'] = {}
        for region in self.regions:
            pos_slack_key = f"slack_pos_{region}"
            neg_slack_key = f"slack_neg_{region}"
            
            total_pos_slack = 0
            total_neg_slack = 0
            
            if pos_slack_key in results['variables']:
                total_pos_slack = sum(results['variables'][pos_slack_key].values())
            
            if neg_slack_key in results['variables']:
                total_neg_slack = sum(results['variables'][neg_slack_key].values())
            
            results['slack_usage'][region] = {
                'positive': total_pos_slack,
                'negative': total_neg_slack,
                'total': total_pos_slack + total_neg_slack
            }

        # after the slack_usage loop
        results['curtailment'] = {}
        for region in self.regions:
            var_key = f"curtail_{region}"
            if var_key in results['variables']:
                results['curtailment'][region] = sum(results['variables'][var_key].values())

        
        return results
