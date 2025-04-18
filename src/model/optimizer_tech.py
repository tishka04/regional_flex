import pandas as pd
import numpy as np
from pulp import *
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional, Union
import time

logger = logging.getLogger(__name__)

class RegionalFlexOptimizerTech:
    """Multi-regional energy flexibility optimization model with technology-specific variables."""
    
    def __init__(self, config: Dict):
        """Initialize the RegionalFlexOptimizer.
        
        Args:
            config: Configuration dictionary containing:
                - regions: List of region names
                - constraints: Dictionary of constraint values
                - costs: Dictionary of cost values
        """
        self.config = config
        self.model = LpProblem("RegionalFlexibility", LpMinimize)
        self.variables = None
        self.regions = config["regions"]
        
        # Define dispatchable technologies
        self.dispatch_techs = ["hydro", "nuclear", "thermal", "biofuel"]
        
        # Define storage technologies
        self.storage_techs = ["battery", "phs"]  # phs = pumped hydro storage
        
    def build_model(self, regional_data: Dict[str, pd.DataFrame]):
        """Build the optimization model with progress tracking.
        
        Args:
            regional_data: Dictionary of preprocessed regional data
        """
        logger.info("Building optimization model...")
        
        # Make sure the data is properly formatted
        if not isinstance(regional_data, dict):
            logger.warning(f"regional_data is not a dictionary, it's a {type(regional_data)}")
            if isinstance(regional_data, pd.DataFrame):
                logger.warning("Converting DataFrame to dictionary")
                # Assuming the DataFrame has a column for each region
                regional_data = {region: regional_data[[col for col in regional_data.columns if region in col]] 
                                for region in self.regions}
        
        # Extract time periods from data
        if isinstance(regional_data, dict) and len(regional_data) > 0:
            first_region = list(regional_data.keys())[0]
            if isinstance(regional_data[first_region], pd.DataFrame):
                time_periods = list(range(len(regional_data[first_region])))
            else:
                logger.error(f"Data for {first_region} is not a DataFrame")
                time_periods = []
        else:
            logger.error("Empty regional data")
            time_periods = []
            
        logger.info(f"Building model with {len(time_periods)} time periods")
        
        # Reset the model and variables
        self.model = LpProblem("RegionalFlexibilityModel", LpMinimize)
        
        # Create progress bar for model building steps
        steps = ['Adding variables', 'Adding objective function', 'Adding constraints']
        with tqdm(total=len(steps), desc="Building model", position=0) as pbar:
            # Add decision variables
            pbar.set_description("Adding variables")
            self._add_variables(regional_data, time_periods)
            pbar.update(1)
            
            # Add objective function
            pbar.set_description("Adding objective function")
            self._add_objective(regional_data, time_periods)
            pbar.update(1)
            
            # Add constraints
            pbar.set_description("Adding constraints")
            self._add_constraints(regional_data, time_periods)
            pbar.update(1)
            
        logger.info("Optimization model built successfully")
        
    def _add_variables(self, data: Dict[str, pd.DataFrame], time_periods):
        """Add decision variables to the model.
        
        Args:
            data: Dictionary of DataFrames by region
            time_periods: Number of time periods or list of time periods
        """
        # Convert time_periods to a list if it's an integer
        if isinstance(time_periods, int):
            T = list(range(time_periods))
        else:
            T = time_periods
            
        logger.info(f"Adding variables for {len(self.regions)} regions and {len(T) if T else 0} time periods")
        
        # Initialize variables dictionary
        self.variables = {}
        
        # Create a progress bar for variable types
        variable_types = ["Technology-specific Dispatch", "Storage Technologies", "Demand response", 
                          "Interregional exchange", "Transportation network exchange", "Slack"]
        
        with tqdm(total=len(variable_types), desc="Adding variables", position=1, leave=False) as pbar:
            # 1. Dispatch variables for each technology and region
            pbar.set_description("Technology-specific Dispatch variables")
            for region in self.regions:
                for tech in self.dispatch_techs:
                    self.variables[f"dispatch_{tech}_{region}"] = LpVariable.dicts(
                        f"dispatch_{tech}_{region}", 
                        T, 
                        lowBound=0,  # No negative dispatch
                        upBound=self.config["constraints"]["max_dispatch"]
                    )
            pbar.update(1)
            
            # 2. Storage variables for each storage technology and region
            pbar.set_description("Storage Technology variables")
            for region in self.regions:
                for storage_tech in self.storage_techs:
                    # Get storage capacity for this technology and region
                    storage_capacity = self.config["constraints"]["storage_capacity"]
                    
                    # Default capacity ratio: 70% battery, 30% PHS 
                    # (adjust these values according to your data/requirements)
                    tech_capacity_ratio = 0.7 if storage_tech == "battery" else 0.3
                    
                    # Check for region-specific storage capacity
                    if "regional_constraints" in self.config and region in self.config["regional_constraints"]:
                        if "storage_capacity" in self.config["regional_constraints"][region]:
                            storage_capacity = self.config["regional_constraints"][region]["storage_capacity"]
                    
                    # Apply technology ratio to get technology-specific capacity
                    tech_storage_capacity = storage_capacity * tech_capacity_ratio
                    
                    # Storage level variables for this technology
                    self.variables[f"storage_level_{storage_tech}_{region}"] = LpVariable.dicts(
                        f"storage_level_{storage_tech}_{region}", 
                        T, 
                        lowBound=0, 
                        upBound=tech_storage_capacity
                    )
                    
                    # Charge and discharge variables for this technology
                    max_charge_rate = tech_storage_capacity * 0.3  # Assume max 30% charge rate
                    
                    self.variables[f"storage_charge_{storage_tech}_{region}"] = LpVariable.dicts(
                        f"storage_charge_{storage_tech}_{region}", 
                        T, 
                        lowBound=0, 
                        upBound=max_charge_rate
                    )
                    
                    self.variables[f"storage_discharge_{storage_tech}_{region}"] = LpVariable.dicts(
                        f"storage_discharge_{storage_tech}_{region}", 
                        T, 
                        lowBound=0, 
                        upBound=max_charge_rate  # Assume same rate for discharge
                    )
            pbar.update(1)
            
            # 3. Demand response variables
            pbar.set_description("Demand response variables")
            for region in self.regions:
                self.variables[f"demand_response_{region}"] = LpVariable.dicts(
                    f"demand_response_{region}", 
                    T, 
                    lowBound=-self.config["constraints"]["max_demand_response"],
                    upBound=self.config["constraints"]["max_demand_response"]
                )
            pbar.update(1)
            
            # 4. Interregional exchange variables
            pbar.set_description("Interregional exchange variables")
            for i, r1 in enumerate(self.regions):
                for r2 in self.regions[i+1:]:
                    # Directed exchange from r1 to r2
                    self.variables[f"exchange_{r1}_{r2}"] = LpVariable.dicts(
                        f"exchange_{r1}_{r2}", 
                        T, 
                        lowBound=-self.config["constraints"]["exchange_capacity"],
                        upBound=self.config["constraints"]["exchange_capacity"]
                    )
                    
                    # Absolute value of exchange for cost calculation
                    self.variables[f"abs_exchange_{r1}_{r2}"] = LpVariable.dicts(
                        f"abs_exchange_{r1}_{r2}", 
                        T, 
                        lowBound=0
                    )
            pbar.update(1)
            
            # 5. Transportation network exchange variables
            pbar.set_description("Transportation network exchange variables")
            for i, r1 in enumerate(self.regions):
                for r2 in self.regions[i+1:]:
                    # Transport network exchange (e.g., for electric vehicles)
                    self.variables[f"transport_{r1}_{r2}"] = LpVariable.dicts(
                        f"transport_{r1}_{r2}", 
                        T, 
                        lowBound=-self.config["constraints"]["transport_capacity"],
                        upBound=self.config["constraints"]["transport_capacity"]
                    )
                    
                    # Absolute value for cost calculation
                    self.variables[f"abs_transport_{r1}_{r2}"] = LpVariable.dicts(
                        f"abs_transport_{r1}_{r2}", 
                        T, 
                        lowBound=0
                    )
            pbar.update(1)
            
            # 6. Slack variables for infeasibility handling
            pbar.set_description("Slack variables")
            for region in self.regions:
                # Positive slack (excess generation)
                self.variables[f"slack_pos_{region}"] = LpVariable.dicts(
                    f"slack_pos_{region}", 
                    T, 
                    lowBound=0,
                    upBound=self.config["constraints"]["max_slack"]
                )
                
                # Negative slack (unmet demand)
                self.variables[f"slack_neg_{region}"] = LpVariable.dicts(
                    f"slack_neg_{region}", 
                    T, 
                    lowBound=0,
                    upBound=self.config["constraints"]["max_slack"]
                )
            pbar.update(1)
    
    def _add_objective(self, data: Dict[str, pd.DataFrame], time_periods):
        """Add objective function to the model.
        
        Args:
            data: Dictionary of DataFrames by region
            time_periods: Number of time periods or list of time periods
        """
        # Convert time_periods to a list if it's an integer
        if isinstance(time_periods, int):
            T = list(range(time_periods))
        else:
            T = time_periods
            
        logger.info("Adding objective function...")
        
        objective = 0
        
        # Define technology-specific costs
        # These would ideally come from config, but we'll define them here for now
        tech_costs = {
            "hydro": self.config["costs"]["dispatch"] * 0.8,  # 20% cheaper than base dispatch
            "nuclear": self.config["costs"]["dispatch"] * 0.9,  # 10% cheaper than base dispatch
            "thermal": self.config["costs"]["dispatch"] * 1.2,  # 20% more expensive than base dispatch
            "biofuel": self.config["costs"]["dispatch"] * 1.1  # 10% more expensive than base dispatch
        }
        
        storage_tech_costs = {
            "battery": {
                "level": self.config["costs"]["storage"] * 1.1,  # 10% more expensive for battery storage
                "charge": self.config["costs"]["storage_charge"] * 1.0,
                "discharge": self.config["costs"]["storage_discharge"] * 1.0
            },
            "phs": {
                "level": self.config["costs"]["storage"] * 0.9,  # 10% cheaper for pumped hydro storage
                "charge": self.config["costs"]["storage_charge"] * 1.1,  # 10% more expensive for PHS charge
                "discharge": self.config["costs"]["storage_discharge"] * 0.9  # 10% cheaper for PHS discharge
            }
        }
        
        # 1. Technology-specific dispatch costs for each region
        for region in self.regions:
            # Check for region-specific dispatch cost multiplier
            cost_multiplier = 1.0
            if "regional_constraints" in self.config and region in self.config["regional_constraints"]:
                if "dispatch_cost_multiplier" in self.config["regional_constraints"][region]:
                    cost_multiplier = self.config["regional_constraints"][region]["dispatch_cost_multiplier"]
            
            try:
                for tech in self.dispatch_techs:
                    for t in T:
                        if f"dispatch_{tech}_{region}" in self.variables:
                            objective += self.variables[f"dispatch_{tech}_{region}"][t] * tech_costs[tech] * cost_multiplier
                        else:
                            logger.warning(f"Variable dispatch_{tech}_{region} not found in objective function")
            except Exception as e:
                logger.warning(f"Error adding dispatch cost for {region}: {e}")
        
        # 2. Storage technology-specific costs
        for region in self.regions:
            try:
                for storage_tech in self.storage_techs:
                    for t in T:
                        if f"storage_level_{storage_tech}_{region}" in self.variables:
                            objective += self.variables[f"storage_level_{storage_tech}_{region}"][t] * storage_tech_costs[storage_tech]["level"]
                        else:
                            logger.warning(f"Variable storage_level_{storage_tech}_{region} not found in objective function")
                        
                        if f"storage_charge_{storage_tech}_{region}" in self.variables:
                            objective += self.variables[f"storage_charge_{storage_tech}_{region}"][t] * storage_tech_costs[storage_tech]["charge"]
                        else:
                            logger.warning(f"Variable storage_charge_{storage_tech}_{region} not found in objective function")
                        
                        if f"storage_discharge_{storage_tech}_{region}" in self.variables:
                            objective += self.variables[f"storage_discharge_{storage_tech}_{region}"][t] * storage_tech_costs[storage_tech]["discharge"]
                        else:
                            logger.warning(f"Variable storage_discharge_{storage_tech}_{region} not found in objective function")
            except Exception as e:
                logger.warning(f"Error adding storage cost for {region}: {e}")
        
        # 3. Demand response costs
        for region in self.regions:
            try:
                for t in T:
                    if f"demand_response_{region}" in self.variables:
                        # Use absolute value of demand response for cost
                        objective += abs(self.variables[f"demand_response_{region}"][t]) * self.config["costs"]["demand_response"]
                    else:
                        logger.warning(f"Variable demand_response_{region} not found in objective function")
            except Exception as e:
                logger.warning(f"Error adding demand response cost for {region}: {e}")
        
        # 4. Exchange costs
        for r1 in self.regions:
            for r2 in self.regions:
                if r1 < r2:  # Avoid duplicates
                    try:
                        for t in T:
                            if f"abs_exchange_{r1}_{r2}" in self.variables:
                                objective += self.variables[f"abs_exchange_{r1}_{r2}"][t] * self.config["costs"]["exchange"]
                            else:
                                logger.warning(f"Variable abs_exchange_{r1}_{r2} not found in objective function")
                    except Exception as e:
                        logger.warning(f"Error adding exchange cost for {r1}-{r2}: {e}")
        
        # 5. Transport costs
        for r1 in self.regions:
            for r2 in self.regions:
                if r1 < r2:  # Avoid duplicates
                    try:
                        for t in T:
                            if f"abs_transport_{r1}_{r2}" in self.variables:
                                objective += self.variables[f"abs_transport_{r1}_{r2}"][t] * self.config["costs"]["transport_exchange"]
                            else:
                                logger.warning(f"Variable abs_transport_{r1}_{r2} not found in objective function")
                    except Exception as e:
                        logger.warning(f"Error adding transport cost for {r1}-{r2}: {e}")
        
        # 6. Slack penalties (high cost to discourage use)
        for region in self.regions:
            try:
                for t in T:
                    if f"slack_pos_{region}" in self.variables and f"slack_neg_{region}" in self.variables:
                        objective += (self.variables[f"slack_pos_{region}"][t] + 
                                     self.variables[f"slack_neg_{region}"][t]) * self.config["costs"]["slack_penalty"]
                    else:
                        logger.warning(f"Slack variables for {region} not found in objective function")
            except Exception as e:
                logger.warning(f"Error adding slack penalty for {region}: {e}")
        
        # Set the objective function
        self.model += objective
        
    def _add_constraints(self, data: Dict[str, pd.DataFrame], time_periods):
        """Add constraints to the model.
        
        Args:
            data: Dictionary of DataFrames by region
            time_periods: Number of time periods or list of time periods
        """
        # Convert time_periods to a list if it's an integer
        if isinstance(time_periods, int):
            T = list(range(time_periods))
        else:
            T = time_periods
            
        logger.info(f"Adding constraints for {len(self.regions)} regions and {len(T) if T else 0} time periods")
        
        # Create a progress bar for constraint types
        constraint_types = ["Regional balance", "Storage balance", "Capacity", 
                           "Exchange network", "Demand response", "Absolute value", "Ramping"]
        
        with tqdm(total=len(constraint_types), desc="Adding constraints", position=1, leave=False) as pbar:
            # 1. Regional energy balance constraints
            pbar.set_description("Regional balance constraints")
            for region in self.regions:
                try:
                    for t in T:
                        # Get net load (demand - renewable generation)
                        if region in data and isinstance(data[region], pd.DataFrame):
                            if 'net_load' in data[region].columns:
                                net_load_value = data[region]['net_load'].iloc[t].item() if isinstance(data[region]['net_load'].iloc[t], np.ndarray) else data[region]['net_load'].iloc[t]
                            else:
                                logger.warning(f"'net_load' column not found for region {region}, using 0")
                                net_load_value = 0
                        else:
                            logger.warning(f"No data found for region {region}, using 0 for net_load")
                            net_load_value = 0
                        
                        # Calculate all energy flows for this region at time t
                        lhs = 0  # Left-hand side of constraint equation
                        
                        # Add technology-specific dispatch for this region
                        for tech in self.dispatch_techs:
                            if f"dispatch_{tech}_{region}" in self.variables:
                                lhs += self.variables[f"dispatch_{tech}_{region}"][t]
                        
                        # Add technology-specific storage discharge and charge
                        for storage_tech in self.storage_techs:
                            if f"storage_discharge_{storage_tech}_{region}" in self.variables:
                                lhs += self.variables[f"storage_discharge_{storage_tech}_{region}"][t]
                            if f"storage_charge_{storage_tech}_{region}" in self.variables:
                                lhs -= self.variables[f"storage_charge_{storage_tech}_{region}"][t]
                        
                        # Add demand response (positive = demand reduction, negative = demand increase)
                        if f"demand_response_{region}" in self.variables:
                            lhs += self.variables[f"demand_response_{region}"][t]
                        
                        # Add exchange with other regions
                        for other_region in self.regions:
                            if region != other_region:
                                # Determine the correct variable name based on alphabetical order
                                if region < other_region:
                                    exchange_var = f"exchange_{region}_{other_region}"
                                    # Positive value = export from region to other_region
                                    if exchange_var in self.variables:
                                        lhs -= self.variables[exchange_var][t]
                                else:
                                    exchange_var = f"exchange_{other_region}_{region}"
                                    # Positive value = export from other_region to region
                                    if exchange_var in self.variables:
                                        lhs += self.variables[exchange_var][t]
                        
                        # Add transport network exchange
                        for other_region in self.regions:
                            if region != other_region:
                                # Determine the correct variable name based on alphabetical order
                                if region < other_region:
                                    transport_var = f"transport_{region}_{other_region}"
                                    if transport_var in self.variables:
                                        lhs -= self.variables[transport_var][t]
                                else:
                                    transport_var = f"transport_{other_region}_{region}"
                                    if transport_var in self.variables:
                                        lhs += self.variables[transport_var][t]
                        
                        # Add slack variables
                        if f"slack_pos_{region}" in self.variables:
                            lhs += self.variables[f"slack_pos_{region}"][t]
                        if f"slack_neg_{region}" in self.variables:
                            lhs -= self.variables[f"slack_neg_{region}"][t]
                        
                        # Energy balance: supply = demand
                        self.model += (lhs == net_load_value, f"balance_{region}_{t}")
                except Exception as e:
                    logger.warning(f"Error adding balance constraint for {region} at time {t}: {e}")
            pbar.update(1)
            
            # 2. Storage balance constraints for each technology
            pbar.set_description("Storage balance constraints")
            for region in self.regions:
                try:
                    # Get base storage capacity and initial level
                    storage_capacity = self.config["constraints"]["storage_capacity"]
                    
                    # Check for region-specific storage capacity
                    if "regional_constraints" in self.config and region in self.config["regional_constraints"]:
                        if "storage_capacity" in self.config["regional_constraints"][region]:
                            storage_capacity = self.config["regional_constraints"][region]["storage_capacity"]
                    
                    # Storage efficiency parameters
                    storage_efficiency = self.config["constraints"]["storage_efficiency"]
                    charge_efficiency = self.config["constraints"]["charge_efficiency"]
                    discharge_efficiency = self.config["constraints"]["discharge_efficiency"]
                    
                    # Apply balance constraints for each storage technology
                    for storage_tech in self.storage_techs:
                        # Get technology-specific capacity ratio
                        tech_capacity_ratio = 0.7 if storage_tech == "battery" else 0.3
                        tech_storage_capacity = storage_capacity * tech_capacity_ratio
                        initial_level = tech_storage_capacity * self.config["constraints"]["initial_storage_level"]
                        
                        # First time period uses initial level
                        if len(T) > 0:
                            if (f"storage_level_{storage_tech}_{region}" in self.variables and 
                                f"storage_charge_{storage_tech}_{region}" in self.variables and 
                                f"storage_discharge_{storage_tech}_{region}" in self.variables):
                                self.model += (
                                    self.variables[f"storage_level_{storage_tech}_{region}"][T[0]] == 
                                    initial_level + 
                                    self.variables[f"storage_charge_{storage_tech}_{region}"][T[0]] * charge_efficiency - 
                                    self.variables[f"storage_discharge_{storage_tech}_{region}"][T[0]] * (1.0 / discharge_efficiency),
                                    f"storage_balance_{storage_tech}_{region}_{T[0]}"
                                )
                        
                        # Subsequent time periods
                        for i in range(1, len(T)):
                            if (f"storage_level_{storage_tech}_{region}" in self.variables and 
                                f"storage_charge_{storage_tech}_{region}" in self.variables and 
                                f"storage_discharge_{storage_tech}_{region}" in self.variables):
                                self.model += (
                                    self.variables[f"storage_level_{storage_tech}_{region}"][T[i]] == 
                                    self.variables[f"storage_level_{storage_tech}_{region}"][T[i-1]] * storage_efficiency + 
                                    self.variables[f"storage_charge_{storage_tech}_{region}"][T[i]] * charge_efficiency - 
                                    self.variables[f"storage_discharge_{storage_tech}_{region}"][T[i]] * (1.0 / discharge_efficiency),
                                    f"storage_balance_{storage_tech}_{region}_{T[i]}"
                                )
                except Exception as e:
                    logger.warning(f"Error adding storage balance constraint for {region}: {e}")
            pbar.update(1)
            
            # 3. Capacity constraints
            pbar.set_description("Capacity constraints")
            # Already incorporated in variable bounds
            pbar.update(1)
            
            # 4. Exchange network constraints
            pbar.set_description("Exchange network constraints")
            # Absolute value constraints for exchange
            for i, r1 in enumerate(self.regions):
                for r2 in self.regions[i+1:]:
                    try:
                        for t in T:
                            if f"exchange_{r1}_{r2}" in self.variables and f"abs_exchange_{r1}_{r2}" in self.variables:
                                # Constraints to model absolute value: abs_var >= var and abs_var >= -var
                                self.model += (self.variables[f"abs_exchange_{r1}_{r2}"][t] >= 
                                               self.variables[f"exchange_{r1}_{r2}"][t], 
                                               f"abs_exch_pos_{r1}_{r2}_{t}")
                                self.model += (self.variables[f"abs_exchange_{r1}_{r2}"][t] >= 
                                               -self.variables[f"exchange_{r1}_{r2}"][t], 
                                               f"abs_exch_neg_{r1}_{r2}_{t}")
                    except Exception as e:
                        logger.warning(f"Error adding exchange constraints for {r1}-{r2}: {e}")
            
            # Absolute value constraints for transport exchange
            for i, r1 in enumerate(self.regions):
                for r2 in self.regions[i+1:]:
                    try:
                        for t in T:
                            if f"transport_{r1}_{r2}" in self.variables and f"abs_transport_{r1}_{r2}" in self.variables:
                                # Constraints to model absolute value: abs_var >= var and abs_var >= -var
                                self.model += (self.variables[f"abs_transport_{r1}_{r2}"][t] >= 
                                               self.variables[f"transport_{r1}_{r2}"][t], 
                                               f"abs_transport_pos_{r1}_{r2}_{t}")
                                self.model += (self.variables[f"abs_transport_{r1}_{r2}"][t] >= 
                                               -self.variables[f"transport_{r1}_{r2}"][t], 
                                               f"abs_transport_neg_{r1}_{r2}_{t}")
                    except Exception as e:
                        logger.warning(f"Error adding transport constraints for {r1}-{r2}: {e}")
            pbar.update(1)
            
            # 5. Demand response constraints
            pbar.set_description("Demand response constraints")
            if "max_daily_demand_response" in self.config["constraints"]:
                # Calculate number of periods per day assuming half-hourly data (48 periods/day)
                periods_per_day = 48
                num_days = len(T) // periods_per_day
                
                for region in self.regions:
                    # Limit total daily demand response
                    for day in range(num_days):
                        day_start = day * periods_per_day
                        day_end = min((day + 1) * periods_per_day, len(T))
                        day_periods = T[day_start:day_end]
                        
                        # Sum all positive demand response for the day
                        if f"demand_response_{region}" in self.variables:
                            try:
                                # Constraint: Sum of absolute demand response per day <= max daily DR
                                # This can be tricky since we need absolute values
                                pos_dr = []
                                neg_dr = []
                                
                                for t in day_periods:
                                    # Create variables for positive and negative parts
                                    pos_dr_var = LpVariable(f"pos_dr_{region}_{t}", lowBound=0)
                                    neg_dr_var = LpVariable(f"neg_dr_{region}_{t}", lowBound=0)
                                    
                                    # Add constraints to define positive and negative parts
                                    self.model += (self.variables[f"demand_response_{region}"][t] == 
                                                  pos_dr_var - neg_dr_var, 
                                                  f"dr_split_{region}_{t}")
                                    
                                    pos_dr.append(pos_dr_var)
                                    neg_dr.append(neg_dr_var)
                                
                                # Add constraint on total daily demand response
                                self.model += (lpSum(pos_dr) + lpSum(neg_dr) <= 
                                              self.config["constraints"]["max_daily_demand_response"], 
                                              f"daily_dr_{region}_{day}")
                            except Exception as e:
                                logger.warning(f"Error adding daily demand response constraint for {region}, day {day}: {e}")
            pbar.update(1)
            
            # 6. Absolute value constraints
            pbar.set_description("Absolute value constraints")
            # Already handled in exchange network constraints
            pbar.update(1)
            
            # 7. Ramping constraints for dispatchable technologies
            pbar.set_description("Ramping constraints")
            if "max_ramp_up" in self.config["constraints"] and "max_ramp_down" in self.config["constraints"]:
                max_ramp_up = self.config["constraints"]["max_ramp_up"]
                max_ramp_down = self.config["constraints"]["max_ramp_down"]
                
                # Add ramping constraints for each technology and region
                for region in self.regions:
                    for tech in self.dispatch_techs:
                        for i in range(1, len(T)):
                            if f"dispatch_{tech}_{region}" in self.variables:
                                # Ramp up limit
                                self.model += (
                                    self.variables[f"dispatch_{tech}_{region}"][T[i]] - 
                                    self.variables[f"dispatch_{tech}_{region}"][T[i-1]] <= max_ramp_up,
                                    f"ramp_up_{tech}_{region}_{T[i]}"
                                )
                                
                                # Ramp down limit
                                self.model += (
                                    self.variables[f"dispatch_{tech}_{region}"][T[i-1]] - 
                                    self.variables[f"dispatch_{tech}_{region}"][T[i-1]] <= max_ramp_down,
                                    f"ramp_down_{tech}_{region}_{T[i]}"
                                )
            pbar.update(1)

    def solve_model(self):
        """Solve the optimization model with progress tracking.
        
        Returns:
            Dictionary of optimization results or infeasibility analysis
        """
        logger.info("Solving optimization model...")
        
        # Start timing
        start_time = time.time()
        
        # Try to solve the model
        try:
            # Set up the solver
            solver = PULP_CBC_CMD(msg=False, timeLimit=3600)  # 1-hour time limit
            
            # Solve the model with progress tracking
            status = self.model.solve(solver)
            
            # Calculate solving time
            solve_time = time.time() - start_time
            logger.info(f"Model solved in {solve_time:.2f} seconds with status: {LpStatus[status]}")
            
            # Check if the solution is optimal
            if status == LpStatusOptimal:
                logger.info("Model solved optimally.")
                return self.get_results()
            else:
                logger.warning(f"Model not solved optimally. Status: {LpStatus[status]}")
                
                # Analyze infeasibility if the model is infeasible
                if status == LpStatusInfeasible:
                    logger.warning("Model is infeasible. Analyzing infeasibility...")
                    return self._analyze_infeasibility()
                else:
                    return {"status": LpStatus[status], "message": "Model not solved optimally or reached time limit."}
                
        except Exception as e:
            logger.error(f"Error solving model: {e}")
            return {"status": "Error", "message": str(e)}
    
    def _analyze_infeasibility(self):
        """Analyze the infeasibility to identify problematic constraints.
        
        Returns:
            Dictionary with infeasibility analysis information
        """
        logger.info("Analyzing infeasibility...")
        
        # Create a new model with slack variables for each constraint
        infeas_model = LpProblem("InfeasibilityAnalysis", LpMinimize)
        
        # Copy all variables from the original model
        for var in self.model.variables():
            # PuLP doesn't provide a direct way to copy variables between models
            # So we need to recreate them with the same properties
            var_name = var.name
            var_lowBound = var.lowBound
            var_upBound = var.upBound
            var_cat = var.cat
            
            # Skip if already in the model
            if var_name in [v.name for v in infeas_model.variables()]:
                continue
                
            # Recreate the variable
            new_var = LpVariable(var_name, lowBound=var_lowBound, upBound=var_upBound, cat=var_cat)
            
        # Add slack variables for each constraint and modify constraints
        constraint_violations = {}
        constraint_count = 0
        
        # Process each constraint in the original model
        for name, constraint in self.model.constraints.items():
            constraint_count += 1
            
            # Create a slack variable for this constraint
            slack_var = LpVariable(f"infeas_slack_{name}", lowBound=0)
            
            # Get the constraint expression
            expr = constraint.expr
            sense = constraint.sense
            rhs = -expr.constant  # The constant part is on the left side in PuLP's internal representation
            
            # Modify the constraint with slack
            if sense == LpConstraintEQ:  # Equality constraint (==)
                # We need two slack variables for equality: one for positive and one for negative deviation
                slack_pos = LpVariable(f"infeas_slack_pos_{name}", lowBound=0)
                slack_neg = LpVariable(f"infeas_slack_neg_{name}", lowBound=0)
                
                # Add the modified constraint: expr + slack_pos - slack_neg == rhs
                infeas_model += (expr + slack_pos - slack_neg == rhs, f"relaxed_{name}")
                
                # Add the slacks to the objective
                infeas_model += 1000 * (slack_pos + slack_neg)
                
                # Track this constraint
                constraint_violations[name] = {"type": "equality", "slack_pos": slack_pos, "slack_neg": slack_neg}
                
            elif sense == LpConstraintLE:  # Less than or equal constraint (<=)
                # Add the modified constraint: expr + slack <= rhs
                infeas_model += (expr + slack_var <= rhs, f"relaxed_{name}")
                
                # Add the slack to the objective
                infeas_model += 1000 * slack_var
                
                # Track this constraint
                constraint_violations[name] = {"type": "less_equal", "slack": slack_var}
                
            elif sense == LpConstraintGE:  # Greater than or equal constraint (>=)
                # Add the modified constraint: expr - slack >= rhs
                infeas_model += (expr - slack_var >= rhs, f"relaxed_{name}")
                
                # Add the slack to the objective
                infeas_model += 1000 * slack_var
                
                # Track this constraint
                constraint_violations[name] = {"type": "greater_equal", "slack": slack_var}
        
        logger.info(f"Created infeasibility analysis model with {constraint_count} relaxed constraints")
        
        # Solve the relaxed model
        try:
            solver = PULP_CBC_CMD(msg=False, timeLimit=1800)  # 30-minute time limit
            status = infeas_model.solve(solver)
            
            if status == LpStatusOptimal:
                logger.info("Infeasibility analysis completed successfully.")
                
                # Find constraints with non-zero slack
                problematic_constraints = []
                
                for name, info in constraint_violations.items():
                    if info["type"] == "equality":
                        slack_pos_value = info["slack_pos"].value()
                        slack_neg_value = info["slack_neg"].value()
                        
                        if slack_pos_value > 1e-6 or slack_neg_value > 1e-6:
                            problematic_constraints.append({
                                "constraint": name,
                                "type": "equality",
                                "violation": max(slack_pos_value, slack_neg_value),
                                "direction": "positive" if slack_pos_value > slack_neg_value else "negative"
                            })
                    else:
                        slack_value = info["slack"].value()
                        
                        if slack_value > 1e-6:
                            problematic_constraints.append({
                                "constraint": name,
                                "type": info["type"],
                                "violation": slack_value
                            })
                
                # Sort by violation magnitude
                problematic_constraints.sort(key=lambda x: x["violation"], reverse=True)
                
                # Limit to top 20 most problematic constraints
                top_constraints = problematic_constraints[:20]
                
                return {
                    "status": "Infeasible",
                    "message": "Model is infeasible. Here are the most problematic constraints:",
                    "problematic_constraints": top_constraints,
                    "suggestions": [
                        "Consider increasing storage capacity in the affected regions",
                        "Relax ramping constraints by increasing max_ramp_up and max_ramp_down",
                        "Increase exchange capacity between regions",
                        "Allow more demand response flexibility",
                        "Check for data quality issues in the affected time periods",
                        "Increase max_slack parameter to allow more flexibility in balance constraints"
                    ]
                }
            else:
                logger.warning(f"Infeasibility analysis model not solved optimally. Status: {LpStatus[status]}")
                return {
                    "status": "Analysis failed",
                    "message": f"Could not analyze infeasibility. Status: {LpStatus[status]}"
                }
        except Exception as e:
            logger.error(f"Error in infeasibility analysis: {e}")
            return {"status": "Error", "message": f"Error in infeasibility analysis: {str(e)}"}
    
    def get_results(self):
        """Get optimization results.
        
        Returns:
            Dictionary of results
        """
        logger.info("Extracting optimization results...")
        
        # Check if the model has been solved
        if self.model.status != LpStatusOptimal:
            logger.warning(f"Model not solved optimally. Status: {LpStatus[self.model.status]}")
            return {"status": LpStatus[self.model.status], "message": "Results may not be optimal."}
        
        results = {"status": "Optimal", "objective_value": value(self.model.objective)}
        
        # Extract results for all variable types
        var_groups = {
            "dispatch": {},
            "storage_level": {},
            "storage_charge": {},
            "storage_discharge": {},
            "demand_response": {},
            "exchange": {},
            "transport": {},
            "slack": {}
        }
        
        # Extract technology-specific dispatch results
        for region in self.regions:
            # Process each dispatch technology
            for tech in self.dispatch_techs:
                var_name = f"dispatch_{tech}_{region}"
                if var_name in self.variables:
                    values = {t: self.variables[var_name][t].value() for t in self.variables[var_name]}
                    var_groups["dispatch"][f"{tech}_{region}"] = values
            
            # Process each storage technology
            for storage_tech in self.storage_techs:
                # Storage level
                var_name = f"storage_level_{storage_tech}_{region}"
                if var_name in self.variables:
                    values = {t: self.variables[var_name][t].value() for t in self.variables[var_name]}
                    var_groups["storage_level"][f"{storage_tech}_{region}"] = values
                
                # Storage charge
                var_name = f"storage_charge_{storage_tech}_{region}"
                if var_name in self.variables:
                    values = {t: self.variables[var_name][t].value() for t in self.variables[var_name]}
                    var_groups["storage_charge"][f"{storage_tech}_{region}"] = values
                
                # Storage discharge
                var_name = f"storage_discharge_{storage_tech}_{region}"
                if var_name in self.variables:
                    values = {t: self.variables[var_name][t].value() for t in self.variables[var_name]}
                    var_groups["storage_discharge"][f"{storage_tech}_{region}"] = values
            
            # Demand response
            var_name = f"demand_response_{region}"
            if var_name in self.variables:
                values = {t: self.variables[var_name][t].value() for t in self.variables[var_name]}
                var_groups["demand_response"][region] = values
            
            # Slack variables
            var_name_pos = f"slack_pos_{region}"
            var_name_neg = f"slack_neg_{region}"
            if var_name_pos in self.variables and var_name_neg in self.variables:
                pos_values = {t: self.variables[var_name_pos][t].value() for t in self.variables[var_name_pos]}
                neg_values = {t: self.variables[var_name_neg][t].value() for t in self.variables[var_name_neg]}
                var_groups["slack"][f"pos_{region}"] = pos_values
                var_groups["slack"][f"neg_{region}"] = neg_values
        
        # Exchange variables
        for i, r1 in enumerate(self.regions):
            for r2 in self.regions[i+1:]:
                var_name = f"exchange_{r1}_{r2}"
                if var_name in self.variables:
                    values = {t: self.variables[var_name][t].value() for t in self.variables[var_name]}
                    var_groups["exchange"][f"{r1}_{r2}"] = values
                
                var_name = f"transport_{r1}_{r2}"
                if var_name in self.variables:
                    values = {t: self.variables[var_name][t].value() for t in self.variables[var_name]}
                    var_groups["transport"][f"{r1}_{r2}"] = values
        
        # Add all variable groups to results
        results["variables"] = var_groups
        
        # Calculate summary statistics
        summary = {
            "total_dispatch": {tech: {} for tech in self.dispatch_techs},
            "total_storage_usage": {tech: {} for tech in self.storage_techs},
            "total_demand_response": {},
            "total_exchange": {},
            "total_slack": {}
        }
        
        # Calculate summary for each technology and region
        for region in self.regions:
            # Dispatch by technology
            for tech in self.dispatch_techs:
                tech_dispatch = var_groups["dispatch"].get(f"{tech}_{region}", {})
                if tech_dispatch:
                    summary["total_dispatch"][tech][region] = sum(tech_dispatch.values())
            
            # Storage by technology
            for storage_tech in self.storage_techs:
                tech_charge = var_groups["storage_charge"].get(f"{storage_tech}_{region}", {})
                tech_discharge = var_groups["storage_discharge"].get(f"{storage_tech}_{region}", {})
                if tech_charge and tech_discharge:
                    summary["total_storage_usage"][storage_tech][region] = {
                        "charge": sum(tech_charge.values()),
                        "discharge": sum(tech_discharge.values())
                    }
            
            # Demand response
            dr_values = var_groups["demand_response"].get(region, {})
            if dr_values:
                pos_dr = sum(max(0, val) for val in dr_values.values())
                neg_dr = sum(min(0, val) for val in dr_values.values())
                summary["total_demand_response"][region] = {"positive": pos_dr, "negative": neg_dr}
            
            # Slack
            pos_slack = var_groups["slack"].get(f"pos_{region}", {})
            neg_slack = var_groups["slack"].get(f"neg_{region}", {})
            if pos_slack and neg_slack:
                summary["total_slack"][region] = {"positive": sum(pos_slack.values()), "negative": sum(neg_slack.values())}
        
        # Add summary to results
        results["summary"] = summary
        
        logger.info("Results extraction completed.")
        return results
    
    def safe_get_values(self, var_name, region, time_range):
        """Safely get values for a variable, handling possible missing keys."""
        if var_name not in self.variables or region not in self.variables[var_name]:
            return [0] * len(time_range)
        return [self.variables[var_name][region][t].value() or 0 for t in time_range]
