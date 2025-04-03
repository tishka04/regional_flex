import pandas as pd
import numpy as np
from pulp import *
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

class RegionalFlexOptimizer:
    """Multi-regional energy flexibility optimization model."""
    
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
    
    def build_model(self, regional_data: Dict[str, pd.DataFrame]):
        """Build the optimization model with progress tracking.
        
        Args:
            regional_data: Dictionary of preprocessed regional data
        """
        # Check the type of regional_data and handle accordingly
        logger.info(f"Type of regional_data: {type(regional_data)}")
        
        # Handle different possible types of regional_data
        if isinstance(regional_data, dict):
            # If it's a dictionary as expected
            if len(regional_data) > 0:
                first_df = next(iter(regional_data.values()))
                time_periods = len(first_df)
            else:
                logger.warning("Empty regional_data dictionary")
                time_periods = 0
        elif isinstance(regional_data, np.ndarray):
            # If it's a numpy array
            time_periods = len(regional_data)
        else:
            # For any other type
            try:
                time_periods = len(regional_data)
            except:
                logger.warning(f"Could not determine time periods from regional_data of type {type(regional_data)}")
                time_periods = 0
        
        logger.info(f"Building optimization model for {len(self.regions)} regions and {time_periods} time periods...")
        
        # Create a progress bar for model building steps
        with tqdm(total=3, desc="Building model", position=0, leave=False) as pbar:
            # Create time periods list
            T = list(range(time_periods))
            
            # Add variables
            pbar.set_description("Adding variables")
            self._add_variables(regional_data, T)
            pbar.update(1)
            
            # Add objective function
            pbar.set_description("Adding objective function")
            self._add_objective(regional_data, T)
            pbar.update(1)
            
            # Add constraints
            pbar.set_description("Adding constraints")
            self._add_constraints(regional_data, T)
            pbar.update(1)
        
        logger.info("Optimization model built successfully")
    
    def _add_variables(self, data: Dict[str, pd.DataFrame], T: List[int]):
        """Add decision variables to the model with progress tracking.
        
        Args:
            data: Dictionary of regional data
            T: List of time periods
        """
        logger.info(f"Adding variables for {len(self.regions)} regions and {len(T)} time periods")
        
        # Initialize variables dictionary
        self.variables = {}
        
        # Create a progress bar for variable types
        variable_types = ["Dispatch", "Storage", "Demand response", "Interregional exchange", 
                          "Transportation network exchange", "Slack"]
        
        with tqdm(total=len(variable_types), desc="Adding variables", position=1, leave=False) as pbar:
            # 1. Dispatch variables for each region
            pbar.set_description("Dispatch variables")
            for region in self.regions:
                self.variables[f"dispatch_{region}"] = LpVariable.dicts(
                    f"dispatch_{region}", 
                    T, 
                    lowBound=self.config["constraints"]["min_dispatch"],
                    upBound=self.config["constraints"]["max_dispatch"]
                )
            pbar.update(1)
            
            # 2. Storage variables for each region
            pbar.set_description("Storage variables")
            for region in self.regions:
                # Storage level variables
                storage_capacity = self.config["constraints"]["storage_capacity"]
                # Check for region-specific storage capacity
                if "regional_constraints" in self.config and region in self.config["regional_constraints"]:
                    if "storage_capacity" in self.config["regional_constraints"][region]:
                        storage_capacity = self.config["regional_constraints"][region]["storage_capacity"]
                
                self.variables[f"storage_level_{region}"] = LpVariable.dicts(
                    f"storage_level_{region}", 
                    T, 
                    lowBound=0, 
                    upBound=storage_capacity
                )
                
                # Charge and discharge variables
                self.variables[f"storage_charge_{region}"] = LpVariable.dicts(
                    f"storage_charge_{region}", 
                    T, 
                    lowBound=0, 
                    upBound=storage_capacity * 0.3  # Assume max 30% charge rate
                )
                
                self.variables[f"storage_discharge_{region}"] = LpVariable.dicts(
                    f"storage_discharge_{region}", 
                    T, 
                    lowBound=0, 
                    upBound=storage_capacity * 0.3  # Assume max 30% discharge rate
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
    
    def _add_objective(self, data: Dict[str, pd.DataFrame], T: List[int]):
        """Add objective function to minimize costs.
        
        Args:
            data: Dictionary of regional data
            T: List of time periods
        """
        print("Adding objective function...")
        
        objective = 0
        
        # 1. Dispatch costs for each region
        for region in self.regions:
            # Check for region-specific dispatch cost multiplier
            cost_multiplier = 1.0
            if "regional_constraints" in self.config and region in self.config["regional_constraints"]:
                if "dispatch_cost_multiplier" in self.config["regional_constraints"][region]:
                    cost_multiplier = self.config["regional_constraints"][region]["dispatch_cost_multiplier"]
            
            try:
                for t in T:
                    if f"dispatch_{region}" in self.variables:
                        objective += self.variables[f"dispatch_{region}"][t] * self.config["costs"]["dispatch"] * cost_multiplier
                    else:
                        logger.warning(f"Variable dispatch_{region} not found in objective function")
            except Exception as e:
                logger.warning(f"Error adding dispatch cost for {region}: {e}")
        
        # 2. Storage costs
        for region in self.regions:
            try:
                for t in T:
                    if f"storage_level_{region}" in self.variables:
                        objective += self.variables[f"storage_level_{region}"][t] * self.config["costs"]["storage"]
                    else:
                        logger.warning(f"Variable storage_level_{region} not found in objective function")
                    
                    if f"storage_charge_{region}" in self.variables:
                        objective += self.variables[f"storage_charge_{region}"][t] * self.config["costs"]["storage_charge"]
                    else:
                        logger.warning(f"Variable storage_charge_{region} not found in objective function")
                    
                    if f"storage_discharge_{region}" in self.variables:
                        objective += self.variables[f"storage_discharge_{region}"][t] * self.config["costs"]["storage_discharge"]
                    else:
                        logger.warning(f"Variable storage_discharge_{region} not found in objective function")
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
    
    def _add_constraints(self, data: Dict[str, pd.DataFrame], T: List[int]):
        """Add constraints to the model with progress tracking.
        
        Args:
            data: Dictionary of regional data
            T: List of time periods
        """
        logger.info(f"Adding constraints for {len(self.regions)} regions and {len(T)} time periods")
        
        # Create a progress bar for constraint types
        constraint_types = ["Regional balance", "Storage balance", "Capacity", 
                           "Exchange network", "Demand response"]
        
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
                        
                        # Add dispatch
                        if f"dispatch_{region}" in self.variables:
                            lhs += self.variables[f"dispatch_{region}"][t]
                        
                        # Add storage discharge (adds to supply) and charge (adds to demand)
                        if f"storage_discharge_{region}" in self.variables:
                            lhs += self.variables[f"storage_discharge_{region}"][t]
                        if f"storage_charge_{region}" in self.variables:
                            lhs -= self.variables[f"storage_charge_{region}"][t]
                        
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
            
            # 2. Storage balance constraints
            pbar.set_description("Storage balance constraints")
            for region in self.regions:
                try:
                    # Initial storage level
                    initial_level = self.config["constraints"]["storage_capacity"] * self.config["constraints"]["initial_storage_level"]
                    
                    # Check for region-specific storage capacity
                    storage_capacity = self.config["constraints"]["storage_capacity"]
                    if "regional_constraints" in self.config and region in self.config["regional_constraints"]:
                        if "storage_capacity" in self.config["regional_constraints"][region]:
                            storage_capacity = self.config["regional_constraints"][region]["storage_capacity"]
                            initial_level = storage_capacity * self.config["constraints"]["initial_storage_level"]
                    
                    # Storage efficiency parameters
                    storage_efficiency = self.config["constraints"]["storage_efficiency"]
                    charge_efficiency = self.config["constraints"]["charge_efficiency"]
                    discharge_efficiency = self.config["constraints"]["discharge_efficiency"]
                    
                    # First time period uses initial level
                    if len(T) > 0:
                        if f"storage_level_{region}" in self.variables and f"storage_charge_{region}" in self.variables and f"storage_discharge_{region}" in self.variables:
                            self.model += (
                                self.variables[f"storage_level_{region}"][T[0]] == 
                                initial_level + 
                                self.variables[f"storage_charge_{region}"][T[0]] * charge_efficiency - 
                                self.variables[f"storage_discharge_{region}"][T[0]] / discharge_efficiency,
                                f"storage_balance_{region}_{T[0]}"
                            )
                    
                    # Subsequent time periods
                    for i in range(1, len(T)):
                        if f"storage_level_{region}" in self.variables and f"storage_charge_{region}" in self.variables and f"storage_discharge_{region}" in self.variables:
                            self.model += (
                                self.variables[f"storage_level_{region}"][T[i]] == 
                                self.variables[f"storage_level_{region}"][T[i-1]] * storage_efficiency + 
                                self.variables[f"storage_charge_{region}"][T[i]] * charge_efficiency - 
                                self.variables[f"storage_discharge_{region}"][T[i]] / discharge_efficiency,
                                f"storage_balance_{region}_{T[i]}"
                            )
                except Exception as e:
                    logger.warning(f"Error adding storage balance constraint for {region}: {e}")
            pbar.update(1)
            
            # 3. Capacity constraints
            pbar.set_description("Capacity constraints")
            for r1 in self.regions:
                for r2 in self.regions:
                    if r1 < r2:  # Avoid duplicates
                        try:
                            for t in T:
                                # Absolute value constraints for exchange
                                if f"exchange_{r1}_{r2}" in self.variables and f"abs_exchange_{r1}_{r2}" in self.variables:
                                    self.model += (
                                        self.variables[f"abs_exchange_{r1}_{r2}"][t] >= self.variables[f"exchange_{r1}_{r2}"][t],
                                        f"abs_exchange_pos_{r1}_{r2}_{t}"
                                    )
                                    self.model += (
                                        self.variables[f"abs_exchange_{r1}_{r2}"][t] >= -self.variables[f"exchange_{r1}_{r2}"][t],
                                        f"abs_exchange_neg_{r1}_{r2}_{t}"
                                    )
                                
                                # Absolute value constraints for transport
                                if f"transport_{r1}_{r2}" in self.variables and f"abs_transport_{r1}_{r2}" in self.variables:
                                    self.model += (
                                        self.variables[f"abs_transport_{r1}_{r2}"][t] >= self.variables[f"transport_{r1}_{r2}"][t],
                                        f"abs_transport_pos_{r1}_{r2}_{t}"
                                    )
                                    self.model += (
                                        self.variables[f"abs_transport_{r1}_{r2}"][t] >= -self.variables[f"transport_{r1}_{r2}"][t],
                                        f"abs_transport_neg_{r1}_{r2}_{t}"
                                    )
                        except Exception as e:
                            logger.warning(f"Error adding capacity constraint for {r1}-{r2}: {e}")
            pbar.update(1)
            
            # 4. Exchange network constraints
            pbar.set_description("Exchange network constraints")
            # Add any additional network constraints here
            pbar.update(1)
            
            # 5. Demand response constraints
            pbar.set_description("Demand response constraints")
            for region in self.regions:
                try:
                    # Limit daily demand response (sum of absolute values)
                    daily_periods = 24 * 2  # Assuming 30-minute resolution
                    max_daily = self.config["constraints"]["max_daily_demand_response"]
                    
                    # For each day
                    for day in range(len(T) // daily_periods):
                        day_start = day * daily_periods
                        day_end = min((day + 1) * daily_periods, len(T))
                        
                        if f"demand_response_{region}" in self.variables:
                            # Sum of absolute values of demand response for the day
                            day_sum = 0
                            for t in range(day_start, day_end):
                                # We need to use auxiliary variables for the absolute value
                                pos_var = LpVariable(f"dr_pos_{region}_{t}", lowBound=0)
                                neg_var = LpVariable(f"dr_neg_{region}_{t}", lowBound=0)
                                
                                # Add constraints to define the absolute value
                                self.model += (
                                    self.variables[f"demand_response_{region}"][t] == pos_var - neg_var,
                                    f"dr_abs_def_{region}_{t}"
                                )
                                
                                # Add to the daily sum
                                day_sum += pos_var + neg_var
                            
                            # Constraint on daily sum
                            self.model += (
                                day_sum <= max_daily,
                                f"dr_daily_limit_{region}_{day}"
                            )
                except Exception as e:
                    logger.warning(f"Error adding demand response constraint for {region}: {e}")
            pbar.update(1)
    
    def solve_model(self) -> Dict:
        """Solve the optimization model with progress tracking.
        
        Returns:
            Dictionary of optimization results
        """
        logger.info("Solving optimization model...")
        
        # Create a progress spinner for the solver
        solver_progress = tqdm(total=100, desc="Solving model", position=0)
        # Since we can't track PuLP solver progress directly, we'll update every second
        # This will show an indeterminate progress bar
        
        # Solve the model
        import threading
        import time
        
        # Function to update progress bar while solver is running
        def update_progress():
            progress = 0
            while progress < 100 and self.model.status == 0:  # 0 = Not Solved
                time.sleep(0.5)
                # Make progress jump in small increments to show activity
                increment = min(2, 100 - progress)
                progress += increment
                solver_progress.n = progress
                solver_progress.refresh()
        
        # Start progress updater in background thread
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        # Start solver
        self.model.solve()
        
        # Set progress to 100% when done
        solver_progress.n = 100
        solver_progress.refresh()
        solver_progress.close()
        
        # Check if solved
        status_str = LpStatus[self.model.status]
        if self.model.status != 1:
            logger.error(f"Failed to find optimal solution. Status: {status_str}")
            return None
        else:
            logger.info(f"Optimization solved successfully with status: {status_str}")
        
        # Get the actual time periods used in the model
        T = 0
        if self.regions and len(self.regions) > 0:
            first_region = self.regions[0]
            first_var_name = f"dispatch_{first_region}"
            if first_var_name in self.variables:
                # Find out how many time periods we actually have
                T = len(self.variables[first_var_name])
                print(f"Getting results for {T} time periods")
            else:
                print("Warning: No variables found for dispatches")
        else:
            logger.warning("No regions available for result extraction")
        
        results = {
            'objective_value': value(self.model.objective),
            'regional_results': {},
            'time_periods': T
        }
        
        # Helper function to safely get variable values 
        def safe_get_values(var_name, region, time_range):
            try:
                if var_name.format(region=region) in self.variables:
                    return [value(self.variables[var_name.format(region=region)][t]) for t in range(time_range)]
                else:
                    print(f"Warning: Variable {var_name.format(region=region)} not found")
                    return [0.0] * time_range
            except Exception as e:
                print(f"Error getting values for {var_name.format(region=region)}: {e}")
                return [0.0] * time_range
        
        # Get regional results
        if self.regions:
            for region in self.regions:
                results['regional_results'][region] = {
                    'dispatch': safe_get_values("dispatch_{region}", region, T),
                    'storage_charge': safe_get_values("storage_charge_{region}", region, T),
                    'storage_discharge': safe_get_values("storage_discharge_{region}", region, T),
                    'soc': safe_get_values("storage_level_{region}", region, T),
                    'slack_pos': safe_get_values("slack_pos_{region}", region, T),
                    'slack_neg': safe_get_values("slack_neg_{region}", region, T)
                }
            
            # Get exchange results
            results['exchange'] = {}
            for i, region1 in enumerate(self.regions):
                for region2 in self.regions[i+1:]:
                    key = f"{region1}_{region2}"
                    var_name = f"exchange_{region1}_{region2}"
                    if var_name in self.variables:
                        results['exchange'][key] = [value(self.variables[var_name][t]) for t in range(T)]
                    else:
                        results['exchange'][key] = [0.0] * T
        
        return results
    
    def get_results(self) -> Dict:
        """Get optimization results.
        
        Returns:
            Dictionary of results
        """
        # This is a placeholder for more detailed results processing
        return self.solve_model()
