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
        
        # Initialize dictionaries for regional parameters
        self.regional_multipliers = {}
        self.tech_params = {}
        
        # Track which variables should be binary in standard model
        # This allows us to relax them to continuous when using LP relaxation
        self.binary_vars = []
        
        if 'regional_costs' in self.config:
            self.regional_costs = self.config['regional_costs']
            logger.info(f"Loaded regional costs for {len(self.regional_costs)} regions")

        # --- harmonise TOUS les noms de région (espace -> underscore) -------------
        def _norm(r):
            return r.replace(" ", "_")  \
            .replace("'", "")   \
            .replace("-", "_")  # if you also want to normalize dashes

        self.regions           = [_norm(r) for r in self.regions]
        self.tech_capacities   = {_norm(r): caps for r, caps in self.tech_capacities.items()}
        self.regional_costs    = {_norm(r): c    for r, c    in self.regional_costs.items()}
        self.storage_capacities= {_norm(r): s    for r, s    in self.storage_capacities.items()}

    # ------------------------------------------------------------
    def _loss(self, region_from, region_to) -> float:
        """Retourne la fraction de pertes entre deux régions."""
        dist = self.config.get('regional_distances', {}) \
                         .get(region_from, {}) \
                         .get(region_to, 0.0)
        return dist * self.config.get('loss_factor_per_km', 0.0)
    # ------------------------------------------------------------

            
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
    
    # ------------------------------------------------------------
        # ---------------------------------------------------------------------
    def _init_variables(
        self,
        data: Dict[str, pd.DataFrame],
        time_periods: Optional[List[Union[int, pd.Timestamp]]]
    ) -> None:
        """
        Initialise toutes les variables du modèle :
        - dispatch techno
        - stockage (soc / charge / décharge)
        - demand-response
        - slack ±
        - curtailment
        - flux dirigés inter-régions (flow_out_i_j ≥ 0)
        - flag d’activation DR (binaire ou relaxé)
        """
        logger.info("Initializing optimization variables")

        # ---- option LP-relax --------------------------------------------
        using_lp_relax = (
            self.use_simplified_model
            and self.simplification_options.get("lp_relaxation", False)
        )

        # ---- horizon temporel -------------------------------------------
        if isinstance(time_periods, list):
            T = time_periods
        else:
            first_df = next(iter(data.values()))
            T = list(range(len(first_df)))

        # (ré)initialisation propres
        self.variables   = {}
        self.binary_vars = []

        # ---- 1. variables intra-région -----------------------------------
        for region in self.regions:

            # a) dispatch
            for tech in self.dispatch_techs:
                k = f"dispatch_{tech}_{region}"
                self.variables[k] = {t: LpVariable(f"{k}_{t}", lowBound=0) for t in T}

            # b) stockage
            for st in self.storage_techs:
                for prefix in ("storage_soc", "storage_charge", "storage_discharge"):
                    k = f"{prefix}_{st}_{region}"
                    self.variables[k] = {t: LpVariable(f"{k}_{t}", lowBound=0) for t in T}

            # c) demand-response (±)
            k = f"demand_response_{region}"
            self.variables[k] = {t: LpVariable(f"{k}_{t}") for t in T}

            # d) slack ±
            for sign in ("pos", "neg"):
                k = f"slack_{sign}_{region}"
                self.variables[k] = {t: LpVariable(f"{k}_{t}", lowBound=0) for t in T}

            # e) curtailment
            k = f"curtail_{region}"
            self.variables[k] = {t: LpVariable(f"{k}_{t}", lowBound=0) for t in T}

            # f) flag DR
            k = f"dr_active_{region}"
            self.variables[k] = {}
            for t in T:
                if using_lp_relax:
                    v = LpVariable(f"{k}_{t}", lowBound=0, upBound=1, cat="Continuous")
                else:
                    v = LpVariable(f"{k}_{t}", cat="Binary")
                    self.binary_vars.append(v)
                self.variables[k][t] = v

        # ---- 2. flux dirigés inter-régions -------------------------------
        for i in self.regions:
            for j in self.regions:
                if i == j:
                    continue
                k = f"flow_out_{i}_{j}"
                self.variables[k] = {t: LpVariable(f"{k}_{t}", lowBound=0) for t in T}

        logger.info(
            f"Created {len(self.variables)} variable groups "
            f"({sum(len(v) for v in self.variables.values())} scalar vars)"
        )


    # ---------------------------------------------------------------------------
    def _add_objective(
            self,
            data: Dict[str, pd.DataFrame],
            time_periods: Optional[List[Union[int, pd.Timestamp]]]
        ) -> None:
        """
        Crée la fonction objectif du modèle : minimisation du coût total
        (dispatch, stockage, DR, flux, slack, curtailment).
        """

        logger.info("Adding objective function")

        # -------- horizon temporel ---------------------------------------------
        if isinstance(time_periods, list):
            T = time_periods
        else:
            first_df = next(iter(data.values()))
            T = list(range(len(first_df)))

        # -------- barème de coûts ----------------------------------------------
        # récupère ceux de la config, sinon valeur par défaut
        costs = dict(self.config.get("costs", {}))

        default_costs = {
            # production
            "hydro": 30.0, "nuclear": 40.0,
            "thermal_gas": 80.0, "thermal_coal": 90.0,
            "biofuel": 70.0,

            # stockage (€/MWh d’énergie)
            "storage_charge": 4.0, "storage_discharge": 3.0,

            # flexibilité
            "demand_response": 120.0,
            "flow": 25.0,                 # coût de transport par MWh
            "flow_km_coeff": 0.0,         # (optionnel) €/MWh·km pour pertes / péage

            # pénalités
            "slack_penalty": 50_000.0,
            "curtailment_penalty": 10_000.0
        }
        # complète les manquants
        for k, v in default_costs.items():
            costs.setdefault(k, v)

        # -------- initialisation de l’objectif ---------------------------------
        objective = 0

        # -- 1. Dispatch ---------------------------------------------------------
        for region in self.regions:
            for tech in self.dispatch_techs:
                tech_cost = costs.get(tech, default_costs[tech])

                # éventuel coût régional spécifique
                if region in getattr(self, "regional_costs", {}) and \
                   tech   in self.regional_costs[region]:
                    tech_cost = self.regional_costs[region][tech]

                var_key = f"dispatch_{tech}_{region}"
                if var_key not in self.variables:
                    continue

                for t, var in self.variables[var_key].items():
                    if t in T:
                        objective += var * tech_cost

        # -- 2. Stockage ---------------------------------------------------------
        for region in self.regions:
            for st in self.storage_techs:
                charge_key = f"storage_charge_{st}_{region}"
                discharge_key = f"storage_discharge_{st}_{region}"

                charge_c  = costs.get(f"storage_{st}_charge",
                                      costs["storage_charge"])
                discharge_c = costs.get(f"storage_{st}_discharge",
                                        costs["storage_discharge"])

                for t in T:
                    if charge_key in self.variables and t in self.variables[charge_key]:
                        objective += self.variables[charge_key][t] * charge_c
                    if discharge_key in self.variables and t in self.variables[discharge_key]:
                        objective += self.variables[discharge_key][t] * discharge_c

        # -- 3. Demand-response (valeur absolue) ---------------------------------
        dr_cost = costs["demand_response"]
        for region in self.regions:
            dr_key = f"demand_response_{region}"
            if dr_key not in self.variables:
                continue
            for t in T:
                if t not in self.variables[dr_key]:
                    continue
                pos = LpVariable(f"pos_dr_{region}_{t}", lowBound=0)
                neg = LpVariable(f"neg_dr_{region}_{t}", lowBound=0)
                # valeur absolue
                self.model += self.variables[dr_key][t] == pos - neg, \
                              f"abs_DR_{region}_{t}"
                objective += (pos + neg) * dr_cost

        # -- 4. Flux inter-régions ----------------------------------------------
        flow_cost      = costs["flow"]
        flow_km_coeff  = costs["flow_km_coeff"]
        distances_km   = self.config.get("distances_km", {})  # optionnel

        for i in self.regions:
            for j in self.regions:
                if i == j:
                    continue
                key = f"flow_out_{i}_{j}"
                if key not in self.variables:
                    continue
                # distance éventuelle pour moduler le coût
                d_ij = distances_km.get(i, {}).get(j, 0.0)
                unit_cost = flow_cost + flow_km_coeff * d_ij
                for t, var in self.variables[key].items():
                    if t in T:
                        objective += var * unit_cost


        # -- 6. Slack et curtailment ---------------------------------------------
        slack_pen = costs["slack_penalty"]
        curt_pen  = costs["curtailment_penalty"]

        for region in self.regions:
            for t in T:
                if (f"slack_pos_{region}" in self.variables and
                        t in self.variables[f"slack_pos_{region}"]):
                    objective += self.variables[f"slack_pos_{region}"][t] * slack_pen
                if (f"slack_neg_{region}" in self.variables and
                        t in self.variables[f"slack_neg_{region}"]):
                    objective += self.variables[f"slack_neg_{region}"][t] * slack_pen
                if (f"curtail_{region}" in self.variables and
                        t in self.variables[f"curtail_{region}"]):
                    objective += self.variables[f"curtail_{region}"][t] * curt_pen

        # -------- affectation à PuLP -------------------------------------------
        self.model += objective
        logger.info("Objective function added.")
# ---------------------------------------------------------------------------

        
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
            
            # 6. Flexibility diversity constraints to ensure balanced use (temporarily skipped)
            # pbar.set_description("Flexibility diversity constraints")
            # self._add_flexibility_diversity_constraints(data, T)
            # pbar.update(1)
    
        # ---------------------------------------------------------------------
    def _add_energy_balance_constraints(
        self,
        data: Dict[str, pd.DataFrame],
        T: List[Union[int, pd.Timestamp]]
    ) -> None:
        """
        Production + décharges – charges – exports + imports(1-perte)
        + DR – curtail + slack_pos – slack_neg = demande
        """
        logger.info("Adding energy-balance constraints")

        for region in self.regions:
            df = data.get(region)
            if df is None:
                logger.warning(f"No data for region {region}")
                continue

            for t in T:
                if t >= len(df):
                    continue
                demand = next(
                    (float(df.iloc[t][c]) for c in ("consumption", "demand", "load")
                     if c in df.columns),
                    0.0
                )

                # --- composantes ------------------------------------------
                dispatch_terms  = [
                    self.variables[f"dispatch_{tec}_{region}"][t]
                    for tec in self.dispatch_techs
                ]
                discharge_terms = [
                    self.variables[f"storage_discharge_{st}_{region}"][t]
                    for st in self.storage_techs
                ]
                charge_terms    = [
                    self.variables[f"storage_charge_{st}_{region}"][t]
                    for st in self.storage_techs
                ]
                dr_term      = self.variables[f"demand_response_{region}"][t]
                curtail_term = self.variables[f"curtail_{region}"][t]
                slack_pos    = self.variables[f"slack_pos_{region}"][t]
                slack_neg    = self.variables[f"slack_neg_{region}"][t]

                # --- flux et pertes ---------------------------------------
                exports = lpSum(
                    self.variables[f"flow_out_{region}_{oth}"][t]
                    for oth in self.regions if oth != region
                )
                imports = lpSum(
                    self.variables[f"flow_out_{oth}_{region}"][t]
                    * (1.0 - self._loss(oth, region))
                    for oth in self.regions if oth != region
                )

                # --- contrainte de bilan ----------------------------------
                self.model += (
                    lpSum(dispatch_terms) +
                    lpSum(discharge_terms) -
                    lpSum(charge_terms)   -
                    exports               +
                    imports               +
                    dr_term               -
                    curtail_term          +
                    slack_pos             -
                    slack_neg             == demand,
                    f"balance_{region}_{t}"
                )


        # ---------------------------------------------------------------------
    def _add_exchange_constraints(
        self,
        data: Dict[str, pd.DataFrame],
        T: List[Union[int, pd.Timestamp]]
    ) -> None:
        """
        Capacité des lignes : flow_out_i_j ≤ cap_{i→j}
        """
        logger.info("Adding network-capacity constraints")

        caps = self.config.get("regional_transport_capacities", {})

        for i in self.regions:
            for j in self.regions:
                if i == j:
                    continue
                cap = caps.get(i, {}).get(j, caps.get(j, {}).get(i, 0.0))
                if cap <= 0:
                    continue

                k = f"flow_out_{i}_{j}"
                for t in T:
                    self.model += (
                        self.variables[k][t] <= cap,
                        f"cap_flow_{i}_{j}_{t}"
                    )


    def _add_dr_and_ramping_constraints(self, data: Dict[str, pd.DataFrame], T):
        """Add demand response and ramping constraints for each region.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary mapping region names to data frames
            T (List): List of time periods
        """
        logger.info("Adding demand response and ramping constraints")
        
        # Get constraints from config
        constraints = self.config.get('constraints') or {}
        
        # Skip ramping constraints if enabled in simplification options
        skip_ramping = self.use_simplified_model and self.simplification_options.get("skip_ramping", False)
        if skip_ramping:
            logger.info("Skipping ramping constraints due to skip_ramping simplification")
        
        # 1. Add demand response constraints
        for region in self.regions:
            # Get demand response parameters from config for this region
            dr_params = self.config.get('demand_response') or {}.get(region, {})
            
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
        constraints = self.config.get('constraints') or {}
        
        # Add capacity constraints for each region and technology
        for region in self.regions:
            # Get regional capacities
            regional_caps = self.tech_capacities.get(region, {})
            
            # Enforce minimum nuclear dispatch if specified in config
            min_frac = self.config.get('min_nuclear_capacity_fraction', 0.0)
            if 'nuclear' in regional_caps and min_frac > 0:
                nuclear_cap = regional_caps['nuclear']
                for t in T:
                    varname = f"dispatch_nuclear_{region}"
                    if varname in self.variables and t in self.variables[varname]:
                        self.model += (
                            self.variables[varname][t] >= min_frac * nuclear_cap,
                            f"min_nuclear_dispatch_{region}_{t}"
                        )
            
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
                        max_capacity = (self.storage_capacities.get(region) or {}).get(storage_key, constraints.get('max_storage', 5000.0))
                        
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
        
        # Get storage-related constraints from config
        constraints = self.config.get('constraints') or {}
        
        # Map config 'max_storage' to 'max_storage_capacity' if present
        if 'max_storage' in constraints and 'max_storage_capacity' not in constraints:
            constraints['max_storage_capacity'] = float(constraints['max_storage'])
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
                # Get regional storage capacities from config
                storage_region_caps = self.storage_capacities.get(region, {})
                max_energy_capacity = float(storage_region_caps.get(
                    f"{storage_tech}_stockage_MWh", constraints.get('max_storage_capacity', 10000)
                ))
                max_charge_rate = float(storage_region_caps.get(
                    f"{storage_tech}_puissance_MW", constraints.get('max_storage_charge_rate', max_energy_capacity)
                ))
                max_discharge_rate = float(storage_region_caps.get(
                    f"{storage_tech}_puissance_MW", constraints.get('max_storage_discharge_rate', max_charge_rate)
                ))
                
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
                        self.variables[f"storage_soc_{storage_tech}_{region}"][0] == initial_storage_level,
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
                        f"storage_soc_{storage_tech}_{region}",
                        f"storage_charge_{storage_tech}_{region}", 
                        f"storage_discharge_{storage_tech}_{region}"
                    ]):
                        self.model += (
                            self.variables[f"storage_soc_{storage_tech}_{region}"][t_next] == \
                            self.variables[f"storage_soc_{storage_tech}_{region}"][t] + \
                            self.variables[f"storage_charge_{storage_tech}_{region}"][t] * charge_efficiency - \
                            self.variables[f"storage_discharge_{storage_tech}_{region}"][t] * (1.0 / discharge_efficiency),
                            f"storage_evolution_{storage_tech}_{region}_{t}_{uuid.uuid4().hex[:8]}"
                        )
                
                # 4. Optional: cyclical constraint (end level = start level)
                skip_cyclical = self.use_simplified_model and self.simplification_options.get("skip_cyclical_storage", False)
                if not skip_cyclical and self.config.get('use_cyclical_storage', False) and T and T[0] in self.variables[f"storage_soc_{storage_tech}_{region}"] and T[-1] in self.variables[f"storage_soc_{storage_tech}_{region}"]:
                    # Add the constraint that the final storage level should be close to the initial
                    self.model += (
                        self.variables[f"storage_soc_{storage_tech}_{region}"][T[-1]] >= \
                        self.variables[f"storage_soc_{storage_tech}_{region}"][T[0]] * 0.95,
                        f"cyclical_storage_min_{storage_tech}_{region}_{uuid.uuid4().hex[:8]}"
                    )
                    
                    self.model += (
                        self.variables[f"storage_soc_{storage_tech}_{region}"][T[-1]] <= \
                        self.variables[f"storage_soc_{storage_tech}_{region}"][T[0]] * 1.05,
                        f"cyclical_storage_max_{storage_tech}_{region}_{uuid.uuid4().hex[:8]}"
                    )

    # ---------------------------------------------------------------------
    #  STORAGE CONSTRAINTS  (corrected version – uses regional_storage keys)
    # ---------------------------------------------------------------------
    def _add_storage_constraints(
            self,
            data: Dict[str, pd.DataFrame],
            T: List[Union[int, pd.Timestamp]]
        ) -> None:
        """Add power- and energy-related constraints for all storage techs."""
        logger.info("Adding storage constraints")

        if not self.storage_techs:
            logger.warning("No storage technologies defined, skipping section")
            return
        for region in self.regions:
            region_store_cfg = self.storage_capacities.get(region, {})
            for tech in self.storage_techs:
                # ---------- efficiencies ----------------------------------
                sp = self.config.get('storage_params', {}).get(tech, {})
                charge_eff = sp.get('charge_efficiency', 0.95)
                discharge_eff = sp.get('discharge_efficiency', 0.95)
                if 'efficiency' in sp:                       # round-trip
                    charge_eff = discharge_eff = sp['efficiency'] ** 0.5
                # ---------- capacities ------------------------------------
                max_capacity = float(
                    region_store_cfg.get(f"{tech}_stockage_MWh", 0)
                )
                if max_capacity == 0 and \
                    region in self.tech_capacities and tech in self.tech_capacities[region]:
                    max_capacity = float(self.tech_capacities[region][tech])
                if max_capacity <= 0:
                    continue
                max_power = float(
                    region_store_cfg.get(
                        f"{tech}_puissance_MW",
                        sp.get('max_power_ratio', 1.0) * max_capacity
                    )
                )
                if region in self.regional_multipliers:
                    mult = self.regional_multipliers[region]
                    max_capacity *= mult.get(f"{tech}_capacity_multiplier", 1.0)
                    max_power    *= mult.get(f"{tech}_power_multiplier",    1.0)
                has_vars = all(
                    f"{prefix}_{tech}_{region}" in self.variables
                    for prefix in ("storage_charge", "storage_discharge", "storage_soc")
                )
                if not has_vars:
                    continue
        # ---------- POWER limits -----------------------------------
                for t in T:
                    if t not in self.variables[f"storage_charge_{tech}_{region}"]:
                        continue
                    self.model += (
                        self.variables[f"storage_charge_{tech}_{region}"][t] <= max_power,
                        f"stor_max_charge_{tech}_{region}_{t}_{uuid.uuid4().hex[:6]}"
                    )
                    self.model += (
                        self.variables[f"storage_discharge_{tech}_{region}"][t] <= max_power,
                        f"stor_max_discharge_{tech}_{region}_{t}_{uuid.uuid4().hex[:6]}"
                    )
    # ---------- SOC bounds & dynamics --------------------------
                for i, t in enumerate(T):
                    if t not in self.variables[f"storage_soc_{tech}_{region}"]:
                        continue
                    self.model += (
                        self.variables[f"storage_soc_{tech}_{region}"][t] <= max_capacity,
                           f"stor_soc_max_{tech}_{region}_{t}_{uuid.uuid4().hex[:6]}"
                    )
                    self.model += (
                        self.variables[f"storage_soc_{tech}_{region}"][t] >= 0,
                        f"stor_soc_min_{tech}_{region}_{t}_{uuid.uuid4().hex[:6]}"
                    )
                if i < len(T) - 1:
                    t_next = T[i + 1]
                    if all(
                        t_next in self.variables[k] 
                        for k in (
                            f"storage_soc_{tech}_{region}",
                            f"storage_charge_{tech}_{region}",
                            f"storage_discharge_{tech}_{region}",
                            )
                        ):
                            self.model += (
                                self.variables[f"storage_soc_{tech}_{region}"][t_next]
                                == self.variables[f"storage_soc_{tech}_{region}"][t]
                                + self.variables[f"storage_charge_{tech}_{region}"][t] * charge_eff
                                - self.variables[f"storage_discharge_{tech}_{region}"][t] * (1.0 / discharge_eff),
                                f"stor_dyn_{tech}_{region}_{t}_{uuid.uuid4().hex[:6]}"
                            )
        # ---------- cyclical constraint (optional) -----------------
        if self.config.get('use_cyclical_storage', False) and len(T) >= 2:
            self.model += (
                self.variables[f"storage_soc_{tech}_{region}"][T[0]]
                == self.variables[f"storage_soc_{tech}_{region}"][T[-1]],
                f"stor_cyc_{tech}_{region}_{uuid.uuid4().hex[:6]}"
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
        constraints = self.config.get('constraints') or {}
        min_storage_utilization = constraints.get('min_storage_utilization', 0.15)  # 15% minimum storage contribution
        min_dr_utilization = constraints.get('min_dr_utilization', 0.10)  # 10% minimum demand response contribution
        min_exchange_utilization = constraints.get('min_exchange_utilization', 0.15)  # 15% minimum exchange contribution
        
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
        
        if self.use_simplified_model and self.simplification_options.get("lp_relaxation", False):
            logger.info("Converting MILP to LP by relaxing integer constraints")
            for var in self.binary_vars:
                var.cat = pulp.LpContinuous    # ← plus de référence à pl
                var.lowBound = 0
                var.upBound  = 1
        
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
            return pulp.LpStatusNotSolved, time.time() - start_time

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
