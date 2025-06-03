"""Utility functions to compute emissions for Regional Flex results."""
from typing import Dict
import pandas as pd


def calculate_emissions(results: Dict, config: Dict) -> Dict:
    """Compute emissions by technology and region.

    Parameters
    ----------
    results : dict
        Dict returned by ``RegionalFlexOptimizer.get_results``.
    config : dict
        Loaded configuration containing ``emission_factors`` and
        ``startup_emissions`` dictionaries. Optionally a
        ``ramp_emission_penalty`` scalar.

    Returns
    -------
    dict
        Dictionary with per-tech timeseries and aggregated totals.
    """
    emission_factors = config.get("emission_factors", {})
    startup_emissions = config.get("startup_emissions", {})
    ramp_penalty = config.get("ramp_emission_penalty", 0.0)

    timeseries = {}
    regions = results.get("regions", [])
    techs = results.get("dispatch_techs", [])

    total_by_region = {r: 0.0 for r in regions}
    total_by_tech = {t: 0.0 for t in techs}

    for region in regions:
        for tech in techs:
            dispatch_key = f"dispatch_{tech}_{region}"
            if dispatch_key not in results["variables"]:
                continue
            dispatch = pd.Series(results["variables"][dispatch_key]).sort_index()
            factor = emission_factors.get(tech, 0.0)
            emis = dispatch * factor

            # additional emissions from ramping
            if ramp_penalty > 0 and len(dispatch) > 1:
                ramp = dispatch.diff().abs().fillna(0)
                emis += ramp * factor * ramp_penalty

            start_key = f"startup_{tech}_{region}"
            if start_key in results["variables"]:
                start_series = pd.Series(results["variables"][start_key]).sort_index()
                emis += start_series * startup_emissions.get(tech, 0.0)

            timeseries[f"emission_{tech}_{region}"] = emis.to_dict()
            total = emis.sum()
            total_by_region[region] += total
            total_by_tech[tech] += total

    return {
        "timeseries": timeseries,
        "total_by_region": total_by_region,
        "total_by_tech": total_by_tech,
    }
