import pandas as pd
import numpy as np

def rolling_horizon_indices(total_steps, window_size, stride):
    """
    Generate (start, end) index pairs for a rolling horizon.
    Each window is [start, end) (Python slice convention).
    """
    indices = []
    start = 0
    while start < total_steps:
        end = min(start + window_size, total_steps)
        indices.append((start, end))
        if end == total_steps:
            break
        start += stride
    return indices


def extract_final_states(results, state_vars, end_idx):
    """
    Extract the final value of each state variable at the end of a window.
    Args:
        results (dict): Results from optimizer.get_results(), must have 'variables'.
        state_vars (list): List of variable names (keys in results['variables']) to extract.
        end_idx (int): The last global time index of the window (exclusive, i.e., end_idx-1 is last step).
    Returns:
        dict: {var_name: final_value}
    """
    final_states = {}
    for var in state_vars:
        series = results['variables'].get(var, {})
        if not series:
            continue
        # Find the max key <= end_idx-1
        last_t = max([t for t in series if t <= end_idx-1], default=None)
        if last_t is not None:
            final_states[var] = series[last_t]
    return final_states


def prepare_initial_states(previous_states):
    """
    Prepare a dictionary of initial values for the next window.
    Args:
        previous_states (dict): {var_name: value} from previous window's extract_final_states.
    Returns:
        dict: {var_name: value} for initializing next window's model.
    """
    return previous_states.copy() if previous_states else {}
