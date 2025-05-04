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
