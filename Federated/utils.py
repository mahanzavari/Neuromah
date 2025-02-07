import numpy as np
"""
Federated Learning Utility Functions.

This module provides helper functions for federated learning operations.
It includes:
- `federated_average()`: Aggregates multiple client model updates.

Features:
- Uses weighted averaging to update global model parameters.

Dependencies:
- NumPy for numerical operations.

Usage:
```python
aggregated_weights = federated_average(client_updates, sample_counts)
"""

def federated_average(updates, num_samples):
    total_samples = sum(num_samples)
    weighted_updates = [np.array(update) * (samples / total_samples) for update, samples in zip(updates, num_samples)]
    return np.sum(weighted_updates, axis=0).tolist()
