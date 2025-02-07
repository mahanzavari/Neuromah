import numpy as np

class FLTrainer:
     """
Federated Learning Training Module.

This module provides a class (`FLTrainer`) for local model training on a client.
It includes:
- A method to receive the global model.
- A function to train the model locally.
- A function to return the updated model parameters.

Features:
- Simulates local training with simple weight updates.
- Returns updated model parameters and sample count.

Dependencies:
- NumPy for mathematical computations.

Usage:
```python
trainer = FLTrainer(global_model)
updated_model, num_samples = trainer.train_local_model()
"""
     def __init__(self, global_model):
         self.model_weights = np.array(global_model["weights"])
 
     def train_local_model(self):
         # Simulate local training (add small updates)
         new_weights = self.model_weights + np.random.normal(0, 0.01, size=self.model_weights.shape)
         return {"weights": new_weights.tolist()}, 100
