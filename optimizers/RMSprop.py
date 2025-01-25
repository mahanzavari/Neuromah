import numpy as np
from typing import Dict, Tuple

class Optimizer_RMSprop:
    """
    RMSprop optimizer.

    RMSprop (Root Mean Square Propagation) adapts the learning rate by dividing it by
    an exponentially decaying average of squared gradients.

    Attributes:
        learning_rate (float): The initial learning rate.
        current_learning_rate (float): The current learning rate, which can be adjusted by decay.
        decay (float): The decay rate for the learning rate.
        iterations (int): The number of iterations performed.
        epsilon (float): A small constant for numerical stability.
        rho (float): The decay rate for the moving average of squared gradients.
    """

    def __init__(self, learning_rate: float = 0.001, decay: float = 0., epsilon: float = 1e-7, rho: float = 0.9):
        # Validate hyperparameters
        if not isinstance(learning_rate, (float, int)) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive float.")
        if not isinstance(decay, (float, int)) or decay < 0:
            raise ValueError("decay must be a non-negative float.")
        if not isinstance(epsilon, (float, int)) or epsilon <= 0:
            raise ValueError("epsilon must be a positive float.")
        if not isinstance(rho, (float, int)) or rho < 0 or rho >= 1:
            raise ValueError("rho must be a float in the range [0, 1).")

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
        self.caches: Dict[str, np.ndarray] = {}  # Dictionary to store moving averages of squared gradients

    def pre_update_params(self) -> None:
        """
        Adjusts the learning rate if decay is applied.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer) -> None:
        """
        Updates the layer's parameters using the RMSprop algorithm.

        Args:
            layer: The layer whose parameters (weights and biases) are to be updated.
        """
        # Check if the layer has the required method
        if not hasattr(layer, 'get_parameters') or not callable(getattr(layer, 'get_parameters')):
            raise AttributeError("The layer must have a 'get_parameters' method.")

        # Retrieve parameters from the layer
        parameters = layer.get_parameters()

        # Validate the parameters
        if not isinstance(parameters, Dict):
            raise TypeError("The 'get_parameters' method must return a dictionary.")
        if not parameters:
            raise ValueError("The parameters dictionary is empty.")

        for param_name, param_values in parameters.items():
            # Validate parameter name
            if not isinstance(param_name, str):
                raise TypeError(f"Parameter name '{param_name}' must be a string.")

            # Validate parameter values
            if not isinstance(param_values, Tuple) or len(param_values) != 2:
                raise ValueError(f"Parameter values for '{param_name}' must be a tuple of (parameter, gradient).")

            param, gradient = param_values

            # Validate parameter and gradient types
            if not isinstance(param, np.ndarray):
                raise TypeError(f"Parameter '{param_name}' must be a numpy array.")
            if not isinstance(gradient, np.ndarray):
                raise TypeError(f"Gradient for '{param_name}' must be a numpy array.")

            # Validate parameter and gradient shapes
            if param.shape != gradient.shape:
                raise ValueError(f"Parameter and gradient for '{param_name}' must have the same shape.")

            # Check for NaN or Inf in gradients
            if np.any(np.isnan(gradient)):
                raise ValueError(f"Gradient for '{param_name}' contains NaN values.")
            if np.any(np.isinf(gradient)):
                raise ValueError(f"Gradient for '{param_name}' contains Inf values.")

            # Initialize cache for the parameter if not already present
            if param_name not in self.caches:
                self.caches[param_name] = np.zeros_like(param)

            # Update the moving average of squared gradients
            self.caches[param_name] = (
                self.rho * self.caches[param_name] + (1.0 - self.rho) * gradient**2
            )

            # Update parameters
            param -= self.current_learning_rate * gradient / (np.sqrt(self.caches[param_name]) + self.epsilon)

    def post_update_params(self) -> None:
        """
        Increments the iteration count after parameter updates.
        """
        self.iterations += 1