import numpy as np

    
from typing import Dict, Tuple

class Optimizer_Nadam:
    """
    Nesterov-accelerated Adaptive Moment Estimation (Nadam) optimizer.
    Supports arbitrary trainable parameters, making it compatible with
    various layer types (e.g., Dense, CNN, RNN, LSTM, GRU).

    Nadam combines the benefits of Adam with Nesterov momentum for improved convergence.

    Attributes:
        learning_rate (float): The initial learning rate.
        current_learning_rate (float): Adjusted learning rate after decay.
        decay (float): Decay rate for learning rate.
        iterations (int): Number of updates performed.
        epsilon (float): Small constant for numerical stability.
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
    
    Supports arbitrary trainable parameters, 
    making it compatible with
    various layer types (e.g., Dense, CNN, RNN, LSTM, GRU).
    """

    def __init__(self, 
                 learning_rate: float = 1e-3, 
                 decay: float = 0.0, 
                 beta1: float = 0.9, 
                 beta2: float = 0.999, 
                 epsilon: float = 1e-8) -> None:
        # Input and params validations
        if not isinstance(learning_rate, (int, float)):
            raise TypeError("learning_rate must be a number")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(decay, (int, float)):
            raise TypeError("decay must be a number")
        if decay < 0:
            raise ValueError("decay must be non-negative")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not 0 < beta1 < 1:
            raise ValueError("beta1 must be in range (0, 1)")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not 0 < beta2 < 1:
            raise ValueError("beta2 must be in range (0, 1)")
        if not isinstance(epsilon, float):
            raise TypeError("epsilon must be a float")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

        # Store momentum and cache for all parameter keys
        self.momentums: Dict[str, np.ndarray] = {}
        self.caches: Dict[str, np.ndarray] = {}

    def pre_update_params(self) -> None:
        """Adjust the current learning rate based on the decay schedule."""
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer) -> None:
        """
        Update the trainable parameters of a layer using the Nadam optimization algorithm.
        
        Args:
            layer (Layer): The layer to update. It must have a `get_parameters()` method 
                           that returns a dictionary of parameters and their gradients.
        
        Raises:
            AttributeError: If the layer does not have a `get_parameters()` method.
            TypeError: If the get_parameters method doesn't return a dictionary.
            ValueError: If the parameter value and gradient don't have the same shape.
        """
        # Check if the layer has a get_parameters method
        if not hasattr(layer, "get_parameters") or not callable(getattr(layer, "get_parameters")):
            raise AttributeError("Layer must have a 'get_parameters' method")
        
        # Retrieve trainable parameters and their gradients
        parameters = layer.get_parameters()

        if not isinstance(parameters, dict):
            raise TypeError("Layer's get_parameters() must return a dictionary.")
        
        for param_name, param_values in parameters.items():
            # Check if the layer's parameters are valid
            if not isinstance(param_name, str):
                raise TypeError("Parameter name must be a string")
            if not isinstance(param_values, tuple) or len(param_values) != 2:
                raise ValueError("Parameter values must be a tuple of (parameter, gradient)")
                
            # Unpack weights and gradients
            param, gradient = param_values
                
            if not isinstance(param, np.ndarray) or not isinstance(gradient, np.ndarray):
                raise TypeError("Parameter and gradient must be numpy arrays")
            
            if param.shape != gradient.shape:
                raise ValueError("Parameter and gradient must have the same shape")

            # Initialize momentums and caches for this parameter if not done
            if param_name not in self.momentums:
                self.momentums[param_name] = np.zeros_like(param)
                self.caches[param_name] = np.zeros_like(param)

            # Update momentums and caches
            self.momentums[param_name] = (
                self.beta1 * self.momentums[param_name] + (1 - self.beta1) * gradient
            )
            self.caches[param_name] = (
                self.beta2 * self.caches[param_name] + (1 - self.beta2) * gradient**2
            )

            # Bias correction
            beta1_power = 1 - self.beta1 ** (self.iterations + 1)
            beta2_power = 1 - self.beta2 ** (self.iterations + 1)
            m_hat = self.momentums[param_name] / (beta1_power + self.epsilon)
            v_hat = self.caches[param_name] / (beta2_power + self.epsilon)

            # Nesterov correction
            m_nesterov = (
                self.beta1 * m_hat + (1 - self.beta1) * gradient / (beta1_power + self.epsilon)
            )

            # Update parameter values
            param -= (
                self.current_learning_rate * m_nesterov / (np.sqrt(v_hat) + self.epsilon)
            )

    def post_update_params(self) -> None:
        """Increment the iteration counter."""
        self.iterations += 1