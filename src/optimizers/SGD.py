import numpy as np
from typing import Dict, Tuple, Union

class Optimizer_SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    SGD updates the parameters using the gradient of the loss function with respect
    to the parameters. It can optionally use momentum to accelerate convergence.

    Attributes:
        learning_rate (float): The initial learning rate.
        current_learning_rate (float): The current learning rate, which can be adjusted by decay.
        decay (float): The decay rate for the learning rate.
        iterations (int): The number of iterations performed.
        momentum_factor (float): The momentum factor, if momentum is used.
        momentums (Dict[str, np.ndarray]): Stores momentum values for each parameter.
    """
    def __init__(self, learning_rate: Union[int, float] = 1., decay: Union[int, float] = 0., momentum: Union[int, float] = 0.):
        """
        Initializes the SGD optimizer.

        Args:
            learning_rate (float): The initial learning rate. Default is 1.0.
            decay (float): The decay rate for the learning rate. Default is 0.0.
            momentum (float): The momentum factor. Default is 0.0.
        """
        if not isinstance(learning_rate, (int, float)):
            raise TypeError(f"learning_rate must be a number, got {type(learning_rate)}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate should be a positive number, got {learning_rate}")
        if not isinstance(decay, (int, float)):
            raise TypeError(f"decay must be a number, got {type(decay)}")
        if decay < 0:
            raise ValueError(f"decay must be non-negative, got {decay}")
        if not isinstance(momentum, (int, float)):
            raise TypeError(f"momentum must be a number, got {type(momentum)}")
        if momentum < 0:
            raise ValueError(f"momentum must be non-negative, got {momentum}")

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum_factor = momentum
        self.momentums = {}

    def pre_update_params(self):
        """
        Adjusts the learning rate if decay is applied.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Updates the layer's parameters using the SGD algorithm.

        Args:
            layer: The layer whose parameters (weights and biases) are to be updated.
        """
        if not hasattr(layer, 'get_parameters') or not callable(getattr(layer, 'get_parameters')):
            raise AttributeError(f"The layer: {layer} does not implement get_parameters method")

        parameters = layer.get_parameters()
        if not isinstance(parameters, Dict):
            raise TypeError(f"The get_parameters() method should return a Dict")

        for param_name, param_values in parameters.items():
            if not isinstance(param_name, str):
                raise TypeError(f"The parameter name for the layer:{layer} should be a string")
            if not isinstance(param_values, Tuple) or len(param_values) != 2:
                raise ValueError("The parameter values must be a tuple: (parameter, gradients)")

            param, gradients = param_values
            if not isinstance(param, np.ndarray) or not isinstance(gradients, np.ndarray):
                raise TypeError("The parameters and gradients for the layer must be numpy arrays")
            if param.shape != gradients.shape:
                raise ValueError("Parameter and gradient must have the same shape")
            if np.any(np.isnan(gradients)):
                raise ValueError(f"Gradient for '{param_name}' contains NaN values.")
            if np.any(np.isinf(gradients)):
                raise ValueError(f"Gradient for '{param_name}' contains Inf values.")

            if self.momentum_factor:
                if param_name not in self.momentums:
                    self.momentums[param_name] = np.zeros_like(param)
                self.momentums[param_name] = (
                    self.momentums[param_name] * self.momentum_factor + self.current_learning_rate * gradients
                )
                param += -self.momentums[param_name]
            else:
                param += -self.current_learning_rate * gradients

    def post_update_params(self):
        """
        Increments the iteration count after parameter updates.
        """
        self.iterations += 1