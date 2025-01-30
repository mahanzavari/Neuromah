import numpy as np
from typing import Dict

class Optimizer_Adam:
    """
    Adam optimizer.

    Adam (Adaptive Moment Estimation) combines the benefits of Adagrad and RMSprop,
    and also includes bias corrections for the first and second moment estimates.

    Attributes:
        learning_rate (float): The initial learning rate.
        current_learning_rate (float): The current learning rate, which can be adjusted by decay.
        decay (float): The decay rate for the learning rate.
        iterations (int): The number of iterations performed.
        epsilon (float): A small constant for numerical stability.
        beta_1 (float): The exponential decay rate for the first moment estimates.
        beta_2 (float): The exponential decay rate for the second moment estimates.
    """

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        if not (0 <= beta_1 < 1) or not (0 <= beta_2 < 1):
            raise ValueError("beta values must be in the interval [0 , 1)")
        # add warning for unlogical values
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.momentums = {}
        self.caches = {}

    def pre_update_params(self):
        """
        Adjusts the learning rate if decay is applied.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
        # no assumption is made about the Layers parameter in order to make it a generic code
    def update_params(self, layer):
        """
        Updates the layer's parameters using the Adam algorithm.

        Args:
            layer: The layer whose parameters (weights and biases) are to be updated.
        """
        
        if not hasattr(layer, 'get_parameters') or not callable(getattr(layer, 'get_parameters')):
            raise AttributeError("Layer must have a get_parameters method")
        
        parameters = layer.get_parameters()

        if not isinstance(parameters, Dict):
            raise TypeError("The get_parameters method must return a dictionary")
        
        for param_name, param_values in parameters.items():
            if not isinstance(param_name, str):
                raise ValueError("Parameter name must be a string")
            if not isinstance(param_values, tuple) or len(param_values) != 2:
                raise ValueError("The parameter values must be in tuples")
            # if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
            #     raise ValueError("Gradients contain NaN/inf values.")

            param, gradient = param_values
            if not isinstance(param, np.ndarray) or not isinstance(gradient, np.ndarray):
                raise ValueError("The parameters and gradients must be numpy arrays")
            if param.shape != gradient.shape:
                raise ValueError("The parameter and gradient must have the same shape")

            if param_name not in self.momentums:
                self.momentums[param_name] = np.zeros_like(param)
                self.caches[param_name] = np.zeros_like(param)

            # Update momentums and caches
            self.momentums[param_name] = (
                self.beta_1 * self.momentums[param_name] + (1.0 - self.beta_1) * gradient
            )
            self.caches[param_name] = (
                self.beta_2 * self.caches[param_name] + (1.0 - self.beta_2) * gradient ** 2
            )

            # Bias correction
            m_hat = self.momentums[param_name] / (1.0 - self.beta_1 ** (self.iterations + 1))
            v_hat = self.caches[param_name] / (1.0 - self.beta_2 ** (self.iterations + 1))

            # Update parameters
            param -= (
                self.current_learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            )
            
        def get_state(self):
            return {
                'momentums' : self.momentums,
                'caches' : self.caches,
                'iterations' : self.iterations
            }
        
        def laod_state(self , state: Dict[str , any]):
            self.momentums = state['momentums']
            self.cache = state['cache']
            self.iterations = state['iterations']

    def post_update_params(self):
        """
        Increments the iteration count after parameter updates.
        """
        self.iterations += 1