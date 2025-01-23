import numpy as np
from typing import Dict , Tuple

class Optimizer_Adagrad:
    """
    Adagrad optimizer.

    Adagrad adapts the learning rate to the parameters, performing smaller updates
    for parameters associated with frequently occurring features, and larger updates
    for parameters associated with infrequent features.

    Attributes:
        learning_rate (float): The initial learning rate.
        current_learning_rate (float): The current learning rate, which can be adjusted by decay.
        decay (float): The decay rate for the learning rate.
        iterations (int): The number of iterations performed.
        epsilon (float): A small constant for numerical stability.
    """

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.momentums = {}

    # Call once before any parameter updates
    def pre_update_params(self):
        """
        Adjusts the learning rate if decay is applied.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        """
        Updates the layer's parameters using the Adagrad algorithm.

        Args:
            layer: The layer whose parameters (weights and biases) are to be updated.
        """

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2


        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        # / indicates next line whereas / indicates division
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        """
        Increments the iteration count after parameter updates.
        """
        self.iterations += 1
