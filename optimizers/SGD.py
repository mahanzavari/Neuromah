import numpy as np


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
        momentum (float): The momentum factor, if momentum is used.
    """
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        """
        Initializes the SGD optimizer.

        Args:
            learning_rate (float): The initial learning rate. Default is 1.0.
            decay (float): The decay rate for the learning rate. Default is 0.0.
            momentum (float): The momentum factor. Default is 0.0.
        """
        if not isinstance(learning_rate , (int , float)):
            raise TypeError("learning rate must be a number")
        if learning_rate <= 0:
            raise ValueError("learning rate should be a positive number")
        if not isinstance(decay, (int, float)):
            raise TypeError("decay must be a number")
        if decay < 0:
            raise ValueError("decay must be non-negative")

        if not isinstance(momentum, (int, float)):
            raise TypeError("momentum must be a number")
        if momentum < 0:
            raise ValueError("momentum must be non-negative")
        
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Is called before any parameter updates
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
        Updates the layer's parameters using the SGD algorithm.

        Args:
            layer: The layer whose parameters (weights and biases) are to be updated.
        """
        if not hasattr(layer , "dweights") or not hasattr(layer , 'dbiases'):
            raise AttributeError('Layer Object must have weights and biases')
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'): # or hasatrr(layer , 'bias_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates


        # Vanilla SGD updates
        else:
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        """
        Increments the iteration count after parameter updates.
        """
        self.iterations += 1
