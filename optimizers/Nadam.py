import numpy as np

class Optimizer_Nadam:
    """
    Nesterov-accelerated Adaptive Moment Estimation (Nadam) optimizer.

    Nadam combines the benefits of Adam with Nesterov momentum, which often
    leads to faster and more stable convergence.

    Attributes:
        learning_rate (float): The initial learning rate.
        current_learning_rate (float): The current learning rate, which can be adjusted by decay.
        decay (float): The decay rate for the learning rate.
        iterations (int): The number of iterations performed.
        epsilon (float): A small constant added for numerical stability.
        beta_1 (float): The exponential decay rate for the first moment estimates.
        beta_2 (float): The exponential decay rate for the second moment estimates.
    """
    def __init__(self , learning_rate = 1e-3 , decay = 1. , beta1 = 0.9 , beta2 = 0.999 , epsilon = 1e-8):
        """
        Initialize the Nadam optimizer.

        Parameters:
            learning_rate (float): The learning rate.
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): Small constant to avoid division by zero.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        self.iter_num = 0
    def pre_update(self):
        """
        Adjusts the learning rate if decay is applied.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iter_num))
    def update(self , layer):
        """
        Updates the layer's parameters using the Nadam algorithm.

        Args:
            layer: The layer whose parameters (weights and biases) are to be updated.
        """
        if not hasattr(layer , 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # update momentums
        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * \
            layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + \
            (1 - self.beta1) * layer.dbiases
            
        # get corrected momentums
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (self.iter_num + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (self.iter_num + 1))
        
        layer.weight_cache = self.beta2 * layer.weight_cache + \
            (1 - self.beta2) * layer.dweights ** 2
        layer.bias_cache = self.beta2 * layer.bias_cache + \
            (1 - self.beta2) * layer.dbiases **2
            
        weight_cache_corrected = layer.weight_cache / ((1 - self.beta2) ** (self.iter_num + 1))
        bias_cache_corrected = layer.bias_cache / ((1 - self.beta2) ** (self.iter_num + 1))
        
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)
            
    def post_update_params(self):
        """
        Increments the iteration count after parameter updates.
        """
        self.iterations += 1