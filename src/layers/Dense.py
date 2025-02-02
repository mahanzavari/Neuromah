import numpy as np
from typing import Dict, Tuple, Optional

class Layer_Dense:
    """
A class representing a Dense (fully connected) layer in a neural network.

Parameters
----------
    n_inputs : int
        The number of input features for the layer. Must be a positive integer.
    n_neurons : int
        The number of neurons in the layer. Must be a positive integer.
    initializer : Optional[object]
        An initializer object to initialize the weights. Defaults to None, in which case weights are initialized with small random values.
    activation : Optional[object]
        An activation function object to apply to the layer's outputs. Defaults to None, meaning no activation function is applied.
    weight_regularizer_l1 : float
        The L1 regularization coefficient for the weights. Defaults to 0.
    weight_regularizer_l2 : float
        The L2 regularization coefficient for the weights. Defaults to 0."""
        
    def __init__(self, n_inputs: int, n_neurons: int, 
                 initializer: Optional[object] = None,
                 activation: Optional[object] = None,
                 weight_regularizer_l1: float = 0, 
                 weight_regularizer_l2: float = 0) -> None:
        # Validate inputs
        if not isinstance(n_inputs, int) or n_inputs <= 0:
            raise ValueError("n_inputs must be a positive integer")
        if not isinstance(n_neurons, int) or n_neurons <= 0:
            raise ValueError("n_neurons must be a positive integer")
        # Initialize weights and biases
        if initializer is None:
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        else:
            self.weights = initializer.initialize((n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))  # Always zero-initialize biases!!!!

        self.activation = activation  
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        # using regularization for biases is not recommneded and might harm the performance and reduce the model's flexibility

    # def forward(self, inputs: np.ndarray, training: bool) -> None:
    def forward(self, inputs: np.ndarray , training : bool) -> None:
        """
    Compute the forward pass of the layer.
    
    Args:
        inputs (np.ndarray): Input data of shape (batch_size, n_inputs).
        training (bool): Whether the layer is in training mode. Affects activation behavior (e.g., dropout).
        """
        if not isinstance(inputs , np.ndarray):
            raise TypeError("inputs must be a numpy array")
        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Input features mismatch. Expected {self.weights.shape[0]}, got {inputs.shape[1]}")
        if np.any(np.isnan(inputs)) or np.any(np.isinf(inputs)):
            raise ValueError("Input contains NaN/Inf values")
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

        # Forward pass through activation
        if self.activation is not None:
            self.activation.forward(self.output , training = training)
            self.output = self.activation.output 

    def backward(self, dvalues: np.ndarray) -> None:
        # Backward pass through activation first 
        if self.activation is not None:
            self.activation.backward(dvalues)
            dvalues = self.activation.dinputs  

        # Gradients on parameters
        batch_size = dvalues.shape[0]
        self.dweights = np.dot(self.inputs.T, dvalues) / batch_size  # Normalized
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) / batch_size

        # L1/L2 regularization for weights (biases excluded : line 26)
        if self.weight_regularizer_l1 > 0:
            dL1 = np.sign(self.weights)
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # Gradient on inputs
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {
            'weights': (self.weights, self.dweights),
            'biases': (self.biases, self.dbiases)
        }

    def set_parameters(self, weights: np.ndarray, biases: np.ndarray) -> None:
        if weights.shape != self.weights.shape:
            raise ValueError(f"Expected weights shape {self.weights.shape}, got {weights.shape}")
        if biases.shape != self.biases.shape:
            raise ValueError(f"Expected biases shape {self.biases.shape}, got {biases.shape}")
        self.weights = weights
        self.biases = biases