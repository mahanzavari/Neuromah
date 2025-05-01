import numpy as np
import cupy as cp
from typing import Optional, Union
from ....initializers import Initializer
from ....initializers.Initializer import RandomNormalInitializer, XavierInitializer, HeInitializer

class DenseCUDA:
    def __init__(self, n_inputs: int, n_neurons: int,
                 weight_initializer: Optional[Union[object, str]] = None,
                 activation: Optional[object] = None,
                 weight_regularizer_l1: float = 0,
                 weight_regularizer_l2: float = 0):
        # Validate inputs
        if not isinstance(n_inputs, int) or n_inputs <= 0:
            raise ValueError("n_inputs must be a positive integer")
        if not isinstance(n_neurons, int) or n_neurons <= 0:
            raise ValueError("n_neurons must be a positive integer")

        # Initialize weights and biases on GPU
        if weight_initializer is None:
            self.weight_initializer = RandomNormalInitializer(xp=cp)
        elif isinstance(weight_initializer, str):
            if weight_initializer.lower() == 'xavier':
                self.weight_initializer = XavierInitializer(xp=cp)
            elif weight_initializer.lower() == "he":
                self.weight_initializer = HeInitializer(xp=cp)
            else:
                raise ValueError(f"Unknown initializer: {weight_initializer}")
        elif isinstance(weight_initializer, Initializer):
            self.weight_initializer = weight_initializer
        else:
            raise ValueError("weight_initializer must be a string or an Initializer instance")

        self.weights = cp.asarray(self.weight_initializer.initialize((n_inputs, n_neurons)))
        self.biases = cp.zeros((1, n_neurons))
        
        self.activation = activation
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        
        # Initialize CUDA streams
        self.stream = cp.cuda.Stream()
        
        # Set up kernel parameters
        self.block_size = (32, 32)
        self.grid_size = lambda x, y: ((x + 31) // 32, (y + 31) // 32)

    def forward(self, inputs: cp.ndarray, training: bool) -> None:
        if not isinstance(inputs, cp.ndarray):
            inputs = cp.asarray(inputs)
            
        self.inputs = inputs
        batch_size = inputs.shape[0]
        
        # Allocate output array
        self.output = cp.empty((batch_size, self.weights.shape[1]), dtype=cp.float32)
        
        # Launch forward kernel
        with self.stream:
            # Compute matrix multiplication
            self.output = cp.dot(inputs, self.weights) + self.biases
            
            # Apply activation if exists
            if self.activation is not None:
                self.activation.forward(self.output, training=training)
                self.output = self.activation.output

    def backward(self, dvalues: cp.ndarray) -> None:
        if not isinstance(dvalues, cp.ndarray):
            dvalues = cp.asarray(dvalues)
            
        batch_size = dvalues.shape[0]
        
        # Backward pass through activation first
        if self.activation is not None:
            self.activation.backward(dvalues)
            dvalues = self.activation.dinputs
            
        # Compute gradients
        with self.stream:
            # Gradients on parameters
            self.dweights = cp.dot(self.inputs.T, dvalues) / batch_size
            self.dbiases = cp.sum(dvalues, axis=0, keepdims=True) / batch_size
            
            # L1/L2 regularization
            if self.weight_regularizer_l1 > 0:
                self.dweights += self.weight_regularizer_l1 * cp.sign(self.weights)
            if self.weight_regularizer_l2 > 0:
                self.dweights += 2 * self.weight_regularizer_l2 * self.weights
                
            # Gradient on inputs
            self.dinputs = cp.dot(dvalues, self.weights.T)

    def get_parameters(self) -> dict:
        return {
            'weights': (self.weights, self.dweights),
            'biases': (self.biases, self.dbiases)
        }

    def set_parameters(self, weights: cp.ndarray, biases: cp.ndarray) -> None:
        if weights.shape != self.weights.shape:
            raise ValueError(f"Expected weights shape {self.weights.shape}, got {weights.shape}")
        if biases.shape != self.biases.shape:
            raise ValueError(f"Expected biases shape {self.biases.shape}, got {biases.shape}")
        self.weights = cp.asarray(weights)
        self.biases = cp.asarray(biases) 