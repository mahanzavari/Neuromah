from typing import Dict, Tuple, Optional, Union
from ...initializers import Initializer
from ...initializers.Initializer import RandomNormalInitializer, XavierInitializer, HeInitializer
from ..core.base import BaseLayer
import cupy as cp # type: ignore
import numpy as np

class Dense(BaseLayer):
    """
    A class representing a Dense (fully connected) layer in a neural network.
    This is a wrapper class that can use either CPU or GPU implementation.

    Parameters
    ----------
    input_size : int
        The number of input features.
    output_size : int
        The number of output features.
    weight_initializer : Optional[Union[object, str]]
        An initializer object to initialize the weights. Defaults to None.
    bias_initializer : Optional[Union[object, str]]
        An initializer object to initialize the biases. Defaults to None.
    use_gpu : bool
        Whether to use GPU implementation. Defaults to False.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        weight_initializer: Optional[Union[object, str]] = None,
        bias_initializer: Optional[Union[object, str]] = None,
        use_gpu: bool = False
    ):
        super().__init__(use_gpu)
        
        # Import the appropriate implementation
        if use_gpu:
            try:
                from .cuda.dense_cuda import DenseCUDA
                self.impl = DenseCUDA(
                    input_size=input_size,
                    output_size=output_size,
                    weight_initializer=weight_initializer,
                    bias_initializer=bias_initializer
                )
            except ImportError:
                raise ImportError("CUDA implementation not available. Please install cupy and ensure CUDA is properly set up.")
        else:
            # Use CPU implementation
            self.input_size = input_size
            self.output_size = output_size
            
            if weight_initializer is None:
                self.weight_initializer = RandomNormalInitializer(xp=self.xp)
            elif isinstance(weight_initializer, str):
                if weight_initializer.lower() == 'xavier':
                    self.weight_initializer = XavierInitializer(xp=self.xp)
                elif weight_initializer.lower() == "he":
                    self.weight_initializer = HeInitializer(xp=self.xp)
                else:
                    raise ValueError(f"Unknown initializer: {weight_initializer}")
            elif isinstance(weight_initializer, Initializer):
                self.weight_initializer = weight_initializer
            else:
                raise ValueError("weight_initializer must be a string or an Initializer instance")
                
            self.weights = self.weight_initializer.initialize((output_size, input_size))
            self.biases = self.xp.zeros((output_size, 1))

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Compute the forward pass of the layer.
        
        Args:
            inputs (np.ndarray): Input data of shape (batch_size, input_size).
            training (bool): Whether the layer is in training mode.
        """
        if self.use_gpu:
            self.impl.forward(inputs, training)
            self.output = self.impl.output
        else:
            if not isinstance(inputs, self.xp.ndarray):
                inputs = self.xp.asarray(inputs)
                
            self.inputs = inputs
            self.output = self.xp.dot(inputs, self.weights.T) + self.biases.T

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Compute the backward pass of the layer.
        
        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the layer's output.
        """
        if self.use_gpu:
            self.impl.backward(dvalues)
            self.dinputs = self.impl.dinputs
            self.dweights = self.impl.dweights
            self.dbiases = self.impl.dbiases
        else:
            if not isinstance(dvalues, self.xp.ndarray):
                dvalues = self.xp.asarray(dvalues)
                
            # Gradients on parameters
            self.dweights = self.xp.dot(dvalues.T, self.inputs)
            self.dbiases = self.xp.sum(dvalues, axis=0, keepdims=True).T
            
            # Gradient on values
            self.dinputs = self.xp.dot(dvalues, self.weights)

    def get_parameters(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get the layer's parameters and their gradients."""
        if self.use_gpu:
            return self.impl.get_parameters()
        return {
            'weights': (self.weights, self.dweights),
            'biases': (self.biases, self.dbiases)
        }

    def set_parameters(self, weights: np.ndarray, biases: np.ndarray) -> None:
        """Set the layer's parameters."""
        if self.use_gpu:
            self.impl.set_parameters(weights, biases)
        else:
            if weights.shape != self.weights.shape:
                raise ValueError(f"Expected weights shape {self.weights.shape}, got {weights.shape}")
            if biases.shape != self.biases.shape:
                raise ValueError(f"Expected biases shape {self.biases.shape}, got {biases.shape}")
            self.weights = self.xp.asarray(weights)
            self.biases = self.xp.asarray(biases) 