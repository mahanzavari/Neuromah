from typing import Dict, Tuple, Optional
from ..core.base import BaseLayer
import cupy as cp

class Dropout(BaseLayer):
    """
    A class representing a Dropout layer in a neural network.
    This is a wrapper class that can use either CPU or GPU implementation.

    Parameters
    ----------
    rate : float
        The dropout rate (probability of dropping a unit). Defaults to 0.1.
    use_gpu : bool
        Whether to use GPU implementation. Defaults to False.
    """
    
    def __init__(
        self,
        rate: float = 0.1,
        use_gpu: bool = False
    ):
        super().__init__(use_gpu)
        
        # Import the appropriate implementation
        if use_gpu:
            try:
                from .cuda.dropout_cuda import DropoutCUDA
                self.impl = DropoutCUDA(rate=rate)
            except ImportError:
                raise ImportError("CUDA implementation not available. Please install cupy and ensure CUDA is properly set up.")
        else:
            # Use CPU implementation
            self.rate = rate
            self.scale = 1.0 / (1.0 - rate)

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Compute the forward pass of the layer.
        
        Args:
            inputs (np.ndarray): Input data.
            training (bool): Whether the layer is in training mode.
        """
        if self.use_gpu:
            self.impl.forward(inputs, training)
            self.output = self.impl.output
        else:
            if not isinstance(inputs, self.xp.ndarray):
                inputs = self.xp.asarray(inputs)
                
            self.inputs = inputs
            
            if training:
                # Generate a random mask
                self.mask = self.xp.random.binomial(1, 1.0 - self.rate, size=inputs.shape)
                # Scale the inputs by the mask and the scale factor
                self.output = inputs * self.mask * self.scale
            else:
                # During inference, no dropout is applied
                self.output = inputs

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Compute the backward pass of the layer.
        
        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the layer's output.
        """
        if self.use_gpu:
            self.impl.backward(dvalues)
            self.dinputs = self.impl.dinputs
        else:
            if not isinstance(dvalues, self.xp.ndarray):
                dvalues = self.xp.asarray(dvalues)
                
            # Scale the gradients by the mask and the scale factor
            self.dinputs = dvalues * self.mask * self.scale 