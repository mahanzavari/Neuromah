import numpy as np
from typing import Optional

class Layer_Flatten:
    """
    Flatten layer that converts multi-dimensional input into 1D vectors.
    Used for transitioning from convolutional/pooling layers to dense layers.
    
    Attributes:
        xp: Array module (NumPy or CuPy) for device-agnostic operations
        input_shape: Shape of the input to the layer (stored for backward pass)
    """
    
    def __init__(self, xp=np):
        self.xp = xp
        self.input_shape = None
        self.output_shape = None
        self.dinputs = None
        self.output = None

    def forward(self, inputs: np.ndarray, training: bool) -> np.ndarray:
        """
        Forward pass of the flatten layer.
        
        Args:
            inputs: Input tensor of shape (batch_size, dim1, dim2, ...)
            training: Flag indicating training mode (unused here, but kept for API consistency)
            
        Returns:
            Flattened output of shape (batch_size, product_of_dimensions)
        """
        self.input_shape = inputs.shape
        batch_size = inputs.shape[0]
        
        # Flatten all dimensions except batch dimension
        self.output = inputs.reshape(batch_size, -1)
        self.output_shape = self.output.shape
        
        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """
        Backward pass of the flatten layer.
        
        Args:
            dvalues: Gradient of the loss with respect to the output of this layer
            
        Returns:
            Gradient reshaped to match original input dimensions
        """
        # Reshape gradients to match original input shape
        self.dinputs = dvalues.reshape(self.input_shape)
        return self.dinputs

    def get_parameters(self) -> dict:
        """Return parameters (none for flatten layer)"""
        return {}  # No trainable parameters

    def set_parameters(self, params: dict) -> None:
        """Set parameters (no-op for flatten layer)"""
        pass  # No parameters to set

    def __repr__(self) -> str:
        return f"Layer_Flatten(input_shape={self.input_shape})"