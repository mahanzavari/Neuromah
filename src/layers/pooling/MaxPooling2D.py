import numpy as np
from ..compiled import _MaxPooling2DBackend_cpp

class Layer_MaxPooling2D:
    def __init__(self, pool_size, strides=None, padding='valid'):
        """
        Initializes a 2D max pooling layer.

        Args:
            pool_size (int or tuple): Size of the pooling window (e.g., 2 for a 2x2 window).
            strides (int or tuple, optional): Stride of the pooling operation. 
                                              If None, defaults to pool_size.
            padding (str): "valid" or "same" padding.
        """
        # Ensure pool_size is a tuple.
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        elif isinstance(pool_size, tuple) and len(pool_size) == 2:
            self.pool_size = pool_size
        else:
            raise ValueError("pool_size must be an int or a tuple of two ints")
        
        # If strides is not provided, default to pool_size.
        if strides is None:
            self.strides = self.pool_size
        elif isinstance(strides, int):
            self.strides = (strides, strides)
        elif isinstance(strides, tuple) and len(strides) == 2:
            self.strides = strides
        else:
            raise ValueError("strides must be an int or a tuple of two ints")
        
        if padding not in ['valid', 'same']:
            raise ValueError("padding must be 'valid' or 'same'")
        self.padding = padding

    def forward(self, inputs, training=True):
        """
        Performs the forward pass of the max pooling operation.

        Args:
            inputs (np.ndarray): Input data of shape (batch_size, channels, height, width).
            training (bool): Whether the layer is in training mode (not used here, but provided for API consistency).

        Returns:
            np.ndarray: The pooled output.
        """
        self.inputs = inputs
        # Call the C++ backend forward function.
        # It returns a tuple (output, max_indices) where:
        #   - output is the pooled result,
        #   - max_indices is an integer array with the indices of the maximum values.
        output, max_indices = _MaxPooling2DBackend_cpp.maxpool2d_forward_cpu(
            inputs,
            self.pool_size[0], self.pool_size[1],
            self.strides[0], self.strides[1],
            self.padding
        )
        self.output = output
        self.max_indices = max_indices  # Save indices for use in backward pass.
        return output

    def backward(self, dvalues):
        """
        Performs the backward pass of the max pooling operation.

        Args:
            dvalues (np.ndarray): Gradient from the next layer of shape (batch_size, channels, out_height, out_width).

        Returns:
            np.ndarray: The gradient with respect to the input.
        """
        dinputs = _MaxPooling2DBackend_cpp.maxpool2d_backward_cpu(
            dvalues,
            self.max_indices,
            self.inputs,
            self.pool_size[0], self.pool_size[1],
            self.strides[0], self.strides[1],
            self.padding
        )
        
        self.dinputs = dinputs

        return dinputs
