import numpy as np
from typing import Dict, Tuple, Literal, Optional
import cupy as cp

class Layer_Pooling:
    """
    A class representing a 2D Pooling layer in a neural network.
    This is a wrapper class that can use either CPU or GPU implementation.

    Parameters
    ----------
    pool_size : int
        The size of the pooling window. Defaults to 2.
    stride : int
        The stride of the pooling operation. Defaults to 2.
    padding : int
        The padding size. Defaults to 0.
    mode : Literal['max', 'avg']
        The pooling mode. Can be either 'max' or 'avg'. Defaults to 'max'.
    use_gpu : bool
        Whether to use GPU implementation. Defaults to False.
    """
    
    def __init__(
        self,
        pool_size: int = 2,
        stride: int = 2,
        padding: int = 0,
        mode: Literal['max', 'avg'] = 'max',
        use_gpu: bool = False
    ):
        self.use_gpu = use_gpu
        self.xp = cp if use_gpu else np
        
        # Import the appropriate implementation
        if use_gpu:
            try:
                from ..cuda.pooling_cuda import PoolingCUDA
                self.impl = PoolingCUDA(
                    pool_size=pool_size,
                    stride=stride,
                    padding=padding,
                    mode=mode
                )
            except ImportError:
                raise ImportError("CUDA implementation not available. Please install cupy and ensure CUDA is properly set up.")
        else:
            # Use CPU implementation
            self.pool_size = pool_size
            self.stride = stride
            self.padding = padding
            self.mode = mode

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Compute the forward pass of the layer.
        
        Args:
            inputs (np.ndarray): Input data of shape (batch_size, channels, height, width).
            training (bool): Whether the layer is in training mode.
        """
        if self.use_gpu:
            self.impl.forward(inputs, training)
            self.output = self.impl.output
        else:
            if not isinstance(inputs, self.xp.ndarray):
                inputs = self.xp.asarray(inputs)
                
            self.inputs = inputs
            batch_size = inputs.shape[0]
            channels = inputs.shape[1]
            input_height = inputs.shape[2]
            input_width = inputs.shape[3]
            
            output_height = (input_height + 2 * self.padding - self.pool_size) // self.stride + 1
            output_width = (input_width + 2 * self.padding - self.pool_size) // self.stride + 1
            
            self.output = self.xp.zeros((batch_size, channels, output_height, output_width))
            
            # Apply padding if needed
            if self.padding > 0:
                inputs_padded = self.xp.pad(
                    inputs,
                    ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                    mode='constant'
                )
            else:
                inputs_padded = inputs
                
            # Perform pooling
            for b in range(batch_size):
                for c in range(channels):
                    for h_out in range(output_height):
                        for w_out in range(output_width):
                            h_start = h_out * self.stride
                            w_start = w_out * self.stride
                            
                            # Extract the input patch
                            patch = inputs_padded[b, c, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size]
                            
                            # Apply pooling
                            if self.mode == 'max':
                                self.output[b, c, h_out, w_out] = self.xp.max(patch)
                            else:  # avg
                                self.output[b, c, h_out, w_out] = self.xp.mean(patch)

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
                
            batch_size = dvalues.shape[0]
            channels = dvalues.shape[1]
            input_height = self.inputs.shape[2]
            input_width = self.inputs.shape[3]
            
            self.dinputs = self.xp.zeros_like(self.inputs)
            
            # Apply padding if needed
            if self.padding > 0:
                dinputs_padded = self.xp.zeros((
                    batch_size,
                    channels,
                    input_height + 2 * self.padding,
                    input_width + 2 * self.padding
                ))
            else:
                dinputs_padded = self.dinputs
                
            # Compute gradients
            for b in range(batch_size):
                for c in range(channels):
                    for h_out in range(dvalues.shape[2]):
                        for w_out in range(dvalues.shape[3]):
                            h_start = h_out * self.stride
                            w_start = w_out * self.stride
                            
                            # Extract the input patch
                            patch = self.inputs[b, c, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size]
                            
                            if self.mode == 'max':
                                # Find the position of the maximum value
                                max_pos = self.xp.unravel_index(
                                    self.xp.argmax(patch),
                                    patch.shape
                                )
                                # Set gradient only at the maximum position
                                dinputs_padded[b, c, h_start+max_pos[0], w_start+max_pos[1]] += dvalues[b, c, h_out, w_out]
                            else:  # avg
                                # Distribute gradient equally
                                avg_grad = dvalues[b, c, h_out, w_out] / (self.pool_size * self.pool_size)
                                dinputs_padded[b, c, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size] += avg_grad
            
            # Remove padding from input gradients if needed
            if self.padding > 0:
                self.dinputs = dinputs_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
            else:
                self.dinputs = dinputs_padded 