import numpy as np
import cupy as cp
from typing import Optional, Union, Tuple

class Conv2DCUDA:
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        weight_initializer: Optional[Union[object, str]] = None,
        bias_initializer: Optional[Union[object, str]] = None
    ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and biases
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
            
        self.weights = cp.asarray(self.weight_initializer.initialize(
            (output_channels, input_channels, kernel_size, kernel_size)
        ))
        self.biases = cp.zeros((output_channels, 1))
        
        # Initialize CUDA streams
        self.stream = cp.cuda.Stream()
        
        # Set up kernel parameters
        self.block_size = (16, 16, 1)
        self.grid_size = lambda x, y, z: ((x + 15) // 16, (y + 15) // 16, z)

    def forward(self, inputs: cp.ndarray, training: bool) -> None:
        if not isinstance(inputs, cp.ndarray):
            inputs = cp.asarray(inputs)
            
        batch_size = inputs.shape[0]
        input_height = inputs.shape[2]
        input_width = inputs.shape[3]
        
        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        self.inputs = inputs
        self.output = cp.empty((batch_size, self.output_channels, output_height, output_width), dtype=cp.float32)
        
        with self.stream:
            # Launch forward kernel
            grid_size = self.grid_size(output_height, output_width, batch_size)
            conv2d_forward_kernel[grid_size, self.block_size](
                inputs.ravel(),
                self.weights.ravel(),
                self.output.ravel(),
                batch_size,
                self.input_channels,
                input_height,
                input_width,
                self.output_channels,
                self.kernel_size,
                self.stride,
                self.padding
            )
            
            # Add biases
            self.output += self.biases.reshape(1, -1, 1, 1)

    def backward(self, dvalues: cp.ndarray) -> None:
        if not isinstance(dvalues, cp.ndarray):
            dvalues = cp.asarray(dvalues)
            
        batch_size = dvalues.shape[0]
        input_height = self.inputs.shape[2]
        input_width = self.inputs.shape[3]
        
        self.dweights = cp.zeros_like(self.weights)
        self.dbiases = cp.sum(dvalues, axis=(0, 2, 3), keepdims=True)
        self.dinputs = cp.zeros_like(self.inputs)
        
        with self.stream:
            # Launch backward kernel
            grid_size = (self.output_channels, self.input_channels, self.kernel_size)
            block_size = (self.kernel_size, self.kernel_size, 1)
            conv2d_backward_kernel[grid_size, block_size](
                self.inputs.ravel(),
                dvalues.ravel(),
                self.dweights.ravel(),
                self.dinputs.ravel(),
                batch_size,
                self.input_channels,
                input_height,
                input_width,
                self.output_channels,
                self.kernel_size,
                self.stride,
                self.padding
            )

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