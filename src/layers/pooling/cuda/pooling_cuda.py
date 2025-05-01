import numpy as np
import cupy as cp
from typing import Literal, Optional

class PoolingCUDA:
    def __init__(
        self,
        pool_size: int = 2,
        stride: int = 2,
        padding: int = 0,
        mode: Literal['max', 'avg'] = 'max'
    ):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.mode = mode
        
        # Initialize CUDA streams
        self.stream = cp.cuda.Stream()
        
        # Set up kernel parameters
        self.block_size = (16, 16, 1)
        self.grid_size = lambda x, y, z: ((x + 15) // 16, (y + 15) // 16, z)

    def forward(self, inputs: cp.ndarray, training: bool) -> None:
        if not isinstance(inputs, cp.ndarray):
            inputs = cp.asarray(inputs)
            
        batch_size = inputs.shape[0]
        channels = inputs.shape[1]
        input_height = inputs.shape[2]
        input_width = inputs.shape[3]
        
        output_height = (input_height + 2 * self.padding - self.pool_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.pool_size) // self.stride + 1
        
        self.inputs = inputs
        self.output = cp.empty((batch_size, channels, output_height, output_width), dtype=cp.float32)
        
        with self.stream:
            if self.mode == 'max':
                # For max pooling, we need to store indices for backward pass
                self.indices = cp.empty_like(self.output, dtype=cp.int32)
                
                # Launch max pooling forward kernel
                grid_size = self.grid_size(output_height, output_width, batch_size)
                max_pool2d_forward_kernel[grid_size, self.block_size](
                    inputs.ravel(),
                    self.output.ravel(),
                    self.indices.ravel(),
                    batch_size,
                    channels,
                    input_height,
                    input_width,
                    self.pool_size,
                    self.stride
                )
            else:  # avg pooling
                # Launch average pooling forward kernel
                grid_size = self.grid_size(output_height, output_width, batch_size)
                avg_pool2d_forward_kernel[grid_size, self.block_size](
                    inputs.ravel(),
                    self.output.ravel(),
                    batch_size,
                    channels,
                    input_height,
                    input_width,
                    self.pool_size,
                    self.stride
                )

    def backward(self, dvalues: cp.ndarray) -> None:
        if not isinstance(dvalues, cp.ndarray):
            dvalues = cp.asarray(dvalues)
            
        batch_size = dvalues.shape[0]
        channels = dvalues.shape[1]
        input_height = self.inputs.shape[2]
        input_width = self.inputs.shape[3]
        output_height = dvalues.shape[2]
        output_width = dvalues.shape[3]
        
        self.dinputs = cp.zeros_like(self.inputs)
        
        with self.stream:
            if self.mode == 'max':
                # Launch max pooling backward kernel
                grid_size = self.grid_size(output_height, output_width, batch_size)
                max_pool2d_backward_kernel[grid_size, self.block_size](
                    dvalues.ravel(),
                    self.indices.ravel(),
                    self.dinputs.ravel(),
                    batch_size,
                    channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width
                )
            else:  # avg pooling
                # Launch average pooling backward kernel
                grid_size = self.grid_size(output_height, output_width, batch_size)
                avg_pool2d_backward_kernel[grid_size, self.block_size](
                    dvalues.ravel(),
                    self.dinputs.ravel(),
                    batch_size,
                    channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    self.pool_size,
                    self.stride
                ) 