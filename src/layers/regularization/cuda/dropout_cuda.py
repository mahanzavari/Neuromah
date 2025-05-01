import numpy as np
import cupy as cp
from typing import Optional
import random

class DropoutCUDA:
    def __init__(self, rate: float = 0.1):
        self.rate = rate
        self.scale = 1.0 / (1.0 - rate)
        
        # Initialize CUDA streams
        self.stream = cp.cuda.Stream()
        
        # Set up kernel parameters
        self.block_size = 256
        self.grid_size = lambda x: (x + self.block_size - 1) // self.block_size
        
        # Initialize random states
        self.states = None
        self.mask = None

    def forward(self, inputs: cp.ndarray, training: bool) -> None:
        if not isinstance(inputs, cp.ndarray):
            inputs = cp.asarray(inputs)
            
        self.inputs = inputs
        
        if not training:
            self.output = inputs.copy()
            return
            
        size = inputs.size
        
        # Initialize random states if not already done
        if self.states is None or self.states.size < size:
            self.states = cp.empty(size, dtype=cp.uint64)
            self.mask = cp.empty(size, dtype=cp.int32)
            
            # Initialize random states
            seed = random.getrandbits(64)
            grid_size = self.grid_size(size)
            init_curand_states_kernel[grid_size, self.block_size](
                self.states,
                seed,
                size
            )
        
        # Allocate output
        self.output = cp.empty_like(inputs)
        
        with self.stream:
            # Launch dropout forward kernel
            grid_size = self.grid_size(size)
            dropout_forward_kernel[grid_size, self.block_size](
                inputs.ravel(),
                self.output.ravel(),
                self.mask.ravel(),
                size,
                self.rate,
                self.states
            )

    def backward(self, dvalues: cp.ndarray) -> None:
        if not isinstance(dvalues, cp.ndarray):
            dvalues = cp.asarray(dvalues)
            
        size = dvalues.size
        self.dinputs = cp.empty_like(dvalues)
        
        with self.stream:
            # Launch dropout backward kernel
            grid_size = self.grid_size(size)
            dropout_backward_kernel[grid_size, self.block_size](
                dvalues.ravel(),
                self.mask.ravel(),
                self.dinputs.ravel(),
                size,
                self.rate
            ) 