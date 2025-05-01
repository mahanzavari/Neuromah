from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import cupy as cp
from abc import ABC, abstractmethod
from numba import cuda
import threading
from concurrent.futures import ThreadPoolExecutor
from .cuda_kernels import (
    sgd_update_kernel,
    momentum_update_kernel,
    rmsprop_update_kernel,
    adam_update_kernel
)

class BaseOptimizer(ABC):
    """Base class for all optimizers with CUDA support and performance optimizations."""
    
    def __init__(
        self,
        learning_rate: float = 1.0,
        decay: float = 0.0,
        epsilon: float = 1e-7,
        device: str = 'cpu',
        use_cuda: bool = False,
        num_threads: Optional[int] = None
    ):
        """
        Initialize the optimizer.
        
        Args:
            learning_rate: Learning rate
            decay: Learning rate decay
            epsilon: Small value to prevent division by zero
            device: 'cpu' or 'cuda'
            use_cuda: Whether to use CUDA kernels
            num_threads: Number of threads for CPU parallelization
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
        self.device = device
        self.use_cuda = use_cuda and device == 'cuda'
        self.num_threads = num_threads or threading.active_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)
        
        if self.use_cuda:
            self._init_cuda()
            
    def _init_cuda(self) -> None:
        """Initialize CUDA-specific attributes."""
        self.stream = cp.cuda.Stream()
        self.block_size = (32, 32)  # Optimal for most modern GPUs
        
    def _get_grid_size(self, shape: Tuple[int, ...]) -> Tuple[int, int]:
        """Calculate grid size for CUDA kernels."""
        if len(shape) == 1:
            return (shape[0] + self.block_size[0] - 1) // self.block_size[0], 1
        return (
            (shape[0] + self.block_size[0] - 1) // self.block_size[0],
            (shape[1] + self.block_size[1] - 1) // self.block_size[1]
        )
        
    def _to_device(self, array: np.ndarray) -> Union[np.ndarray, cp.ndarray]:
        """Move array to the appropriate device."""
        if self.use_cuda:
            return cp.asarray(array)
        return array
        
    def _to_host(self, array: Union[np.ndarray, cp.ndarray]) -> np.ndarray:
        """Move array back to host if needed."""
        if self.use_cuda:
            return cp.asnumpy(array)
        return array
        
    def pre_update_params(self) -> None:
        """Update learning rate if decay is used."""
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )
            
    def post_update_params(self) -> None:
        """Increment iteration counter."""
        self.iterations += 1
        
    def _parallel_update(self, layer, update_func) -> None:
        """Update parameters in parallel."""
        if self.use_cuda:
            # Use CUDA kernel
            self._cuda_update(layer, update_func)
        else:
            # Use CPU parallelization
            futures = []
            if layer.weights is not None:
                futures.append(
                    self.thread_pool.submit(
                        update_func,
                        layer.weights,
                        layer.dweights,
                        self.current_learning_rate
                    )
                )
            if layer.biases is not None:
                futures.append(
                    self.thread_pool.submit(
                        update_func,
                        layer.biases,
                        layer.dbiases,
                        self.current_learning_rate
                    )
                )
            for future in futures:
                future.result()
                
    def _cuda_update(self, layer, update_func) -> None:
        """Update parameters using CUDA kernel."""
        if layer.weights is not None:
            grid_size = self._get_grid_size(layer.weights.shape)
            update_func[grid_size, self.block_size](
                layer.weights,
                layer.biases,
                layer.dweights,
                layer.dbiases,
                self.current_learning_rate
            )
            
    @abstractmethod
    def update_params(self, layer) -> None:
        """
        Update layer parameters.
        
        Args:
            layer: Layer to update
        """
        pass
        
    def __del__(self):
        """Clean up resources."""
        self.thread_pool.shutdown() 