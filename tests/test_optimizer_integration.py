import unittest
import numpy as np
import cupy as cp
from src.optimizers.base import BaseOptimizer
from src.optimizers.cuda_kernels import (
    sgd_update_kernel,
    momentum_update_kernel,
    rmsprop_update_kernel,
    adam_update_kernel
)

class TestLayer:
    """Mock layer for testing optimizers."""
    def __init__(self, shape):
        self.weights = np.random.randn(*shape)
        self.biases = np.random.randn(shape[0])
        self.dweights = np.random.randn(*shape)
        self.dbiases = np.random.randn(shape[0])
        self.weight_regularizer_l1 = 0.01
        self.weight_regularizer_l2 = 0.01

class TestOptimizerIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.shape = (64, 32)  # Typical layer shape
        self.layer = TestLayer(self.shape)
        self.learning_rate = 0.001
        
    def test_cuda_kernel_initialization(self):
        """Test CUDA kernel initialization and basic functionality."""
        # Test SGD kernel
        weights = cp.asarray(self.layer.weights)
        biases = cp.asarray(self.layer.biases)
        dweights = cp.asarray(self.layer.dweights)
        dbiases = cp.asarray(self.layer.dbiases)
        
        # Launch kernel
        grid_size = ((weights.shape[0] + 31) // 32, (weights.shape[1] + 31) // 32)
        block_size = (32, 32)
        sgd_update_kernel[grid_size, block_size](
            weights, biases, dweights, dbiases, self.learning_rate
        )
        
        # Verify results
        self.assertTrue(cp.all(weights != self.layer.weights))
        self.assertTrue(cp.all(biases != self.layer.biases))
        
    def test_optimizer_device_management(self):
        """Test device management in base optimizer."""
        # Test CPU mode
        optimizer = BaseOptimizer(
            learning_rate=self.learning_rate,
            device='cpu',
            use_cuda=False
        )
        self.assertEqual(optimizer.device, 'cpu')
        self.assertFalse(optimizer.use_cuda)
        
        # Test CUDA mode
        if cp.cuda.is_available():
            optimizer = BaseOptimizer(
                learning_rate=self.learning_rate,
                device='cuda',
                use_cuda=True
            )
            self.assertEqual(optimizer.device, 'cuda')
            self.assertTrue(optimizer.use_cuda)
            
    def test_parallel_update(self):
        """Test parallel parameter updates."""
        optimizer = BaseOptimizer(
            learning_rate=self.learning_rate,
            device='cpu',
            use_cuda=False,
            num_threads=4
        )
        
        # Store original values
        original_weights = self.layer.weights.copy()
        original_biases = self.layer.biases.copy()
        
        # Perform update
        optimizer._parallel_update(self.layer, lambda w, dw, lr: w - lr * dw)
        
        # Verify updates
        self.assertTrue(np.any(self.layer.weights != original_weights))
        self.assertTrue(np.any(self.layer.biases != original_biases))
        
    def test_learning_rate_decay(self):
        """Test learning rate decay functionality."""
        optimizer = BaseOptimizer(
            learning_rate=self.learning_rate,
            decay=0.1,
            device='cpu'
        )
        
        # Initial learning rate
        self.assertEqual(optimizer.current_learning_rate, self.learning_rate)
        
        # After one iteration
        optimizer.post_update_params()
        optimizer.pre_update_params()
        expected_lr = self.learning_rate * (1.0 / (1.0 + 0.1 * 1))
        self.assertAlmostEqual(optimizer.current_learning_rate, expected_lr)
        
    def test_cuda_memory_management(self):
        """Test CUDA memory management."""
        if not cp.cuda.is_available():
            self.skipTest("CUDA not available")
            
        optimizer = BaseOptimizer(
            learning_rate=self.learning_rate,
            device='cuda',
            use_cuda=True
        )
        
        # Test memory allocation and deallocation
        weights = optimizer._to_device(self.layer.weights)
        self.assertIsInstance(weights, cp.ndarray)
        
        # Test memory cleanup
        del weights
        cp.cuda.Stream.null.synchronize()
        
    def test_multi_threading(self):
        """Test multi-threading functionality."""
        optimizer = BaseOptimizer(
            learning_rate=self.learning_rate,
            device='cpu',
            use_cuda=False,
            num_threads=4
        )
        
        # Verify thread pool initialization
        self.assertEqual(optimizer.thread_pool._max_workers, 4)
        
        # Test parallel execution
        def square(x):
            return x * x
            
        futures = []
        for _ in range(10):
            futures.append(optimizer.thread_pool.submit(square, 2))
            
        results = [f.result() for f in futures]
        self.assertEqual(results, [4] * 10)
        
    def test_resource_cleanup(self):
        """Test resource cleanup."""
        optimizer = BaseOptimizer(
            learning_rate=self.learning_rate,
            device='cpu'
        )
        
        # Verify thread pool is active
        self.assertFalse(optimizer.thread_pool._shutdown)
        
        # Clean up resources
        optimizer.__del__()
        
        # Verify thread pool is shut down
        self.assertTrue(optimizer.thread_pool._shutdown)
        
    def test_integration_with_all_kernels(self):
        """Test integration with all CUDA kernels."""
        if not cp.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Test SGD
        self._test_kernel_integration(sgd_update_kernel, [self.learning_rate])
        
        # Test Momentum
        momentum = 0.9
        momentum_weights = cp.zeros_like(self.layer.weights)
        momentum_biases = cp.zeros_like(self.layer.biases)
        self._test_kernel_integration(
            momentum_update_kernel,
            [momentum_weights, momentum_biases, self.learning_rate, momentum]
        )
        
        # Test RMSprop
        decay_rate = 0.99
        cache_weights = cp.zeros_like(self.layer.weights)
        cache_biases = cp.zeros_like(self.layer.biases)
        self._test_kernel_integration(
            rmsprop_update_kernel,
            [cache_weights, cache_biases, self.learning_rate, decay_rate, optimizer.epsilon]
        )
        
        # Test Adam
        beta1, beta2 = 0.9, 0.999
        m_weights = cp.zeros_like(self.layer.weights)
        m_biases = cp.zeros_like(self.layer.biases)
        v_weights = cp.zeros_like(self.layer.weights)
        v_biases = cp.zeros_like(self.layer.biases)
        self._test_kernel_integration(
            adam_update_kernel,
            [m_weights, m_biases, v_weights, v_biases, self.learning_rate,
             beta1, beta2, optimizer.epsilon, 1]
        )
        
    def _test_kernel_integration(self, kernel, additional_args):
        """Helper method to test kernel integration."""
        weights = cp.asarray(self.layer.weights)
        biases = cp.asarray(self.layer.biases)
        dweights = cp.asarray(self.layer.dweights)
        dbiases = cp.asarray(self.layer.dbiases)
        
        # Store original values
        original_weights = weights.copy()
        original_biases = biases.copy()
        
        # Launch kernel
        grid_size = ((weights.shape[0] + 31) // 32, (weights.shape[1] + 31) // 32)
        block_size = (32, 32)
        kernel[grid_size, block_size](
            weights, biases, dweights, dbiases, *additional_args
        )
        
        # Verify updates
        self.assertTrue(cp.any(weights != original_weights))
        self.assertTrue(cp.any(biases != original_biases))

if __name__ == '__main__':
    unittest.main() 