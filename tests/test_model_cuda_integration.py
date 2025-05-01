import unittest
import numpy as np
import cupy as cp
from src.api.model import Model
from src.api.data import ArrayDataset, DataLoader
from src.layers import Dense, ReLU, Softmax
from src.optimizers import Adam
from src.losses import CategoricalCrossEntropy
from src.metrics import Accuracy

class TestModelCudaIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        if not cp.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Generate synthetic data
        self.n_samples = 1000
        self.n_features = 20
        self.n_classes = 3
        
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randint(0, self.n_classes, size=self.n_samples)
        
        # Create dataset and dataloader
        self.dataset = ArrayDataset(self.X, self.y)
        self.dataloader = DataLoader(self.dataset, batch_size=32)
        
    def test_model_to_cuda(self):
        """Test moving model to CUDA."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Move model to CUDA
        model.to('cuda')
        
        # Verify model is on CUDA
        for layer in model.layers:
            if hasattr(layer, 'weights'):
                self.assertTrue(isinstance(layer.weights, cp.ndarray))
                self.assertTrue(isinstance(layer.biases, cp.ndarray))
                
    def test_cuda_forward_pass(self):
        """Test forward pass on CUDA."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Move model to CUDA
        model.to('cuda')
        
        # Get a batch of data
        batch_X, batch_y = next(iter(self.dataloader))
        
        # Move data to CUDA
        batch_X = cp.asarray(batch_X)
        batch_y = cp.asarray(batch_y)
        
        # Perform forward pass
        output = model.forward(batch_X)
        
        # Verify output is on CUDA
        self.assertTrue(isinstance(output, cp.ndarray))
        
    def test_cuda_backward_pass(self):
        """Test backward pass on CUDA."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Move model to CUDA
        model.to('cuda')
        
        # Get a batch of data
        batch_X, batch_y = next(iter(self.dataloader))
        
        # Move data to CUDA
        batch_X = cp.asarray(batch_X)
        batch_y = cp.asarray(batch_y)
        
        # Perform forward and backward pass
        output = model.forward(batch_X)
        dvalues = model.backward(batch_y)
        
        # Verify gradients are on CUDA
        for layer in model.layers:
            if hasattr(layer, 'dweights'):
                self.assertTrue(isinstance(layer.dweights, cp.ndarray))
                self.assertTrue(isinstance(layer.dbiases, cp.ndarray))
                
    def test_cuda_training(self):
        """Test training on CUDA."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        model.compile(
            optimizer=Adam(learning_rate=0.001, device='cuda', use_cuda=True),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Move model to CUDA
        model.to('cuda')
        
        # Train model
        history = model.fit(self.dataloader, epochs=1)
        
        # Verify training completed
        self.assertIn('loss', history)
        self.assertGreater(len(history['loss']), 0)
        
    def test_cuda_optimizer(self):
        """Test CUDA optimizer integration."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        # Create CUDA optimizer
        optimizer = Adam(learning_rate=0.001, device='cuda', use_cuda=True)
        
        model.compile(
            optimizer=optimizer,
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Move model to CUDA
        model.to('cuda')
        
        # Get a batch of data
        batch_X, batch_y = next(iter(self.dataloader))
        
        # Move data to CUDA
        batch_X = cp.asarray(batch_X)
        batch_y = cp.asarray(batch_y)
        
        # Perform forward and backward pass
        output = model.forward(batch_X)
        dvalues = model.backward(batch_y)
        
        # Update parameters
        model.optimizer.update_params(model.layers)
        
        # Verify parameters are on CUDA
        for layer in model.layers:
            if hasattr(layer, 'weights'):
                self.assertTrue(isinstance(layer.weights, cp.ndarray))
                self.assertTrue(isinstance(layer.biases, cp.ndarray))
                
    def test_cuda_memory_management(self):
        """Test CUDA memory management."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        model.compile(
            optimizer=Adam(learning_rate=0.001, device='cuda', use_cuda=True),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Move model to CUDA
        model.to('cuda')
        
        # Get initial memory usage
        initial_memory = cp.cuda.Device().mem_info()[0]
        
        # Train model
        model.fit(self.dataloader, epochs=1)
        
        # Get final memory usage
        final_memory = cp.cuda.Device().mem_info()[0]
        
        # Verify memory is freed
        self.assertLess(final_memory, initial_memory)
        
    def test_cuda_performance(self):
        """Test CUDA performance."""
        # Create CPU model
        cpu_model = Model()
        cpu_model.add(Dense(64, input_shape=(self.n_features,)))
        cpu_model.add(ReLU())
        cpu_model.add(Dense(self.n_classes))
        cpu_model.add(Softmax())
        
        cpu_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Create CUDA model
        cuda_model = Model()
        cuda_model.add(Dense(64, input_shape=(self.n_features,)))
        cuda_model.add(ReLU())
        cuda_model.add(Dense(self.n_classes))
        cuda_model.add(Softmax())
        
        cuda_model.compile(
            optimizer=Adam(learning_rate=0.001, device='cuda', use_cuda=True),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Move CUDA model to CUDA
        cuda_model.to('cuda')
        
        # Get a batch of data
        batch_X, batch_y = next(iter(self.dataloader))
        
        # Time CPU forward pass
        import time
        start_time = time.time()
        cpu_model.forward(batch_X)
        cpu_time = time.time() - start_time
        
        # Time CUDA forward pass
        batch_X_cuda = cp.asarray(batch_X)
        start_time = time.time()
        cuda_model.forward(batch_X_cuda)
        cuda_time = time.time() - start_time
        
        # Verify CUDA is faster
        self.assertLess(cuda_time, cpu_time)
        
    def test_cuda_mixed_precision(self):
        """Test mixed precision training on CUDA."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        model.compile(
            optimizer=Adam(learning_rate=0.001, device='cuda', use_cuda=True),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Move model to CUDA with mixed precision
        model.to('cuda', dtype=cp.float16)
        
        # Get a batch of data
        batch_X, batch_y = next(iter(self.dataloader))
        
        # Move data to CUDA with mixed precision
        batch_X = cp.asarray(batch_X, dtype=cp.float16)
        batch_y = cp.asarray(batch_y, dtype=cp.float16)
        
        # Perform forward pass
        output = model.forward(batch_X)
        
        # Verify mixed precision
        self.assertEqual(output.dtype, cp.float16)
        for layer in model.layers:
            if hasattr(layer, 'weights'):
                self.assertEqual(layer.weights.dtype, cp.float16)
                self.assertEqual(layer.biases.dtype, cp.float16)

if __name__ == '__main__':
    unittest.main() 