import unittest
import numpy as np
import cupy as cp
from src.api.model import Model
from src.api.data import ArrayDataset, DataLoader
from src.layers import Dense, ReLU, Softmax
from src.optimizers import SGD, Adam
from src.losses import CategoricalCrossEntropy
from src.metrics import Accuracy

class TestModelOptimizerIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        # Generate synthetic data
        self.n_samples = 1000
        self.n_features = 20
        self.n_classes = 3
        
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randint(0, self.n_classes, size=self.n_samples)
        
        # Create dataset and dataloader
        self.dataset = ArrayDataset(self.X, self.y)
        self.dataloader = DataLoader(self.dataset, batch_size=32)
        
    def test_model_optimizer_initialization(self):
        """Test model initialization with different optimizers."""
        # Test with SGD
        model_sgd = Model()
        model_sgd.add(Dense(64, input_shape=(self.n_features,)))
        model_sgd.add(ReLU())
        model_sgd.add(Dense(self.n_classes))
        model_sgd.add(Softmax())
        
        model_sgd.compile(
            optimizer=SGD(learning_rate=0.01),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Test with Adam
        model_adam = Model()
        model_adam.add(Dense(64, input_shape=(self.n_features,)))
        model_adam.add(ReLU())
        model_adam.add(Dense(self.n_classes))
        model_adam.add(Softmax())
        
        model_adam.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Verify optimizer types
        self.assertIsInstance(model_sgd.optimizer, SGD)
        self.assertIsInstance(model_adam.optimizer, Adam)
        
    def test_training_with_optimizer(self):
        """Test model training with optimizer."""
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
        
        # Store initial weights
        initial_weights = [layer.weights.copy() for layer in model.layers if hasattr(layer, 'weights')]
        
        # Train for one epoch
        model.fit(self.dataloader, epochs=1)
        
        # Verify weights have been updated
        for layer, initial_weight in zip(
            [layer for layer in model.layers if hasattr(layer, 'weights')],
            initial_weights
        ):
            self.assertTrue(np.any(layer.weights != initial_weight))
            
    def test_optimizer_state_persistence(self):
        """Test optimizer state persistence during training."""
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
        
        # Train for one epoch
        model.fit(self.dataloader, epochs=1)
        
        # Store optimizer state
        optimizer_state = model.optimizer.get_state()
        
        # Train for another epoch
        model.fit(self.dataloader, epochs=1)
        
        # Verify optimizer state has changed
        new_optimizer_state = model.optimizer.get_state()
        self.assertNotEqual(optimizer_state, new_optimizer_state)
        
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling during training."""
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        initial_lr = 0.001
        model.compile(
            optimizer=Adam(learning_rate=initial_lr),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Store initial learning rate
        initial_learning_rate = model.optimizer.learning_rate
        
        # Train for one epoch
        model.fit(self.dataloader, epochs=1)
        
        # Verify learning rate has been updated
        self.assertNotEqual(model.optimizer.learning_rate, initial_learning_rate)
        
    def test_cuda_optimizer_integration(self):
        """Test CUDA optimizer integration."""
        if not cp.cuda.is_available():
            self.skipTest("CUDA not available")
            
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
        
        # Train for one epoch
        model.fit(self.dataloader, epochs=1)
        
        # Verify CUDA usage
        self.assertTrue(model.optimizer.use_cuda)
        self.assertEqual(model.optimizer.device, 'cuda')
        
    def test_parallel_optimization(self):
        """Test parallel optimization."""
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        model.compile(
            optimizer=Adam(learning_rate=0.001, num_threads=4),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Train for one epoch
        model.fit(self.dataloader, epochs=1)
        
        # Verify parallel execution
        self.assertEqual(model.optimizer.num_threads, 4)
        
    def test_optimizer_metrics(self):
        """Test optimizer metrics during training."""
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
        
        # Train for one epoch
        history = model.fit(self.dataloader, epochs=1)
        
        # Verify metrics are recorded
        self.assertIn('loss', history)
        self.assertIn('accuracy', history)
        self.assertGreater(len(history['loss']), 0)
        self.assertGreater(len(history['accuracy']), 0)

if __name__ == '__main__':
    unittest.main() 