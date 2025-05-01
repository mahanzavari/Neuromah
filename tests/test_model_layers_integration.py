import unittest
import numpy as np
from src.api.model import Model
from src.api.data import ArrayDataset, DataLoader
from src.layers import (
    Dense, ReLU, LeakyReLU, ELU, Softmax, Dropout,
    BatchNormalization, LayerNormalization, Layer
)
from src.optimizers import Adam
from src.losses import CategoricalCrossEntropy
from src.metrics import Accuracy

class TestModelLayersIntegration(unittest.TestCase):
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
        
    def test_dense_layer_integration(self):
        """Test Dense layer integration with model."""
        # Create model with Dense layers
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(Dense(32))
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Train model
        history = model.fit(self.dataloader, epochs=1)
        
        # Verify training completed
        self.assertIn('loss', history)
        self.assertGreater(len(history['loss']), 0)
        
    def test_activation_layers_integration(self):
        """Test activation layers integration with model."""
        # Create model with different activation layers
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(16))
        model.add(ELU(alpha=1.0))
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Train model
        history = model.fit(self.dataloader, epochs=1)
        
        # Verify training completed
        self.assertIn('loss', history)
        self.assertGreater(len(history['loss']), 0)
        
    def test_dropout_layer_integration(self):
        """Test Dropout layer integration with model."""
        # Create model with Dropout layers
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dropout(rate=0.5))
        model.add(Dense(32))
        model.add(ReLU())
        model.add(Dropout(rate=0.3))
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Train model
        history = model.fit(self.dataloader, epochs=1)
        
        # Verify training completed
        self.assertIn('loss', history)
        self.assertGreater(len(history['loss']), 0)
        
    def test_normalization_layers_integration(self):
        """Test normalization layers integration with model."""
        # Create model with normalization layers
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dense(32))
        model.add(LayerNormalization())
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Train model
        history = model.fit(self.dataloader, epochs=1)
        
        # Verify training completed
        self.assertIn('loss', history)
        self.assertGreater(len(history['loss']), 0)
        
    def test_custom_layer_integration(self):
        """Test custom layer integration with model."""
        class CustomLayer(Layer):
            def __init__(self, units):
                super().__init__()
                self.units = units
                
            def build(self, input_shape):
                self.weights = np.random.randn(input_shape[-1], self.units)
                self.biases = np.random.randn(self.units)
                
            def forward(self, inputs):
                return np.dot(inputs, self.weights) + self.biases
                
            def backward(self, dvalues):
                self.dweights = np.dot(self.inputs.T, dvalues)
                self.dbiases = np.sum(dvalues, axis=0)
                return np.dot(dvalues, self.weights.T)
        
        # Create model with custom layer
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(CustomLayer(32))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Train model
        history = model.fit(self.dataloader, epochs=1)
        
        # Verify training completed
        self.assertIn('loss', history)
        self.assertGreater(len(history['loss']), 0)
        
    def test_complex_architecture_integration(self):
        """Test complex layer architecture integration with model."""
        # Create complex model architecture
        model = Model()
        model.add(Dense(128, input_shape=(self.n_features,)))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(rate=0.3))
        model.add(Dense(64))
        model.add(LayerNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(rate=0.2))
        model.add(Dense(32))
        model.add(ELU(alpha=1.0))
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Train model
        history = model.fit(self.dataloader, epochs=1)
        
        # Verify training completed
        self.assertIn('loss', history)
        self.assertGreater(len(history['loss']), 0)
        
    def test_layer_parameter_updates(self):
        """Test layer parameter updates during training."""
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
        
        # Store initial weights
        initial_weights = [layer.weights.copy() for layer in model.layers if hasattr(layer, 'weights')]
        
        # Train model
        model.fit(self.dataloader, epochs=1)
        
        # Verify weights have been updated
        for layer, initial_weight in zip(
            [layer for layer in model.layers if hasattr(layer, 'weights')],
            initial_weights
        ):
            self.assertTrue(np.any(layer.weights != initial_weight))
            
    def test_layer_forward_backward_pass(self):
        """Test layer forward and backward pass integration."""
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
        
        # Get a batch of data
        batch_X, batch_y = next(iter(self.dataloader))
        
        # Perform forward pass
        output = model.forward(batch_X)
        
        # Verify output shape
        self.assertEqual(output.shape, (batch_X.shape[0], self.n_classes))
        
        # Perform backward pass
        dvalues = model.backward(batch_y)
        
        # Verify gradients
        for layer in model.layers:
            if hasattr(layer, 'dweights'):
                self.assertIsNotNone(layer.dweights)
                self.assertIsNotNone(layer.dbiases)

if __name__ == '__main__':
    unittest.main() 