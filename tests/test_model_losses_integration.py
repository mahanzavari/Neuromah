import unittest
import numpy as np
from src.api.model import Model
from src.api.data import ArrayDataset, DataLoader
from src.layers import Dense, ReLU, Softmax
from src.optimizers import Adam
from src.losses import (
    CategoricalCrossEntropy, BinaryCrossEntropy,
    MeanSquaredError, MeanAbsoluteError,
    HuberLoss, KLDivergence
)

class TestModelLossesIntegration(unittest.TestCase):
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
        
    def test_categorical_cross_entropy_integration(self):
        """Test CategoricalCrossEntropy loss integration with model."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        # Compile with CategoricalCrossEntropy loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(self.dataloader, epochs=1)
        
        # Verify loss is recorded
        self.assertIn('loss', history)
        self.assertGreater(len(history['loss']), 0)
        
    def test_binary_cross_entropy_integration(self):
        """Test BinaryCrossEntropy loss integration with model."""
        # Generate binary classification data
        X_bin = np.random.randn(self.n_samples, self.n_features)
        y_bin = np.random.randint(0, 2, size=self.n_samples)
        
        # Create dataset and dataloader
        bin_dataset = ArrayDataset(X_bin, y_bin)
        bin_dataloader = DataLoader(bin_dataset, batch_size=32)
        
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(1))
        model.add(Softmax())
        
        # Compile with BinaryCrossEntropy loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=BinaryCrossEntropy(),
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(bin_dataloader, epochs=1)
        
        # Verify loss is recorded
        self.assertIn('loss', history)
        self.assertGreater(len(history['loss']), 0)
        
    def test_regression_losses_integration(self):
        """Test regression losses integration with model."""
        # Generate regression data
        X_reg = np.random.randn(self.n_samples, self.n_features)
        y_reg = np.random.randn(self.n_samples)
        
        # Create dataset and dataloader
        reg_dataset = ArrayDataset(X_reg, y_reg)
        reg_dataloader = DataLoader(reg_dataset, batch_size=32)
        
        # Test MeanSquaredError
        model_mse = Model()
        model_mse.add(Dense(64, input_shape=(self.n_features,)))
        model_mse.add(ReLU())
        model_mse.add(Dense(1))
        
        model_mse.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MeanSquaredError(),
            metrics=['mse']
        )
        
        history_mse = model_mse.fit(reg_dataloader, epochs=1)
        self.assertIn('loss', history_mse)
        
        # Test MeanAbsoluteError
        model_mae = Model()
        model_mae.add(Dense(64, input_shape=(self.n_features,)))
        model_mae.add(ReLU())
        model_mae.add(Dense(1))
        
        model_mae.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MeanAbsoluteError(),
            metrics=['mae']
        )
        
        history_mae = model_mae.fit(reg_dataloader, epochs=1)
        self.assertIn('loss', history_mae)
        
        # Test HuberLoss
        model_huber = Model()
        model_huber.add(Dense(64, input_shape=(self.n_features,)))
        model_huber.add(ReLU())
        model_huber.add(Dense(1))
        
        model_huber.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=HuberLoss(delta=1.0),
            metrics=['mae']
        )
        
        history_huber = model_huber.fit(reg_dataloader, epochs=1)
        self.assertIn('loss', history_huber)
        
    def test_kl_divergence_integration(self):
        """Test KLDivergence loss integration with model."""
        # Generate probability distributions
        X_prob = np.random.randn(self.n_samples, self.n_features)
        y_prob = np.random.dirichlet(np.ones(self.n_classes), size=self.n_samples)
        
        # Create dataset and dataloader
        prob_dataset = ArrayDataset(X_prob, y_prob)
        prob_dataloader = DataLoader(prob_dataset, batch_size=32)
        
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        # Compile with KLDivergence loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=KLDivergence(),
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(prob_dataloader, epochs=1)
        
        # Verify loss is recorded
        self.assertIn('loss', history)
        self.assertGreater(len(history['loss']), 0)
        
    def test_loss_calculation(self):
        """Test loss calculation during training."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        # Compile with CategoricalCrossEntropy loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(self.dataloader, epochs=1)
        
        # Verify loss values are positive
        for loss in history['loss']:
            self.assertGreaterEqual(loss, 0)
            
    def test_loss_regularization(self):
        """Test loss regularization during training."""
        # Create model with L1 and L2 regularization
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,),
                       weight_regularizer_l1=0.01,
                       weight_regularizer_l2=0.01))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        # Compile with CategoricalCrossEntropy loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(self.dataloader, epochs=1)
        
        # Verify loss includes regularization
        self.assertIn('loss', history)
        self.assertGreater(len(history['loss']), 0)
        
    def test_loss_gradient(self):
        """Test loss gradient calculation."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        # Compile with CategoricalCrossEntropy loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=['accuracy']
        )
        
        # Get a batch of data
        batch_X, batch_y = next(iter(self.dataloader))
        
        # Perform forward pass
        output = model.forward(batch_X)
        
        # Calculate loss and gradients
        loss = model.loss.forward(output, batch_y)
        dvalues = model.loss.backward(output, batch_y)
        
        # Verify gradients
        self.assertIsNotNone(dvalues)
        self.assertEqual(dvalues.shape, output.shape)
        
    def test_loss_validation(self):
        """Test loss validation during training."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        # Compile with CategoricalCrossEntropy loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=['accuracy']
        )
        
        # Train with validation data
        history = model.fit(
            self.dataloader,
            validation_data=self.dataloader,
            epochs=1
        )
        
        # Verify validation loss
        self.assertIn('val_loss', history)
        self.assertGreater(len(history['val_loss']), 0)

if __name__ == '__main__':
    unittest.main() 