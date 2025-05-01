import unittest
import numpy as np
from src.api.model import Model
from src.api.data import ArrayDataset, DataLoader
from src.layers import Dense, ReLU, Softmax
from src.optimizers import Adam
from src.losses import CategoricalCrossEntropy
from src.metrics import (
    Accuracy, Precision, Recall, F1Score,
    MeanSquaredError, MeanAbsoluteError,
    R2Score, ConfusionMatrix
)

class TestModelMetricsIntegration(unittest.TestCase):
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
        
    def test_classification_metrics_integration(self):
        """Test classification metrics integration with model."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        # Compile with multiple classification metrics
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[
                Accuracy(),
                Precision(),
                Recall(),
                F1Score(),
                ConfusionMatrix()
            ]
        )
        
        # Train model
        history = model.fit(self.dataloader, epochs=1)
        
        # Verify metrics are recorded
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            self.assertIn(metric, history)
            self.assertGreater(len(history[metric]), 0)
            
    def test_regression_metrics_integration(self):
        """Test regression metrics integration with model."""
        # Generate regression data
        X_reg = np.random.randn(self.n_samples, self.n_features)
        y_reg = np.random.randn(self.n_samples)
        
        # Create dataset and dataloader
        reg_dataset = ArrayDataset(X_reg, y_reg)
        reg_dataloader = DataLoader(reg_dataset, batch_size=32)
        
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(1))
        
        # Compile with regression metrics
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=[
                MeanSquaredError(),
                MeanAbsoluteError(),
                R2Score()
            ]
        )
        
        # Train model
        history = model.fit(reg_dataloader, epochs=1)
        
        # Verify metrics are recorded
        for metric in ['mse', 'mae', 'r2_score']:
            self.assertIn(metric, history)
            self.assertGreater(len(history[metric]), 0)
            
    def test_metric_calculation(self):
        """Test metric calculation during training."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        # Compile with accuracy metric
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Train model
        history = model.fit(self.dataloader, epochs=1)
        
        # Verify accuracy values are between 0 and 1
        for acc in history['accuracy']:
            self.assertGreaterEqual(acc, 0)
            self.assertLessEqual(acc, 1)
            
    def test_metric_reset(self):
        """Test metric reset between epochs."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        # Compile with accuracy metric
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Train for two epochs
        history = model.fit(self.dataloader, epochs=2)
        
        # Verify metrics are reset between epochs
        self.assertEqual(len(history['accuracy']), 2)
        self.assertNotEqual(history['accuracy'][0], history['accuracy'][1])
        
    def test_custom_metric_integration(self):
        """Test custom metric integration with model."""
        class CustomMetric:
            def __init__(self):
                self.accumulated_sum = 0
                self.count = 0
                
            def compare(self, y_pred, y_true):
                return np.mean(np.abs(y_pred - y_true))
                
            def calculate(self, y_pred, y_true):
                return self.compare(y_pred, y_true)
                
            def calculate_accumulated(self):
                return self.accumulated_sum / self.count if self.count > 0 else 0
                
            def new_pass(self):
                self.accumulated_sum = 0
                self.count = 0
                
            def set_model(self, model):
                self.model = model
        
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        # Compile with custom metric
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[CustomMetric()]
        )
        
        # Train model
        history = model.fit(self.dataloader, epochs=1)
        
        # Verify custom metric is recorded
        self.assertIn('custom_metric', history)
        self.assertGreater(len(history['custom_metric']), 0)
        
    def test_metric_aggregation(self):
        """Test metric aggregation across batches."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        # Compile with accuracy metric
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Train model
        history = model.fit(self.dataloader, epochs=1)
        
        # Get batch metrics
        batch_metrics = []
        for batch_X, batch_y in self.dataloader:
            y_pred = model.forward(batch_X)
            batch_metrics.append(Accuracy().compare(y_pred, batch_y))
        
        # Verify aggregated metric matches batch metrics
        avg_batch_metric = np.mean(batch_metrics)
        self.assertAlmostEqual(history['accuracy'][0], avg_batch_metric, places=4)
        
    def test_metric_tracking(self):
        """Test metric tracking during training."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        # Compile with multiple metrics
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[
                Accuracy(),
                Precision(),
                Recall(),
                F1Score()
            ]
        )
        
        # Train model
        history = model.fit(self.dataloader, epochs=2)
        
        # Verify all metrics are tracked
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            self.assertIn(metric, history)
            self.assertEqual(len(history[metric]), 2)
            
    def test_metric_validation(self):
        """Test metric validation during training."""
        # Create model
        model = Model()
        model.add(Dense(64, input_shape=(self.n_features,)))
        model.add(ReLU())
        model.add(Dense(self.n_classes))
        model.add(Softmax())
        
        # Compile with accuracy metric
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=CategoricalCrossEntropy(),
            metrics=[Accuracy()]
        )
        
        # Train with validation data
        history = model.fit(
            self.dataloader,
            validation_data=self.dataloader,
            epochs=1
        )
        
        # Verify validation metrics
        self.assertIn('val_accuracy', history)
        self.assertGreater(len(history['val_accuracy']), 0)

if __name__ == '__main__':
    unittest.main() 