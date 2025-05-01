import unittest
import numpy as np
from src.api.model import Model
from src.api.data import ArrayDataset, DataLoader, Preprocessor
from src.layers import Dense, ReLU, Softmax
from src.optimizers import Adam
from src.losses import CategoricalCrossEntropy
from src.metrics import Accuracy

class TestModelDataIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        # Generate synthetic data
        self.n_samples = 1000
        self.n_features = 20
        self.n_classes = 3
        
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randint(0, self.n_classes, size=self.n_samples)
        
    def test_dataset_creation(self):
        """Test dataset creation and basic functionality."""
        dataset = ArrayDataset(self.X, self.y)
        
        # Test length
        self.assertEqual(len(dataset), self.n_samples)
        
        # Test item access
        sample, label = dataset[0]
        self.assertEqual(sample.shape, (self.n_features,))
        self.assertIsInstance(label, (int, np.integer))
        
    def test_dataloader_functionality(self):
        """Test dataloader functionality."""
        dataset = ArrayDataset(self.X, self.y)
        batch_size = 32
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Test batch iteration
        for batch_X, batch_y in dataloader:
            self.assertEqual(batch_X.shape[0], batch_size)
            self.assertEqual(batch_y.shape[0], batch_size)
            
    def test_preprocessing_integration(self):
        """Test preprocessing integration with model."""
        # Preprocess data
        X_normalized = Preprocessor.normalize(self.X)
        y_one_hot = Preprocessor.one_hot_encode(self.y, self.n_classes)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = Preprocessor.train_val_test_split(
            X_normalized, y_one_hot, train_size=0.7, val_size=0.15
        )
        
        # Create datasets and dataloaders
        train_dataset = ArrayDataset(X_train, y_train)
        val_dataset = ArrayDataset(X_val, y_val)
        test_dataset = ArrayDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Create and compile model
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
        
        # Train model
        history = model.fit(
            train_loader,
            validation_data=val_loader,
            epochs=1
        )
        
        # Verify training history
        self.assertIn('loss', history)
        self.assertIn('val_loss', history)
        self.assertIn('accuracy', history)
        self.assertIn('val_accuracy', history)
        
    def test_batch_size_handling(self):
        """Test different batch size handling."""
        dataset = ArrayDataset(self.X, self.y)
        
        # Test different batch sizes
        for batch_size in [1, 16, 32, 64, 128]:
            dataloader = DataLoader(dataset, batch_size=batch_size)
            
            # Create and compile model
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
            history = model.fit(dataloader, epochs=1)
            
            # Verify training completed
            self.assertIn('loss', history)
            self.assertGreater(len(history['loss']), 0)
            
    def test_data_shuffling(self):
        """Test data shuffling functionality."""
        dataset = ArrayDataset(self.X, self.y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Get first batch
        first_batch_X, first_batch_y = next(iter(dataloader))
        
        # Get second batch
        second_batch_X, second_batch_y = next(iter(dataloader))
        
        # Verify batches are different
        self.assertFalse(np.array_equal(first_batch_X, second_batch_X))
        
    def test_data_preprocessing_pipeline(self):
        """Test complete data preprocessing pipeline."""
        # Normalize features
        X_normalized = Preprocessor.normalize(self.X)
        
        # One-hot encode labels
        y_one_hot = Preprocessor.one_hot_encode(self.y, self.n_classes)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = Preprocessor.train_val_test_split(
            X_normalized, y_one_hot, train_size=0.7, val_size=0.15
        )
        
        # Create datasets
        train_dataset = ArrayDataset(X_train, y_train)
        val_dataset = ArrayDataset(X_val, y_val)
        test_dataset = ArrayDataset(X_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Create and compile model
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
        
        # Train model
        history = model.fit(
            train_loader,
            validation_data=val_loader,
            epochs=1
        )
        
        # Evaluate model
        test_metrics = model.evaluate(test_loader)
        
        # Verify results
        self.assertIn('loss', test_metrics)
        self.assertIn('accuracy', test_metrics)
        self.assertIsInstance(test_metrics['loss'], float)
        self.assertIsInstance(test_metrics['accuracy'], float)

if __name__ == '__main__':
    unittest.main() 