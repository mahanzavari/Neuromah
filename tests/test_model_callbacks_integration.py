import unittest
import numpy as np
import os
import tempfile
from src.api.model import Model
from src.api.data import ArrayDataset, DataLoader
from src.api.callbacks import (
    Callback, CallbackList, TensorBoard, EarlyStopping,
    ModelCheckpoint, ReduceLROnPlateau
)
from src.layers import Dense, ReLU, Softmax
from src.optimizers import Adam
from src.losses import CategoricalCrossEntropy
from src.metrics import Accuracy

class TestModelCallbacksIntegration(unittest.TestCase):
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
        
        # Create temporary directory for TensorBoard logs and model checkpoints
        self.temp_dir = tempfile.mkdtemp()
        
    def test_callback_list_integration(self):
        """Test callback list integration with model."""
        # Create callbacks
        callbacks = [
            TensorBoard(log_dir=os.path.join(self.temp_dir, 'logs')),
            EarlyStopping(monitor='val_loss', patience=3),
            ModelCheckpoint(
                filepath=os.path.join(self.temp_dir, 'model.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=2
            )
        ]
        
        # Create callback list
        callback_list = CallbackList(callbacks)
        
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
        
        # Set callbacks for model
        model.callbacks = callback_list
        
        # Train model
        history = model.fit(self.dataloader, epochs=5)
        
        # Verify callbacks were executed
        self.assertIn('loss', history)
        self.assertGreater(len(history['loss']), 0)
        
    def test_tensorboard_callback(self):
        """Test TensorBoard callback integration."""
        # Create TensorBoard callback
        tensorboard = TensorBoard(
            log_dir=os.path.join(self.temp_dir, 'tensorboard_logs'),
            histogram_freq=1
        )
        
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
        
        # Set callback for model
        model.callbacks = CallbackList([tensorboard])
        
        # Train model
        history = model.fit(self.dataloader, epochs=2)
        
        # Verify TensorBoard logs were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'tensorboard_logs')))
        
    def test_early_stopping_callback(self):
        """Test EarlyStopping callback integration."""
        # Create EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=2,
            min_delta=0.001
        )
        
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
        
        # Set callback for model
        model.callbacks = CallbackList([early_stopping])
        
        # Train model
        history = model.fit(self.dataloader, epochs=10)
        
        # Verify early stopping worked
        self.assertLess(len(history['loss']), 10)
        
    def test_model_checkpoint_callback(self):
        """Test ModelCheckpoint callback integration."""
        # Create ModelCheckpoint callback
        checkpoint_path = os.path.join(self.temp_dir, 'checkpoint.h5')
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='loss',
            save_best_only=True
        )
        
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
        
        # Set callback for model
        model.callbacks = CallbackList([model_checkpoint])
        
        # Train model
        model.fit(self.dataloader, epochs=2)
        
        # Verify checkpoint was created
        self.assertTrue(os.path.exists(checkpoint_path))
        
    def test_reduce_lr_callback(self):
        """Test ReduceLROnPlateau callback integration."""
        # Create ReduceLROnPlateau callback
        reduce_lr = ReduceLROnPlateau(
            monitor='loss',
            factor=0.1,
            patience=1,
            min_lr=0.00001
        )
        
        # Create and compile model
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
        
        # Set callback for model
        model.callbacks = CallbackList([reduce_lr])
        
        # Train model
        model.fit(self.dataloader, epochs=3)
        
        # Verify learning rate was reduced
        self.assertLess(model.optimizer.learning_rate, initial_lr)
        
    def test_custom_callback(self):
        """Test custom callback integration."""
        class CustomCallback(Callback):
            def __init__(self):
                self.epochs = []
                self.batches = []
                
            def on_epoch_begin(self, epoch, logs=None):
                self.epochs.append(epoch)
                
            def on_batch_begin(self, batch, logs=None):
                self.batches.append(batch)
        
        # Create custom callback
        custom_callback = CustomCallback()
        
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
        
        # Set callback for model
        model.callbacks = CallbackList([custom_callback])
        
        # Train model
        model.fit(self.dataloader, epochs=2)
        
        # Verify custom callback was executed
        self.assertGreater(len(custom_callback.epochs), 0)
        self.assertGreater(len(custom_callback.batches), 0)
        
    def test_multiple_callbacks(self):
        """Test integration of multiple callbacks."""
        # Create multiple callbacks
        callbacks = [
            TensorBoard(log_dir=os.path.join(self.temp_dir, 'multi_logs')),
            EarlyStopping(monitor='loss', patience=2),
            ModelCheckpoint(
                filepath=os.path.join(self.temp_dir, 'multi_checkpoint.h5'),
                monitor='loss'
            ),
            ReduceLROnPlateau(monitor='loss', patience=1)
        ]
        
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
        
        # Set callbacks for model
        model.callbacks = CallbackList(callbacks)
        
        # Train model
        history = model.fit(self.dataloader, epochs=3)
        
        # Verify all callbacks were executed
        self.assertIn('loss', history)
        self.assertGreater(len(history['loss']), 0)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'multi_logs')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'multi_checkpoint.h5')))
        self.assertLess(model.optimizer.learning_rate, 0.001)

if __name__ == '__main__':
    unittest.main() 