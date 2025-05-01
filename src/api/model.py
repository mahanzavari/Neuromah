from typing import List, Dict, Optional, Union, Tuple, Callable
import numpy as np
from ..core.Model import Model as BaseModel
from ..layers.core.base import BaseLayer
from ..optimizers.base import BaseOptimizer
from ..losses.base import BaseLoss
from ..metrics.base import BaseMetric
from ..graph import ExecutionMode
from ..graph.layer_integration import GraphModel
from .callbacks import CallbackList, Callback
from .data import DataLoader

class Model:
    """High-level model interface similar to PyTorch and TensorFlow."""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the model.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self._base_model = BaseModel(device=device)
        self._graph_model = None
        self._compiled = False
        self._optimizer = None
        self._loss = None
        self._metrics = []
        self._callbacks = CallbackList()
        self._training_history = []
        
    def add(self, layer: BaseLayer) -> None:
        """Add a layer to the model."""
        self._base_model.add(layer)
        
    def compile(
        self,
        optimizer: Union[str, BaseOptimizer],
        loss: Union[str, BaseLoss],
        metrics: Optional[List[Union[str, BaseMetric]]] = None,
        execution_mode: ExecutionMode = ExecutionMode.EAGER
    ) -> None:
        """
        Configure the model for training.
        
        Args:
            optimizer: Optimizer instance or string identifier
            loss: Loss function instance or string identifier
            metrics: List of metric instances or string identifiers
            execution_mode: Execution mode (EAGER, GRAPH, HYBRID)
        """
        # Set optimizer
        if isinstance(optimizer, str):
            optimizer = self._get_optimizer_by_name(optimizer)
        self._optimizer = optimizer
        
        # Set loss
        if isinstance(loss, str):
            loss = self._get_loss_by_name(loss)
        self._loss = loss
        
        # Set metrics
        self._metrics = []
        if metrics:
            for metric in metrics:
                if isinstance(metric, str):
                    metric = self._get_metric_by_name(metric)
                self._metrics.append(metric)
                
        # Set up the base model
        self._base_model.set(
            loss=self._loss,
            optimizer=self._optimizer,
            accuracy=self._metrics[0] if self._metrics else None
        )
        self._base_model.finalize()
        
        # Create graph model if needed
        if execution_mode != ExecutionMode.EAGER:
            self._graph_model = GraphModel(self._base_model)
            self._graph_model.compile(mode=execution_mode)
            
        self._compiled = True
        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 1,
        batch_size: Optional[int] = None,
        callbacks: Optional[List[Callback]] = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train the model.
        
        Args:
            X: Training data
            y: Training labels
            validation_data: Tuple of (X_val, y_val)
            epochs: Number of epochs
            batch_size: Batch size
            callbacks: List of callbacks
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if not self._compiled:
            raise RuntimeError("Model must be compiled before training")
            
        # Set up callbacks
        if callbacks:
            self._callbacks = CallbackList(callbacks)
        self._callbacks.set_model(self)
        
        # Create data loader
        train_loader = DataLoader(X, y, batch_size=batch_size, shuffle=True)
        val_loader = None
        if validation_data:
            val_loader = DataLoader(*validation_data, batch_size=batch_size)
            
        # Training loop
        self._callbacks.on_train_begin()
        for epoch in range(epochs):
            self._callbacks.on_epoch_begin(epoch)
            
            # Training
            train_metrics = self._train_epoch(train_loader, verbose)
            
            # Validation
            val_metrics = {}
            if val_loader:
                val_metrics = self._validate_epoch(val_loader, verbose)
                
            # Update history
            epoch_history = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
            self._training_history.append(epoch_history)
            
            self._callbacks.on_epoch_end(epoch, epoch_history)
            
        self._callbacks.on_train_end()
        return self._training_history
        
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: Optional[int] = None,
        verbose: int = 1
    ) -> Dict:
        """
        Evaluate the model.
        
        Args:
            X: Test data
            y: Test labels
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Evaluation metrics
        """
        if not self._compiled:
            raise RuntimeError("Model must be compiled before evaluation")
            
        loader = DataLoader(X, y, batch_size=batch_size)
        return self._validate_epoch(loader, verbose)
        
    def predict(
        self,
        X: np.ndarray,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input data
            batch_size: Batch size
            
        Returns:
            Predictions
        """
        if not self._compiled:
            raise RuntimeError("Model must be compiled before prediction")
            
        loader = DataLoader(X, batch_size=batch_size)
        predictions = []
        
        for batch_X in loader:
            if self._graph_model:
                batch_pred = self._graph_model.predict(batch_X)
            else:
                batch_pred = self._base_model.predict(batch_X)
            predictions.append(batch_pred)
            
        return np.vstack(predictions)
        
    def save(self, path: str) -> None:
        """Save the model."""
        if self._graph_model:
            self._graph_model.save(path)
        else:
            self._base_model.save(path)
            
    @classmethod
    def load(cls, path: str) -> 'Model':
        """Load a saved model."""
        model = cls()
        model._base_model = BaseModel.load(path)
        model._compiled = True
        return model
        
    def to(self, device: str) -> None:
        """Move model to device."""
        self._base_model.to(device)
        if self._graph_model:
            self._graph_model.to(device)
            
    def _train_epoch(self, loader: DataLoader, verbose: int) -> Dict:
        """Train for one epoch."""
        metrics = {metric.__class__.__name__: 0.0 for metric in self._metrics}
        metrics['loss'] = 0.0
        
        for batch_X, batch_y in loader:
            self._callbacks.on_batch_begin(len(metrics))
            
            # Forward pass
            if self._graph_model:
                outputs = self._graph_model.train_step(batch_X, batch_y)
                batch_loss = outputs['loss']
                batch_metrics = {k: v for k, v in outputs.items() if k != 'loss'}
            else:
                outputs = self._base_model.forward(batch_X, training=True)
                batch_loss = self._loss.calculate(outputs, batch_y)
                predictions = self._base_model.output_layer_activation.predictions(outputs)
                batch_metrics = {
                    metric.__class__.__name__: metric.calculate(predictions, batch_y)
                    for metric in self._metrics
                }
                
            # Update metrics
            metrics['loss'] += batch_loss
            for name, value in batch_metrics.items():
                metrics[name] += value
                
            self._callbacks.on_batch_end(len(metrics), metrics)
            
        # Average metrics
        n_batches = len(loader)
        return {k: v / n_batches for k, v in metrics.items()}
        
    def _validate_epoch(self, loader: DataLoader, verbose: int) -> Dict:
        """Validate for one epoch."""
        metrics = {metric.__class__.__name__: 0.0 for metric in self._metrics}
        metrics['loss'] = 0.0
        
        for batch_X, batch_y in loader:
            # Forward pass
            if self._graph_model:
                outputs = self._graph_model.predict(batch_X)
                batch_loss = self._loss.calculate(outputs, batch_y)
                predictions = self._base_model.output_layer_activation.predictions(outputs)
            else:
                outputs = self._base_model.forward(batch_X, training=False)
                batch_loss = self._loss.calculate(outputs, batch_y)
                predictions = self._base_model.output_layer_activation.predictions(outputs)
                
            # Update metrics
            metrics['loss'] += batch_loss
            for metric in self._metrics:
                metrics[metric.__class__.__name__] += metric.calculate(predictions, batch_y)
                
        # Average metrics
        n_batches = len(loader)
        return {k: v / n_batches for k, v in metrics.items()}
        
    def _get_optimizer_by_name(self, name: str) -> BaseOptimizer:
        """Get optimizer by name."""
        optimizers = {
            'adam': self._base_model.optimizers.Optimizer_Adam,
            'sgd': self._base_model.optimizers.Optimizer_SGD,
            'rmsprop': self._base_model.optimizers.Optimizer_RMSprop
        }
        return optimizers[name.lower()]()
        
    def _get_loss_by_name(self, name: str) -> BaseLoss:
        """Get loss by name."""
        losses = {
            'categorical_crossentropy': self._base_model.losses.Loss_CategoricalCrossentropy,
            'binary_crossentropy': self._base_model.losses.Loss_BinaryCrossentropy,
            'mse': self._base_model.losses.Loss_MeanSquaredError
        }
        return losses[name.lower()]()
        
    def _get_metric_by_name(self, name: str) -> BaseMetric:
        """Get metric by name."""
        metrics = {
            'accuracy': self._base_model.metrics.Accuracy_Categorical,
            'precision': self._base_model.metrics.Precision,
            'recall': self._base_model.metrics.Recall
        }
        return metrics[name.lower()]() 