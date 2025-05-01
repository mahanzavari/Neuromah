from typing import List, Dict, Optional, Any
import numpy as np
from abc import ABC, abstractmethod
import os
import time
import json
from ..utils import TensorMonitor

class Callback(ABC):
    """Base class for callbacks."""
    
    def __init__(self):
        self.model = None
        
    def set_model(self, model):
        """Set the model for the callback."""
        self.model = model
        
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training."""
        pass
        
    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training."""
        pass
        
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of an epoch."""
        pass
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of an epoch."""
        pass
        
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Called at the beginning of a batch."""
        pass
        
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of a batch."""
        pass

class CallbackList:
    """Container for callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
        self.model = None
        
    def set_model(self, model):
        """Set the model for all callbacks."""
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)
            
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
            
    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training."""
        for callback in self.callbacks:
            callback.on_train_end(logs)
            
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of an epoch."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
            
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of an epoch."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
            
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Called at the beginning of a batch."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
            
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of a batch."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

class TensorBoard(Callback):
    """TensorBoard callback for visualization."""
    
    def __init__(
        self,
        log_dir: str = './logs',
        histogram_freq: int = 0,
        write_graph: bool = True,
        write_images: bool = False,
        update_freq: str = 'epoch'
    ):
        super().__init__()
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.write_images = write_images
        self.update_freq = update_freq
        self.tensorboard = None
        self.epoch = 0
        
    def set_model(self, model):
        """Set up TensorBoard."""
        super().set_model(model)
        self.tensorboard = TensorMonitor(log_dir=self.log_dir)
        self.tensorboard.start_run(model)
        
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Start epoch in TensorBoard."""
        self.epoch = epoch
        self.tensorboard.start_epoch(epoch)
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log epoch metrics."""
        if logs:
            for name, value in logs.items():
                self.tensorboard.log_scalar(f'epoch/{name}', value)
                
        # Log histograms if enabled
        if self.histogram_freq > 0 and epoch % self.histogram_freq == 0:
            for layer in self.model._base_model.trainable_layers:
                self.tensorboard.log_layer_parameters(layer)
                
        self.tensorboard.end_epoch()
        
    def on_train_end(self, logs: Optional[Dict] = None):
        """Save TensorBoard logs."""
        self.tensorboard.save_logs()

class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving."""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0,
        patience: int = 0,
        mode: str = 'auto',
        baseline: Optional[float] = None,
        restore_best_weights: bool = False
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Initialize early stopping."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Check if training should be stopped."""
        if logs is None:
            return
            
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.best is None:
            self.best = current
            if self.restore_best_weights:
                self.best_weights = self.model._base_model.get_parameters()
            return
            
        if self.mode == 'min':
            if current < self.best - self.min_delta:
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model._base_model.get_parameters()
            else:
                self.wait += 1
        else:  # mode == 'max'
            if current > self.best + self.min_delta:
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model._base_model.get_parameters()
            else:
                self.wait += 1
                
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model._base_model.stop_training = True
            if self.restore_best_weights:
                self.model._base_model.set_parameters(self.best_weights)

class ModelCheckpoint(Callback):
    """Save the model after every epoch."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = False,
        mode: str = 'auto',
        save_freq: str = 'epoch'
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq
        self.best = None
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Save the model if needed."""
        if logs is None:
            return
            
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.best is None:
            self.best = current
            self._save_model(epoch, logs)
            return
            
        if self.mode == 'min':
            if current < self.best:
                self.best = current
                self._save_model(epoch, logs)
        else:  # mode == 'max'
            if current > self.best:
                self.best = current
                self._save_model(epoch, logs)
                
    def _save_model(self, epoch: int, logs: Dict):
        """Save the model."""
        filepath = self.filepath.format(epoch=epoch, **logs)
        self.model.save(filepath)

class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving."""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 0,
        mode: str = 'auto',
        min_delta: float = 0.0001
    ):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.min_delta = min_delta
        
        self.wait = 0
        self.best = None
        
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Initialize learning rate reduction."""
        self.wait = 0
        self.best = None
        
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Check if learning rate should be reduced."""
        if logs is None:
            return
            
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.best is None:
            self.best = current
            return
            
        if self.mode == 'min':
            if current < self.best - self.min_delta:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
        else:  # mode == 'max'
            if current > self.best + self.min_delta:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                
        if self.wait >= self.patience:
            old_lr = self.model._base_model.optimizer.learning_rate
            new_lr = max(old_lr * self.factor, self.min_lr)
            self.model._base_model.optimizer.learning_rate = new_lr
            self.wait = 0 