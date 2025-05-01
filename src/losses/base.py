from typing import Optional, Union, Dict, Any
import numpy as np
from abc import ABC, abstractmethod

class BaseLoss(ABC):
    """Base class for all loss functions."""
    
    def __init__(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
        self.model = None
        
    def set_model(self, model):
        """Set the model for the loss function."""
        self.model = model
        
    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Forward pass of the loss function.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Loss values
        """
        pass
        
    @abstractmethod
    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Backward pass of the loss function.
        
        Args:
            dvalues: Gradient values
            y_true: True values
            
        Returns:
            Gradient of the loss
        """
        pass
        
    def calculate(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        include_regularization: bool = False
    ) -> float:
        """
        Calculate the loss.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            include_regularization: Whether to include regularization loss
            
        Returns:
            Loss value
        """
        sample_losses = self.forward(y_pred, y_true)
        data_loss = np.mean(sample_losses)
        
        if include_regularization and self.model is not None:
            regularization_loss = self.regularization_loss()
            return data_loss + regularization_loss
            
        return data_loss
        
    def regularization_loss(self) -> float:
        """Calculate regularization loss."""
        regularization_loss = 0
        
        if self.model is not None:
            for layer in self.model.trainable_layers:
                if layer.weight_regularizer_l1 > 0:
                    regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
                if layer.weight_regularizer_l2 > 0:
                    regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
                    
        return regularization_loss
        
    def calculate_accumulated(
        self,
        include_regularization: bool = False
    ) -> float:
        """
        Calculate accumulated loss.
        
        Args:
            include_regularization: Whether to include regularization loss
            
        Returns:
            Accumulated loss value
        """
        data_loss = self.accumulated_sum / self.accumulated_count
        
        if include_regularization and self.model is not None:
            regularization_loss = self.regularization_loss()
            return data_loss + regularization_loss
            
        return data_loss
        
    def new_pass(self) -> None:
        """Reset accumulated loss values."""
        self.accumulated_sum = 0
        self.accumulated_count = 0 