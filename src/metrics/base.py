from typing import Optional, Union, Dict, Any
import numpy as np
from abc import ABC, abstractmethod

class BaseMetric(ABC):
    """Base class for all metrics."""
    
    def __init__(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
        self.model = None
        
    def set_model(self, model):
        """Set the model for the metric."""
        self.model = model
        
    @abstractmethod
    def compare(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compare predictions to true values.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Comparison results
        """
        pass
        
    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate the metric.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            Metric value
        """
        comparisons = self.compare(y_pred, y_true)
        return np.mean(comparisons)
        
    def calculate_accumulated(self) -> float:
        """
        Calculate accumulated metric.
        
        Returns:
            Accumulated metric value
        """
        return self.accumulated_sum / self.accumulated_count
        
    def new_pass(self) -> None:
        """Reset accumulated metric values."""
        self.accumulated_sum = 0
        self.accumulated_count = 0 