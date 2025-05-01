from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional, Union
import cupy as cp

class BaseLayer(ABC):
    """
    Abstract base class for all neural network layers.
    Defines the common interface that all layers must implement.
    """
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.xp = cp if use_gpu else np
        self.inputs = None
        self.output = None
        self.dinputs = None
    
    @abstractmethod
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Compute the forward pass of the layer.
        
        Args:
            inputs (np.ndarray): Input data.
            training (bool): Whether the layer is in training mode.
        """
        pass
    
    @abstractmethod
    def backward(self, dvalues: np.ndarray) -> None:
        """
        Compute the backward pass of the layer.
        
        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the layer's output.
        """
        pass
    
    def get_parameters(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the layer's parameters and their gradients.
        Returns an empty dict by default, should be overridden by layers with parameters.
        """
        return {}
    
    def set_parameters(self, **kwargs) -> None:
        """
        Set the layer's parameters.
        Does nothing by default, should be overridden by layers with parameters.
        """
        pass 