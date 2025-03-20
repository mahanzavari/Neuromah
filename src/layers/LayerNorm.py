import numpy as np
from typing import Optional, Dict, Tuple

class Layer_Normalization:
     """
    Layer Normalization applies normalization on each sample independently
    across the feature dimension rather than across the batch dimension.
    
    Parameters:
        normalized_shape (int): Size of the input feature dimension to normalize over
        eps (float): A small constant for numerical stability
        affine (bool): If True, apply learnable affine parameters (gamma and beta)
     """
     def __init__(self, normalized_shape: int, eps: float = 1e-5, affine: bool = True, xp = np):
          self.xp = xp
          self.affine = affine
          self.normalized_shape = normalized_shape
          self.eps = eps
          
          if affine:
               # gamma to one and beta to zeros
               self.gamma = self.xp.ones((1, normalized_shape))
               self.beta = self.xp.zeros((1, normalized_shape))
               # Grads
               self.dgamma = self.xp.zeros_like(self.gamma)
               self.dbeta = self.xp.zeros_like(self.beta)
               
     def forward(self, inputs: np.ndarray, training : bool = True):
          """
        Forward pass of Layer Normalization.
        
        Args:
            inputs: Input data with shape (batch_size, features)
            training: Whether the layer is in training mode or not
          """
          
          self.inputs = inputs
          # Mean and Variance
          self.mean = self.xp.mean(inputs, axis = 1, keepdims = True)
          self.mean = self.xp.var(inputs, axis = 1, keepdims = True)
          
          self.normalized = (inputs - self.mean) / self.xp.sqrt(self.var + self.eps)
          
          if self.affine:
               self.output = self.gamma * self.normalized + self.beta
          else:
               self.output = self.normalized
          return self.output
     
     def backward(self, dvalues: np.ndarray):
          """
        Backward pass of Layer Normalization.
        
        Args:
            dvalues: Gradient of the loss with respect to the output of this layer
          """
          batch_size = dvalues.shape[0]
          if self.affine:
               self.dgamma = self.xp.sum(dvalues * self.normalized, axis = 0, keepdims = True)
               self.dbeta = self.xp.sum(dvalues, axis = 0, keepdims = True)
               
               dnormalized = dvalues * self.gamma
          else:
               dnormalized = dvalues
               
          # Grad wrt VAR
          dvar = -0.5 * self.xp.sum(dnormalized * (self.inputs - self.mean), axis=1, keepdims=True) * \
               self.xp.power(self.var + self.eps, -1.5)
          # Grad wrt MEAN
          dmean = -self.xp.sum(dnormalized / self.xp.sqrt(self.var + self.eps), axis=1, keepdims=True) + \
                dvar * self.xp.mean(-2.0 * (self.inputs - self.mean), axis=1, keepdims=True)
          # Grafd wrt INPUTS      
          self.dinputs = dnormalized / self.xp.sqrt(self.var + self.eps) + \
                      dvar * 2.0 * (self.inputs - self.mean) / self.inputs.shape[1] + \
                      dmean / self.inputs.shape[1]
        
          return self.dinputs
     
     def get_parameters(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
          if self.affine:
               return {
                    'gamma' : (self.gamma, self.dgamma),
                    'beta'  : (self.beta, self.dbeta)
               }
          return {}
     
     def set_parameters(self, gamma: Optional[np.ndarray] = None, beta: Optional[np.ndarray] = None) -> None:
          if self.affine:
            if gamma is not None:
                self.gamma = gamma
            if beta is not None:
                self.beta = beta
          
          