import numpy as np
from typing import Dict, Tuple

class PositionwiseFeedForward:
     """
    Position-wise Feed-Forward Network as described in 'Attention Is All You Need'.
    
    Each position in the sequence is processed independently with the same fully
    connected feed-forward network, consisting of two linear transformations with ReLU
    activation in between.
    
    Formula: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    
    Parameters:
        d_model (int): Input and output dimension
        d_ff (int): Hidden dimension of the feed-forward layer
     """
     
     def __init__(self, d_model: int, d_ff: int , xp = np):
          from .Dense import Layer_Dense
          from ..activations import Activation_ReLU
               #    from ..activations.Activation_ReLU import Activation_ReLU
          self.xp = xp 
          self.d_model = d_model
          self.d_ff = d_ff
          
          self.dense1 = Layer_Dense(d_model, d_ff, xp = np)
          self.relu = Activation_ReLU()
          self.dense2 = Layer_Dense(d_ff, d_model, xp = np)
     
     def forward(self, x, training : bool = True):
          """
        Forward pass for the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            training: Whether in training mode
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
          """
          
          # self.original_shape = x.shape
          
          batch_size , seq_len, d_model = x.shape
          reshaped_input = x.reshape(-1, d_model)
          
          self.dense1.forward(reshaped_input, training)
          self.relu.forward(self.dense1.output, training)
          self.dense2.forward(self.relu.output, training)
          
          self.output = self.dense2.output.reshape(batch_size, seq_len, d_model)
          
          return self.output
     
     def backward(self, dvalues):
          """
        Backward pass for the feed-forward network.
        
        Args:
            dvalues: Gradient of loss with respect to output
            
        Returns:
            Gradients with respect to input
          """
          
          batch_size , seq_len, d_model = dvalues.shape
          reshaped_dvalues = dvalues.reshape(-1, d_model)
          
          self.dense2.backward(reshaped_dvalues)
          self.relu.backward(self.dense2.dinputs)
          self.dense1.backward(self.relu.dinputs)
          self.dinputs = self.dense1.dinputs.reshape(batch_size, seq_len, d_model)
          
          return self.dinputs

     def get_parameters(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
          params = {}
          # dense1 params
          dense1_params = self.dense1.get_parameters()          
          for name, (param, grad) in dense1_params.items():
               params[f"dense1_{name}"] = (param, grad)
          # dense2 params
          dense2_params = self.dense2.get_parameters()
          for name, (parma, grad) in dense2_params.items():
               param[f"dense2_{name}"] = (parma, grad)
               
          return param
     
     def set_patameters(self, **kwargs) -> None:
          """
          Set parameters for dense layers.
          
          Expected parameter names:
          - dense1_weights, dense1_biases
          - dense2_weights, dense2_biases
          """ 
          # dense1
          if 'dense1_weights' in kwargs and 'dense1_biases' in kwargs:
              self.dense1.set_parameters(kwargs['dense1_weights'], kwargs['dense1_biases'])
          
          # dense2
          if 'dense2_weights' in kwargs and 'dense2_biases' in kwargs:
              self.dense2.set_parameters(kwargs['dense2_weights'], kwargs['dense2_biases'])

          
          

