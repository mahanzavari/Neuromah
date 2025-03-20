import numpy as np
from typing import Dict, Tuple, Optional

class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention as described in 'Attention Is All You Need' papaer by Google.
    
    Computes attention scores between queries and keys, then uses these scores
    to create a weighted sum of values.
    
    Formula: Attention(Q, K, V) = softmax((Q·K^T)/√d_k)·V
    """
    
    def __init__(self, mask: Optional[np.ndarray] = None, xp = np):
      self.xp = xp
      self.mask = mask
      self.attention_weights = None      
    
    def forward(self, queries, keys, values, training = True):
          """
        Forward pass for scaled dot-product attention.
        
        Args:
            queries: Query tensor of shape (..., seq_len_q, depth)
            keys: Key tensor of shape (..., seq_len_k, depth)
            values: Value tensor of shape (..., seq_len_k, depth_v)
            training: Whether in training mode
            
        Returns:
            Output tensor and attention weights
          """
          self.queries = queries
          self.keys = keys
          self.values = values
          
          d_k = queries.shape[-1]
          
          attention_logits = self.xp.matmul(queries, keys.transpose(0, 2, 1)) / self.xp.sqrt(d_k)
          
          if self.mask is not None:
            attention_logits += (self.mask * -float('inf'))
            # attention_logits += (self.mask * -1e9)
            
            from ..activations import Activation_Softmax
            self.softmax = Activation_Softmax()
            
            original_shape = attention_logits.shape
            reshaped_logits = attention_logits.reshape(-1, original_shape[-1])
            
            self.softmax.forward(reshaped_logits, training=training)
            attention_weights = self.softmax.output.reshape(original_shape)
            
            self.attention_weights = attention_weights
            
            self.output = self.xp.matmul(attention_weights, values)
            
            return self.output
          
    def backward(self, dvalues):
        """
        Backward pass for scaled dot-product attention.
        
        Args:
            dvalues: Gradient of loss with respect to output
            
        Returns:
            Gradients with respect to queries, keys, and values
        """
        
        d_attention_weights = self.xp.matmul(dvalues, self.values.transpose(0, 2, 1))
        
        original_shape = d_attention_weights.shape
        reshaped_d_weights = d_attention_weights.reshape(-1, original_shape[-1])
        
        self.softmax.backward(reshaped_d_weights)
        d_attention_logits = self.softmax.dinputs.reshape(original_shape)
        
        d_k = self.queries.shape[-1]
        d_attention_logits /= self.xp.sqrt(d_k)
        # Grad wrt Queries, Keys and Values
        self.dqueries = self.xp.matmul(d_attention_logits, self.keys) 
        self.dkeys = self.xp.matmul(d_attention_logits.transpose(0, 2, 1), self.queries)
        self.dvalues = self.xp.matmul(d_attention_weights.transpose(0, 2, 1), dvalues) 
        
        return self.dqueries, self.dkeys , self.dvalues    
      

        