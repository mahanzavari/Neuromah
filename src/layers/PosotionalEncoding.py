import numpy as np
from typing import Dict, Tuple, Optional

class PosotionalEncoding:
     """
    Positional encoding for transformer architecture.
    
    This adds information about the position of tokens in the sequence
    using sine and cosine functions of different frequencies.
    
    Parameters:
        d_model (int): Embedding dimension
        max_seq_length (int): Maximum sequence length to pre-compute
        dropout_rate (float): Dropout rate to apply to the positional encodings
     """
     
     def __init__(self, d_model: int, max_seq_length: int = 5000, xp = np):
          self.d_model = d_model
          self.xp = self.xp
          self.max_seq_length = max_seq_length
          
          # Positional Encoding Precomputation
          self.pe = self.xp.zeros((max_seq_length, d_model))
          position = self.xp.arange(0, max_seq_length).reshape(-1 , 1)
          div_term = self.xp.exp(self.xp.arange(0, d_model, 2) * -(self.xp.log(10000.0) / d_model))

          self.pe[:, 0::2] = self.xp.sin(position * div_term)
          self.pe[:, 1::2] = self.xp.cos(position * div_term)
          # add BATCH DIM 
          self.pe = self.pe.reshape(1, max_seq_length, d_model)
          
     def forward(self, x, training : bool = True):
          """
          Add positional encoding to the input embeddings.
          
          Args:
              x: Input embeddings with shape (batch_size, seq_length, d_model)
              training: Whether in training mode (affects dropout)
          """
          self.inputs = x
          seq_len = x.shape[1]
          
          assert seq_len <= self.max_seq_length, f"Input sequence length {seq_len} exceeds maximum allowed {self.max_seq_length}"
          
          self.output = x + self.pe[:, :seq_len, :]
          
          return self.output
     
     def backward(self, dvalues):
          """
        Backward pass for positional encoding.
        
        Args:
            dvalues: Gradients from the next layer
          """
          self.dinputs = dvalues
          return self.dinputs
     
     def get_parameters(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]
          # NO Trainable params
          return {}
