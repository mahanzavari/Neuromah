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
            
            from ...activations import Activation_Softmax
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
      

class MultiHeadAttention:
    """
    Multi-Head Attention as described in 'Attention Is All You Need', by Google, 2017.
    
    Runs multiple attention heads in parallel and concatenates the results.
    Each head focuses on different parts of the input.
    
    Parameters:
        d_model (int): The model dimension
        num_heads (int): Number of attention heads
        mask (Optional[np.ndarray]): Optional mask for masked attention (for decoder)
    """
    def __init__(self, d_model: int, num_heads: int, mask: Optional[np.ndarray] = None, xp = np):
        from ..dense import Dense
        self.xp = xp
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_model must be divisable by num_heads" 

        self.depth = d_model // num_heads
        # Linear Projections for Q, K and V
        self.wq = Dense(d_model, d_model, xp = xp)
        self.wk = Dense(d_model, d_model, xp = xp)
        self.wv = Dense(d_model, d_model, xp = xp)
        # linear projection for output
        self.dense = Dense(d_model, d_model, xp = xp)
        # Attention
        self.attention = ScaledDotProductAttention(mask = mask, xp = xp)
    
    def _split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth)
        and transpose to (batch_size, num_heads, seq_len, depth)
        """
        x = x.reshape(batch_size, -1 , self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)
    def _combine_heads(self, x, batch_size):
        """
        Transpose from (batch_size, num_heads, seq_len, depth)
        to (batch_size, seq_len, d_model)
        """
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, -1, self.d_model)
    
    def forward(self, q, k, v, training = False):
        """
        Forward pass for multi-head attention.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len_q, d_model)
            k: Key tensor of shape (batch_size, seq_len_k, d_model)
            v: Value tensor of shape (batch_size, seq_len_v, d_model)
            training: Whether in training mode
            
        Returns:
            Output of shape (batch_size, seq_len_q, d_model)
        """
        self.q_in = q
        self.k_in = k
        self.v_in = v
        batch_size = q.shape[0]
        # Linear Projections
        self.wq.forward(q, training=training)
        self.wk.forward(k, training=training)
        self.wv.forward(v, training=training)
        # Attention on all proj vectors in batch
        self.attention.forward(q, k, v, training)
        # Head Calculation and Final Linear proj
        concat_attention = self._combine_heads(self.attention.output, batch_size)
        self.dense.forward(concat_attention, training)
        self.output = self.dense.ouptut
        
        return self.output
    def backward(self, dvalues):
        """
        Backward pass for multi-head attention.
        
        Args:
            dvalues: Gradient of loss with respect to output
            
        Returns:
            Gradients with respect to q, k, and v
        """
        self.dense.backward(dvalues)
        dconcatenated = self.dense.dinputs
        
        batch_size = dconcatenated.shape[0]        
        # split concat grads into heads
        d_concat_attention = self._split_heads(dconcatenated, batch_size)
        self.attention.backward(d_concat_attention)
        dq_heads, dk_heads, dv_heads = (
          self.attention.dqueries,
          self.attention.dkeys,
          self.attention.dvaleus
        )
        # combine heads back
        dq = self._combine_heads(dq_heads, batch_size)
        dk = self._combine_heads(dk_heads, batch_size)
        dv = self._combine_heads(dv_heads, batch_size)
        # Linear Proj backwrd pass
        self.wq.backward(dq)
        self.wk.backward(dk)
        self.wv.backward(dv)
        # Grads wrt inputs
        self.dinputs_q = self.wq.dinputs
        self.dinputs_k = self.wk.dinputs
        self.dinputs_v = self.wv.dinputs
        
        return self.dinputs_q, self.dinputs_k, self.dinputs_v
        
    def get_parameters(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        params = {}
        
        # Get parameters from WQ
        wq_params = self.wq.get_parameters()
        for name, (param, grad) in wq_params.items():
            params[f'wq_{name}'] = (param, grad)
        
        # Get parameters from WK
        wk_params = self.wk.get_parameters()
        for name, (param, grad) in wk_params.items():
            params[f'wk_{name}'] = (param, grad)
        
        # Get parameters from WV
        wv_params = self.wv.get_parameters()
        for name, (param, grad) in wv_params.items():
            params[f'wv_{name}'] = (param, grad)
        
        # Get parameters from dense output layer
        dense_params = self.dense.get_parameters()
        for name, (param, grad) in dense_params.items():
            params[f'dense_{name}'] = (param, grad)
        
        return params
      
      
    def set_parameters(self, **kwargs) -> None:
        """
        Set parameters for all sub-layers.
        
        Expected parameter names:
        - wq_weights, wq_biases
        - wk_weights, wk_biases
        - wv_weights, wv_biases
        - dense_weights, dense_biases
        """
        # Extract and set parameters for WQ
        if 'wq_weights' in kwargs and 'wq_biases' in kwargs:
            self.wq.set_parameters(kwargs['wq_weights'], kwargs['wq_biases'])
        
        # Extract and set parameters for WK
        if 'wk_weights' in kwargs and 'wk_biases' in kwargs:
            self.wk.set_parameters(kwargs['wk_weights'], kwargs['wk_biases'])
        
        # Extract and set parameters for WV
        if 'wv_weights' in kwargs and 'wv_biases' in kwargs:
            self.wv.set_parameters(kwargs['wv_weights'], kwargs['wv_biases'])
        
        # Extract and set parameters for dense output layer
        if 'dense_weights' in kwargs and 'dense_biases' in kwargs:
            self.dense.set_parameters(kwargs['dense_weights'], kwargs['dense_biases'])   