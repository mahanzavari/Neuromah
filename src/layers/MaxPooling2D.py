import numpy as np

class Layer_MaxPooling2D:
     def __init__(self , pool_size , strides = None , padding = 'valid'):
          """
        Initializes a 2D max pooling layer.

        Args:
            pool_size (int or tuple): Size of the pooling window (e.g., 2 for a 2x2 window).
            stride (int or tuple): Stride of the pooling operation. If None, defaults to pool_size.
            padding (str): "valid" or "same" padding.
          """
          self.pool_size = (
               pool_size if isinstance(pool_size , tuple) else (pool_size, pool_size)
          )
          self.stride = (
            strides if isinstance(strides, tuple) else (strides, strides)
          )
          self.padding = padding
          
          
     def forward(self , inputs , training):
          """
          Performs the forward pass of the max pooling op.

          Args:
              inputs (np.ndarray): input data of shape (batch_size , in_channels , in_height , in_width)
              training (bool)):  whether the layer is in the training mode
          """
          self.inputs = inputs
          batch_size , in_channels , in_height , in_width = inputs.shape
          # calculate output shape 
          if self.padding == 'same':
               out_height = int(np.ceil(in_height / self.stride[0]))
               out_width = int(np.ceil(in_width / self.stride[1]))
          else: # 'valid'
               out_height = (in_height - self.pool_size[0]) // self.stride[0] + 1
               out_width = (in_width - self.pool_size[1]) // self.stride[1] + 1
          # padding
          padded_inputs = inputs 
          if self.padding == 'same':
               pad_height = (out_height - 1) * self.stride[0] + self.pool_size[0] - in_height 
               pad_width = (out_width - 1) * self.stride[1] + self.pool_size[1] - in_width
               pad_top = pad_height // 2
               pad_bottom = pad_height - pad_top
               pad_left = pad_width // 2
               pad_right = pad_width = pad_left
               
               padded_inputs = np.pad(
                inputs,
                ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
               )
               # update input shape with padding
               batch_size , in_channels , in_height , in_width = padded_inputs.shape
          
          self.output = np.zeros((batch_size , in_channels , out_height , out_width))
          self.max_indices = np.zeros((batch_size , in_channels , out_height , out_width , 2) , dtype='int')
             
          for b in range(batch_size):
               for c in range(in_channels):
                    for h in range(out_height):
                         for w in range(out_width):
                              h_start = h * self.stride[0]
                              h_end = h_start + self.pool_size[0]
                              w_start = w * self.stride[1]
                              w_end = w_start + self.pool_size[1]
                              pool_region = padded_inputs[b, c, h_start:h_end, w_start:w_end]
                              self.output[b, c, h, w] = np.max(pool_region)
                              max_index = np.unravel_index(
                                  np.argmax(pool_region), pool_region.shape
                              )
                              self.max_indices[b, c, h, w] = (
                                  h_start + max_index[0],
                                  w_start + max_index[1],
                              )   
     def backward(self , dvalues):
          """
        Performs the backward pass of the max pooling operation.

        Args:
            dvalues (numpy.ndarray): Gradients from the next layer, shape (batch_size, in_channels, out_height, out_width).
          """
          batch_size , in_channels , in_height , in_width = self.inputs.shape
          _, _, out_height, out_width = dvalues.shape 
          
          self.dinputs = np.zeros_like(self.inputs)
          if self.padding == "same":
               pad_height = (out_height - 1) * self.stride[0] + self.pool_size[0] - in_height
               pad_width = (out_width - 1) * self.stride[1] + self.pool_size[1] - in_width
               pad_top = pad_height // 2
               pad_bottom = pad_height - pad_top
               pad_left = pad_width // 2
               pad_right = pad_width - pad_left   
               # Initialize a zero-padded dinputs for the padded input shape
               self.dinputs = np.zeros((batch_size, in_channels, in_height + pad_top + pad_bottom, in_width + pad_left + pad_right))

          for b in range(batch_size):
               for c in range(in_channels):
                    for h in range(out_height):
                         for w in range(out_width):
                              max_index = self.max_indices[b, c, h, w]
                              self.dinputs[b, c, max_index[0], max_index[1]] += dvalues[b, c, h, w]

        # Remove padding from dinputs
          if self.padding == "same":
               self.dinputs = self.dinputs[
                   :, :, pad_top:-pad_bottom, pad_left:-pad_right
               ]
          
          