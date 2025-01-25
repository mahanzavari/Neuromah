import numpy as np

class Layer_Conv2D:
     def __init__(self , in_channels , out_channels , kernel_size , strides = 1 , padding = 'valid'):
          self.out_channels = out_channels
          self.kernel_size = kernel_size
          self.strides = strides
          self.padding = padding # either valid or same
          
          #initializations
          self.weights = np.random.randn(out_channels , in_channels , kernel_size , kernel_size * 1e-2)
          self.biases = np.zeros((out_channels , 1))
          
          self.weight_momentums = np.zeros_like(self.weights)
          self.bias_momentums = np.zeros_like(self.biases)
          
     def forward(self , inputs , training):
          self.inputs = inputs
          batch_size , in_channels , in_height , in_width = inputs.shape
          # what are these?
          out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1  # Output size calculation
          out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1
          if self.padding:
               if self.padding != 'valid' or self.padding != 'same':
                    raise ValueError("incorrect value for padding\nuse 'same' or 'valid' ")
          if self.padding == 'same':
               pad_height =  max((out_height - 1) * self.strides + self.kernel_size - in_height , 0)
          