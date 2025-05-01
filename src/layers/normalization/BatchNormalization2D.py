import numpy as np

class Layer_BatchNormalization:
     def __init__(self, epsilon=1e-5, momentum=0.9):
          """
          Initializes a Batch Normalization layer.
  
          Args:
              epsilon (float): Small constant to prevent division by zero.
              momentum (float): Momentum for the moving average of mean and variance.
          """
          self.epsilon = epsilon
          self.momentum = momentum
          self.gamma = None  
          self.beta = None   
  
          # Moving average of mean and variance (for inference)
          self.moving_mean = None  
          self.moving_variance = None  
 
     def init_params(self, input_shape):
          # Initialize gamma and beta, and the moving average of mean and variance
          if len(input_shape) == 2:  # for Dense layer
              self.gamma = np.ones((1, input_shape[1]))
              self.beta = np.zeros((1, input_shape[1]))
              self.moving_mean = np.zeros((1, input_shape[1]))
              self.moving_variance = np.ones((1, input_shape[1]))
          elif len(input_shape) == 4:  # for Conv2D
              self.gamma = np.ones((1, input_shape[1], 1, 1))
              self.beta = np.zeros((1, input_shape[1], 1, 1))
              self.moving_mean = np.zeros((1, input_shape[1], 1, 1))
              self.moving_variance = np.ones((1, input_shape[1], 1, 1))
          else:
              raise ValueError("Unsupported input shape for Batch Normalization.")
  
     def forward(self, inputs, training):
          """
          Performs the forward pass of batch normalization.
  
          Args:
             inputs (numpy.ndarray): Input data.
             training (bool): Whether the layer is in training mode.
          """
          self.inputs = inputs
  
          if self.gamma is None:
              self.init_params(inputs.shape)
  
          if training:
              # Calculate batch mean and variance
              self.mean = np.mean(inputs, axis=0, keepdims=True)
              self.variance = np.var(inputs, axis=0, keepdims=True)
  
              # Update moving average of mean and variance
              self.moving_mean = (
                  self.momentum * self.moving_mean + (1 - self.momentum) * self.mean
              )
              self.moving_variance = (
                  self.momentum * self.moving_variance
                  + (1 - self.momentum) * self.variance
              )
  
              # Normalize
              self.inputs_normalized = (inputs - self.mean) / np.sqrt(self.variance + self.epsilon)
          else:
              # Normalize using moving average (inference)
              self.inputs_normalized = (inputs - self.moving_mean) / np.sqrt(
                  self.moving_variance + self.epsilon
              )
  
          # Scale and shift
          self.output = self.gamma * self.inputs_normalized + self.beta
  
     def backward(self, dvalues):
          """
          Performs the backward pass of batch normalization.
  
          Args:
             dvalues (numpy.ndarray): Gradients from the next layer.
          """
          if len(self.inputs.shape) == 2:
              # For fully connected layers
              num_examples = dvalues.shape[0]
  
              # Gradient wrt normalized inputs
              dinputs_normalized = dvalues * self.gamma
  
              # Gradients wrt gamma and beta
              self.dgamma = np.sum(dvalues * self.inputs_normalized, axis=0, keepdims=True)
              self.dbeta = np.sum(dvalues, axis=0, keepdims=True)
  
              # Gradients wrt inputs
              dvariance = np.sum(
                  dinputs_normalized * (self.inputs - self.mean) * -0.5 * (self.variance + self.epsilon) ** (-1.5),
                  axis=0,
                  keepdims=True,
              )
              dmean = np.sum(
                  dinputs_normalized * -1 / np.sqrt(self.variance + self.epsilon), axis=0, keepdims=True
              ) + dvariance * np.mean(-2 * (self.inputs - self.mean), axis=0, keepdims=True)
  
              self.dinputs = (
                  dinputs_normalized / np.sqrt(self.variance + self.epsilon)
                  + dvariance * 2 * (self.inputs - self.mean) / num_examples
                  + dmean / num_examples
              )
          elif len(self.inputs.shape) == 4:
              # For convolutional layers
              num_examples = np.prod(self.inputs.shape) / self.inputs.shape[1]
  
              # Gradient wrt normalized inputs
              dinputs_normalized = dvalues * self.gamma
  
              # Gradients wrt gamma and beta
              self.dgamma = np.sum(dvalues * self.inputs_normalized, axis=(0, 2, 3), keepdims=True)
              self.dbeta = np.sum(dvalues, axis=(0, 2, 3), keepdims=True)
  
              # Gradients wrt inputs
              dvariance = np.sum(
                  dinputs_normalized
                  * (self.inputs - self.mean)
                  * -0.5
                  * (self.variance + self.epsilon) ** (-1.5),
                  axis=(0, 2, 3),
                  keepdims=True,
              )
              dmean = np.sum(
                  dinputs_normalized * -1 / np.sqrt(self.variance + self.epsilon),
                  axis=(0, 2, 3),
                  keepdims=True,
              ) + dvariance * np.mean(
                  -2 * (self.inputs - self.mean), axis=(0, 2, 3), keepdims=True
              )
  
              self.dinputs = (
                  dinputs_normalized / np.sqrt(self.variance + self.epsilon)
                  + dvariance * 2 * (self.inputs - self.mean) / num_examples
                  + dmean / num_examples
              )
          else:
              raise ValueError("Unsupported input shape for Batch Normalization backward pass.")
  
     def get_parameters(self):
          return self.gamma, self.beta
  
     def set_parameters(self, gamma, beta):
          self.gamma = gamma
          self.beta = beta