import numpy as np

class Initializer:
     """
     Base class for weight initializers in a neural network library.
 
     This class provides the foundation for different weight initialization strategies.
     Derived classes should implement the `initialize` method to provide specific
     initialization logic. This design allows users to easily switch between
     different initialization methods or create their own custom initializers.
 
     Args:
         xp (module): The array module to use (NumPy or CuPy). This ensures
             compatibility with both CPU and GPU computations. Defaults to NumPy.
 
     Attributes:
         xp (module): The array module (NumPy or CuPy) used for array operations.
 
     Methods:
         initialize(shape): Abstract method that must be implemented by subclasses.
             It takes the shape of the weight matrix as input and returns the
             initialized weight array.
 
     Raises:
         NotImplementedError: If the `initialize` method is not overridden in a
             derived class.
 
     Usage Example (in derived classes):
 
         class MyCustomInitializer(Initializer):
             def initialize(self, shape):
                 # Your custom initialization logic here.
                 # Example: Initialize with a specific constant.
                 return self.xp.full(shape, 0.1)
 
     Why Use Initializers?
 
     Proper weight initialization is crucial for training neural networks effectively.
     It can help prevent issues like vanishing/exploding gradients, speed up
     convergence, and improve the overall performance of the model. Different
     initialization methods are suited for different network architectures and
     activation functions.
 
     How to use with Layers (e.g., Layer_Dense, Layer_Conv2D):
 
     Layers in this library accept an `weight_initializer` parameter during
     construction. You can pass:
 
     1.  An instance of an `Initializer` subclass (e.g., `XavierInitializer()`,
         `HeInitializer()`, or your own custom initializer).
     2.  A string representing a pre-defined initializer (e.g., "xavier", "he").
         This is a convenient shortcut for common initializers.
     3.  `None` (or omit the argument) to use the default initializer
         (typically `RandomNormalInitializer`).
 
     Example:
 
         # Using a pre-defined initializer (string)
         dense_layer = Layer_Dense(..., weight_initializer="xavier")
 
         # Using a specific initializer instance
         he_init = HeInitializer()
         conv_layer = Layer_Conv2D(..., weight_initializer=he_init)
 
         # Using a custom initializer
         class MyInit(Initializer):
             def initialize(self, shape):
                 return self.xp.random.uniform(-0.1, 0.1, size=shape)
 
         my_init = MyInit()
         dense_layer2 = Layer_Dense(..., weight_initializer=my_init)
 
         # Using the default initializer (RandomNormalInitializer):
         dense_layer3 = Layer_Dense(...) # Omit weight_initializer
     """
     def __init__(self , xp = np):
          self.xp = xp
     def initialize(self, shape):
          """
          Initializes weights given a shape.

          Args:
              shape (tuple): Shape of the weight matrix.

          Raises:
              NotImplementedError: This method must be implemented by derived classes.
          """
          raise NotImplementedError("Derived classes must implement the 'initialize' method.")

class RandomNormalInitializer(Initializer):
     """Initializes weights with small random values from a normal distribution."""
     def __init__(self, stddev=0.01 , xp = np):
          super().__init__(xp)
          self.stddev = stddev 
     def initialize(self, shape):
          return self.xp.random.randn(*shape) * self.stddev

class XavierInitializer(Initializer):
     """Initializes weights using the Xavier (Glorot) initialization."""
     def __init__(self , xp = np):
         super().__init__(xp)
     
     def initialize(self, shape):
          if len(shape) == 2:  # Dense layer
               n_input = shape[0]
               n_output = shape[1]
          elif len(shape) == 4:  # Conv2D layer
               n_input = shape[1] * shape[2] * shape[3]  # fan_in
               n_output = shape[0] * shape[2] * shape[3] # fan_out
          else:
               raise ValueError("Unsupported shape for Xavier initialization.")   
          limit = self.xp.sqrt(6.0 / (n_input + n_output))
          return self.xp.random.uniform(-limit, limit, size=shape)

class HeInitializer(Initializer):
     """Initializes weights using the He initialization."""
     def __init__(self, xp = np):
          super().__init__(xp)
     
     def initialize(self, shape):
          if len(shape) == 2:  # Dense Layer
               n_input = shape[0]
          elif len(shape) == 4: # Conv2D
               n_input = shape[1] * shape[2] * shape[3]  # fan_in
          else:
               raise ValueError("Unsupported shape for He initialization.")
          stddev = self.xp.sqrt(2.0 / n_input)
          return self.xp.random.normal(0, stddev, size=shape)

class CustomInitializer(Initializer):
     """Allows the user to provide a custom initialization function."""
     def __init__(self, init_func , xp = np):
          super().__init__(xp)
          if not callable(init_func):
              raise TypeError("init_func must be a callable function.")
          self.init_func = init_func
     def initialize(self, shape):
          try:
               initialized_weights = self.init_func(shape , xp = self.xp)
               if not isinstance(initialized_weights, self.xp.ndarray) or initialized_weights.shape != shape:
                 raise ValueError(f"Custom init_func must return a numpy array of shape {shape}.")
               return initialized_weights
          except Exception as e:
               raise RuntimeError(f"Error in custom init_func: {e}")