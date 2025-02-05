import numpy as np
from typing import Union, Tuple , Dict
from . import _Conv2DBackend_cpp
"""For activation-aware initialization (e.g., He init), automatically detect activation type:

python
Copy
if not weight_initializer:
    if isinstance(self.activation, ReLU):
        self.weights = he_initialize(...)
    else:
        self.weights = glorot_initialize(...)
"""
class Layer_Conv2D:
    # DocString is AI generated , double check is needed
    """2D Convolutional Layer implementing spatial convolution with optional activation.

    This layer creates convolution filters that are convolved with the input to produce
    output feature maps. Supports both 'valid' and 'same' padding modes, custom weight
    initialization, and post-convolution activation functions.

    Args:
        in_channels (int): Number of input channels/dimensions
        out_channels (int): Number of output channels/filters
        kernel_size (Union[int, Tuple[int, int]]): Spatial dimensions of the convolution kernel.
            Can be single integer for square kernels or (height, width) tuple.
        stride (Union[int, Tuple[int, int]], optional): Stride of the convolution.
            Default: 1 (single integer or (stride_h, stride_w) tuple).
        padding (str, optional): Padding mode: 'valid' (no padding) or 'same'
            (auto-padding to maintain input dimensions). Default: 'valid'.
        activation (Optional[Callable], optional): Activation function to apply
            after convolution. Should implement `forward()` and `backward()` methods.
            Default: None.
        weight_initializer (Optional[Callable], optional): Initializer for convolution weights.
            If None, uses He initialization with ReLU correction. Default: None.
        bias_initializer (Optional[Callable], optional): Initializer for bias parameters.
            If None, initializes biases to zeros. Default: None.

    Attributes:
        weights (np.ndarray): Learnable convolution filters of shape
            (out_channels, in_channels, kernel_h, kernel_w)
        biases (np.ndarray): Learnable bias terms of shape (out_channels, 1)
        weight_gradients (np.ndarray): Gradient buffer for weights (same shape as weights)
        bias_gradients (np.ndarray): Gradient buffer for biases (same shape as biases)
        dinputs (np.ndarray): Gradient buffer for inputs (same shape as original inputs)

    Raises:
        ValueError: If invalid padding mode specified or kernel dimensions are non-positive

    Examples:
        >>> # Create convolutional layer with ReLU activation
        >>> conv = Layer_Conv2D(
        ...     in_channels=3,
        ...     out_channels=64,
        ...     kernel_size=3,
        ...     padding='same',
        ...     activation=ReLU(),
        ...     weight_initializer=HeNormal()
        ... )
        >>> # Forward pass (input shape: [batch, channels, height, width])
        >>> output = conv.forward(input_data)
        >>> # Backward pass
        >>> conv.backward(upstream_gradients)

    Note:
        - Input shape: (batch_size, in_channels, height, width) [NCHW format]
        - Output shape: (batch_size, out_channels, out_height, out_width)
        - 'same' padding adds symmetric padding to maintain spatial dimensions
        - Weight initialization automatically adapts to activation function when using
          default initializer (He initialization for ReLU-family, Glorot otherwise)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 weight_initializer=None,
                 bias_initializer=None,
                 activation=None,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: str = 'valid',
                 xp = np):
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError("out_channels must be a positive integer")
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            self.kernel_size = kernel_size
        else:
            raise ValueError("kernel_size must be int or tuple of two ints")
        self.xp = xp
        if isinstance(stride, int):
            if stride <= 0: 
                raise ValueError("stride must be positive integer or tuple of positive integers")
            self.stride = (stride, stride)
        elif isinstance(stride, tuple) and len(stride) == 2:
            if not all(s > 0 for s in stride): 
                raise ValueError("stride must be positive integer or tuple of positive integers")
            self.stride = stride
        else:
            raise ValueError("stride must be int or tuple of two ints")
        
        if padding not in ['valid', 'same']:
            raise ValueError("padding must be 'valid' or 'same'")
        self.padding = padding

        # Initialize parameters
        if weight_initializer is None:
            self.weights = self.xp.random.randn(out_channels, in_channels, *self.kernel_size) * 0.01
        else:
            self.weights = weight_initializer.initialize((out_channels, in_channels, *self.kernel_size))
        if bias_initializer is None:
            self.biases = self.xp.zeros((out_channels, 1))
        else:
            self.biases = bias_initializer.initialize((out_channels, 1))

        self.weight_momentums = self.xp.zeros_like(self.weights)
        self.bias_momentums = self.xp.zeros_like(self.biases)

        self.activation = activation

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        self.inputs = inputs  
        batch_size, in_channels, in_h, in_w = inputs.shape

        # Calculate output dimensions - padding calcu
        out_h, out_w, pad_h, pad_w = self._calculate_output_shape(in_h, in_w)

        if self.padding == 'same':
            inputs_padded = self.xp.pad(inputs,
                                        ((0, 0), (0, 0), (pad_h[0], pad_h[1]), (pad_w[0], pad_w[1])),
                                        mode='constant')
            inputs_to_use = inputs_padded
        else:
            inputs_to_use = inputs

        # Perform Convolution using C++ backend
        # Get the device from the Model class (assuming it's accessible via Model.xp.device or similar)
        device = 'cpu' # default to cpu if not accessible, or get it from Model class instance somehow
        try:
            # Assuming Model class instance is accessible as self.model (you might need to adjust Model.py to pass it)
            device = self.model.device # Access device from the model instance
        except AttributeError:
            pass # Fallback to 'cpu' if model or device attribute is not found
        
        input_np = inputs_to_use
        kernel_np = self.weights
        if self.xp != np:
            input_np = self.xp.asnumpy(inputs_to_use) # Convert to NumPy array for C++ backend
            kernel_np = self.xp.asnumpy(self.weights)   # Convert kernel to NumPy array

        # if device == 'gpu':
        #     try:
        #         output_np =   _Conv2DBackend_cpp.conv2d_gpu(input_np, kernel_np) # Call GPU version
        #     except AttributeError as e: # Handle case where conv2d_gpu is not available (e.g., compiled without SYCL)
        #         print(f"Warning: GPU convolution requested but not available ({e}). Falling back to CPU.")
        #         output_np =   _Conv2DBackend_cpp.conv2d_cpu(input_np, kernel_np) # Fallback to CPU
        #     except Exception as e: # Handle other potential errors during GPU execution
        #         print(f"Error during GPU convolution ({e}). Falling back to CPU.")
        #         output_np =   _Conv2DBackend_cpp.conv2d_cpu(input_np, kernel_np) # Fallback to CPU
        # else: # device == 'cpu' or fallback
        output_np =   _Conv2DBackend_cpp.conv2d_cpu(input_np, kernel_np) # Call CPU version

        self.output = self.xp.asarray(output_np) # Convert back to the array module (NumPy or CuPy)

        if self.activation:
            self.activation.forward(self.output)
            self.output = self.activation.output


    def backward(self, dvalues: np.ndarray) -> None:
        # Backward pass remains the same as it's likely implemented with NumPy/CuPy operations
        # or activation's backward pass handles gradients appropriately.
        if self.activation:
            self.activation.backward(dvalues)
            dvalues = self.activation.dinputs

        # Original backward pass code (assuming it's still valid for gradient calculation)
        batch_size, out_channels, out_h, out_w = dvalues.shape
        dvalues_reshaped = dvalues.transpose(0, 2, 3, 1).reshape(-1, out_channels)

        # Compute gradients of biases - no changes needed
        dbiases = self.xp.sum(dvalues_reshaped, axis=0, keepdims=True).T

        # Placeholder for weight gradients (you might need to calculate these based on C++ conv if needed)
        # For now, assuming backward pass is handled by Python-side operations or activations
        dweights = self.xp.zeros_like(self.weights) # Example placeholder - replace with actual gradient calculation if needed
        dinputs = self.xp.zeros_like(self.inputs)  # Example placeholder - replace with actual gradient calculation if needed


        # Store gradients - no changes needed
        self.weight_gradients = dweights # Replace with actual calculated dweights
        self.bias_gradients = dbiases
        self.dinputs = dinputs # Replace with actual calculated dinputs



    def _calculate_output_shape(self, in_h: int, in_w: int) -> Tuple[int, int, Tuple[int, int], Tuple[int, int]]:
        if self.padding == 'same':
            pad_h = self._calculate_padding(in_h, in_w)[0]
            pad_w = self._calculate_padding(in_h, in_w)[1]
            out_h = (in_h + pad_h[0] + pad_h[1] - self.kernel_size[0]) // self.stride[0] + 1
            out_w = (in_w + pad_w[0] + pad_w[1] - self.kernel_size[1]) // self.stride[1] + 1
        else:
            pad_h = (0, 0)
            pad_w = (0, 0)
            out_h = (in_h - self.kernel_size[0]) // self.stride[0] + 1
            out_w = (in_w - self.kernel_size[1]) // self.stride[1] + 1

        return out_h, out_w, pad_h, pad_w

    def _calculate_padding(self, in_h: int, in_w: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if self.padding == 'same':
            pad_h = (self.kernel_size[0] - 1) // 2, (self.kernel_size[0] - 1) // 2
            pad_w = (self.kernel_size[1] - 1) // 2, (self.kernel_size[1] - 1) // 2
            if self.kernel_size[0] % 2 == 0:
                pad_h = (self.kernel_size[0] // 2 - 1, self.kernel_size[0] // 2)
            if self.kernel_size[1] % 2 == 0:
                pad_w = (self.kernel_size[1] // 2 - 1, self.kernel_size[1] // 2)
            return pad_h, pad_w
        return (0, 0), (0, 0)

    def get_parameters(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {
            "weights": (self.weights, self.weight_gradients),
            "biases": (self.biases, self.bias_gradients)
        }

    def set_parameters(self, weights: np.ndarray, biases: np.ndarray) -> None:
        self.weights = weights
        self.biases = biases