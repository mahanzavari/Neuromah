
import numpy as np
from typing import Union, Tuple , Dict
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
        # Parameter validation
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
            self.stride = (stride, stride)
        elif isinstance(stride, tuple) and len(stride) == 2:
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

        # Cache for im2col matrices
        self.col_input = None
        self.col_weights = None

        self.weight_momentums = self.xp.zeros_like(self.weights)
        self.bias_momentums = self.xp.zeros_like(self.biases)

        self.activation = activation

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        self.inputs = inputs  # original inputs
        # print(self.xp , 'mahan')
        batch_size, in_channels, in_h, in_w = inputs.shape

        # Calculate output dimensions
        out_h, out_w, pad_h, pad_w = self._calculate_output_shape(in_h, in_w)

        # Apply padding if needed and store it
        if self.padding == 'same':
            self.inputs_padded = self.xp.pad(inputs,
                                        ((0, 0), (0, 0), (pad_h[0], pad_h[1]), (pad_w[0], pad_w[1])),
                                        mode='constant')
            inputs_to_use = self.inputs_padded
        else:
            inputs_to_use = inputs

        # im2col transformation on padded inputs
        self.col_input = self._im2col(inputs_to_use, self.kernel_size, self.stride)
        col_weights = self.weights.reshape(self.weights.shape[0], -1).T

        # Matrix multiplication
        output = self.col_input @ col_weights + self.biases.T
        self.output = output.reshape(batch_size, out_h, out_w, -1).transpose(0, 3, 1, 2)
        if self.activation:
            self.activation.forward(self.output)
            self.output = self.activation.output


    def backward(self, dvalues: np.ndarray) -> None:
        if self.activation:
            self.activation.backward(dvalues)
            dvalues = self.activation.dinputs

        # Reshape dvalues to match the column matrix format
        batch_size, out_channels, out_h, out_w = dvalues.shape
        dvalues_reshaped = dvalues.transpose(0, 2, 3, 1).reshape(-1, out_channels)

        # Compute gradients of weights
        dweights = self.col_input.T @ dvalues_reshaped
        dweights = dweights.T.reshape(self.weights.shape)

        # Compute gradients of biases
        dbiases = self.xp.sum(dvalues_reshaped, axis=0, keepdims=True).T

        # Compute gradients of inputs
        dinputs_col = dvalues_reshaped @ self.weights.reshape(self.weights.shape[0], -1)

        # Use the padded shape in _col2im if padding was applied
        if self.padding == 'same':
            padded_shape = self.inputs_padded.shape
            dinputs_padded = self._col2im(dinputs_col, padded_shape, self.kernel_size, self.stride)
            # Remove the padding to recover gradients corresponding to the original input
            pad_h, pad_w = self._calculate_padding(self.inputs.shape[2], self.inputs.shape[3])
            dinputs = dinputs_padded[:, :,
                                       pad_h[0]: dinputs_padded.shape[2] - pad_h[1],
                                       pad_w[0]: dinputs_padded.shape[3] - pad_w[1]]
        else:
            dinputs = self._col2im(dinputs_col, self.inputs.shape, self.kernel_size, self.stride)

        # Store gradients
        self.weight_gradients = dweights
        self.bias_gradients = dbiases
        self.dinputs = dinputs

    def _im2col(self, inputs: np.ndarray, kernel_size: Tuple[int, int], stride: Tuple[int, int]) -> np.ndarray:
        batch_size, in_channels, in_h, in_w = inputs.shape
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride

        # Calculate output dimensions
        out_h = (in_h - kernel_h) // stride_h + 1
        out_w = (in_w - kernel_w) // stride_w + 1

        # Use stride tricks to create a view of the input as sliding windows
        shape = (batch_size, in_channels, out_h, out_w, kernel_h, kernel_w)
        strides = (inputs.strides[0], inputs.strides[1],
                   inputs.strides[2] * stride_h, inputs.strides[3] * stride_w,
                   inputs.strides[2], inputs.strides[3])

        strided = self.xp.lib.stride_tricks.as_strided(
            inputs, shape=shape, strides=strides, writeable=False
        )

        # Reshape to (batch_size * out_h * out_w, in_channels * kernel_h * kernel_w)
        col_matrix = strided.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size * out_h * out_w, -1)
        return col_matrix

    def _col2im(self, col_matrix: np.ndarray, input_shape: Tuple[int, int, int, int],
                kernel_size: Tuple[int, int], stride: Tuple[int, int]) -> np.ndarray:
        batch_size, in_channels, in_h, in_w = input_shape
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride

        # Calculate output dimensions used in im2col
        out_h = (in_h - kernel_h) // stride_h + 1
        out_w = (in_w - kernel_w) // stride_w + 1

        # Reshape the column matrix to (batch, out_h, out_w, in_channels, kernel_h, kernel_w)
        col_reshaped = col_matrix.reshape(batch_size, out_h, out_w, in_channels, kernel_h, kernel_w)
        # Permute to (batch, in_channels, out_h, out_w, kernel_h, kernel_w)
        col_reshaped = col_reshaped.transpose(0, 3, 1, 2, 4, 5)

        # Initialize output tensor
        output = self.xp.zeros((batch_size, in_channels, in_h, in_w), dtype=col_matrix.dtype)

        # Iterate through output spatial dimensions
        for b in range(batch_size):
            for c in range(in_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        # Calculate corresponding input indices
                        h_start = oh * stride_h
                        w_start = ow * stride_w
                        # Extract the corresponding kernel values from col_reshaped
                        kernel_values = col_reshaped[b, c, oh, ow] # (kernel_h, kernel_w)
                        # Add these kernel values to the correct input locations
                        output[b, c, h_start:h_start + kernel_h, w_start:w_start + kernel_w] += kernel_values

        return output


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
