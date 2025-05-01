from typing import Dict, Tuple, Optional, Union
from ...initializers import Initializer
from ...initializers.Initializer import RandomNormalInitializer, XavierInitializer, HeInitializer
from ..base import BaseLayer
import cupy as cp

class Convolutional(BaseLayer):
    """
    A class representing a 2D Convolutional layer in a neural network.
    This is a wrapper class that can use either CPU or GPU implementation.

    Parameters
    ----------
    input_channels : int
        The number of input channels.
    output_channels : int
        The number of output channels.
    kernel_size : int
        The size of the convolution kernel.
    stride : int
        The stride of the convolution. Defaults to 1.
    padding : int
        The padding size. Defaults to 0.
    weight_initializer : Optional[Union[object, str]]
        An initializer object to initialize the weights. Defaults to None.
    bias_initializer : Optional[Union[object, str]]
        An initializer object to initialize the biases. Defaults to None.
    use_gpu : bool
        Whether to use GPU implementation. Defaults to False.
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        weight_initializer: Optional[Union[object, str]] = None,
        bias_initializer: Optional[Union[object, str]] = None,
        use_gpu: bool = False
    ):
        super().__init__(use_gpu)
        
        # Import the appropriate implementation
        if use_gpu:
            try:
                from ...cuda.convolution_cuda import Conv2DCUDA
                self.impl = Conv2DCUDA(
                    input_channels=input_channels,
                    output_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    weight_initializer=weight_initializer,
                    bias_initializer=bias_initializer
                )
            except ImportError:
                raise ImportError("CUDA implementation not available. Please install cupy and ensure CUDA is properly set up.")
        else:
            # Use CPU implementation
            self.input_channels = input_channels
            self.output_channels = output_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            
            if weight_initializer is None:
                self.weight_initializer = RandomNormalInitializer(xp=self.xp)
            elif isinstance(weight_initializer, str):
                if weight_initializer.lower() == 'xavier':
                    self.weight_initializer = XavierInitializer(xp=self.xp)
                elif weight_initializer.lower() == "he":
                    self.weight_initializer = HeInitializer(xp=self.xp)
                else:
                    raise ValueError(f"Unknown initializer: {weight_initializer}")
            elif isinstance(weight_initializer, Initializer):
                self.weight_initializer = weight_initializer
            else:
                raise ValueError("weight_initializer must be a string or an Initializer instance")
                
            self.weights = self.weight_initializer.initialize(
                (output_channels, input_channels, kernel_size, kernel_size)
            )
            self.biases = self.xp.zeros((output_channels, 1))

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Compute the forward pass of the layer.
        
        Args:
            inputs (np.ndarray): Input data of shape (batch_size, channels, height, width).
            training (bool): Whether the layer is in training mode.
        """
        if self.use_gpu:
            self.impl.forward(inputs, training)
            self.output = self.impl.output
        else:
            if not isinstance(inputs, self.xp.ndarray):
                inputs = self.xp.asarray(inputs)
                
            self.inputs = inputs
            batch_size = inputs.shape[0]
            input_height = inputs.shape[2]
            input_width = inputs.shape[3]
            
            output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
            output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1
            
            self.output = self.xp.zeros((batch_size, self.output_channels, output_height, output_width))
            
            # Apply padding if needed
            if self.padding > 0:
                inputs_padded = self.xp.pad(
                    inputs,
                    ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                    mode='constant'
                )
            else:
                inputs_padded = inputs
                
            # Perform convolution
            for b in range(batch_size):
                for c_out in range(self.output_channels):
                    for h_out in range(output_height):
                        for w_out in range(output_width):
                            h_start = h_out * self.stride
                            w_start = w_out * self.stride
                            
                            # Extract the input patch
                            patch = inputs_padded[b, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                            
                            # Compute the convolution
                            self.output[b, c_out, h_out, w_out] = self.xp.sum(
                                patch * self.weights[c_out]
                            ) + self.biases[c_out]

    def backward(self, dvalues: np.ndarray) -> None:
        """
        Compute the backward pass of the layer.
        
        Args:
            dvalues (np.ndarray): Gradient of the loss with respect to the layer's output.
        """
        if self.use_gpu:
            self.impl.backward(dvalues)
            self.dinputs = self.impl.dinputs
            self.dweights = self.impl.dweights
            self.dbiases = self.impl.dbiases
        else:
            if not isinstance(dvalues, self.xp.ndarray):
                dvalues = self.xp.asarray(dvalues)
                
            batch_size = dvalues.shape[0]
            input_height = self.inputs.shape[2]
            input_width = self.inputs.shape[3]
            
            # Initialize gradients
            self.dweights = self.xp.zeros_like(self.weights)
            self.dbiases = self.xp.sum(dvalues, axis=(0, 2, 3), keepdims=True)
            self.dinputs = self.xp.zeros_like(self.inputs)
            
            # Apply padding if needed
            if self.padding > 0:
                inputs_padded = self.xp.pad(
                    self.inputs,
                    ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                    mode='constant'
                )
                dinputs_padded = self.xp.zeros_like(inputs_padded)
            else:
                inputs_padded = self.inputs
                dinputs_padded = self.dinputs
                
            # Compute gradients
            for b in range(batch_size):
                for c_out in range(self.output_channels):
                    for h_out in range(dvalues.shape[2]):
                        for w_out in range(dvalues.shape[3]):
                            h_start = h_out * self.stride
                            w_start = w_out * self.stride
                            
                            # Extract the input patch
                            patch = inputs_padded[b, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                            
                            # Update weight gradients
                            self.dweights[c_out] += patch * dvalues[b, c_out, h_out, w_out]
                            
                            # Update input gradients
                            dinputs_padded[b, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size] += (
                                self.weights[c_out] * dvalues[b, c_out, h_out, w_out]
                            )
            
            # Remove padding from input gradients if needed
            if self.padding > 0:
                self.dinputs = dinputs_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
            else:
                self.dinputs = dinputs_padded

    def get_parameters(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get the layer's parameters and their gradients."""
        if self.use_gpu:
            return self.impl.get_parameters()
        return {
            'weights': (self.weights, self.dweights),
            'biases': (self.biases, self.dbiases)
        }

    def set_parameters(self, weights: np.ndarray, biases: np.ndarray) -> None:
        """Set the layer's parameters."""
        if self.use_gpu:
            self.impl.set_parameters(weights, biases)
        else:
            if weights.shape != self.weights.shape:
                raise ValueError(f"Expected weights shape {self.weights.shape}, got {weights.shape}")
            if biases.shape != self.biases.shape:
                raise ValueError(f"Expected biases shape {self.biases.shape}, got {biases.shape}")
            self.weights = self.xp.asarray(weights)
            self.biases = self.xp.asarray(biases) 