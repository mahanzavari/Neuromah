import unittest
import numpy as np
from Neuromah.src.layers import Layer_Conv2D
class TestLayerConv2D(unittest.TestCase
                      ):
    
    def test_forward_valid_padding(self):
        layer =  Layer_Conv2D(in_channels=3, out_channels=2, kernel_size=3, padding='valid')
        
        inputs = np.random.randn(1, 3, 5, 5)
        
        layer.forward(inputs, training=True)
        
        output_shape = layer.output.shape
        self.assertEqual(output_shape, (1, 2, 3, 3), "Output shape mismatch for 'valid' padding")
    
    def test_forward_same_padding(self):
        layer = Layer_Conv2D(in_channels=3, out_channels=2, kernel_size=3, padding='same')
        
        inputs = np.random.randn(1, 3, 5, 5)
        
        layer.forward(inputs, training=True)
        
        output_shape = layer.output.shape
        self.assertEqual(output_shape, (1, 2, 5, 5), "Output shape mismatch for 'same' padding")
    
    def test_weight_initialization(self):
        layer = Layer_Conv2D(in_channels=3, out_channels=2, kernel_size=3, padding='valid')
        
        self.assertEqual(layer.weights.shape, (2, 3, 3, 3), "Weight shape mismatch")
        self.assertTrue(np.allclose(np.mean(layer.weights), 0, atol=0.1), "Weight initialization is incorrect")
    
    def test_backward(self):
        layer = Layer_Conv2D(in_channels=3, out_channels=2, kernel_size=3, padding='valid')
        
        inputs = np.random.randn(1, 3, 5, 5)
        dvalues = np.random.randn(1, 2, 3, 3)  # gradient of the output
        
        layer.forward(inputs, training=True)
        
        layer.backward(dvalues)
        
        self.assertEqual(layer.weight_gradients.shape, (2, 3, 3, 3), "Weight gradients shape mismatch")
        self.assertEqual(layer.bias_gradients.shape, (2, 1), "Bias gradients shape mismatch")
    
    def test_activation_forward_and_backward(self):
        class MockActivation:
            def forward(self, inputs):
                self.output = np.maximum(0, inputs)  # ReLU activation
            
            def backward(self, dvalues):
                self.dinputs = dvalues * (self.output > 0)  # Derivative of ReLU

        layer =      Layer_Conv2D(in_channels=3, out_channels=2, kernel_size=3, padding='valid', activation=MockActivation())
        
        inputs = np.random.randn(1, 3, 5, 5)
        
        layer.forward(inputs, training=True)
        self.assertTrue(np.all(layer.output >= 0), "Activation function didn't work correctly in forward pass")
        
        dvalues = np.random.randn(1, 2, 3, 3)
        layer.backward(dvalues)
        self.assertTrue(np.all(layer.dinputs >= 0), "Activation function didn't work correctly in backward pass")
    
    def test_invalid_padding(self):
        with self.assertRaises(ValueError):
            layer =      Layer_Conv2D(in_channels=3, out_channels=2, kernel_size=3, padding='invalid')
    
    def test_invalid_kernel_size(self):
        with self.assertRaises(ValueError):
            layer =      Layer_Conv2D(in_channels=3, out_channels=2, kernel_size=(3, -1))
    
    def test_invalid_stride(self):
        with self.assertRaises(ValueError):
            layer =      Layer_Conv2D(in_channels=3, out_channels=2, kernel_size=3, stride=(1, -1))
    
    def test_invalid_channels(self):
        with self.assertRaises(ValueError):
            layer =      Layer_Conv2D(in_channels=-1, out_channels=2, kernel_size=3)
        
        with self.assertRaises(ValueError):
            layer =      Layer_Conv2D(in_channels=3, out_channels=-2, kernel_size=3)

if __name__ == '__main__':
    unittest.main()
