import unittest
import numpy as np
from Neuromah.src.layers import Layer_MaxPooling2D


class TestLayer_MaxPooling2D(unittest.TestCase):
    def setUp(self):
        # Initialize a 2x2 max pooling layer with stride 2 and 'valid' padding
        self.pool_size = (2, 2)
        self.stride = (2, 2)
        self.padding = 'valid'
        self.layer = Layer_MaxPooling2D(pool_size=self.pool_size, strides=self.stride, padding=self.padding)

    def test_forward_pass(self):
        inputs = np.array([[
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12],
             [13, 14, 15, 16]]
        ]], dtype=np.float32)

        # Perform the forward pass
        self.layer.forward(inputs, training=True)

        expected_output = np.array([[
            [[6, 8],
             [14, 16]]
        ]], dtype=np.float32)

        np.testing.assert_array_equal(self.layer.output, expected_output)

    def test_backward_pass(self):
        inputs = np.array([[
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12],
             [13, 14, 15, 16]]
        ]], dtype=np.float32)

        self.layer.forward(inputs, training=True)

        dvalues = np.array([[
            [[1, 2],
             [3, 4]]
        ]], dtype=np.float32)

        self.layer.backward(dvalues)

        # Expected gradient after backward pass
        expected_dinputs = np.array([[
            [[0, 0, 0, 0],
             [0, 1, 0, 2],
             [0, 0, 0, 0],
             [0, 3, 0, 4]]
        ]], dtype=np.float32)

        np.testing.assert_array_equal(self.layer.dinputs, expected_dinputs)

if __name__ == '__main__':
    unittest.main()
