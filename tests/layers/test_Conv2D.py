import unittest
import numpy as np
from Neuromah.src.layers.Conv2DBackend import Layer_Conv2D  # Assuming Layer_Conv2D is in this path
from Neuromah.src.activations import Activation_ReLU  # Import ReLU for testing activation


class TestLayer_Conv2D(unittest.TestCase):
     def test_inintialization(self):
          conv = Layer_Conv2D(in_channels= 3 , out_channels = 64 , kernel_size = 3)
          self.assertEqual(conv.stride , (1 , 1))
          self.assertEqual(conv.weights.shape , (64 , 3 , 3 , 3))
          self.assertEqual(conv.biases.shape , (64  , 1))
          self.assertEqual(conv.padding , 'valid')
          self.assertEqual(conv.kernel_size , (3 , 3))
          
          conv_tuple_kernel = Layer_Conv2D(in_channels= 2 , out_channels= 32 , kernel_size= (3 , 3),
                                           stride= (1 , 1) , padding= 'same')
          self.assertEqual(conv_tuple_kernel.kernel_size , (3 , 3))
          self.assertEqual(conv_tuple_kernel.stride , (1 , 1))
          self.assertEqual(conv_tuple_kernel.padding , 'same')
          self.assertEqual(conv_tuple_kernel.weights.shape , (32 , 2 , 3 , 3))
     def test_invalid_init(self):
          
          with self.assertRaisesRegex(ValueError, "in_channels must be a positive integer"):
               Layer_Conv2D(in_channels=0, out_channels=64, kernel_size=3)
          with self.assertRaisesRegex(ValueError, "in_channels must be a positive integer"):
               Layer_Conv2D(in_channels=-1, out_channels=64, kernel_size=3)
          with self.assertRaisesRegex(ValueError, "in_channels must be a positive integer"):
               Layer_Conv2D(in_channels=1.5, out_channels=64, kernel_size=3)
          with self.assertRaisesRegex(ValueError, "out_channels must be a positive integer"):
               Layer_Conv2D(in_channels=3, out_channels=0, kernel_size=3)
          with self.assertRaisesRegex(ValueError, "out_channels must be a positive integer"):
               Layer_Conv2D(in_channels=3, out_channels=-1, kernel_size=3)
          with self.assertRaisesRegex(ValueError, "out_channels must be a positive integer"):
               Layer_Conv2D(in_channels=3, out_channels=1.5, kernel_size=3)

          with self.assertRaisesRegex(ValueError , "kernel_size must be int or tuple of two ints"):
               Layer_Conv2D(in_channels=4 , out_channels= 4 , kernel_size= (3 , 3 , 3))
          with self.assertRaisesRegex(ValueError , "kernel_size must be int or tuple of two ints"):
               Layer_Conv2D(in_channels=4 , out_channels= 4 , kernel_size= [3 , 3])
          with self.assertRaisesRegex(ValueError , "kernel_size must be int or tuple of two ints"):
               Layer_Conv2D(in_channels=4 , out_channels= 4 , kernel_size= "invalid")
               
          with self.assertRaisesRegex(ValueError, "stride must be int or tuple of two ints"):
               Layer_Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=(1, 2, 3))
          with self.assertRaisesRegex(ValueError, "stride must be int or tuple of two ints"):
               Layer_Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride="invalid")

          with self.assertRaisesRegex(ValueError, "padding must be 'valid' or 'same'"):
               Layer_Conv2D(in_channels=3, out_channels=64, kernel_size=3, padding='invalid_padding')
     def test_forward_valid_padding(self):
          conv_valid_padding = Layer_Conv2D(in_channels= 3, out_channels=64 , kernel_size=3 , padding= 'valid')
          inputs = np.random.randn(1 , 3 , 32 , 32) # a 32*32 with bath size of 1 and three channels
          conv_valid_padding.forward(inputs , training=True)
          self.assertEqual(conv_valid_padding.output.shape , (1 , 64 , 30 , 30)) # 32 - (k.h or k,w - 1)
     
     def test_forward_same(self):
          conv_same_padding = Layer_Conv2D(in_channels= 3, out_channels= 32 , kernel_size=(5 , 5),
                                           padding = 'same')
          input_data_same_kernel_5 = np.random.randn(1 , 3 , 32 , 32)
          conv_same_padding.forward(inputs= input_data_same_kernel_5 , training=True)
          self.assertEqual(conv_same_padding.output.shape , (1 , 32 , 32 , 32))
          
     def test_forward_with_activation(self):
          conv_forward_activation = Layer_Conv2D(in_channels= 3, out_channels= 4 , kernel_size=(3 , 3) , activation=Activation_ReLU())
          input_data_forward_activation = np.random.randn(1 , 3 , 32 , 32)
          conv_forward_activation.forward(input_data_forward_activation , training=True)
          self.assertIsNotNone(conv_forward_activation.activation)
          self.assertEqual(conv_forward_activation.output.shape , (1 , 4 , 30 , 30 ))
          self.assertTrue(np.all(conv_forward_activation.output >=0)) # relu output
          
     def test_backward_shape(self):
          conv_layer = Layer_Conv2D(in_channels=3, out_channels=2, kernel_size=3)
          inputs = np.random.randn(1 , 3 , 32 , 32)
          conv_layer.forward(inputs = inputs , training=True)
          dvalues = np.random.randn(1 , 2, 30 , 30)
          
          conv_layer.backward(dvalues= dvalues)
          self.assertEqual(conv_layer.dinputs.shape , inputs.shape)
          self.assertEqual(conv_layer.weight_gradients.shape, conv_layer.weights.shape)
          self.assertEqual(conv_layer.bias_gradients.shape, conv_layer.biases.shape)
          
     def test_get_set_params(self):
          conv_get_set = Layer_Conv2D(in_channels= 3 , out_channels= 4 , kernel_size= (3 , 3))
          initial_weights , initial_biases = conv_get_set.get_parameters()     
          
          new_weights = np.random.randn(*initial_weights.shape)
          new_bias = np.random.randn(*initial_biases.shape)
          
          conv_get_set.set_parameters(new_weights , new_bias)
          curr_weights , curr_biases = conv_get_set.get_parameters()
          
          np.testing.assert_array_equal(curr_weights , new_weights)
          np.testing.assert_array_equal(curr_biases , new_bias)
          
          
          
if __name__ == '__main__':
     
    unittest.main()