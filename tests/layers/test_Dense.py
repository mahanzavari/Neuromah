import numpy as np
from src.layers import Layer_Dense
import unittest


class test_Dense(unittest.TestCase):
     def setUp(self):
          self.n_inputs = 3 
          self.n_neurons = 3
     def test_forward(self):
          inputs = np.array([
               [1 , 2, 3],
               [4 , 5, 6],
               [7 , 8, 9],
          ])
          
          self.layer.forward(inputs)
     
          expected_output = np.array([
               self.weights * input + self.biases
          ])
          
          np.testing.assert_array_equal(inputs , expected_output)