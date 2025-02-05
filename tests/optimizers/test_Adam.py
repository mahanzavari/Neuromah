import unittest
import numpy as np
from typing import Dict
from Neuromah.src.optimizers.Adam import Optimizer_Adam
class MockLayer:
     def __init__(self , parameters : Dict):
          self.parameters
          
     def get_parameters(self):
          return self.get_parameters
     
class test_OptimizerAdam(unittest.TestCase):
     def test_initilizations(self):
          optimizer = Optimizer_Adam()
          self.assertEqual(optimizer.learning_rate  , 0.001)
          self.assertEqual(optimizer.decay , 0.0)
          self.assertEqual(optimizer.beta_1 ,  0.9)
          self.assertEqual(optimizer.beta_2 , 0.999)
          self.assertEqual(optimizer.epsilon , 1e-7)
          self.assertEqual(optimizer.iterations , 0)
          
     def test_initializations_custom_values(self):
          optimizer = Optimizer_Adam(learning_rate= 0.2 , epsilon= 1e-5,
                                     beta_1= 0.8 , beta_2= 0.888 , decay= 0.5)
          self.assertEqual(optimizer.learning_rate , 0.2)
          self.assertEqual(optimizer.beta_1 , 0.8)
          self.assertEqual(optimizer.beta_2 , 0.888)
          self.assertEqual(optimizer.epsilon , 1e-5)
          self.assertEqual(optimizer.decay , 0.5)
     
     def test_invalid_betas(self):
          with self.assertRaises(ValueError):
               Optimizer_Adam(beta_1= 1.2 , beta_2= 0.5)
          with self.assertRaises(ValueError):
               Optimizer_Adam(beta_1= 0.8 , beta_2= -0.5)
               