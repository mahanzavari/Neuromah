import numpy as np


class Layer_BatchNormalization2D:
     def __init__(self , epsilon = 1e-5 , momentum = 0.9):
          self.epsilon = epsilon
          