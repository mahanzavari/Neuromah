import numpy as np
from ..core.Accuracy import Accuracy

class Accuracy_Categorical(Accuracy):
     def __init__(self, * , binary = False , model):
          super().__init__(model)
          self.binary = binary
          
     def init(self , y):
          pass
     

     def comare(self , predictions , y):
          if not self.binary and len(y.shape) == 2:
               y = np.argmax(y , axis = 1)
          return np.equal(predictions , y)