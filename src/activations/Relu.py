import numpy as np
from ..core.Activation import Activation 

class Activation_ReLU(Activation):
    # def forward(self, inputs , training: bool):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)


    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0  # Gradient is 0 for negative inputs

    def predictions(self, outputs):
        return outputs  