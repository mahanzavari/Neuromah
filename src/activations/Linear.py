import numpy as np
from ..core.Activation import Activation  

class Activation_Linear(Activation):
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        # Derivative is 1
        self.dinputs = dvalues.copy()  

    def predictions(self, outputs):
        return outputs  