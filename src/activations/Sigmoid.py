import numpy as np
from core import Activation 

class Activation_Sigmoid(Activation):
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        # Threshold at 0.5 for binary classification
        return (outputs > 0.5) * 1 