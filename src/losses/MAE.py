import numpy as np
from ..core.Loss import Loss

class Loss_MeanAbsoluteError(Loss): # L1 loss
    # forward pass
    def __init__(self, model):
        super().__init__(model)
        
    
    def forward(self, y_pred, y_true):

        
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=tuple(range(1, y_pred.ndim)))
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # calculate gradients
        # d|x| = sign(x) 
        self.dinputs = np.sign(y_true - dvalues)

        # normalize b the number of samples and shape dimentions product
        norm_factor = np.prod(dvalues.shape) / samples
        self.dinputs = self.dinputs / norm_factor