import numpy as np
from ..core.Loss import Loss

class Loss_MeanSquaredError(Loss):  # L2 loss
    
    def __init__(self, model):
        super().__init__(model)

    # forward pass
    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=tuple(range(1 , y_pred.ndim)))
        return sample_losses

    # backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # calculate gradient
        # d(y - x)**2 == -2(y - x)
        self.dinputs = -2 * (y_true - dvalues)
        # dvalues.shape == total number of elemnts
        norm_factor = np.prod(dvalues.shape) / samples
        # normalization
        self.dinputs = self.dinputs / norm_factor
        