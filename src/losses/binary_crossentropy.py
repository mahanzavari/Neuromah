import numpy as np
from ..core.Loss import Loss

class Loss_BinaryCrossentropy(Loss):
    
    def __init__(self, model):
        super().__init__(model)
    
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

        # Take the mean over all dimensions except the batch dimension
        sample_losses = np.mean(sample_losses, axis=tuple(range(1, sample_losses.ndim)))

        return sample_losses

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # Clip data to prevent division by 0
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues))

        # Normalize by the number of samples and the product of the shape dimensions
        normalization_factor = np.prod(dvalues.shape) / samples
        self.dinputs = self.dinputs / normalization_factor