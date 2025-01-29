import numpy as np

class Layer_Dropout:
    def __init__(self, rate):
        """
        Initializes a Dropout layer.

        Args:
            rate (float): Probability of setting a neuron to zero (dropout rate).
        """
        # Probability of keeping a unit active. Higher = less dropout.
        self.rate = 1 - rate

    def forward(self, inputs, training):
        """
        Performs the forward pass of dropout.

        Args:
            inputs (numpy.ndarray): Input data.
            training (bool): Whether the layer is in training mode.
        """
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return
        # Generate a mask of units to keep
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply the mask
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        """
        Performs the backward pass of dropout.

        Args:
            dvalues (numpy.ndarray): Gradients from the next layer.
        """
        # Gradient is scaled by the mask
        self.dinputs = dvalues * self.binary_mask