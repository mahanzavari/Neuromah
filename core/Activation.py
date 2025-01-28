class Activation:
    def forward(self, inputs, training):
        """
        Performs the forward pass of the activation function.

        Args:
            inputs: Input values.
            training: Boolean flag indicating if the network is in training mode.

        Raises:
            NotImplementedError: This method must be implemented by derived classes.
        """
        raise NotImplementedError("Derived classes must implement the 'forward' method.")

    def backward(self, dvalues):
        """
        Performs the backward pass (calculates gradients) of the activation function.

        Args:
            dvalues: Gradient of the loss with respect to the output of the activation function.

        Raises:
            NotImplementedError: This method must be implemented by derived classes.
        """
        raise NotImplementedError("Derived classes must implement the 'backward' method.")

    def predictions(self, outputs):
        """
        Returns the prediction from the output of the activation function.

        Args:
            outputs: Output of the activation function.

        Raises:
            NotImplementedError: This method must be implemented by derived classes.
        """
        raise NotImplementedError("Derived classes must implement the 'predictions' method.")