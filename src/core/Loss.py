import numpy as np
from typing import Dict

class Loss:
    def __init__(self , model):
        self.model = model

    def regularization_loss(self):
        regularization_loss = 0
        for layer in self.model.trainable_layers:
            # L1 regularization - weights
            # calculate only when factor greater than 0
            if hasattr(layer ,'weight_regularizer_l1') and layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                    np.sum(np.abs(layer.weights))
            # L2 regularization - weights
            if hasattr(layer , 'weight_regularizer_l1') and layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                       np.sum(layer.weights * layer.weights)
                


            # # L1 regularization - biases
            # # calculate only when factor greater than 0
            # if layer.bias_regularizer_l1 > 0:
            #     regularization_loss += layer.bias_regularizer_l1 * \
            #                            np.sum(np.abs(layer.biases))

            # # L2 regularization - biases
            # if layer.bias_regularizer_l2 > 0:
            #     regularization_loss += layer.bias_regularizer_l2 * \
            #                            np.sum(layer.biases * layer.biases)

        return regularization_loss

    # # Set/remember trainable layers
    # def remember_trainable_layers(self, trainable_layers):
    #     self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y, *, include_regularization=False) -> Dict:
        # calculate sample losses
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return {"data_loss" : data_loss } # Dict

        # Return the data and regularization losses
        return {"data_loss" : data_loss,"regularization_loss" : self.regularization_loss()}

    # Calculates accumulated loss
    def calculate_accumulated(self, *, include_regularization=False):
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_regularization:
            return {"data_loss" : data_loss}

        return {"data_loss" : data_loss,"regularization_loss" : self.regularization_loss()}

    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
        
    def forward(self , y_pred , y_true):
        """
        Calculates the loss given model predictions and ground truth values.

        Args:
            y_pred: Model predictions.
            y_true: Ground truth values.

        Raises:
            NotImplementedError: This method should be implemented by derived classes.
        """
        raise NotImplementedError("Derived classes must implement the 'forward' method")
    
    def backward(self, dvalues, y_true):
        """
        Performs backpropagation for the loss function.

        Args:
            dvalues: Gradient of the loss with respect to the output of the previous layer.
            y_true: Ground truth values.

        Raises:
            NotImplementedError: This method should be implemented by derived classes.
        """
        raise NotImplementedError("Derived classes must implement the 'backward' method.")


