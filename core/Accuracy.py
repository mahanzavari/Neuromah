import numpy as np

class Accuracy:
    def __init__(self, model):
        self.model = model
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, predictions, y):
        """
        Calculates the accuracy given predictions and ground truth values.

        Args:
            predictions: Predicted values.
            y: Ground truth values.

        Returns:
            float: The accuracy.
        """
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy

    def calculate_accumulated(self):
        """
        Calculates the accumulated accuracy.

        Returns:
            float: The accumulated accuracy.
        """
        return self.accumulated_sum / self.accumulated_count

    def new_pass(self):
        """
        Resets the accumulated accuracy values.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def compare(self, predictions, y):
        """
        Compares predictions to ground truth values.

        Args:
            predictions: Predicted values.
            y: Ground truth values.

        Raises:
            NotImplementedError: This method must be implemented by derived classes.
        """
        raise NotImplementedError("Derived classes must implement the 'compare' method.")