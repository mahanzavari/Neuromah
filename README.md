# Neuromah: A Modular Deep Learning Framework

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/mahanzavari/Neuromah) 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) 
[![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-compatible-green.svg)](https://numpy.org/)
[![CuPy](https://img.shields.io/badge/cupy-compatible-green.svg)](https://cupy.dev/)  

## Overview

Neuromah is a modular and extensible deep learning framework built from scratch in Python. It supports both CPU (NumPy) and GPU (CuPy) execution, allowing you to seamlessly switch between backends for development and deployment.  The framework prioritizes clarity, ease of use, and flexibility, making it suitable for both learning and building custom neural network architectures.  It's inspired by the principles demonstrated in the book "Neural Networks from Scratch" (NNFS) but extends those concepts with additional features and optimizations.

## Key Features

*   **Modular Design:**  Build complex networks by combining individual, well-defined layers (Dense, Conv2D, MaxPooling2D, Dropout, Flatten), activations (ReLU, Softmax, Sigmoid, Tanh, Linear), optimizers (SGD, Adagrad, RMSprop, Adam, Nadam), and loss functions (CategoricalCrossentropy, BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError).
*   **CPU and GPU Support:**  Switch between NumPy (CPU) and CuPy (GPU) backends with a single configuration change, enabling efficient training and inference on different hardware.
*   **Customizable Layers and Components:** Easily create your own layers, activations, loss functions, and optimizers by inheriting from the provided base classes and implementing the required methods.
*   **Automatic Differentiation:**  The framework handles backpropagation automatically, calculating gradients for all trainable parameters.
*   **Weight Initializers:**  Includes several weight initialization strategies (RandomNormal, Xavier/Glorot, He) to improve training stability and convergence.
*   **Regularization:**  Supports L1 and L2 regularization for weights to prevent overfitting.
*   **Model Serialization:** Save and load trained models and parameters using built-in `save()` and `load()` methods (leveraging `pickle`).
*   **TensorBoard Integration (with `TensorMonitor`):** Easily visualize training progress, layer parameters, and gradients using TensorBoard.
*   **Optimized Convolutional Layers:** Uses a C++ backend (`_Conv2DBackend_cpp`) for significantly faster convolution operations.
*   **Max Pooling Layer:** Implements a Max Pooling 2D layer with a C++ backend (`_MaxPooling2DBackend_cpp`).
*   **Clear API and Documentation:**  Well-documented code with type hints and comprehensive explanations, making it easy to understand and extend.
*   **Batch Training:** Supports mini-batch training for improved efficiency.
*   **Verbose Training Output:** Provides detailed training progress information, including loss, accuracy, learning rate, and epoch time.
*   **Validation Support:** Evaluate model performance on a separate validation dataset during training.


## Installation

1.  **Prerequisites:**
    *   Python 3.7+
    *   NumPy (required)
    *   CuPy (optional, for GPU support) - [Installation Instructions](https://docs.cupy.dev/en/stable/install.html)
    *   tqdm (for progress bars)
    *   Tensorboard (optional)

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/yourrepository.git  # Replace with your repo URL
    cd yourrepository
    ```

3.  **Install Dependencies:**

    ```bash
    pip install numpy tqdm
    ```
    If you want to use GPU support, install cupy:
    ```
     pip install cupy-cuda11x  # Replace 11x with your CUDA version (e.g., 10x, 12x)
    ```
    
    If you intend to use tensorboard, install it:
    ```bash
     pip install tensorboard
    ```


### Basic Usage
```python
import numpy as np
from neuromah.models import Model
from neuromah.layers import Layer_Dense, Layer_Conv2D, Layer_Flatten, Layer_MaxPooling2D, Layer_Dropout
from neuromah.activations import Activation_ReLU, Activation_Softmax, Activation_Sigmoid
from neuromah.losses import Loss_CategoricalCrossentropy, Loss_BinaryCrossentropy
from neuromah.optimizers import Optimizer_Adam
from neuromah.metrics import Accuracy_Categorical
from neuromah.utils import TensorMonitor
# from sklearn.model_selection import train_test_split # If you need train/test split

# Sample Dataset (replace with your actual data loading)
def create_data(n_samples, n_features):
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] > 0).astype(int)  # Example binary classification
    return X, y

X, y = create_data(1000, 5)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize TensorMonitor (optional, for TensorBoard visualization)
tensorboard = TensorMonitor(log_dir='logs/') # dir for tb logs

# Create a Model
model = Model(device='cpu')  # Or 'gpu' if CuPy is installed and you have a CUDA-enabled GPU

# Add layers
model.add(Layer_Dense(5, 64, weight_initializer='he'))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 2, weight_initializer='he'))
model.add(Activation_Softmax())  # For multi-class classification


# Set loss, optimizer, and accuracy
model.set(
    loss=Loss_CategoricalCrossentropy, # Use categorical crossentropy for multi-class
    optimizer=Optimizer_Adam(learning_rate=0.001, decay=1e-3),
    accuracy=Accuracy_Categorical,
    tensorMonitor = tensorboard
)

# Finalize the model (connects layers, sets up loss and optimizer)
model.finalize()

# Train the model
model.train(X, y, epochs=10, batch_size=32, verbose=2) #, validation_data=(X_test, y_test))

# Evaluate the model
print("\nEvaluation:")
model.evaluate(X, y, batch_size=32) # X_test , y_test

# Make predictions
predictions = model.predict(X)
# print("Predictions:", predictions)


# --- CNN Example ---
def create_image_data(n_samples, height, width, channels):
    X = np.random.rand(n_samples, height, width, channels)
    y = np.random.randint(0, 10, size=n_samples)  # Example: 10 classes
    return X, y

X_img, y_img = create_image_data(100, 28, 28, 1)  # 100 samples, 28x28, 1 channel
y_img_onehot = np.eye(10)[y_img] # One-hot encode the labels

cnn_model = Model(device='cpu') # or 'gpu'
cnn_model.add(Layer_Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding='same', weight_initializer='he'))
cnn_model.add(Activation_ReLU())
cnn_model.add(Layer_MaxPooling2D(pool_size=2, strides=2))
cnn_model.add(Layer_Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding='same', weight_initializer='he'))
cnn_model.add(Activation_ReLU())
cnn_model.add(Layer_MaxPooling2D(pool_size=2, strides=2))
cnn_model.add(Layer_Flatten())
cnn_model.add(Layer_Dense(64 * 7 * 7, 128 , weight_initializer='he'))  # Adjust based on input and pooling
cnn_model.add(Activation_ReLU())
cnn_model.add(Layer_Dropout(0.5)) # Dropout layer
cnn_model.add(Layer_Dense(128, 10, weight_initializer='he'))
cnn_model.add(Activation_Softmax())

cnn_model.set(
    loss=Loss_CategoricalCrossentropy,
    optimizer=Optimizer_Adam(learning_rate=0.001),
    accuracy=Accuracy_Categorical,
    tensorMonitor = tensorboard
)

cnn_model.finalize()
cnn_model.train(X_img, y_img_onehot, epochs=15, batch_size=32 , verbose = 1)

# --- Saving and Loading ---
model.save_parameters('my_model_params.pkl')
cnn_model.save('my_cnn_model.pkl')

# Load the parameters into a new model (must have the same architecture)
new_model = Model()
# ... add layers to new_model (same as the original) ...
new_model.set(
    loss=Loss_CategoricalCrossentropy, # Use categorical crossentropy for multi-class
    optimizer=Optimizer_Adam(learning_rate=0.001, decay=1e-3),
    accuracy=Accuracy_Categorical
)

new_model.finalize()

new_model.load_parameters('my_model_params.pkl')

# Load a full model
loaded_cnn_model = Model.load('my_cnn_model.pkl')
```

## API Documentation

This section provides a brief overview of the main classes and their methods.  For detailed docstrings, refer to the source code.

### `Model`

*   **`__init__(self, device='cpu')`**: Initializes the model.  `device` can be 'cpu' (default) or 'gpu'.
*   **`add(self, layer)`**: Adds a layer to the model.
*   **`set(self, *, loss, optimizer, accuracy, tensorMonitor=None)`**: Sets the loss function, optimizer, accuracy metric, and TensorBoard monitor.
*   **`finalize(self)`**: Finalizes the model, connecting layers and setting up the training process.  Must be called before training.
*   **`train(self, X, y, *, epochs=1, batch_size=None, verbose=1, validation_data=None)`**: Trains the model.
    *   `X`: Training data.
    *   `y`: Training labels.
    *   `epochs`: Number of training epochs.
    *   `batch_size`: Batch size.  If `None`, uses the entire dataset as a single batch.
    *   `verbose`:  Verbosity level (0: silent, 1: progress bar, 2: epoch summary).
    *    `validation_data`: Tuple `(X_val, y_val)` for validation.
*   **`evaluate(self, X_val, y_val, *, batch_size=None)`**: Evaluates the model on validation data.
*   **`predict(self, X, *, batch_size=None)`**: Makes predictions on new data.
*   **`save_parameters(self, path)`**: Saves the model's parameters to a file.
*   **`load_parameters(self, path)`**: Loads parameters from a file into the model.
*   **`save(self, path)`**: Saves the entire model (architecture and parameters) to a file.
*   **`load(path)`**:  Static method. Loads a model from a file.  Usage: `model = Model.load('model.pkl')`
*   **`forward(self, X, training)`**: Performs a forward pass through the network.
*    **`backward(self, output, y)`**: Performs a backward pass (backpropagation).
* **`get_parameters(self)`**: retrieves parameters of traininable layers.
*  **`set_parameters(self, parameters)`**: sets the model with new parameters.


### Layers

*   **`Layer_Dense(n_inputs, n_neurons, weight_initializer=None, activation=None, weight_regularizer_l1=0, weight_regularizer_l2=0)`**:  A fully connected layer.
    * `weight_initializer`: accepts 'xavier' , 'he' or an object of the class `Initializer`
*   **`Layer_Conv2D(in_channels, out_channels, kernel_size, weight_initializer=None, activation=None, stride=1, padding='valid')`**:  A 2D convolutional layer.
    *   `kernel_size`:  Int or tuple (height, width).
    *   `stride`: Int or tuple (height, width).
    *    `padding`: 'valid' or 'same'.
    * `weight_initializer`: accepts 'xavier' , 'he' or an object of the class `Initializer`.
*   **`Layer_MaxPooling2D(pool_size, strides=None, padding='valid')`**:  A 2D max pooling layer.
    *   `pool_size`: Int or tuple (height, width).
    *   `strides`:  Int or tuple (height, width). Defaults to `pool_size`.
*   **`Layer_Flatten()`**:  Flattens the input.
*   **`Layer_Dropout(rate)`**:  Dropout layer.  `rate` is the *drop* probability (e.g., 0.5 for 50% dropout).
*    **`Layer_Input()`**: Input layer
*  All layers have `forward(self, inputs, training)` and `backward(self, dvalues)` methods. Trainable layers also have:
        *   `get_parameters()`:  Returns a dictionary of parameters and their gradients.
        *   `set_parameters(self, weights, biases)`: Sets the layer's parameters.

### Activations

*   **`Activation_ReLU()`**:  Rectified Linear Unit.
*   **`Activation_Softmax()`**:  Softmax activation.
*   **`Activation_Sigmoid()`**:  Sigmoid activation.
*   **`Activation_Tanh()`**: Tanh activation.
* **`Activation_Linear()`**: Linear activation.
*   All activations have `forward(self, inputs, training)` and `backward(self, dvalues)` methods.  They also have a `predictions(self, outputs)` method to get the predicted class.

### Losses

*   **`Loss_CategoricalCrossentropy()`**:  Categorical cross-entropy loss (for multi-class classification).
*   **`Loss_BinaryCrossentropy()`**:  Binary cross-entropy loss (for binary classification).
*   **`Loss_MeanSquaredError()`**:  Mean Squared Error loss (for regression).
*   **`Loss_MeanAbsoluteError()`**:  Mean Absolute Error loss (for regression).
*    All losses have:
    * **`__init__(self , model)`**: init the Loss class and passing model as an argument.
    *   `calculate(self, output, y, *, include_regularization=False)`: Calculates the loss.
    *   `forward(self, y_pred, y_true)`:  Calculates the data loss.
    *   `backward(self, dvalues, y_true)`:  Calculates the gradients.
    *   `regularization_loss(self)`: Calculates the regularization loss (if applicable).
    * `calculate_accumulated(self, *, include_regularization=False)`:Calculates accumulated loss.
    * `new_pass(self)`: Reset variables for accumulated loss.

### Optimizers

*   **`Optimizer_SGD(learning_rate=1.0, decay=0.0, momentum=0.0)`**:  Stochastic Gradient Descent.
*   **`Optimizer_Adagrad(learning_rate=1.0, decay=0.0, epsilon=1e-7)`**:  Adagrad optimizer.
*   **`Optimizer_RMSprop(learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9)`**:  RMSprop optimizer.
*   **`Optimizer_Adam(learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999)`**:  Adam optimizer.
*   **`Optimizer_Nadam(learning_rate=0.001, decay=0.0, beta1=0.9, beta2=0.999, epsilon=1e-8)`**: Nadam optimizer.
*   All optimizers have:
    *   `pre_update_params(self)`:  Adjusts the learning rate (if decay is used).
    *   `update_params(self, layer)`:  Updates the layer's parameters.
    *   `post_update_params(self)`:  Increments the iteration counter.

### Metrics
* **`Accuracy`**: Accuracy base class.
    * **`__init__(self , model)`**: init the class and passing model as an argument.
    *   `calculate(self, predictions, y)`: Calculates the accuracy given predictions and ground truth values.
    *    `calculate_accumulated(self)`: Calculates the accumulated accuracy.
    * `new_pass(self)`: Resets the accumulated accuracy values.
    * `compare(self , predictions , y)`: Compares predictions to ground truth values.
* **`Accuracy_Categorical(*, binary=False)`**: Categorical accuracy (for classification).
* **`Accuracy_Regression()`**: Accuracy for regression.
    *   `init(self, y, reinit=False)`: initialize, or reinitialize, precision.

### Initializers
* **`RandomNormalInitializer(std_dev=0.01)`**: Initializes weights with a normal distribution.
* **`XavierInitializer()`**: Xavier/Glorot initialization.
* **`HeInitializer()`**: He initialization.
* All Initializers Have:
    * **`initialize(self, shape)`**: Initializes weights with the specified shape.

### Utils
* **`TensorMonitor(log_dir='logs/')`**: a class for visualizing model parameters with TensorBoard.
    * **`start_run(self , model)`**: Starts a TensorBoard run, initializing log writers.
    * **`start_epoch(self, epoch)`**: Logs the start of a new epoch.
    * **`log_scalar(self, tag, value)`**: Logs a scalar value.
    * **`log_histogram(self, tag, values)`**: Logs a histogram of values.
    * **`log_layer_parameters(self, layer)`**: Logs the parameters (weights and biases) of a layer as histograms.
    * **`end_epoch(self)`**: Logs the end of an epoch, writing summaries to the log directory.
    * **`end_step(self)`**: increments `step` by one.
    * **`save_logs(self)`**: Saves the logs to the specified directory.

## Contributing

Contributions are welcome! 

1.  **Fork the repository.**
2.  **Create a new branch:** `git checkout -b my-feature-branch`
3.  **Make your changes and write tests.**
4.  **Ensure your code passes all tests** and adheres to the existing coding style (use a linter like `flake8`).
5.  **Commit your changes:** `git commit -m "Add my feature"`
6.  **Push to the branch:** `git push origin my-feature-branch`
7.  **Create a pull request.**

Please include a clear description of your changes and any relevant issue numbers in your pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
