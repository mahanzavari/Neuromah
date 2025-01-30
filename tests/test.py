import numpy as np
from Neuromah.src.core import Model
from Neuromah.src.layers import Layer_Dense
from Neuromah.src.losses.categorical_crossentropy import Loss_CategoricalCrossentropy
from Neuromah.src.optimizers import Optimizer_Adam
from Neuromah.src.activations import Activation_Softmax, Activation_ReLU
from Neuromah.src.metrics.Accuracy_Categorical import Accuracy_Categorical
print(Accuracy_Categorical)
import tensorflow as tf

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Print dataset shapes
print("-" * 30)
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# Normalize and reshape the data
train_images = (train_images.astype(np.float32) / 255.0).reshape(train_images.shape[0], -1)
test_images = (test_images.astype(np.float32) / 255.0).reshape(test_images.shape[0], -1)
train_labels = np.eye(10)[train_labels]
test_labels = np.eye(10)[test_labels]

# Split validation data
val_images = train_images[:10000]
val_labels = train_labels[:10000]

# Create the model
model = Model()

# Add layers with correct input sizes
model.add(Layer_Dense(n_inputs=784, n_neurons=128, activation=Activation_ReLU))
model.add(Layer_Dense(n_inputs=128, n_neurons=64, activation=Activation_ReLU))
model.add(Layer_Dense(n_inputs=64, n_neurons=10, activation=Activation_Softmax))

# Set loss, optimizer
model.set(
    loss=Loss_CategoricalCrossentropy(model=model),
    optimizer=Optimizer_Adam(learning_rate=0.001),  # recommended lr value for Adam
    accuracy = Accuracy_Categorical(model= model)
)

# Finalize the model
model.finalize()  # like summary in Keras

# Train the model with correct training data
model.train(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

# Predict on test data
predictions = model.predict(test_images)
print(predictions)