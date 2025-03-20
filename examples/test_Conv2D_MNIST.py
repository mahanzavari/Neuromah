import numpy as np
import tensorflow as tf
from Neuromah.src.layers import Layer_Conv2D , Layer_MaxPooling2D, Layer_Dense, Layer_Flatten
from Neuromah.src.optimizers import Optimizer_Adam
from Neuromah.src.losses import Loss_CategoricalCrossentropy
from Neuromah.src.core import Model
from Neuromah.src.activations import Activation_ReLU, Activation_Softmax
from Neuromah.src.metrics import Accuracy_Categorical
from Neuromah.src.utils import TensorMonitor
import cProfile
import pstats  

TF_ENABLE_ONEDNN_OPTS = 0

# Load MNIST dataset from TensorFlow/Keras
(train_images_raw, train_labels) , (test_images_raw, test_labels) = tf.keras.datasets.mnist.load_data()

# Print dataset shapes
print("-" * 30)
print(train_images_raw.shape)
print(train_labels.shape)
print(test_images_raw.shape)
print(test_labels.shape)

# Preprocess and reshape data
# Reshape images to NCHW format (batch_size, channels, height, width) and normalize
train_images = (train_images_raw.astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
test_images = (test_images_raw.astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)

# One-hot encode labels
train_labels_onehot = np.eye(10)[train_labels]
test_labels_onehot = np.eye(10)[test_labels]

# Split validation data (using one-hot encoded labels)
val_images = train_images[:128]
val_labels = train_labels_onehot[:128]
train_images = train_images[128:256] # Reduced training data for faster profiling
train_labels_onehot = train_labels_onehot[128:256] # Reduced training data for faster profiling
test_images = test_images[:128]
test_labels_onehot = test_labels_onehot[:128]
# Initialize the Model - using CPU for this example
model = Model(device='cpu')


# Add layers to the model - CNN architecture for MNIST
# Input shape is (1, 28, 28) - 1 channel (grayscale), 28x28 image size
model.add(Layer_Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding='same', activation=Activation_ReLU())) # Output (32, 28, 28)
model.add(Layer_MaxPooling2D(pool_size=2, strides=2)) # Output (32, 14, 14)
model.add(Layer_Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding='same', activation=Activation_ReLU() )) # Output (64, 14, 14)
model.add(Layer_MaxPooling2D(pool_size=2, strides=2)) # Output (64, 7, 7)
model.add(Layer_Flatten()) # Output (64 * 7 * 7 = 3136)
model.add(Layer_Dense(n_inputs=64*7*7, n_neurons=10)) # Output (128)
model.add(Activation_Softmax()) # Output (10) - 10 classes for MNIST


# Set loss and optimizer
model.set(loss=Loss_CategoricalCrossentropy,
          optimizer=Optimizer_Adam(learning_rate=0.001),
          accuracy=Accuracy_Categorical,
          tensorMonitor= TensorMonitor())

# Finalize the model, preparing it for training
model.finalize()

# Train the model
epochs = 2
batch_size = 64

print("Training the model with profiling...")
profiler = cProfile.Profile() # Create a profiler object
profiler.enable() # Start profiling

model.train(train_images, train_labels_onehot, validation_data=(val_images, val_labels), epochs=epochs, batch_size=batch_size, verbose=1)

profiler.disable() # Stop profiling
stats = pstats.Stats(profiler).sort_stats('cumulative') # Sort stats by cumulative time
stats.print_stats(20) # Print top 20 lines of stats

# Optionally save stats to a file for later analysis:
# stats.dump_stats('profile_output.prof')


# After training, you can evaluate on the test set (optional)
# Note: For proper evaluation, you would typically create an Accuracy class
#       specific to classification and use it here.
print("\nEvaluating on test data (basic forward pass for demonstration):")
output_test = model.predict(test_images) # Get predictions on test data
predicted_classes = np.argmax(output_test, axis=1) # Get class indices
true_labels_test = np.argmax(test_labels_onehot, axis=1)

