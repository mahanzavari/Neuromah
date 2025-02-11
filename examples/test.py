"""
Neuromah Federated Learning API - Usage Guide

This example demonstrates how to use Neuromah for federated learning while integrating with the built-in authentication API.

It includes:
1. Authenticating a client with the federated learning server.
2. Loading and preprocessing the MNIST dataset.
3. Defining a CNN model using Neuromah.
4. Training the model in a federated learning setup.
5. Evaluating the model on a test dataset.
6. Submitting trained model updates to the federated server.

Prerequisites:
- Ensure the Neuromah federated learning server (`Server.py`) is running.
- Ensure client nodes (`Client.py`) are configured to participate in federated training.
- API authentication credentials are correctly set up.
"""
"""
Neuromah Federated Learning API - Usage Guide

... (rest of the test.py description is largely the same, adjust as needed) ...
"""

import numpy as np
import tensorflow as tf
from Neuromah.src.layers import Layer_Conv2D, Layer_MaxPooling2D, Layer_Dense, Layer_Flatten
from Neuromah.src.optimizers import Optimizer_Adam
from Neuromah.src.losses import Loss_CategoricalCrossentropy
from Neuromah.src.core import Model
from Neuromah.src.activations import Activation_ReLU, Activation_Softmax
from Neuromah.src.metrics import Accuracy_Categorical
from Neuromah.src.utils import TensorMonitor 
from Neuromah.Federated.Client import FederatedClient  

# Disable OneDNN optimizations (same as before)
TF_ENABLE_ONEDNN_OPTS = 0

# ------------------------------------------------------------------------------------------------
# Step 1: Authenticate with the Federated Learning Server
# ------------------------------------------------------------------------------------------------

# Initialize federated learning client - authentication is handled in the constructor
SERVER_URL = "http://localhost:8000" # Corrected Server URL to match Server.py and be consistent
USERNAME = "testuser" # Define username for authentication
PASSWORD = "testpass" # Define password for authentication

try:
    federated_client = FederatedClient(server_url=SERVER_URL, username=USERNAME, password=PASSWORD)
    print("✅ Successfully authenticated with the federated learning server.")
except Exception as e:
    print(f"Authentication failed: {e}")
    exit()

# ------------------------------------------------------------------------------------------------
# Step 2 & 3: Load Data and Initialize Model - These parts are kept as in your original test.py
# ------------------------------------------------------------------------------------------------

# Load MNIST dataset (same as before)
(train_images_raw, train_labels), (test_images_raw, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape images (same as before)
train_images = (train_images_raw.astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
test_images = (test_images_raw.astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)

# One-hot encode labels (same as before)
train_labels_onehot = np.eye(10)[train_labels]
test_labels_onehot = np.eye(10)[test_labels]

# Split validation set (same as before)
val_images, val_labels = train_images[:128], train_labels_onehot[:128]
train_images, train_labels_onehot = train_images[128:256], train_labels_onehot[128:256]

# Initialize the Neuromah model (same as before - assuming Neuromah.src is available)
model = Model(device='cpu')

# Define CNN architecture (same as before)
model.add(Layer_Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding='same', activation=Activation_ReLU()))
model.add(Layer_MaxPooling2D(pool_size=2, strides=2))
model.add(Layer_Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding='same', activation=Activation_ReLU()))
model.add(Layer_MaxPooling2D(pool_size=2, strides=2))
model.add(Layer_Flatten())
model.add(Layer_Dense(n_inputs=64 * 7 * 7, n_neurons=10))
model.add(Activation_Softmax())

# Set loss, optimizer, and accuracy metric (same as before)
model.set(
    loss=Loss_CategoricalCrossentropy,
    optimizer=Optimizer_Adam(learning_rate=0.001),
    accuracy=Accuracy_Categorical,
    tensorMonitor=TensorMonitor() # Assuming TensorMonitor is available
)

# Finalize the model (same as before)
model.finalize()

# ------------------------------------------------------------------------------------------------
# Step 4 & 5: Federated Learning Setup and Training
# ------------------------------------------------------------------------------------------------

epochs = 2
batch_size = 64

print("🚀 Training in federated learning mode...")

# Train using federated learning via the client's train_federated method
federated_client.train_federated(
    train_images, train_labels_onehot,
    validation_data=(val_images, val_labels),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# ------------------------------------------------------------------------------------------------
# Step 6: Evaluate Model Performance (Same as before)
# ------------------------------------------------------------------------------------------------

print("\n🔍 Evaluating on test data (basic forward pass):")
output_test = model.predict(test_images)
predicted_classes = np.argmax(output_test, axis=1)
true_labels_test = np.argmax(test_labels_onehot, axis=1)
test_accuracy = np.mean(predicted_classes == true_labels_test) * 100

print(f"📊 Test Accuracy: {test_accuracy:.2f}%")

# ------------------------------------------------------------------------------------------------
# Step 7: Submit Updated Model (Dummy in this corrected version, updates are sent during training)
# ------------------------------------------------------------------------------------------------

# Submission is now handled within train_federated, this is just a placeholder
if test_accuracy > 80:
    federated_client.submit_weights(model) # Dummy submit_weights call
    print("✅ Federated learning round completed. (Updates sent during training)")
else:
    print("⚠️ Model performance did not meet the threshold. No separate update sent (updates sent during training).")