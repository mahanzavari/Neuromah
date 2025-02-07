"""
Neuromah Federated Learning API - Usage Guide

This example demonstrates how to use Neuromah for federated learning while integrating with the built-in authentication API. 
It includes steps to:
1. Authenticate a client with the federated learning server.
2. Load and preprocess the MNIST dataset.
3. Define a convolutional neural network (CNN) model using Neuromah.
4. Train the model in a federated learning setup.
5. Evaluate the model on a test dataset.

Prerequisites:
- Ensure the Neuromah federated learning server (`Server.py`) is running.
- Ensure client nodes (`Client.py`) are configured to participate in federated training.
- API authentication credentials are correctly set up.

------------------------------------------------------------------------------------------------
Step 1: Import Required Modules
------------------------------------------------------------------------------------------------
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
from Neuromah.Federated.Client import FederatedClient  # Import federated learning client
from Neuromah.Federated.Auth import AuthAPI  # Import authentication API

TF_ENABLE_ONEDNN_OPTS = 0  # Disable OneDNN optimizations for compatibility

"""
------------------------------------------------------------------------------------------------
Step 2: Authenticate with the Federated Learning Server
------------------------------------------------------------------------------------------------
"""
# Initialize authentication API and login
auth_api = AuthAPI(server_url="http://localhost:5000")  # Replace with actual server URL
token = auth_api.login(username="user1", password="securepassword")

if not token:
    raise Exception("Authentication failed. Check your credentials and server connection.")

print("Successfully authenticated with federated learning server.")

"""
------------------------------------------------------------------------------------------------
Step 3: Load and Preprocess Data
------------------------------------------------------------------------------------------------
"""
# Load MNIST dataset
(train_images_raw, train_labels), (test_images_raw, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape images to (batch_size, channels, height, width)
train_images = (train_images_raw.astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
test_images = (test_images_raw.astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)

# One-hot encode labels
train_labels_onehot = np.eye(10)[train_labels]
test_labels_onehot = np.eye(10)[test_labels]

# Split validation set
val_images, val_labels = train_images[:128], train_labels_onehot[:128]
train_images, train_labels_onehot = train_images[128:256], train_labels_onehot[128:256]

"""
------------------------------------------------------------------------------------------------
Step 4: Initialize Model
------------------------------------------------------------------------------------------------
"""
# Initialize the Neuromah model for federated learning
model = Model(device='cpu')

# Define CNN architecture
model.add(Layer_Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding='same', activation=Activation_ReLU()))
model.add(Layer_MaxPooling2D(pool_size=2, strides=2))
model.add(Layer_Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding='same', activation=Activation_ReLU()))
model.add(Layer_MaxPooling2D(pool_size=2, strides=2))
model.add(Layer_Flatten())
model.add(Layer_Dense(n_inputs=64 * 7 * 7, n_neurons=10))
model.add(Activation_Softmax())

# Set loss, optimizer, and accuracy metric
model.set(loss=Loss_CategoricalCrossentropy,
          optimizer=Optimizer_Adam(learning_rate=0.001),
          accuracy=Accuracy_Categorical,
          tensorMonitor=TensorMonitor())

# Finalize the model
model.finalize()

"""
------------------------------------------------------------------------------------------------
Step 5: Federated Learning Setup
------------------------------------------------------------------------------------------------
"""
# Initialize federated learning client
federated_client = FederatedClient(server_url="http://localhost:5000", auth_token=token)

# Register model with federated server
federated_client.register_model(model)

# Sync model weights with federated server
federated_client.sync_weights()

"""
------------------------------------------------------------------------------------------------
Step 6: Train the Model in Federated Learning Mode
------------------------------------------------------------------------------------------------
"""
epochs = 2
batch_size = 64

print("Training in federated learning mode...")

# Train using federated learning, where training occurs across multiple clients
federated_client.train_federated(train_images, train_labels_onehot, 
                                 validation_data=(val_images, val_labels),
                                 epochs=epochs, batch_size=batch_size, verbose=1)

"""
------------------------------------------------------------------------------------------------
Step 7: Evaluate Model Performance
------------------------------------------------------------------------------------------------
"""
print("\nEvaluating on test data (basic forward pass):")
output_test = model.predict(test_images)  # Get predictions
predicted_classes = np.argmax(output_test, axis=1)  # Convert to class labels
true_labels_test = np.argmax(test_labels_onehot, axis=1)

# Calculate accuracy
test_accuracy = np.mean(predicted_classes == true_labels_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

"""
------------------------------------------------------------------------------------------------
Step 8: Submit Updated Model to the Federated Server
------------------------------------------------------------------------------------------------
"""
# After training, submit updated model weights to federated server
federated_client.submit_weights(model)

print("Federated learning round completed. Updated weights sent to the server.")

"""
------------------------------------------------------------------------------------------------
Summary:
------------------------------------------------------------------------------------------------
- **Authentication:** Users must authenticate using the `AuthAPI` before interacting with the federated server.
- **Federated Learning Setup:** The client registers the model and syncs weights before training.
- **Federated Training:** Training occurs across multiple clients, and updated weights are submitted back.
- **Evaluation:** The model is evaluated locally after training.
- **Federated Updates:** After evaluation, the trained model updates are sent back to the server.

Next Steps:
- Run multiple clients to simulate a real federated learning environment.
- Modify authentication methods for improved security.
- Experiment with different architectures and datasets.

"""
