"""
Neuromah Federated Learning Example

This example demonstrates how to use the updated federated learning implementation in Neuromah.
It shows how to:
1. Create a federated learning setup with server and clients
2. Partition data in different ways (IID, non-IID)
3. Use different aggregation strategies
4. Apply differential privacy
5. Monitor federated learning metrics
6. Handle client dropout and stragglers

The example uses MNIST dataset for simplicity.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import os
import time

# Ensure Neuromah is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Neuromah components
from src.layers import Layer_Dense, Layer_Flatten
from src.activations import Activation_ReLU, Activation_Softmax
from src.losses import Loss_CategoricalCrossentropy
from src.optimizers import Optimizer_Adam
from src.metrics import Accuracy_Categorical
from src.core import Model

# Import federated learning components
from Federated.Federated_server import FederatedServer
from Federated.Federated_client import FederatedClient
from Federated.Federated_client_partitioner import ClientDataPartitioner

# For visualization
def plot_metrics(metrics_history, title="Federated Learning Training"):
    """Plot training metrics over rounds."""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history['train_loss'], label='Train Loss')
    if metrics_history['val_loss'] and any(x is not None for x in metrics_history['val_loss']):
        plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.title('Loss over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history['train_accuracy'], label='Train Accuracy')
    if metrics_history['val_accuracy'] and any(x is not None for x in metrics_history['val_accuracy']):
        plt.plot(metrics_history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot client participation
    plt.subplot(2, 2, 3)
    plt.bar(range(len(metrics_history['client_participation'])), 
           metrics_history['client_participation'])
    plt.title('Client Participation per Round')
    plt.xlabel('Round')
    plt.ylabel('Number of Clients')
    
    # Plot aggregation time
    plt.subplot(2, 2, 4)
    plt.plot(metrics_history['aggregation_time'])
    plt.title('Aggregation Time per Round')
    plt.xlabel('Round')
    plt.ylabel('Time (s)')
    
    plt.tight_layout()
    plt.suptitle(title)
    plt.savefig('federated_metrics.png')
    plt.show()

def load_mnist_data():
    """
    Load and preprocess MNIST dataset.
    
    Returns:
        Tuple: (X_train, y_train), (X_test, y_test)
    """
    try:
        # Try to load from keras if available
        import tensorflow as tf
        print("Loading MNIST from TensorFlow...")
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    except ImportError:
        # Fallback to manual loading
        print("TensorFlow not available. Using scikit-learn to fetch MNIST...")
        from sklearn.datasets import fetch_openml
        print("Loading MNIST from OpenML...")
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.astype('float32')
        y = mnist.target.astype('int')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Reshape to (samples, height, width)
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)
    
    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape to (samples, 1, height, width) for CNN or (samples, height*width) for MLP
    # We'll use MLP for simplicity
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)
    
    # One-hot encode labels
    y_train_onehot = np.eye(10)[y_train]
    y_test_onehot = np.eye(10)[y_test]
    
    print(f"Data loaded: X_train: {X_train.shape}, y_train: {y_train_onehot.shape}")
    print(f"            X_test: {X_test.shape}, y_test: {y_test_onehot.shape}")
    
    return (X_train, y_train_onehot), (X_test, y_test_onehot)

def create_model():
    """
    Create a simple MLP model for MNIST classification.
    
    Returns:
        Model: Initialized model.
    """
    model = Model(device='cpu')  # Use 'gpu' if available
    
    # Simple architecture: Input -> Dense -> ReLU -> Dense -> Softmax
    model.add(Layer_Dense(28*28, 128, weight_initializer='he'))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10, weight_initializer='he'))
    model.add(Activation_Softmax())
    
    # Set loss, optimizer, and accuracy metric
    model.set(
        loss=Loss_CategoricalCrossentropy,
        optimizer=Optimizer_Adam(learning_rate=0.001, decay=1e-4),
        accuracy=Accuracy_Categorical
    )
    
    # Finalize the model
    model.finalize()
    
    return model

def run_federated_learning(num_clients=10, num_rounds=5, aggregation_strategy='fedavg',
                          use_differential_privacy=False, dp_epsilon=None,
                          client_sample_ratio=0.8, data_partition='iid'):
    """
    Run federated learning with specified settings.
    
    Args:
        num_clients: Number of clients to simulate.
        num_rounds: Number of federated learning rounds.
        aggregation_strategy: Strategy for server to aggregate model updates.
        use_differential_privacy: Whether to use differential privacy.
        dp_epsilon: Epsilon parameter for differential privacy.
        client_sample_ratio: Fraction of clients to select in each round.
        data_partition: Type of data partition ('iid', 'non_iid', 'dirichlet', etc.).
    
    Returns:
        Tuple: (server, clients, final_metrics)
    """
    # Load data
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    
    # Create global model
    global_model = create_model()
    
    # Create server
    server = FederatedServer(
        model=global_model,
        aggregation_strategy=aggregation_strategy,
        min_clients_per_round=2,
        client_sample_ratio=client_sample_ratio,
        max_wait_time=30,
        dp_epsilon=dp_epsilon if use_differential_privacy else None,
        dp_delta=1e-5 if use_differential_privacy else None,
        dp_mechanism='gaussian',
        secure_aggregation=False,
        adaptive_aggregation=True
    )
    
    # Partition data among clients
    print(f"Partitioning data using {data_partition} strategy...")
    if data_partition == 'iid':
        client_datasets = ClientDataPartitioner.iid_partition(X_train, y_train, num_clients)
    elif data_partition == 'non_iid':
        client_datasets = ClientDataPartitioner.non_iid_partition(X_train, y_train, num_clients)
    elif data_partition == 'dirichlet':
        client_datasets = ClientDataPartitioner.non_iid_dirichlet_partition(X_train, y_train, num_clients, alpha=0.5)
    elif data_partition == 'pathological':
        client_datasets = ClientDataPartitioner.pathological_non_iid_partition(X_train, y_train, num_clients)
    elif data_partition == 'realistic':
        client_datasets = ClientDataPartitioner.real_world_simulation(X_train, y_train, num_clients)
    else:
        raise ValueError(f"Unknown partition strategy: {data_partition}")
    
    # Create and register clients
    clients = []
    for i, (X_client, y_client) in enumerate(client_datasets):
        # Create a new model instance for each client
        client_model = create_model()
        
        # Create client
        client = FederatedClient(
            model=client_model,
            X_train=X_client,
            y_train=y_client,
            server=server,
            batch_size=32,
            epochs=2,
            differential_privacy=use_differential_privacy,
            dp_clip_norm=1.0,
            dp_noise_scale=0.1 if use_differential_privacy else 0.0,
            local_validation_split=0.1  # Use 10% of client data for local validation
        )
        
        clients.append(client)
        print(f"Client {i} created with {len(X_client)} samples")
    
    # Train for specified number of rounds
    print(f"Starting federated training for {num_rounds} rounds...")
    start_time = time.time()
    
    # Use the test set as global validation
    results = server.train(
        rounds=num_rounds,
        eval_frequency=1,
        eval_data=(X_test, y_test)
    )
    
    total_time = time.time() - start_time
    print(f"Federated training completed in {total_time:.2f} seconds")
    
    # Get final metrics
    metrics_history = server.get_metrics_history()
    
    # Evaluate final model on test set
    output = global_model.forward(X_test, training=False)
    loss = global_model.loss.calculate(output, y_test)
    predictions = global_model.output_layer_activation.predictions(output)
    accuracy = global_model.accuracy.calculate(predictions, y_test)
    
    print(f"Final model performance - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return server, clients, metrics_history

if __name__ == "__main__":
    # IID Federated Learning with FedAvg
    print("\n==== Running Federated Learning with IID data and FedAvg ====")
    server, clients, metrics_history = run_federated_learning(
        num_clients=10,
        num_rounds=10,
        aggregation_strategy='fedavg',
        use_differential_privacy=False,
        client_sample_ratio=0.8,
        data_partition='iid'
    )
    
    # Plot metrics
    plot_metrics(metrics_history, title="Federated Learning (IID, FedAvg)")
    
    # Save the global model
    server.save_global_model("federated_model_iid.pkl")
    
    # Non-IID Federated Learning with Differential Privacy
    print("\n==== Running Federated Learning with Non-IID data and Differential Privacy ====")
    server_dp, clients_dp, metrics_history_dp = run_federated_learning(
        num_clients=10,
        num_rounds=10,
        aggregation_strategy='weighted',
        use_differential_privacy=True,
        dp_epsilon=3.0,
        client_sample_ratio=0.8,
        data_partition='dirichlet'
    )
    
    # Plot metrics
    plot_metrics(metrics_history_dp, title="Federated Learning (Non-IID, DP, Weighted)")
    
    # Save the global model
    server_dp.save_global_model("federated_model_noniid_dp.pkl")
    
    # Realistic simulation
    print("\n==== Running Realistic Federated Learning Simulation ====")
    server_real, clients_real, metrics_history_real = run_federated_learning(
        num_clients=20,
        num_rounds=15,
        aggregation_strategy='median',
        use_differential_privacy=False,
        client_sample_ratio=0.6,
        data_partition='realistic'
    )
    
    # Plot metrics
    plot_metrics(metrics_history_real, title="Federated Learning (Realistic Simulation)")
    
    # Save the global model
    server_real.save_global_model("federated_model_realistic.pkl")
    
    print("\nAll federated learning experiments completed!") 