import tensorflow as tf
import numpy as np
import time
from src.layers.Dense import Layer_Dense
from src.activations import ReLU
from src.activations import Softmax
from src.losses import CategoricalCrossentropy
from src.optimizers import Adam

def load_mnist():
    """Load and preprocess MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Reshape to (samples, features)
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    
    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def create_model(use_gpu=False):
    """Create a neural network model"""
    model = [
        Layer_Dense(28*28, 128, weight_initializer='he', use_gpu=use_gpu),
        ReLU(),
        Layer_Dense(128, 64, weight_initializer='he', use_gpu=use_gpu),
        ReLU(),
        Layer_Dense(64, 10, weight_initializer='he', use_gpu=use_gpu),
        Softmax()
    ]
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=5, batch_size=128):
    """Train the model"""
    # Create optimizer
    optimizer = Adam(learning_rate=0.001, decay=5e-5)
    
    # Create loss function
    loss_function = CategoricalCrossentropy()
    
    # Training loop
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        
        # Shuffle training data
        indices = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Mini-batch training
        for batch_start in range(0, len(x_train), batch_size):
            batch_end = min(batch_start + batch_size, len(x_train))
            x_batch = x_train_shuffled[batch_start:batch_end]
            y_batch = y_train_shuffled[batch_start:batch_end]
            
            # Forward pass
            output = x_batch
            for layer in model:
                output = layer.forward(output, training=True)
            
            # Calculate loss
            loss = loss_function.calculate(output, y_batch)
            
            # Backward pass
            dvalues = loss_function.backward(output, y_batch)
            for layer in reversed(model):
                dvalues = layer.backward(dvalues)
            
            # Update parameters
            optimizer.pre_update_params()
            for layer in model:
                if hasattr(layer, 'weights'):
                    optimizer.update_params(layer)
            optimizer.post_update_params()
        
        # Calculate accuracy on test set
        test_output = x_test
        for layer in model:
            test_output = layer.forward(test_output, training=False)
        
        predictions = np.argmax(test_output, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == true_labels)
        
        print(f'Test accuracy: {accuracy:.4f}')

def main():
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    # Train on CPU
    print("\nTraining on CPU...")
    cpu_model = create_model(use_gpu=False)
    cpu_start_time = time.time()
    train_model(cpu_model, x_train, y_train, x_test, y_test)
    cpu_time = time.time() - cpu_start_time
    print(f"CPU training time: {cpu_time:.2f} seconds")
    
    # Train on GPU
    print("\nTraining on GPU...")
    gpu_model = create_model(use_gpu=True)
    gpu_start_time = time.time()
    train_model(gpu_model, x_train, y_train, x_test, y_test)
    gpu_time = time.time() - gpu_start_time
    print(f"GPU training time: {gpu_time:.2f} seconds")
    
    # Print speedup
    speedup = cpu_time / gpu_time
    print(f"\nGPU speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main() 