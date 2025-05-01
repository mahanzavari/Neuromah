import numpy as np
from src.api import Model, DataLoader, Preprocessor
from src.api.callbacks import EarlyStopping, ModelCheckpoint
from src.layers import Dense, Dropout
from src.activations import ReLU, Softmax
from src.optimizers import Adam
from src.losses import CategoricalCrossEntropy
from src.metrics import Accuracy

def main():
    # Generate synthetic data
    n_samples = 1000
    n_features = 20
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, size=n_samples)
    
    # Preprocess data
    X, mean, std = Preprocessor.normalize(X)
    y = Preprocessor.one_hot(y, n_classes)
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = Preprocessor.split_data(
        X, y, train_size=0.7, val_size=0.15
    )
    
    # Create data loaders
    train_loader = DataLoader(X_train, y_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(X_val, y_val, batch_size=32)
    test_loader = DataLoader(X_test, y_test, batch_size=32)
    
    # Create model
    model = Model()
    
    # Add layers
    model.add(Dense(64, input_shape=(n_features,)))
    model.add(ReLU())
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(ReLU())
    model.add(Dropout(0.2))
    model.add(Dense(n_classes))
    model.add(Softmax())
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=CategoricalCrossEntropy(),
        metrics=[Accuracy()]
    )
    
    # Create callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        train_loader,
        epochs=100,
        validation_data=val_loader,
        callbacks=callbacks
    )
    
    # Evaluate model
    test_loss, test_metrics = model.evaluate(test_loader)
    print(f"\nTest Loss: {test_loss:.4f}")
    for name, value in test_metrics.items():
        print(f"Test {name}: {value:.4f}")
    
    # Make predictions
    predictions = model.predict(X_test)
    print("\nSample predictions:")
    print(predictions[:5])

if __name__ == "__main__":
    main() 