#!/usr/bin/env python3
"""
Script to train the Deep Learning model for code quality prediction.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def load_data():
    """Load training and testing data."""
    dataset_path = Path("dataset")

    X_train = np.load(dataset_path / "X_train.npy")
    y_train = np.load(dataset_path / "y_train.npy")
    X_test = np.load(dataset_path / "X_test.npy")
    y_test = np.load(dataset_path / "y_test.npy")

    # Load scaler and feature names
    with open(dataset_path / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(dataset_path / "feature_names.pkl", 'rb') as f:
        feature_names = pickle.load(f)

    return X_train, y_train, X_test, y_test, scaler, feature_names

def build_model(input_shape):
    """Build the neural network model."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Regression output
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model

def train_model():
    """Train the model."""
    print("Loading data...")
    X_train, y_train, X_test, y_test, scaler, feature_names = load_data()

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Features: {len(feature_names)}")

    # Scale the data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Building model...")
    model = build_model(X_train.shape[1])

    print("Training model...")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Save the model
    models_path = Path("models")
    models_path.mkdir(exist_ok=True)
    model.save(models_path / "final_model.keras")

    # Save performance
    with open(models_path / "model_performance.txt", 'w') as f:
        f.write("==================================================\n")
        f.write("MODEL PERFORMANCE\n")
        f.write("==================================================\n\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"Root Mean Squared Error: {rmse:.4f}\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"Best Epoch: {len(history.history['loss']) - early_stopping.patience}\n")

    print("Model saved to models/final_model.keras")
    print("Performance saved to models/model_performance.txt")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(models_path / "training_results.png")
    plt.show()

if __name__ == "__main__":
    train_model()