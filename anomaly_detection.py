import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ai4i2020.csv")

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

# Select machine operating features
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"].astype(int)

# Split the data
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Use only normal machine data for training
X_train_normal = X_train_all[y_train_all == 0]

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_normal)
X_test_scaled = scaler.transform(X_test)

# Build the autoencoder
inputs = Input(shape=(X_train.shape[1],))
encoded = Dense(8, activation="relu")(inputs)
encoded = Dense(3, activation="relu")(encoded)
decoded = Dense(8, activation="relu")(encoded)
outputs = Dense(X_train.shape[1], activation="linear")(decoded)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.summary()

# Train the model
print("\nTraining the autoencoder...")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = autoencoder.fit(
    X_train,
    X_train,
    epochs=30,
    batch_size=64,
    validation_split=0.15,
    callbacks=[early_stopping],
    verbose=1
)

# Calculate reconstruction errors
train_reconstructed = autoencoder.predict(X_train, verbose=0)
train_errors = np.mean(
    np.square(X_train - train_reconstructed), axis=1
)

test_reconstructed = autoencoder.predict(X_test_scaled, verbose=0)
test_errors = np.mean(
    np.square(X_test_scaled - test_reconstructed), axis=1
)

# Set threshold using normal training data
threshold = np.percentile(train_errors, 95)

# Identify potential anomalies
y_pred = (test_errors > threshold).astype(int)

# Display results
print("\n===== ANOMALY DETECTION RESULTS =====")
print(f"Anomaly threshold: {threshold:.4f}")
print(f"Actual machine failures: {int(y_test.sum())}")
print(f"Potential anomalies detected: {int(y_pred.sum())}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Normal", "Machine Failure"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Autoencoder Training Performance")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "training_loss.png"))
plt.show()

# Plot anomaly detection results
plt.figure(figsize=(9, 5))

normal_mask = y_pred == 0
anomaly_mask = y_pred == 1

plt.scatter(
    np.where(normal_mask)[0],
    test_errors[normal_mask],
    label="Predicted Normal",
    s=15
)

plt.scatter(
    np.where(anomaly_mask)[0],
    test_errors[anomaly_mask],
    label="Predicted Anomaly",
    s=18
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Anomaly Threshold"
)

plt.title("Machine Anomaly Detection Results")
plt.xlabel("Test Sample")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "anomaly_detection_results.png")
)
plt.show()

# Save results
results = X_test.copy()
results["Actual Failure"] = y_test.values
results["Reconstruction Error"] = test_errors
results["Predicted Anomaly"] = y_pred

results.to_csv(
    os.path.join(BASE_DIR, "anomaly_results.csv"),
    index=False
)

print("\nProject completed successfully!")
print("Graphs and results saved in your project folder.")