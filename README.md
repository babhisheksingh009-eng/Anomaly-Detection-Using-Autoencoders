# Anomaly Detection Using Autoencoders

## Project Introduction
This is a Deep Learning project that detects unusual machine behavior using Autoencoders.

The model learns normal machine operating patterns and identifies unusual readings using reconstruction error.

## Technologies Used
- Python
- TensorFlow and Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

## Dataset
AI4I 2020 Predictive Maintenance Dataset.

Dataset features:
1. Air Temperature (K)
2. Process Temperature (K)
3. Rotational Speed (RPM)
4. Torque (Nm)
5. Tool Wear (min)

## Working Process
1. Load the dataset.
2. Preprocess the data.
3. Select normal machine operating data.
4. Train the Autoencoder model.
5. Calculate reconstruction error.
6. Set an anomaly threshold.
7. Enter machine values through the Streamlit interface.
8. Detect whether the machine behavior is normal or anomalous.

## How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/babhisheksingh009-eng/Anomaly-Detection-Using-Autoencoders.git
