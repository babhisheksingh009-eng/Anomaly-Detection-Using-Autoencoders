
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Page configuration
st.set_page_config(
    page_title="Machine Anomaly Detection",
    page_icon="⚙️",
    layout="wide"
)

# Machine features
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Title
st.title("⚙️ Machine Anomaly Detection System")
st.write(
    "Enter machine operating parameters to check "
    "whether the machine shows a potential anomaly."
)

st.info(
    "This project uses a Deep Learning Autoencoder "
    "trained on the AI4I 2020 Predictive Maintenance dataset."
)

# Load and train model only when needed
@st.cache_resource
def train_model():
    # Load dataset
    data = pd.read_csv("ai4i2020.csv")

    # Use only normal machines to train the autoencoder
    normal_data = data[data["Machine failure"] == 0][FEATURES]

    # Scale the data
    scaler = StandardScaler()
    X_normal = scaler.fit_transform(normal_data)

    # Build the autoencoder
    inputs = Input(shape=(len(FEATURES),))
    encoded = Dense(8, activation="relu")(inputs)
    encoded = Dense(3, activation="relu")(encoded)
    decoded = Dense(8, activation="relu")(encoded)
    outputs = Dense(len(FEATURES), activation="linear")(decoded)

    autoencoder = Model(inputs, outputs)

    autoencoder.compile(
        optimizer="adam",
        loss="mse"
    )

    # Train the model
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    autoencoder.fit(
        X_normal,
        X_normal,
        epochs=20,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    # Calculate reconstruction errors on normal training data
    reconstructed = autoencoder.predict(X_normal, verbose=0)
    errors = np.mean(np.square(X_normal - reconstructed), axis=1)

    # Set anomaly threshold
    threshold = float(np.percentile(errors, 95))

    return autoencoder, scaler, threshold


# Train button
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

if st.button("🚀 Train Model", type="primary"):
    try:
        with st.spinner("Training the autoencoder... Please wait."):
            model, scaler, threshold = train_model()

            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.threshold = threshold
            st.session_state.model_ready = True

        st.success("Model trained successfully!")

    except FileNotFoundError:
        st.error(
            "Dataset not found! Make sure ai4i2020.csv "
            "is in the same folder as app.py."
        )
    except Exception as error:
        st.error(f"Training failed: {error}")


st.divider()

# User input section
st.subheader("📝 Enter Machine Parameters")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.1
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=310.0,
        step=0.1
    )

    rotational_speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=0,
        max_value=10000,
        value=1500,
        step=10
    )

with col2:
    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=200.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear (minutes)",
        min_value=0,
        max_value=500,
        value=100,
        step=1
    )

# Detection button
if st.button("🔍 DETECT ANOMALY", use_container_width=True):

    if not st.session_state.model_ready:
        st.warning("Please click 'Train Model' first.")

    else:
        # Prepare user input
        user_data = pd.DataFrame(
            [[
                air_temp,
                process_temp,
                rotational_speed,
                torque,
                tool_wear
            ]],
            columns=FEATURES
        )

        # Scale input and reconstruct it
        scaled_input = st.session_state.scaler.transform(user_data)

        reconstructed = st.session_state.model.predict(
            scaled_input,
            verbose=0
        )

        # Calculate reconstruction error
        error = float(
            np.mean(np.square(scaled_input - reconstructed))
        )

        threshold = st.session_state.threshold

        st.divider()
        st.subheader("📊 Detection Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Reconstruction Error",
                f"{error:.5f}"
            )

        with result_col2:
            st.metric(
                "Anomaly Threshold",
                f"{threshold:.5f}"
            )

        # Compare error with threshold
        if error > threshold:
            st.error(
                "⚠️ POTENTIAL ANOMALY DETECTED"
            )
            st.write(
                "The machine's operating values have a "
                "higher reconstruction error than the "
                "selected threshold. Further inspection may be needed."
            )
        else:
            st.success(
                "✅ NORMAL BEHAVIOR"
            )
            st.write(
                "The machine's operating values are within "
                "the model's selected reconstruction-error threshold."
            )

        # Display error comparison
        chart_data = pd.DataFrame({
            "Measurement": ["Reconstruction Error", "Threshold"],
            "Value": [error, threshold]
        })

        st.bar_chart(
            chart_data,
            x="Measurement",
            y="Value"
        )

        st.caption(
            "Note: This is an educational anomaly-detection model. "
            "Its result indicates unusual operating data, not a "
            "confirmed machine failure."
        )

st.divider()
st.caption("Deep Learning Project | Autoencoder-Based Anomaly Detection")
