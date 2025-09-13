# app_modified.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="🔧 Machine Failure Prediction", layout="wide")

st.title("🔧 Machine Failure Prediction")
st.caption("Predict machine failure using sensor data — now with improved layout and charts.")

MODEL_PATH = "model.pkl"

# Load model with caching
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model = load_model(MODEL_PATH)
if model is None:
    st.warning("⚠️ No pre-trained model found (`model.pkl`). Run `train_model.py` or upload a model.")

st.sidebar.header("Options")
mode = st.sidebar.radio("Choose Mode", ["Single prediction (manual)", "Batch prediction (CSV)"])

# --- SINGLE PREDICTION ---
if mode == "Single prediction (manual)":
    st.subheader("Enter Sensor Values")
    col1, col2 = st.columns(2)

    with col1:
        footfall = st.number_input("Footfall", value=0.0, step=1.0)
        tempMode = st.number_input("Temp Mode", value=25.0, step=1.0)
        AQ = st.number_input("Air Quality Index (AQ)", value=50.0, step=1.0)
        USS = st.number_input("Ultrasonic Sensor (USS)", value=0.0, step=1.0)
        CS = st.number_input("Current Sensor (CS)", value=0.0, step=0.1)
    with col2:
        VOC = st.number_input("VOC (volatile organic compounds)", value=0.0, step=0.1)
        RP = st.number_input("Rotational Position / RPM", value=0.0, step=1.0)
        IP = st.number_input("Input Pressure", value=0.0, step=0.1)
        Temperature = st.number_input("Temperature", value=30.0, step=0.1)

    if st.button("🔍 Predict"):
        features = np.array([[footfall, tempMode, AQ, USS, CS, VOC, RP, IP, Temperature]])
        if model is None:
            st.error("Model not found. Please add `model.pkl` and reload.")
        else:
            pred = model.predict(features)[0]
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(features)[0]
            else:
                prob = [1-pred, pred]  # dummy probability if not available

            # Display result
            if pred == 1:
                st.error("⚠️ MACHINE FAILURE predicted!")
            else:
                st.success("✅ MACHINE SAFE predicted!")

            # Probability bar chart
            fig, ax = plt.subplots()
            ax.bar(["Safe (0)", "Failure (1)"], prob, color=["green", "red"])
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.set_title("Prediction Probabilities")
            st.pyplot(fig)

# --- BATCH PREDICTION ---
else:
    st.subheader("Upload CSV for Batch Predictions")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 **Preview of Uploaded Data**")
        st.dataframe(df.head())

        expected_cols = ["footfall","tempMode","AQ","USS","CS","VOC","RP","IP","Temperature"]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            X = df[expected_cols].apply(pd.to_numeric, errors="coerce").dropna()
            st.info(f"Rows used for prediction after cleaning: {X.shape[0]}")

            if st.button("▶ Run Predictions"):
                if model is None:
                    st.error("Model not found. Please add `model.pkl`.")
                else:
                    preds = model.predict(X)
                    df_result = df.copy()
                    df_result["prediction"] = preds
                    st.dataframe(df_result.head(20))

                    # Chart: counts of predicted outcomes
                    counts = pd.Series(preds).value_counts().sort_index()
                    fig, ax = plt.subplots()
                    ax.bar(["Safe (0)", "Failure (1)"], counts, color=["green","red"])
                    ax.set_ylabel("Count")
                    ax.set_title("Prediction Outcomes")
                    st.pyplot(fig)

                    # Download button
                    csv = df_result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "💾 Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
