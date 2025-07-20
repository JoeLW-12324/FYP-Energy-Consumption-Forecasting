import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from datetime import datetime
import xgboost as xgb
from skops.io import load

trusted_types = [
    "__main__.binary_map",
    "sklearn.compose._column_transformer._RemainderColsList"
]

# Custom transformer to map On/Off and Yes/No to 1/0
def binary_map(X):
    return X.replace({'On': 1, 'Off': 0, 'Yes': 1, 'No': 0}).infer_objects(copy=False).astype(int)

# for implementing safe lag features
def safe_lag(df, lag, default):
    try:
        return df.iloc[-lag]["PredictedEnergy"]
    except (IndexError, KeyError):
        return default


# Load models and preprocessors
xgb_model = xgb.Booster()
xgb_model.load_model("best_xgb_model.json")
gbm_model = joblib.load("best_gbm_model.pkl")
preprocessor = load("preprocessor.skops", trusted=trusted_types)

# --- Streamlit Page Setup ---
st.set_page_config(page_title="⚡ Energy AI", page_icon="⚡", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f7f9fc; }
    .stButton > button { background-color: #4CAF50; color: white; font-size: 16px; border-radius: 10px; }
    .stDownloadButton > button { background-color: #1976D2; color: white; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Energy Consumption Predictor")
st.markdown("Enter building conditions below to forecast energy usage.")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🔧 Settings")
    show_plot = st.checkbox("Show Feature Contribution", value=True)
    st.markdown("""---
    ⚠️ This tool is a predictive AI system trained with advanced ML models.
    """)

# Initialize session state variable
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = pd.DataFrame()

# --- Input Form ---
st.subheader("📥 Enter Input Parameters")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("🌡️ Temperature (°C)", 10.0, 50.0, 25.0)
        humidity = st.slider("💧 Humidity (%)", 10.0, 100.0, 50.0)
        square_footage = st.number_input("🏢 Square Footage", min_value=100.0, value=1500.0)
        occupancy = st.slider("👥 Occupancy", 0, 50, 5)
    with col2:
        hvac_usage = st.selectbox("HVAC Usage", ["On", "Off"])
        lighting_usage = st.selectbox("Lighting Usage", ["On", "Off"])
        renewable_energy = st.slider("Renewable Energy (kWh)", 0.0, 30.0, 5.0)
        holiday = st.selectbox("Holiday", ["Yes", "No"])

    submitted = st.form_submit_button("🔮 Predict Energy Usage")

# --- Prediction Logic ---
if submitted:
    # 1. Timestamp features
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    hour = now.hour
    day = now.day
    month = now.month
    year = now.year
    day_of_week = now.strftime('%A')

    # Historical lag fallback
    hist_df = st.session_state.prediction_history
    lag_1 = safe_lag(hist_df, 1, 75.0)
    lag_2 = safe_lag(hist_df, 2, 75.0)
    lag_24 = safe_lag(hist_df, 24, 75.0)

    rolling_24 = hist_df["PredictedEnergy"].tail(24).mean() if len(hist_df) >= 24 else 75.0

    # 2. Assemble input as DataFrame
    input_data = pd.DataFrame([{
        "Temperature": temperature,
        "Humidity": humidity,
        "SquareFootage": square_footage,
        "Occupancy": occupancy,
        "HVACUsage": hvac_usage,
        "LightingUsage": lighting_usage,
        "RenewableEnergy": renewable_energy,
        "Holiday": holiday,
        "DayOfWeek": day_of_week,
        "Hour": hour,
        "Day": day,
        "Month": month,
        "Year": year,
        "energy_lag_1": lag_1,
        "energy_lag_2": lag_2,
        "energy_lag_24": lag_24,
        "rolling_mean_24": rolling_24
    }])

    # 4. Preprocess input
    input_processed = preprocessor.transform(input_data)

    # 5. Predict from both models
    pred_xgb = xgb_model.predict(input_processed)
    pred_gbm = gbm_model.predict(input_processed)

    # 6. Average predictions
    final_prediction = (pred_xgb + pred_gbm) / 2

    # 7. Display result
    st.success(f"🔋 Estimated Energy Consumption: **{final_prediction.item():.2f} kWh**")

    # Update history
    new_row = input_data.copy()
    new_row["PredictedEnergy"] = final_prediction
    new_row["Timestamp"] = timestamp
    st.session_state.prediction_history = pd.concat(
        [st.session_state.prediction_history, new_row], ignore_index=True)

    # 8. Plot input breakdown
    if show_plot:
        plot_df = pd.DataFrame({
            "Feature": ["Temperature", "Humidity", "SquareFootage", "Occupancy", "RenewableEnergy"],
            "Value": [temperature, humidity, square_footage, occupancy, renewable_energy]
        })
        fig = px.bar(plot_df, x="Feature", y="Value", color="Feature", text="Value")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # 9. Download CSV
    result_df = input_data.copy()
    result_df["Predicted_EnergyConsumption"] = final_prediction
    st.download_button("📥 Download Prediction", result_df.to_csv(index=False), "prediction_result.csv")

    # Plot energy timeline
    st.markdown("---")
    st.subheader("📊 Energy Forecast Over Time")
    fig = px.line(
        st.session_state.prediction_history,
        x="Timestamp", y="PredictedEnergy",
        title="Predicted Energy Consumption",
        markers=True
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# --- About ---
with st.expander("ℹ️ About This Tool"):
    st.write("""
    This smart AI model uses a hybrid of XGBoost and Gradient Boosting Machines (GBM) trained on real energy usage data. 
    The prediction leverages environmental, appliance, and temporal features to estimate energy consumption accurately.

    ✅ Model: Hybrid XGB + GBM  
    ✅ Inputs: Temperature, Humidity, Square Footage, Usage, Renewable Energy, and Time Context  
    ✅ Features: Automatic timestamp detection, one-hot encoding, lag features

    Developed as part of a final year AI energy optimization project.
    """)
