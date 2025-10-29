# This is the code for app.py

import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
try:
    model = joblib.load('flood_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler files not found. Make sure they are in the same directory.")
    st.stop()

# Define the feature names in the order your model expects them
feature_names = [
    'Latitude', 'Longitude', 'Total Deaths', 'Total Affected',
    'duration', 'time', 'Rainfall', 'Elevation', 'Slope', 'distance'
]

# Set up the Streamlit page
st.set_page_config(page_title="Flood Risk Predictor", page_icon="🌊", layout="centered")
st.title('🌊 Flood Risk Prediction Model')
st.write("This app predicts flood risk using a Random Forest model.")

# Create input fields
st.header("Enter Event Features:")

col1, col2 = st.columns(2)
with col1:
    latitude = st.number_input('Latitude', format="%.4f")
    longitude = st.number_input('Longitude', format="%.4f")
    total_deaths = st.number_input('Total Deaths', min_value=0, step=1)
    total_affected = st.number_input('Total Affected', min_value=0, step=1)
    duration = st.number_input('Duration (days)', min_value=0.0, format="%.2f")

with col2:
    time = st.number_input('Time (Year)', min_value=1900, max_value=2100, step=1, value=2024)
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, format="%.2f")
    elevation = st.number_input('Elevation (m)', min_value=0.0, format="%.2f")
    slope = st.number_input('Slope (degrees)', min_value=0.0, format="%.2f")
    distance = st.number_input('Distance (to water body)', min_value=0.0, format="%.2f")

# Create a 'Predict' button
if st.button('**Predict Flood Risk**', type="primary"):
    # Collect inputs
    input_features = [
        latitude, longitude, total_deaths, total_affected,
        duration, time, rainfall, elevation, slope, distance
    ]
    
    # Scale inputs
    input_array = np.array(input_features).reshape(1, -1)
    scaled_features = scaler.transform(input_array)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)

    # Display result
    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        probability = prediction_proba[0][1] * 100
        st.error(f'**High Flood Risk Detected!** (Probability: {probability:.2f}%)')
    else:
        probability = prediction_proba[0][1] * 100
        st.success(f'**Low Flood Risk.** (Flood Probability: {probability:.2f}%)')