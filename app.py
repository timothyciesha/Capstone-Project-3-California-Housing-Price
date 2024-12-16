import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
MODEL_PATH = 'final_lightgbm_model.pkl'
model = joblib.load(MODEL_PATH)

# App title
st.title("California Housing Price Prediction")
st.write("This app predicts the median house value based on input features using the LightGBM model.")

# Sidebar navigation
menu = st.sidebar.selectbox("Navigation", ["Home", "Single Prediction"])

# Home page
if menu == "Home":
    st.header("Summary of Results")
    st.write("**Selected Features:**")
    st.write("['median_income', 'longitude', 'latitude', 'ocean_proximity_INLAND', 'housing_median_age', 'rooms_per_household', 'bedrooms_per_room', 'population_per_household']")
    st.write("**Best Performing Model:** LightGBM")
    st.write("**Test RMSE:** 49006.1362")
    st.write("**Test R²:** 0.8264")
    st.write("**Residuals Mean:** 205.7978")
    st.write("**Residuals Standard Deviation:** 49005.7041")

# Single prediction page
elif menu == "Single Prediction":
    st.header("Single Prediction")
    st.write("Provide input data to get a prediction.")

    # Input fields
    longitude = st.number_input("Longitude", value=-119.79, step=0.01)
    latitude = st.number_input("Latitude", value=36.73, step=0.01)
    housing_median_age = st.number_input("Housing Median Age", value=52.0, step=1.0)
    total_rooms = st.number_input("Total Rooms", value=112.0, step=1.0)
    total_bedrooms = st.number_input("Total Bedrooms", value=28.0, step=1.0)
    population = st.number_input("Population", value=193.0, step=1.0)
    households = st.number_input("Households", value=40.0, step=1.0)
    median_income = st.number_input("Median Income", value=1.9750, step=0.01)
    ocean_proximity_INLAND = st.selectbox("Ocean Proximity is INLAND?", ["Yes", "No"])

    # Preprocess input data
    rooms_per_household = total_rooms / households if households else 0
    bedrooms_per_room = total_bedrooms / total_rooms if total_rooms else 0
    population_per_household = population / households if households else 0

    # One-hot encoding
    ocean_proximity_INLAND = 1 if ocean_proximity_INLAND == "Yes" else 0

    # Prepare input for prediction
    input_data = np.array([
        [median_income, longitude, latitude, ocean_proximity_INLAND, 
         housing_median_age, rooms_per_household, bedrooms_per_room, population_per_household]
    ])

    # Predict button
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Median House Value: ${prediction:,.2f}")
