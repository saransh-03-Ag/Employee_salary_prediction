import streamlit as st
import os

st.write("📂 Working directory:", os.getcwd())
st.write("📁 Files here:", os.listdir())

import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the trained model
model = joblib.load("best_model.pkl")



# Custom Page Title
st.set_page_config(page_title="Income Prediction", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4B8BBE;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .main {
            background-color: #F5F5F5;
            padding: 20px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>💼 Income Prediction App</h1>", unsafe_allow_html=True)
st.markdown("### 👇 Please enter the following details to predict your income class:")

# Input layout using columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=0, max_value=100, value=30)
    fnlwgt = st.number_input("📊 Fnlwgt", min_value=0, max_value=1000000, value=100000)
    educational_num = st.number_input("🎓 Educational Number", min_value=0, max_value=20, value=10)
    marital_status = st.number_input("💍 Marital Status (Encoded)", min_value=0, max_value=10, value=1)
    occupation = st.number_input("🛠️ Occupation (Encoded)", min_value=0, max_value=20, value=5)
    race = st.number_input("🌎 Race (Encoded)", min_value=0, max_value=10, value=1)
    capital_gain = st.number_input("📈 Capital Gain", min_value=0, max_value=100000, value=0)

with col2:
    workclass = st.number_input("🏢 Workclass (Encoded)", min_value=0, max_value=10, value=1)
    relationship = st.number_input("👥 Relationship (Encoded)", min_value=0, max_value=10, value=1)
    gender = st.number_input("⚧️ Gender (0=Female, 1=Male)", min_value=0, max_value=1, value=1)
    capital_loss = st.number_input("📉 Capital Loss", min_value=0, max_value=100000, value=0)
    hours_per_week = st.number_input("🕒 Hours Per Week", min_value=0, max_value=168, value=40)
    native_country = st.number_input("🌐 Native Country (Encoded)", min_value=0, max_value=100, value=1)

# Prediction button
if st.button("🔮 Predict Income"):
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'educational-num': [educational_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

 # After getting the prediction
 
prediction = model.predict(input_data)[0]

st.markdown("---")
st.subheader("🎯 Prediction Result")
st.success(f"**Predicted Income:** {prediction}")

# Add custom comment based on prediction
if prediction == '<=50K':
    st.info("It seems your predicted income is on the lower side. Consider ways to improve your skills or explore higher-paying job opportunities!")
else:
    st.info("Great! Your predicted income is above 50K. Keep up the good work!")

