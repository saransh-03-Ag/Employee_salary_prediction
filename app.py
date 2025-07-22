import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load model and features
model = joblib.load("best_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Set Streamlit page config
st.set_page_config(page_title="Income Prediction", layout="centered")

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

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>💼 Employee Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("### 👇 Please enter the following details to predict your income class:")

# Input layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=0, max_value=100, value=30)
    fnlwgt = st.number_input("📊 Fnlwgt", min_value=0, max_value=1000000, value=100000)
    marital_status = st.text_input("💍 Marital Status", value="Never-married")
    occupation = st.text_input("🛠️ Occupation", value="Exec-managerial")
    race = st.text_input("🌎 Race", value="White")
    capital_gain = st.number_input("📈 Capital Gain", min_value=0, max_value=100000, value=0)

with col2:
    workclass = st.text_input("🏢 Workclass", value="Private")
    relationship = st.text_input("👥 Relationship", value="Not-in-family")
    gender = st.text_input("⚧️ Gender", value="Male")
    capital_loss = st.number_input("📉 Capital Loss", min_value=0, max_value=100000, value=0)
    hours_per_week = st.number_input("🕒 Hours Per Week", min_value=0, max_value=168, value=40)
    native_country = st.text_input("🌐 Native Country", value="United-States")

# Predict button
if st.button("🔮 Predict Income"):
    input_dict = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }

    try:
        input_df = pd.DataFrame([input_dict])[feature_names]
        prediction = model.predict(input_df)[0]

        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        if prediction == 0:
            st.success("**Predicted Income:** <=50K")
        else:
            st.success("**Predicted Income:** >50K")

    except KeyError as e:
        st.error(f"❌ Input mismatch: {e}")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center; color: gray; font-size: 12px;'>
        Created by <strong>Saransh Agnihotri</strong> 🚀
    </p>
""", unsafe_allow_html=True)
