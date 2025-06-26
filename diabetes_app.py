import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Set page config
st.set_page_config(page_title="Diabetes Predictor", layout="centered")

# App title with markdown
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🩺 Diabetes Prediction App</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("""
Welcome to the **Diabetes Prediction App**.  
Enter the following details in the sidebar to check whether the person is likely to have diabetes or not.

---
""")

# Sidebar inputs
st.sidebar.header("📝 Enter Patient Details")

preg = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.slider("Glucose", 0, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 0, 122, 70)
skinthick = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0.0, 900.0, 80.0)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 10, 100, 30)

# Input array and prediction
input_data = np.array([[preg, glucose, bp, skinthick, insulin, bmi, dpf, age]])
scaled_input = scaler.transform(input_data)

# Predict button
if st.button("🔍 Predict Diabetes Status"):
    prediction = model.predict(scaled_input)
    result = "✅ **Not Diabetic**" if prediction[0] == 0 else "🚨 **Diabetic**"
    
    st.markdown("---")
    st.markdown(f"### 🧾 Prediction Result: {result}")
    

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2025 Diabetes Predictor | Powered by Streamlit</p>", unsafe_allow_html=True)
