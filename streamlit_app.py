import streamlit as st
import numpy as np
import pickle
import os

model_path = os.path.join("models", "best_model.pkl")
scaler_path = os.path.join("models", "scaler.pkl")
encoder_path = os.path.join("models", "label_encoder.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
le = pickle.load(open(encoder_path, "rb"))

st.title("🎓 Student Grade Predictor")

st.write("Enter student details to predict final grade")

# Inputs
age = st.number_input("Age")
study_hours = st.number_input("Study Hours")
attendance = st.number_input("Attendance Percentage")
math = st.number_input("Math Score")
science = st.number_input("Science Score")
english = st.number_input("English Score")
overall = st.number_input("Overall Score")

# Predict button
if st.button("Predict Grade"):

    features = np.array([[age, study_hours, attendance,
                          math, science, english, overall]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    grade = le.inverse_transform(prediction)

    st.success(f"Predicted Grade: {grade[0]}")

st.markdown("---")
st.subheader("📊 Model Comparison")

st.image("outputs/model_comparison.png")


st.subheader("🔲 Confusion Matrix")
st.image("outputs/confusion_matrix.png")


st.subheader("📌 Feature Importance")
st.image("outputs/feature_importance.png")
