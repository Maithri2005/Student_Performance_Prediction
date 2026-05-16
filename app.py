import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("student_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Student Performance Prediction", layout="centered")

st.title("🎓 Student Performance Prediction")
st.write("This app predicts student academic performance using Machine Learning.")

st.header("Enter Student Details")

gender = st.selectbox("Gender", ["Male", "Female"])
study_hours = st.number_input("Study Hours per Day", min_value=0, max_value=12, value=3)
attendance = st.number_input("Attendance Percentage", min_value=0, max_value=100, value=75)
previous_marks = st.number_input("Previous Marks", min_value=0, max_value=100, value=70)
internet_access = st.selectbox("Internet Access", ["Yes", "No"])
extra_classes = st.selectbox("Extra Classes", ["Yes", "No"])

if st.button("Predict Performance"):
    input_data = pd.DataFrame({
        "Gender": [gender],
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks],
        "Internet_Access": [internet_access],
        "Extra_Classes": [extra_classes]
    })

    # Encode input data
    input_data["Gender"] = label_encoders["Gender"].transform(input_data["Gender"])
    input_data["Internet_Access"] = label_encoders["Internet_Access"].transform(input_data["Internet_Access"])
    input_data["Extra_Classes"] = label_encoders["Extra_Classes"].transform(input_data["Extra_Classes"])

    prediction = model.predict(input_data)

    result = label_encoders["Performance"].inverse_transform(prediction)

    st.success(f"Predicted Student Performance: {result[0]}")