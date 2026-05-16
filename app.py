
import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("student_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Page settings
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    color: #1E3A8A;
    font-size: 40px;
    font-weight: bold;
}

.sub {
    text-align: center;
    color: gray;
    font-size: 18px;
}

.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    '<p class="title">🎓 Student Performance Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub">Predict student academic performance using Machine Learning</p>',
    unsafe_allow_html=True
)

st.write("---")

# Sidebar
st.sidebar.header("📘 About Project")

st.sidebar.info(
    """
    This project predicts student performance based on:
    
    • Study Hours  
    • Attendance  
    • Previous Marks  
    • Internet Access  
    • Extra Classes
    """
)

# Input fields
gender = st.selectbox(
    "👤 Gender",
    ["Male", "Female"]
)

study_hours = st.slider(
    "📚 Study Hours per Day",
    0, 12, 4
)

attendance = st.slider(
    "📝 Attendance Percentage",
    0, 100, 75
)

previous_marks = st.slider(
    "📊 Previous Marks",
    0, 100, 70
)

internet_access = st.selectbox(
    "🌐 Internet Access",
    ["Yes", "No"]
)

extra_classes = st.selectbox(
    "🏫 Extra Classes",
    ["Yes", "No"]
)

st.write("")

# Prediction button
if st.button("Predict Performance"):

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks],
        "Internet_Access": [internet_access],
        "Extra_Classes": [extra_classes]
    })

    # Encode inputs
    input_data["Gender"] = label_encoders["Gender"].transform(
        input_data["Gender"]
    )

    input_data["Internet_Access"] = label_encoders[
        "Internet_Access"
    ].transform(input_data["Internet_Access"])

    input_data["Extra_Classes"] = label_encoders[
        "Extra_Classes"
    ].transform(input_data["Extra_Classes"])

    # Prediction
    prediction = model.predict(input_data)

    result = label_encoders["Performance"].inverse_transform(
        prediction
    )

    st.success(f"🎯 Predicted Performance: {result[0]}")

    # Result Messages
    if result[0] == "Excellent":
        st.balloons()
        st.success("Excellent academic performance!")

    elif result[0] == "Good":
        st.info("Good performance. Keep improving!")

    elif result[0] == "Average":
        st.warning("Average performance. More focus needed.")

    else:
        st.error("Poor performance. Needs improvement.")

