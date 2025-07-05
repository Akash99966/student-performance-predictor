import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the data
df = pd.read_csv("student_data.csv")

# Preprocessing
X = df[["StudyHours", "Attendance", "PreviousScore", "SleepHours"]]
y = df["Result"].map({"Fail": 0, "Pass": 1})

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# --- Streamlit UI ---
st.title("🎓 Student Performance Predictor")
st.markdown("Predict whether a student will Pass or Fail based on inputs.")

# Sidebar inputs
study = st.number_input("📘 Study Hours per Day", min_value=0.0, max_value=10.0, step=0.1, value=3.0)
attendance = st.number_input("🧑‍🏫 Attendance (%)", min_value=0, max_value=100, step=1, value=75)
score = st.number_input("📝 Previous Score (out of 100)", min_value=0, max_value=100, step=1, value=60)
sleep = st.number_input("🛌 Sleep Hours per Night", min_value=0.0, max_value=12.0, step=0.1, value=6.0)

# Predict
if st.button("Predict Result"):
    input_df = pd.DataFrame([[study, attendance, score, sleep]], columns=X.columns)
    result = model.predict(input_df)[0]
    if result == 1:
        st.success("✅ The student is likely to **Pass**.")
    else:
        st.error("❌ The student is likely to **Fail**.")
