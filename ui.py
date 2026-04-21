import streamlit as st
import requests
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# --- Title ---
st.title("❤️ Heart Disease Prediction App")
st.markdown("### 🩺 Enter Patient Details")

st.markdown("---")

# --- Inputs ---
age = st.number_input("Age", min_value=1, max_value=120, value=30)

sex = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])

trestbps = st.number_input("Resting Blood Pressure", value=120)

chol = st.number_input("Cholesterol Level", value=200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

restecg = st.selectbox("Rest ECG Result", [0, 1, 2])

thalach = st.number_input("Max Heart Rate Achieved", value=150)

exang = st.selectbox("Exercise Induced Angina", [0, 1])

oldpeak = st.number_input("ST Depression", value=1.0)

slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])

ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])

thal = st.selectbox("Thalassemia", [0, 1, 2])

st.markdown("---")

# --- Predict Button ---
if st.button("🔍 Predict"):
    features = [
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]

    url = "https://ml-heart-app.onrender.com/predict"

    try:
        response = requests.post(url, json={"features": features})
        result = response.json()

        # --- Extract probability ---
        prob = result.get("probability", 0) * 100

        st.markdown("## 🧾 Result")

        # --- Result Display ---
        if result["prediction"] == 1:
            st.error(f"⚠️ High Risk of Heart Disease ({prob:.2f}%)")
        else:
            st.success(f"✅ Low Risk ({prob:.2f}%)")

        # --- Risk Visualization ---
        st.markdown("## 📊 Risk Visualization")

        # Bar Chart
        fig, ax = plt.subplots()
        ax.bar(["Risk"], [prob])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        ax.set_title("Heart Disease Risk (%)")

        st.pyplot(fig)

        # Progress bar (Gauge style)
        st.markdown("### Risk Level")
        st.progress(int(prob))

    except Exception as e:
        st.error("❌ API connection failed")
        st.write(e)