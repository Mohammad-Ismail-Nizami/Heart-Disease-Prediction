import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/heart_disease_model.pkl")

model = load_model()




# logo
st.image("assets/heart.png", width=150)

st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app predicts heart disease risk using machine learning.

Developed as a mini project.
""")

# Title Section
st.title("❤️ Heart Disease Prediction")

st.markdown("## 🏥 AI-Based Heart Disease Detection System")

st.info("Enter patient details below to predict the likelihood of heart disease.")

st.markdown("""
This application uses a machine learning model to analyze clinical parameters and predict whether a person is at risk of heart disease.

⚠️ **Note:** This is not a medical diagnosis tool. Please consult a doctor for professional advice.
""")

st.markdown("---")

st.header("📝 Patient Information")

# Layout using columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    # cp = st.selectbox("Chest Pain Type", [
    #     "Typical Angina",
    #     "Atypical Angina",
    #     "Non-anginal Pain",
    #     "Asymptomatic"
    # ])
    cp_option = st.selectbox("Chest Pain Type", [
    "Typical Angina",
    "Atypical Angina",
    "Non-anginal Pain",
    "Asymptomatic"
    ])
    trestbps = st.number_input("Resting Blood Pressure", value=120)
    chol = st.number_input("Cholesterol", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])

with col2:
    restecg = st.selectbox("Rest ECG", [
        "Normal",
        "ST-T wave abnormality",
        "Left ventricular hypertrophy"
    ])
    thalach = st.number_input("Max Heart Rate", value=150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("Oldpeak", value=1.0)
    slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Number of Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [
        "Normal",
        "Fixed Defect",
        "Reversible Defect"
    ])

# Convert inputs
sex = 1 if sex == "Male" else 0
cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp_option)
fbs = 1 if fbs == "Yes" else 0
restecg = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg)
exang = 1 if exang == "Yes" else 0
slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
thal = {"Normal": 1, "Fixed Defect": 6, "Reversible Defect": 7}[thal]

st.markdown("---")

# Prediction

if st.button("🔍 Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    prob_percent = probability * 100

    # Risk Levels
    if prob_percent < 30:
        st.success(f"🟢 Low Risk of Heart Disease\n\nProbability: {prob_percent:.2f}%")
        st.info("The model predicts a low likelihood of heart disease. Maintain a healthy lifestyle.")

    elif prob_percent < 60:
        st.warning(f"🟡 Medium Risk of Heart Disease\n\nProbability: {prob_percent:.2f}%")
        st.info("The model indicates a moderate risk. Consider consulting a doctor.")

    else:
        st.error(f"🔴 High Risk of Heart Disease\n\nProbability: {prob_percent:.2f}%")
        st.info("The model predicts a high risk. Please consult a doctor.")




    st.subheader("📊 Prediction Confidence")

    fig, ax = plt.subplots(figsize=(6,4))

    labels = ["No Disease", "Disease"]
    values = [1 - probability, probability]

    bars = ax.bar(labels, values)

# Add values on top
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
            f"{height*100:.1f}%", ha='center', va='bottom')

    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Model Confidence")
    st.pyplot(fig) 


    st.subheader("📈 Feature Importance")

    importance = np.abs(model.coef_[0])
    features = ["age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal"]
    # Sort features
    sorted_idx = np.argsort(importance)[::-1]

    fig2, ax2 = plt.subplots(figsize=(8,6))
    ax2.barh(
    np.array(features)[sorted_idx],
    importance[sorted_idx])

    ax2.invert_yaxis()
    ax2.set_title("Feature Importance (Sorted)")

    st.pyplot(fig2)


    st.subheader("📄 Download Report")

    report = f"""
    Heart Disease Prediction Report

    Prediction: {"High Risk" if prediction == 1 else "Low Risk"}
    Probability: {prob_percent:.2f}%

    Patient Data:
    Age: {age}
    Sex: {"Male" if sex == 1 else "Female"}
    Chest Pain Type: {cp_option}
    Cholesterol: {chol}
    """

    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="heart_disease_report.txt",
        mime="text/plain"
    )

# st.markdown("---")
st.markdown("<h1 style='text-align: center;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)

st.caption("⚠️ This is a machine learning model and not a medical diagnosis tool.")
