import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# -----------------------------
# 1. Load or Train Model
# -----------------------------
@st.cache_resource
def train_or_load_model():
    if os.path.exists("loan_model.pkl"):
        model = joblib.load("loan_model.pkl")
        return model
    
    # Sample dataset (you can replace with bigger CSV dataset)
    data = {
        "Gender": ["Male","Female","Male","Male","Female","Male","Female","Female","Male","Male"],
        "Married": ["Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes"],
        "Education": ["Graduate","Not Graduate","Graduate","Graduate","Graduate","Not Graduate","Graduate","Graduate","Not Graduate","Graduate"],
        "ApplicantIncome": [5000, 3000, 4000, 2500, 6000, 3500, 2000, 8000, 2600, 4000],
        "LoanAmount": [130, 100, 120, 80, 200, 150, 60, 250, 90, 110],
        "Credit_History": [1,0,1,1,1,1,0,1,1,1],
        "Loan_Status": ["Y","N","Y","Y","Y","N","N","Y","N","Y"]
    }
    df = pd.DataFrame(data)

    # Encoding categorical features
    le = LabelEncoder()
    for col in ["Gender","Married","Education","Loan_Status"]:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "loan_model.pkl")
    return model

# -----------------------------
# 2. Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="AI Loan Approval Prediction", page_icon="💰", layout="centered")
    st.title("💰 AI Based Loan Approval Prediction")
    st.write("Enter applicant details to check **loan approval probability**")

    # Load model
    model = train_or_load_model()

    # Input form
    gender = st.selectbox("Gender", ["Male","Female"])
    married = st.selectbox("Married", ["Yes","No"])
    education = st.selectbox("Education", ["Graduate","Not Graduate"])
    income = st.number_input("Applicant Income (₹)", min_value=1000, max_value=100000, step=500)
    loan_amt = st.number_input("Loan Amount (in thousands)", min_value=10, max_value=500000, step=10)
    credit = st.selectbox("Credit History (1 = Good, 0 = Bad)", [1,0])

    # Encode inputs
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0

    input_data = np.array([[gender, married, education, income, loan_amt, credit]])

    if st.button("🔍 Predict Loan Approval"):
        prediction = model.predict(input_data)[0]
        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
        st.subheader(result)

        # Probability
        prob = model.predict_proba(input_data)[0][1]
        st.progress(int(prob*100))
        st.info(f"Approval Probability: {prob*100:.2f}%")

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    main()
