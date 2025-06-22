import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gzip
from pathlib import Path

model_path = Path(__file__).parent / "model.joblib.gz"
# Load model
with gzip.open(model_path, "rb") as f:
    model = joblib.load(f)


# App title
st.title("Credit Risk Prediction App")

# Input fields
person_age = st.number_input("Age", min_value=18, max_value=100, step=1)
person_income = st.number_input("Annual Income (₹)", min_value=1000, step=500)

person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
home_ownership_map = {"RENT": 0, "OWN": 1, "MORTGAGE": 2}
home_ownership_encoded = home_ownership_map[person_home_ownership]

person_emp_length = st.number_input("Employment Length (in years)", min_value=0, max_value=50, step=1)

loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"])
loan_intent_map = {"PERSONAL": 0, "EDUCATION": 1, "MEDICAL": 2, "VENTURE": 3, "HOMEIMPROVEMENT": 4}
loan_intent_encoded = loan_intent_map[loan_intent]

loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E"])
loan_grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
loan_grade_encoded = loan_grade_map[loan_grade]

loan_amnt = st.number_input("Loan Amount", min_value=500, step=500)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, step=0.1)

loan_percent_income = round(loan_amnt / person_income, 2) if person_income > 0 else 0.0
st.write(f"📊 Loan as Percent of Income: **{loan_percent_income}**")

cb_person_default_on_file = st.selectbox("Has Defaulted Before?", ["Y", "N"])
default_map = {"Y": 1, "N": 0}
default_encoded = default_map[cb_person_default_on_file]

# Combine inputs
input_df = pd.DataFrame([{
    "person_age": person_age,
    "person_income": person_income,
    "person_home_ownership": person_home_ownership,
    "person_emp_length": person_emp_length,
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_default_on_file": cb_person_default_on_file
}])



# Predict button
if st.button("Predict Credit Risk"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ High Risk: The applicant is likely to **default** on the loan.")
    else:
        st.success("✅ Low Risk: The applicant is likely to **repay** the loan.")
