import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# Load Trained Model
@st.cache_resource
def load_model():
    with open("loan_approved_status.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()

# Header
st.title("🏦 Loan Approval Prediction System")
st.write("Fill in the applicant details to evaluate loan eligibility.")
st.markdown("---")

# 3-Column Symmetrical Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

with col2:
    st.subheader("Employment & Area")
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0],
        format_func=lambda x: "Good (1.0)" if x == 1.0 else "Poor (0.0)"
    )

with col3:
    st.subheader("Financial Details")
    applicant_income = st.number_input("Applicant Income (₹/month)", min_value=0, value=50000, step=1000)
    coapplicant_income = st.number_input("Coapplicant Income (₹/month)", min_value=0, value=0, step=1000)
    loan_amount = st.number_input("Loan Amount (in Thousands ₹)", min_value=1, value=200, step=10)
    loan_amount_term = st.number_input("Loan Term (in Months/Days)", min_value=12, max_value=480, value=360, step=12)

st.markdown("---")

# Prediction Logic
if st.button("Predict Loan Status", use_container_width=True):
    # Model ke expected dummy columns ko dictionary form me banana
    input_data = {
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Married_Yes': 1 if married == "Yes" else 0,
        'Dependents_1': 1 if dependents == "1" else 0,
        'Dependents_2': 1 if dependents == "2" else 0,
        'Dependents_3+': 1 if dependents == "3+" else 0,
        'Education_Not Graduate': 1 if education == "Not Graduate" else 0,
        'Self_Employed_Yes': 1 if self_employed == "Yes" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0
    }

    input_df = pd.DataFrame([input_data])

    # Model ke feature names ke saath exactly map karna
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_df)

    # Display Result
    if prediction[0] == 1:
        st.success("### ✅ Loan Status: Approved")
    else:
        st.error("### ❌ Loan Status: Rejected")