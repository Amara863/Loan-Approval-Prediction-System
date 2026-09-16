import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    with open("loan_approval_model.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
feature_names = artifacts["feature_names"]
encoders = artifacts["encoders"]

st.title("🏦 Loan Approval Prediction System")
st.write("Predictive ML Engine for Real-Time Loan Eligibility")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Details")
    gender = st.selectbox("Gender", encoders["gender"].classes_)
    married = st.selectbox("Married", encoders["married"].classes_)
    dependents = st.selectbox("Dependents", encoders["dependents"].classes_)
    education = st.selectbox("Education", encoders["education"].classes_)

with col2:
    st.subheader("Employment & Area")
    self_employed = st.selectbox("Self Employed", encoders["self_employed"].classes_)
    property_area = st.selectbox("Property Area", encoders["property_area"].classes_)
    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0],
        format_func=lambda x: "Good (1.0)" if x == 1.0 else "Poor (0.0)"
    )

with col3:
    st.subheader("Financial Details")
    # Dataset values typical ranges: Income ~ 1,500 to 10,000 | Loan ~ 50 to 300
    applicant_income = st.number_input("Applicant Income (Monthly Units)", min_value=100, value=5000, step=250, help="Standard dataset monthly income scale (e.g., 5000)")
    coapplicant_income = st.number_input("Coapplicant Income (Monthly Units)", min_value=0, value=1500, step=250)
    loan_amount = st.number_input("Loan Amount (in Thousands)", min_value=1, value=120, step=10, help="e.g., 120 = ₹1,20,000")
    loan_term = st.number_input("Loan Term (in Months)", min_value=12, max_value=480, value=360, step=12)

st.markdown("---")

if st.button("Predict Loan Status", use_container_width=True):
    # Prepare input matching feature names exactly
    input_dict = {
        "gender": encoders["gender"].transform([str(gender)])[0],
        "married": encoders["married"].transform([str(married)])[0],
        "dependents": encoders["dependents"].transform([str(dependents)])[0],
        "education": encoders["education"].transform([str(education)])[0],
        "self_employed": encoders["self_employed"].transform([str(self_employed)])[0],
        "applicant_income": float(applicant_income),
        "coapplicant_income": float(coapplicant_income),
        "loan_amount": float(loan_amount),
        "loan_term": float(loan_term),
        "credit_history": float(credit_history),
        "property_area": encoders["property_area"].transform([str(property_area)])[0]
    }

    input_df = pd.DataFrame([input_dict])[feature_names]

    # Scaling features
    input_scaled = scaler.transform(input_df)

    # Predictions
    prediction = int(model.predict(input_scaled)[0])
    prob = model.predict_proba(input_scaled)[0]

    # Model class mapping check (0 = Rejected/N, 1 = Approved/Y)
    class_order = list(model.classes_)
    idx_approved = class_order.index(1) if 1 in class_order else 1
    idx_rejected = class_order.index(0) if 0 in class_order else 0

    approval_chance = prob[idx_approved] * 100
    rejection_chance = prob[idx_rejected] * 100

    st.info(f"**Approval Chance:** {approval_chance:.1f}% | **Rejection Chance:** {rejection_chance:.1f}%")

    if prediction == 1:
        st.success("### ✅ Loan Status: Approved")
    else:
        st.error("### ❌ Loan Status: Rejected")