import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model pipeline
model = joblib.load('churn_predict_model.pkl')

st.set_page_config(page_title="Customer Retention Hub", layout="wide")
st.title("🎯 Customer Churn Risk Predictor")
st.write("Input customer details to analyze their likelihood of leaving the service.")

st.markdown("---")

# Create two columns for a clean input layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Contract & Charges")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=780.0)

with col2:
    st.subheader("🛠️ Services Subscribed")
    internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support Subscribed?", ["Yes", "No", "No internet service"])
    online_security = st.selectbox("Online Security?", ["Yes", "No", "No internet service"])
    paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])

# Logic for prediction
st.markdown("---")
if st.button("🔮 Run Risk Analysis", use_container_width=True):
    # Construct a dataframe matching the raw column names expected by your notebook pipeline
    # Note: We fill dummy values for columns we didn't include in the UI so the model pipeline doesn't crash
    input_dict = {
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': tech_support,
        'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': contract,
        'PaperlessBilling': paperless_billing, 'PaymentMethod': 'Electronic check',
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Calculate probabilities
    probabilities = model.predict_proba(input_df)[0]
    churn_probability = round(probabilities[1] * 100, 2)
    
    # Display Results Dashboard
    st.subheader("📈 Risk Assessment Analysis")
    
    if churn_probability > 70:
        st.error(f"⚠️ High Churn Risk: {churn_probability}%")
        st.markdown("**Management Action Item:** This customer is highly likely to cancel. Recommend immediate outreach offering a loyalty discount or contract upgrade incentives.")
    elif 30 <= churn_probability <= 70:
        st.warning(f"⚡ Medium Churn Risk: {churn_probability}%")
        st.markdown("**Management Action Item:** Monitor account closely. Flag for targeted email engagement campaigns.")
    else:
        st.success(f"✅ Low Churn Risk: {churn_probability}%")
        st.markdown("**Management Action Item:** Account stable. No immediate retention intervention required.")
