import pickle
import streamlit as st
import pandas as pd
import os

# -------------------------------
# Load model + scaler
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# -------------------------------
# Get trained columns
# -------------------------------
model_columns = model.feature_names_in_

# -------------------------------
# UI
# -------------------------------
st.title("📊 Customer Churn Prediction")

# -------- basic inputs --------
tenure = st.number_input("Tenure", 0, 72, 12)
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
Total_Charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

gender = st.selectbox("Gender", ("Male", "Female"))
Partner = st.selectbox("Partner", ("Yes", "No"))
Dependents = st.selectbox("Dependents", ("Yes", "No"))
PhoneService = st.selectbox("Phone Service", ("Yes", "No"))
PaperlessBilling = st.selectbox("Paperless Billing", ("Yes", "No"))

Contract = st.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
InternetService = st.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
PaymentMethod = st.selectbox("Payment Method",
                            ("Electronic check",
                             "Mailed check",
                             "Bank transfer (automatic)",
                             "Credit card (automatic)"))

# -------------------------------
# Encoding
# -------------------------------
gender = 1 if gender == "Male" else 0
Partner = 1 if Partner == "Yes" else 0
Dependents = 1 if Dependents == "Yes" else 0
PhoneService = 1 if PhoneService == "Yes" else 0
PaperlessBilling = 1 if PaperlessBilling == "Yes" else 0

# -------------------------------
# Create FULL input dataframe
# -------------------------------
input_data = pd.DataFrame(columns=model_columns)
input_data.loc[0] = 0  # set all features = 0

# -------- numeric --------
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = MonthlyCharges
input_data['Total_Charges'] = Total_Charges
input_data['gender'] = gender
input_data['Partner'] = Partner
input_data['Dependents'] = Dependents
input_data['PhoneService'] = PhoneService
input_data['PaperlessBilling'] = PaperlessBilling

# -------------------------------
# One-hot encoding mapping
# -------------------------------

# Contract
input_data[f'Contract_{Contract}'] = 1

# Internet
input_data[f'InternetService_{InternetService}'] = 1

# Payment
input_data[f'PaymentMethod_{PaymentMethod}'] = 1
# -------------------------------
# Set default values for missing service features
# -------------------------------
defaults = [
    'OnlineSecurity_Yes',
    'OnlineBackup_Yes',
    'DeviceProtection_Yes',
    'TechSupport_Yes',
    'StreamingTV_No',
    'StreamingMovies_No',
    'MultipleLines_Yes'
]

for col in defaults:
    if col in input_data.columns:
        input_data[col] = 1
# -------------------------------
# Scaling
# -------------------------------
input_data[['tenure','MonthlyCharges','Total_Charges']] = scaler.transform(
    input_data[['tenure','MonthlyCharges','Total_Charges']]
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ Customer Likely to Churn")
    else:
        st.success("✅ Customer Likely to Stay")