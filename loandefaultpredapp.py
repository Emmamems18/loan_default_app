#IMPORT NECESSARY LIBRARIES
import streamlit as st
import pandas as pd
import numpy as np
import joblib

#LOAD YOUR SAVED MODEL AND TRANSFORMER
model = joblib.load("loan_predictor.pkl")
power_transformer = joblib.load("power_transformer.pkl")

#ADD THE STREAMLIT APP TITLE
st.title("🏦 Loan Default Risk Prediction App")
st.write("Enter the applicant's details below to predict the risk of loan default.")

#ACCEPT USER INPUT
st.header("Applicant Information")
age = st.slider('Applicant Age', 18, 65, 25)
zone = st.selectbox('Geographical Zone', ['South West', 'North Central', 'South South', 'South East', 'North West', 'North East', 'Unknown'])

st.header("Loan Details")
loanamount = st.number_input("Loan Amount ($)", min_value=5000, max_value=100000, value=20000)
termdays = st.slider("Loan Term (Days)", 15, 90, 30)

st.header("Applicant's Financial History")
number_previous_loans = st.number_input("Number of Previous Loans", min_value=0,max_value=50, value=5)
average_loan_amount = st.number_input("Average Previous Loan Amount ($)", min_value=0, max_value=100000, value=15000)
on_time_repayment_rate = st.slider("On-Time Repayment Rate", 0.0, 1.0, 0.70, help="Historical rate of on-time payments (1.0 = 100%).")
average_days_repaid_after_due = st.number_input("Average Days Repaid After Due Date", min_value=-30, max_value=100, value=0)
average_termdays = st.slider("Average Term of Previous Loans (Days)", 0.0, 90.0, 30.0)
average_loan_duration_days = st.slider("Average Actual Duration of Previous Loans (Days)", 0.0, 90.0, 30.0)

#PREDICT BUTTON AND LOGIC
if st.button("Predict Loan Default Risk"):

    # Create a dictionary from the user's input.
    user_input_data = {
        'loanamount': float(loanamount),
        'termdays': float(termdays),
        'age': float(age),
        'zone': 'Unknown' if 'Unknown' in zone else zone,
        'number_previous_loans': float(number_previous_loans),
        'average_loan_amount': float(average_loan_amount),
        'average_termdays': float(average_termdays),
        'average_days_repaid_after_due': float(average_days_repaid_after_due),
        'average_loan_duration_days': float(average_loan_duration_days),
        'on_time_repayment_rate': float(on_time_repayment_rate)
    }

    # Convert the dictionary into a single-row DataFrame.
    new_df = pd.DataFrame([user_input_data])

    #Apply the EXACT SAME transformations as in your training notebook ---
    # a) Feature Engineering: Create 'loan_per_day'
    new_df['loan_per_day'] = new_df['loanamount'] / new_df['termdays']
    
    # b) Log Transform (log1p) for all columns with skew > 1.0
    log_transform_cols = [
        'average_loan_amount', 'loanamount', 'number_previous_loans',
        'termdays', 'average_days_repaid_after_due', 'average_loan_duration_days',
        'loan_per_day']
    for col in log_transform_cols:
        #We use clip(lower=0) to prevent errors from negative inputs if any
        new_df[col] = np.log1p(new_df[col].clip(lower=0))
        
    # c) Square Root Transform for all columns with skew > 0.5
    sqrt_transform_cols = ['age']
    for col in sqrt_transform_cols:
        new_df[col] = np.sqrt(new_df[col])

    # d) Transformation: Apply the loaded PowerTransformer to 'on_time_repayment_rate'
    new_df[['on_time_repayment_rate']] = power_transformer.transform(new_df[['on_time_repayment_rate']])

    #Ensure the column order is IDENTICAL to the training data ---
    final_feature = [
        'loanamount', 'termdays', 'age', 'zone', 'number_previous_loans',
        'average_loan_amount', 'average_termdays', 'average_days_repaid_after_due',
        'average_loan_duration_days', 'on_time_repayment_rate', 'loan_per_day'
    ]
    final_df = new_df[final_feature]

    #Make the prediction
    #We predict the probability. [0][1] gives the probability of the "default" class.
    probability_of_default = model.predict_proba(final_df)[0][1]

    #Define your risk threshold
    risk_threshold = 0.30

    #Display the final prediction to the user
    # This provides a clear, informative result, just like your examples but slightly enhanced.
    if probability_of_default >= risk_threshold:
        st.error("⚠️ RISKY LOAN: This applicant is likely to default.")
    else:
        st.success("✅ SAFE LOAN: This applicant is likely to repay.")
