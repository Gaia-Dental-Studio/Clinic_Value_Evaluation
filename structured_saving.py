import streamlit as st
import pandas as pd
import numpy as np
from model_for_strategy import ModelStructuredSaving

def app():
    st.title('Structured Saving')

    st.write("This model simulates the saving of a structured saving account. The user can choose the amount to save, the interest rate, and the number of years to save for.")

    st.markdown("#### Input Parameters")


    scheme = st.radio("Select Scheme", ("Fixed Rate Saving", "Percentage-Based Saving"), help="Select the saving scheme. Fixed Rate Saving is a fixed amount of saving every month. Percentage-Based Saving is a saving amount based on the percentage of the gross profit.")




    col1, col2 = st.columns(2)

    with col1:
        monthly_payment = st.number_input("Amount to Save (Monthly)", min_value=0, value=100)
        return_period = st.number_input("Return Period (Month)", min_value=0, value=12)
        
    with col2:
        interest_rate_ss = st.number_input("Annual Interest Rate (%)", min_value=0, value=12, step=1, key="annual_IR")
        payment_period = st.number_input("Payment Period (Month)", min_value=0, value=24)

        
    if scheme == "Percentage-Based Saving":
        monthly_payment = [1000, 1100, 1200] 

    st.markdown("#### Simulation Results")

    if st.button("Calculate"):

        model = ModelStructuredSaving()


        if scheme == "Fixed Rate Saving":
            
            df = model.calculate_schedule_fixed(monthly_payment, interest_rate_ss, return_period, payment_period)
            
            st.dataframe(df, hide_index=True)


        if scheme == "Percentage-Based Saving":
            df = model.calculate_schedule_percentage(monthly_payment, interest_rate_ss, return_period, payment_period)
            
            st.dataframe(df, hide_index=True)
            
        
        df.to_csv('structured_saving_cashflow.csv', index=False)
            
    
    