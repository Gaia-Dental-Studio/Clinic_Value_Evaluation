import streamlit as st
import pandas as pd
import numpy as np
from model_for_strategy import ModelStructuredSaving

def app(profit_data=None):
    st.title('Structured Saving')

    st.write("This model simulates the saving of a structured saving account. The user can choose the amount to save, the interest rate, and the number of years to save for.")

    st.markdown("#### Input Parameters")


    scheme = st.radio("Select Scheme", ("Fixed Rate Saving", "Profit Percentage-Based Saving"), help="Select the saving scheme. Fixed Rate Saving is a fixed amount of saving every month. Percentage-Based Saving is a saving amount based on the percentage of the gross profit.")




    col1, col2 = st.columns(2)

    with col1:
        
        percentage_to_save = st.number_input("Percentage of Gross Profit to Save (%)", min_value=0, value=10, step=5) if scheme == "Profit Percentage-Based Saving" else 0
        percentage_to_save = percentage_to_save / 100
        
        monthly_payment = st.number_input("Amount to Save (Monthly)", min_value=0, value=100) if scheme == "Fixed Rate Saving" else [x * percentage_to_save for x in profit_data]

        return_period = st.number_input("Return Period (Month)", min_value=0, value=12)
        
    with col2:
        interest_rate_ss = st.number_input("Annual Interest Rate (%)", min_value=0, value=7, step=1, key="annual_IR")
        payment_period = st.number_input("Payment Period (Month)", min_value=0, value=24)

        


    st.markdown("#### Simulation Results")

    if st.button("Calculate", key="calculate structured saving"):

        model = ModelStructuredSaving()


        if scheme == "Fixed Rate Saving":
            
            df = model.calculate_schedule_fixed(monthly_payment, interest_rate_ss, return_period, payment_period)
            
            st.dataframe(df, hide_index=True)


        if scheme == "Profit Percentage-Based Saving":
            

            
            df = model.calculate_schedule_percentage(monthly_payment, interest_rate_ss, return_period, payment_period)
            
            st.dataframe(df, hide_index=True)
            
        
        df.to_csv('structured_saving_cashflow.csv', index=False)
            
    
    