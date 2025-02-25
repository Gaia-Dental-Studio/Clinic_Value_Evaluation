import streamlit as st 
import numpy as np
import pandas as pd

# from model import ModelContributionAmounts
from Model_Initiatives.Contribution_Amount.model import ModelContributionAmounts

def app(transaction_df):
    
    st.title("Contribution Amounts")

    col1, col2 = st.columns(2)

    with col1:
        minimum_eligible_amount = st.number_input("Minimum Eligible Amount", value = 400, step = 10)
        conversion_rate = st.number_input("Conversion Rate (%)", value = 50, step=5, help="Probability of a transaction being keen for conversion")
        conversion_rate /= 100
        
    with col2:
        apy = st.number_input("Annual Percentage Yield", value = 5, step=1, help="Annual Percentage Yield for interest calculations")
        apy /= 100
        monthly_payment_rate = st.number_input("Monthly Payment Rate (%)", value = 10, step=1, help="Percentage of the treatment price as the monthly contribution")
        monthly_payment_rate /= 100


    if st.button("Calculate", key="calculate_contribution_amount"):
        
        # transaction_df = pd.read_csv('Model_Initiatives/Contribution_Amount/forecast_df_treatment.csv')

        contribution_model = ModelContributionAmounts(minimum_eligible_amount, apy)

        # Process transactions
        updated_transactions_df, result_df = contribution_model.process_transactions(
            transaction_df,
            conversion_rate,
            monthly_payment_rate
        )

        st.dataframe(result_df)
        
        print(result_df)


        result_df.to_csv("Model_Initiatives/Contribution_Amount/contribution_schedule_df.csv", index=False)