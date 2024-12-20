import pandas as pd 
import numpy as np 
from Model_Initiatives.Subscription_Scheme.model_subscription import ModelSubscription
# from model_subscription import ModelSubscription
import streamlit as st


def app(transaction_df):

    # transaction_df = pd.read_csv('forecast_df_treatment.csv')

    st.title('Subscription Scheme Model')

    col1, col2 = st.columns(2)

    with col1:
        subscription_fee = st.number_input("Subscription Fee ($)", value=150, step=10)
        days_until_inactive = st.number_input("Days Until Inactive", value=90, step=10)
        
    with col2:
        conversion_rate_monthly = st.number_input("Conversion Rate (%)", value=0.2, step=0.05)
        churn_probability = st.number_input("Churn Probability (%)", value=10, step=1)
        churn_probability = churn_probability / 100
      
    if st.button("Calculate Subscription Scheme"):  
        model = ModelSubscription(transaction_df, subscription_fee, conversion_rate_monthly, days_until_inactive, churn_probability)

        transformed_df = model.transform_to_subscription_scheme()

        aggregated_df = model.aggregate_by_period()
        
        

        st.markdown("### Transformed Data")

        aggregated_df['Period'] = aggregated_df['Period'].dt.to_timestamp()
        aggregated_df_monthly = aggregated_df.groupby(aggregated_df['Period'].dt.to_period('M')).agg({
            'Revenue': 'sum',
            'Expense': 'sum',
        }).reset_index()


        aggregated_df.to_csv('Model_Initiatives/Subscription_Scheme/after_subscription_scheme.csv', index=False)
        # aggregated_df['Period'] = aggregated_df['Period'].dt.to_timestamp()
        
       

        st.dataframe(aggregated_df, height=400)

  