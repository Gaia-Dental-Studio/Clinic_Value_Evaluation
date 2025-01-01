import pandas as pd 
import numpy as np 
from Model_Initiatives.Subscription_Scheme.model_subscription import ModelSubscriptionScheme
# from model_subscription import ModelSubscription
import streamlit as st


def app(transaction_df):

    # transaction_df = pd.read_csv('forecast_df_treatment.csv')

    st.title('Subscription Scheme Model')

    col1, col2 = st.columns(2)

    with col1:
        subscription_fee = st.number_input("Subscription Fee ($)", value=200, step=10)
        days_until_inactive = st.number_input("Days Until Inactive", value=90, step=10)
        
    with col2:
        conversion_rate_monthly = st.number_input("Conversion Rate (%)", value=20, step=5)
        conversion_rate_monthly = conversion_rate_monthly / 100
        churn_probability = st.number_input("Churn Probability (%)", value=10, step=1)
        churn_probability = churn_probability / 100
      
      
    eligible_list = ['015', '018', '013', '114', '121', '141', '022', '521', '941']  
      
    transformed = transaction_df[transaction_df['Treatment'].isin(eligible_list)]
    
    treatment_counts_total = transaction_df['Treatment'].value_counts()
    
    treatment_counts = transformed['Treatment'].value_counts()
    
    # calculate the percentage of the eligible treatments compared to the total
    total = (treatment_counts.sum() / treatment_counts_total.sum()) * 100
    
    # sum the treatment percentage of the eligible treatments compared to the total

    
    st.write(total)
      
      
    if st.button("Calculate Subscription Scheme"):  
        model = ModelSubscriptionScheme(conversion_rate_monthly, subscription_fee ,days_until_inactive, churn_probability)

        transformed_df, subscription_df = model.transform(transaction_df)

        aggregated_df = transformed_df.groupby(transformed_df['Period'].dt.to_period("M")).agg({
            'Revenue': 'sum',
            'Expense': 'sum'
        }).reset_index()
        
        aggregated_subscription_df = subscription_df.groupby(subscription_df['Period'].dt.to_period("M")).agg({
            'Revenue': 'sum',   
            'Expense': 'sum'
        }).reset_index()
        


        st.markdown("### Transformed Data")

        # aggregated_df['Period'] = aggregated_df['Period'].dt.to_timestamp()
        # aggregated_df_monthly = aggregated_df.groupby(aggregated_df['Period'].dt.to_period('M')).agg({
        #     'Revenue': 'sum',
        #     'Expense': 'sum',
        # }).reset_index()


        transformed_df.to_csv('Model_Initiatives/Subscription_Scheme/after_subscription_scheme.csv', index=False)
        subscription_df.to_csv('Model_Initiatives/Subscription_Scheme/subscription_df.csv', index=False)
        
        # aggregated_df['Period'] = aggregated_df['Period'].dt.to_timestamp()
        
       
        st.markdown("### Transformed Data")
        st.dataframe(aggregated_df, height=400)
        
        
        st.markdown("### Subscription Data")
        st.dataframe(aggregated_subscription_df, height=400)
        

  