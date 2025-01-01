import streamlit as st 
import pandas as pd 
import numpy as np 
from Model_Initiatives.Group_Discount.model import ModelGroupDiscount


def app(transaction_df):

    st.title('Group Discount')

    col1, col2 = st.columns(2)

    with col1:
        conversion_rate = st.number_input('Conversion Rate', min_value=0, max_value=100, value=70, step=5)
        conversion_rate /= 100
        max_waiting_days = st.number_input('Max Waiting Days', min_value=1, max_value=30, value=14)
        
    with col2:
        discount_start = st.number_input('Discount Start', min_value=0, max_value=100, value=10, step=5)
        discount_increment = st.number_input('Discount Increment', min_value=0, max_value=100, value=5, step=5)
        discount_start /= 100
        discount_increment /= 100
        
    if st.button('Calculate', key='calculate group discount'):
        model = ModelGroupDiscount(conversion_rate, max_waiting_days)
        # transactions_df = pd.read_csv('forecast_df_treatment.csv')
        output_df = model.process_transactions(transaction_df)
        
        st.dataframe(output_df)
        
        cashback_df = model.calculate_cashback(output_df, discount_start, discount_increment)
        
        cashback_df.to_csv('Model_Initiatives/Group_Discount/cashback_group_discount_df.csv', index=False)
        
        st.dataframe(cashback_df)
        