import streamlit as st 
import numpy as np
import pandas as pd

from Model_Initiatives.Family_Discount.model import ModelFamilyDiscount



def app(transaction_df):

    st.title("Family Group Discount")





    col1, col2 = st.columns(2)


    with col1:
        minimum_eligible_spending = st.number_input("Minimum Eligible Spending", value = 100, step = 10)
        discount_start = st.number_input("Starting Discount (%)", value = 10, step=5, help="Initial discount for at least two member in a family group")
        discount_start /= 100
        
    with col2:
        discount_increment = st.number_input("Discount Increment", value = 5, step=1, help="Discount increment for every +1 added member in a family member") 
        discount_increment /= 100
        
    pmf_family_sizes = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
    
    if st.button("Calculate", key="calculate_family_discount"):
        
        # Initialize the model
        family_model = ModelFamilyDiscount(minimum_eligible_spending=100)

        # Create dummy family DataFrame
        family_df = family_model.create_dummy_family_df(transaction_df, pmf_family_sizes)

        # Process transactions
        cashback_df = family_model.process_transactions(
            transaction_df,
            family_df,
            discount_start,
            discount_increment
        )    
        
        cashback_df.to_csv("Model_Initiatives/Family_Discount/cashback_family_discount_df.csv", index=False)
            

        st.dataframe(cashback_df)
    