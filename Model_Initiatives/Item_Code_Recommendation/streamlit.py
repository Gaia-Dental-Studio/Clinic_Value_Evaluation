import streamlit as st 
import pandas as pd
import numpy as np
from Model_Initiatives.Item_Code_Recommendation.model_recommendation import ModelRecommendation


def app(transaction_data):

    st.title('Item Code Recommendation')



    # transaction_data = pd.read_csv('Model_Initiatives/Item_Code_Recommendation/forecast_df_treatment.csv')
    item_code_details = pd.read_csv('Model_Initiatives/Item_Code_Recommendation/cleaned_item_code.csv')
    recommendation_pairs = pd.read_csv('Model_Initiatives/Item_Code_Recommendation/recommendation_pair.csv')


    col1, col2 = st.columns(2)

    with col1:
        
        starting_conversion_rate = st.number_input(label="Starting Conversion Rate (%)", value=5, step=5)
        starting_conversion_rate = starting_conversion_rate / 100
        
        conversion_growth_increment = st.number_input(label="Conversion Growth Increment (%)", value=3, step=1)
        conversion_growth_increment = conversion_growth_increment / 100
        
        
    with col2:    
        optimal_conversion_rate = st.number_input(label="Optimal Conversion Rate (%)", value=30, step=5)
        optimal_conversion_rate = optimal_conversion_rate / 100
        


    if st.button("Generate Converted Transaction Data"):
        
        model = ModelRecommendation(transaction_data, item_code_details, recommendation_pairs)
        
        converted_transaction_data = model.generate_converted_transaction_data(starting_conversion_rate, conversion_growth_increment, optimal_conversion_rate)
        
        st.dataframe(converted_transaction_data, height=400)
        
        # grouped_converted_transaction_data = model.group_by_period()
        
        
        converted_transaction_data.to_csv('Model_Initiatives/Item_Code_Recommendation/grouped_converted_transaction_data.csv', index=False)
    
    
    
