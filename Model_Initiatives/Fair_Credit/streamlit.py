import streamlit as st 
import pandas as pd
from Model_Initiatives.Fair_Credit.model_fair_credit import ModelFairCredit


def app(transaction_data):

    st.title('Fair Credit Model')
    
    
    transaction_data['Period'] = pd.to_datetime(transaction_data['Period'])
    transaction_data['Revenue'] = transaction_data['Revenue'].astype(float)
    transaction_data['Customer ID'] = transaction_data['Customer ID'].astype(str)

    col1, col2 = st.columns(2)


    with col1:
        minimum_allowable_price = st.number_input('Minimum Allowable Price', value=500, step=50)
        applicant_ratio = st.number_input('Applicant Ratio (%)', value=50, step=5)
        applicant_ratio = applicant_ratio / 100
        percentage_downpayment = st.number_input('Percentage Downpayment (%)', value=30, step=5)
        percentage_downpayment = percentage_downpayment / 100
        
    with col2:
        period_term = st.number_input('Period Term', value=10, step=1)
        monthly_interest = st.number_input('Monthly Interest Rate (%)', value=1.0, step=0.1)
        monthly_interest = monthly_interest / 100
        

    if st.button('Generate Amortization Schedule'):
        
        
        model_fair_credit = ModelFairCredit(transaction_data, minimum_allowable_price=minimum_allowable_price, applicant_ratio=applicant_ratio, period_term=period_term, monthly_interest=monthly_interest, percentage_downpayment=percentage_downpayment)
        
        aggregated_amortization_schedule, adjusted_original_transaction = model_fair_credit.generate_amortization_schedule()
        
        # st.write('Previous Revenue')
        # st.write(transaction_data['Revenue'].sum())
        
        # st.write('New Revenue')
        # st.write(adjusted_original_transaction['Revenue'].sum())
        
        # st.write('Adjusted:', model_fair_credit.adjusted)
        # st.write('Skipped:', model_fair_credit.skipped)
        
        
        # st.dataframe(adjusted_original_transaction[adjusted_original_transaction['Revenue'] == 0])
        
        
        grouped_amortization_schedule = model_fair_credit.group_by_period()
        
        st.dataframe(aggregated_amortization_schedule)
        
        st.dataframe(grouped_amortization_schedule)
        
        
        aggregated_amortization_schedule.to_csv('Model_Initiatives/Fair_Credit/grouped_amortization_schedule.csv', index=False)
        
        adjusted_original_transaction['Period'] = pd.to_datetime(adjusted_original_transaction['Period'])
        
        # grouped_original_transaction_data = model_fair_credit.transaction_data.groupby(model_fair_credit.transaction_data['Period'].dt.to_period("M")).agg({'Revenue':'sum', 'Expense':'sum'}).reset_index()
        
        adjusted_original_transaction.to_csv('Model_Initiatives/Fair_Credit/adjusted_transaction_data.csv', index=False)
        
