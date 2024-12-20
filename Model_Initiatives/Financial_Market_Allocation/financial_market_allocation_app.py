import streamlit as st 
import pandas as pd
import numpy as np
from Model_Initiatives.Financial_Market_Allocation import model
# import model
import json

def app():

    st.title('Dividend Investment Model')

    st.markdown("### Investment Parameters")


    type = st.selectbox("Investment Amount Scheme", ['Fixed','Percentage Profit'], index=1)


    # open json
    with open("Model_Initiatives/Financial_Market_Allocation/without_improvement_profit.json") as f:
        data = json.load(f)

    forecast_period = st.number_input("Forecast Period (months)", value=12, step=1)
        

        
    data = data[:forecast_period]

    first_investment = st.date_input("First Investment Date", pd.Timestamp("2025-01-01"))
    periodic_investment = st.number_input("Monthly Investment ($)", value=1000, step=100) if type == 'Fixed' else data
    percentage_profit = st.number_input("Percentage Profit (%)", value=20, step=5) if type == 'Percentage Profit' else 100


    periodic_investment = [profit * percentage_profit / 100 for profit in periodic_investment] if type == 'Percentage Profit' else periodic_investment


    st.divider()

    col1, col2, col3 = st.columns(3)
        
    with col1:
        dividend_yield_mean = st.number_input("Mean Dividend Yield", 0.05)
        dividend_yield_std = st.number_input("Dividend Yield Standard Deviation", 0.01)
        dividend_frequency = st.selectbox("Dividend Frequency", ['monthly', 'quarterly', 'annually'], index=1)
        
    with col2:
        stock_growth_mean = st.number_input("Mean Stock Growth", 0.06)
        stock_growth_std = st.number_input("Stock Growth Standard Deviation", 0.02)
        sell_frequency = st.selectbox("Sell Frequency", ['monthly', 'quarterly', 'annually'], index=1)
        
    with col3:
        sell_percentage = st.number_input("Sell Percentage", value=80, min_value=0, step=10)
        reinvest_percentage = st.number_input("Reinvest Percentage", value=100, min_value
                                                =0, step=10)
        

    if st.button("Generate Forecast"):
        model_forecast = model.DividendInvestmentModel(
        first_investment,
        periodic_investment,  # Monthly investment
        forecast_period,  # 12 months
        dividend_yield_mean,  # 5% mean dividend yield
        dividend_yield_std,  # 1% standard deviation
        dividend_frequency,  # Dividends paid quarterly
        stock_growth_mean,  # 6% annual stock growth
        stock_growth_std,  # 2% stock growth volatility
        sell_frequency,  # Sell stocks annually
        sell_percentage,  # Sell 100% of stock value
        reinvest_percentage  # Reinvest 50% of the proceeds
    )

        forecast_df = model_forecast.generate_forecast()
        
        # st.dataframe(forecast_df)
        
        
        cashflow_df = model_forecast.transform_to_cashflow(forecast_df)
        
        # cashflow_df['Profit'] = pd.Series(data) + cashflow_df['Revenue'] - cashflow_df['Expense']
        
        
        st.dataframe(cashflow_df)
        
        
        
        #save cashflow_df to csv
        cashflow_df.to_csv("Model_Initiatives/Financial_Market_Allocation/cashflow.csv", index=False)
        
        
        # CoV_after = np.array(cashflow_df['Profit']).std() / np.array(cashflow_df['Profit']).mean()
        # data_array = np.array(data)
        # CoV_before = data_array.std() / data_array.mean()
        
        # st.write(f"CoV after: {CoV_after}")
        # st.write(f"CoV before: {CoV_before}")
        




