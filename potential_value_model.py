import pickle
import pandas as pd
import numpy as np
import streamlit as st
from model import ModelClinicValue


# import scikitlearn for linear regression
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


from model_forecasting import ModelForecastPerformance
from cashflow_plot import ModelCashflow
from streamlit_extras.stateful_button import button

import structured_saving as structured_saving_app
import corporate_wellness_app
import school_outreach_app
from Model_Initiatives.Financial_Market_Allocation import financial_market_allocation_app
from Model_Initiatives.Supplier_Selection import streamlit_per_clinic as supplier_selection_app
from Model_Initiatives.Subscription_Scheme import streamlit as subscription_scheme_app
from Model_Initiatives.Item_Code_Recommendation import streamlit as item_code_recommendation_app
from Model_Initiatives.Fair_Credit import streamlit as fair_credit_app
from Model_Initiatives.Group_Discount import streamlit as group_discount_app
from Model_Initiatives.Family_Discount import streamlit as family_discount_app
from Model_Initiatives.Contribution_Amount import streamlit as contribution_amount_app

import os
    
def app():

    st.title("Potential Value Calculator")
    

    
    # # DUMMY DATASET DATA GENERATION
    # dataset = st.selectbox("Select Dataset", ['Dataset 1', 'Dataset 2', 'Dataset 3'])
    # dataset = int(dataset.split()[-1])
    # clinic_data_set = pickle.load(open(f'dummy_clinic_model/pkl_files/dataset_{dataset}/clinic_value_set.pkl', 'rb'))
    # clinic_historical_demand = pickle.load(open(f'dummy_clinic_model/pkl_files/dataset_{dataset}/clinic_metrics.pkl', 'rb'))
    # selected_clinic = st.selectbox("Select Clinic", list(clinic_data_set.keys()))
    
    # SINGLE DATA GENERATION
    st.success("Data found and loaded successfully")
    clinic_data = pickle.load(open(f'single_clinic_value.pkl', 'rb'))
    clinic_historical_demand = pickle.load(open(f'dummy_clinic_model/pkl_files/dataset_1/clinic_metrics.pkl', 'rb'))
    
    
    
    
    
    with st.container(border=True):
        st.markdown('### Clinic Value to Acquire')
        st.write("Based on Calculation of Clinic Value of observed clinic, here is the value of the clinic to acquire")
        
        # For Single Clinic Data
        item_code_clinic_historical = clinic_historical_demand['N21701']['updated_clinic_item_code']
        clinic_value = clinic_data['Clinic Valuation Adjusted']
        
        # # For Multiple Dummy Clinic Data
        # clinic_value = clinic_data['Clinic Valuation Adjusted']
        # item_code_clinic_historical = clinic_historical_demand[selected_clinic]['updated_clinic_item_code']
        
        
        st.metric("Clinic Value", f"${clinic_value:,.0f}")
        model = ModelForecastPerformance(clinic_data)
        
        
        
        
        st.markdown("#### Acquiring Assumption")
        
        acquiring_funding = st.radio("Acquiring Funding", ["Borrow", "Self-Fund"])
        if acquiring_funding == "Borrow":
            
            col1, col2 = st.columns(2)
            with col1:
                percentage_borrow = st.number_input("% of Clinic Price Value to Borrow", min_value=0, max_value=100, value=50, step=10)
                upfront_payment = st.number_input("Upfront Payment (%)", min_value=0, max_value=100, value=20, step=10)
                
            with col2:
                interest_rate = st.number_input("Annual Interest Rate (%)", min_value=0, value=12, step=1)
                loan_term = st.number_input("Loan Term (years)", min_value=1, value=5, step=1)
                
            buying_year = st.number_input("Buying Year", min_value=2021, value=2025, step=1)
                
            borrowed = (clinic_value * (percentage_borrow/100))
            principal = borrowed - ((upfront_payment/100) * borrowed)
            amortization_df, monthly_payment, total_principal, total_interest = model.loan_amortization_schedule(principal, interest_rate, loan_term, start_year=buying_year)
            
            sliced_amortization_df = amortization_df[['Period', 'Monthly Payment']]
            sliced_amortization_df = sliced_amortization_df.rename(columns={'Monthly Payment':"Expense"})
            sliced_amortization_df['Revenue'] = 0
            
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Monthly Payment", f"${monthly_payment:,.0f}")
                
            with col2:
                st.metric("Total Principal", f"${total_principal:,.0f}")
                
            with col3:
                st.metric("Total Interest", f"${total_interest:,.0f}")

            
        


    st.divider()
    
    st.markdown("### Projection of Current Clinic (without Improvement)")

    show_by = st.radio("Show by", ["Product", "Customer"], index=0)
    
    col1, col2 = st.columns(2)

    with col1:
        period_forecast = st.number_input("Period to Forecast (months)", min_value=1, value=12, step=1)
        
    with col2:
        start_year = st.number_input("Start Year", min_value=2021, value=2025, step=1)
    

    calculate = button("Calculate", key="calculate-existing")

    
    if calculate:
        
        number_of_months = model.total_days_from_start(period_forecast, start_year=start_year) 
        
        model_cashflow = ModelCashflow()
        
        forecast_df = model.forecast_revenue_expenses(period_forecast, start_month=1)
        
        forecast_df['Profit'] = forecast_df['Revenue'] - forecast_df['Expense']
        
        
        # st.dataframe(forecast_df)
        # st.write("Total Revenue: ", forecast_df['Revenue'].sum())
        # st.write("Total Expense: ", forecast_df['Expense'].sum())
        # st.write("Total Profit: ", forecast_df['Profit'].sum())
        
        
        
        # forecast_df.to_csv("forecast_df_deletable.csv", index=False)
        # st.dataframe(forecast_df)
        
        indirect_expense = model.forecast_indirect_cost(period_forecast, start_year=start_year)
        
        
        
        # FORECASTING TREATMENTS
        
        treatment_details = pd.read_csv("treatment_with_details.csv")
        item_code_df = pd.read_csv("dummy_clinic_model\cleaned_item_code.csv")
        
        
        # forecast_df_with_treatments = model.generate_forecast_treatment_df_by_profit(treatment_details, forecast_df)
        forecast_df_with_treatments = model.generate_forecast_item_code_by_profit(item_code_df, forecast_df, item_code_clinic_historical, patient_pool=clinic_data['Patient Pool'], start_year=start_year)
        # forecast_df_with_treatments.to_csv("forecast_df_treatment_deletable.csv", index=False)
        
        # st.dataframe(forecast_df_with_treatments)
        
        # st.plotly_chart(model.summary_table(forecast_df_with_treatments, by=show_by))
        
        forecast_df = forecast_df_with_treatments[['Period', 'Revenue', 'Expense']].groupby('Period').sum().reset_index()
        # st.dataframe(forecast_df)
        
        forecast_df['Period'] = pd.to_datetime(forecast_df['Period'])
        
        projected_gross_profit = forecast_df['Revenue'].sum() - forecast_df['Expense'].sum()
        
        # # change forecast_df_with_treatments column period to month and year
        # # Add a new column for Month-Year formatted as 'YYYY-MM'
        # forecast_df['MonthYear'] = forecast_df['Period'].dt.to_period('M')

        # # Group by Month-Year and calculate the sum for Revenue and Expense
        # forecast_df_groupby_monthyear = (
        #     forecast_df.groupby('MonthYear')[['Revenue', 'Expense']]
        #     .sum()
        #     .reset_index()
        # )
        
        # forecast_df_groupby_monthyear['Profit'] = forecast_df_groupby_monthyear['Revenue'] - forecast_df_groupby_monthyear['Expense']
        
        
        # st.dataframe(forecast_df_groupby_monthyear)
        # st.write("Total Revenue: ", forecast_df_groupby_monthyear['Revenue'].sum())
        # st.write("Total Expense: ", forecast_df_groupby_monthyear['Expense'].sum())
        # st.write("Total Profit: ", forecast_df_groupby_monthyear['Revenue'].sum() - forecast_df_groupby_monthyear['Expense'].sum())
        

        sliced_amortization_df = sliced_amortization_df[:period_forecast] if acquiring_funding == "Borrow" else None
         
        equipment_df = model.equipment_to_cashflow(clinic_data['Equipment Life'], period_forecast, start_year=start_year)
        
        
        fitout_df = model.fitout_to_cashflow(clinic_data['Fitout Value'], clinic_data['Last Fitout Year'] ,period_forecast, start_year=start_year)


        # payload_1 = {
        #     "forecast_df": forecast_df,
        #     "debt_repayment": sliced_amortization_df if acquiring_funding == "Borrow" else None,
        #     "indirect_expense": indirect_expense
        # }

        # forecast_df = pd.read_csv("current_clinic_cashflow_forecast.csv")
        model_cashflow.add_company_data("Gross Profit", forecast_df)
        model_cashflow.add_company_data("Debt Repayment", sliced_amortization_df) if acquiring_funding == "Borrow" else None
        model_cashflow.add_company_data("Indirect Expense", indirect_expense)
        model_cashflow.add_company_data("Equipment Procurement", equipment_df)
        model_cashflow.add_company_data("Fit Out", fitout_df)
        
        without_improvement_df = model_cashflow.merge_dataframes_with_category()
        
        without_improvement_df_profit = model_cashflow.groupby_dataframes_month_year(without_improvement_df)['Profit'].values
        
        # set without_improvement_df_profit to list
        without_improvement_df_profit = without_improvement_df_profit.tolist()
        
        # # save into json
        # with open("Model_Initiatives\Financial_Market_Allocation\without_improvement_profit.json", 'w') as f:
        #     f.write(str(without_improvement_df_profit.tolist()))
        
        def generate_datetime(start_year=2024, start_month=1):
            return f"{start_year}-{start_month}-01"
            
        
        forecast_linechart_daily = model_cashflow.cashflow_plot(number_of_months, start_date=generate_datetime(start_year=start_year))
        forecast_linechart_weekly = model_cashflow.cashflow_plot(number_of_months, granularity='weekly',  start_date=generate_datetime(start_year=start_year))
        forecast_linechart_monthly = model_cashflow.cashflow_plot(number_of_months, granularity='monthly',  start_date=generate_datetime(start_year=start_year))
        forecast_linechart_quarterly = model_cashflow.cashflow_plot(number_of_months, granularity='quarterly', start_date=generate_datetime(start_year=start_year))
        
        
        clinic_value = clinic_data['Clinic Valuation Adjusted']
        clinic_ebit_multiple = clinic_data['EBIT Multiple']


        current_ebit = clinic_data['EBIT']
        current_gross_profit = clinic_data['Gross Profit']
        current_ebit_ratio = clinic_data['EBIT Ratio']
        current_growth = clinic_data['Net Sales Growth']
        current_net_sales_relative_variation = clinic_data['Relative Variation of Net Sales']
        

        
        st.write("")
        st.markdown(f"##### Predicted Cash Flow for the Next {period_forecast} Months")
        
        with st.container(border=True):
            st.markdown("**Approach 1: Foreacast based on previous year's Gross Profit, EBIT Ratio, and YoY Growth**")
            
            st.write("Based on previous years' Gross Profit, EBIT Ratio, Year-on-Year growth and Revenue Relative Variation of the clinic, here is the projected net cashflow for the next 12 months. We call this approach as Fluctuation-Adjusted Net Cashflow Forecast.")

            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Previous year Gross Profit", f"${current_gross_profit:,.0f}")
                
            with col2:
                st.metric("Year-on-Year Growth", f"{np.round(current_growth * 100,2)}%")
                
            with col3:
                st.metric("Monthly Relative Variation", f"{np.round(current_net_sales_relative_variation * 100,2)}%")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Daily", "Weekly", "Monthly", "Quarterly"])
        
            with tab1:
            
                st.plotly_chart(forecast_linechart_daily)
                
            with tab2:  
                st.plotly_chart(forecast_linechart_weekly)
            
            with tab3:
                st.plotly_chart(forecast_linechart_monthly)
                
            with tab4:
                st.plotly_chart(forecast_linechart_quarterly)          
                  
            with st.popover("explain Cash flow for Approach 1"):
                st.write(
                    f"The predicted Cash flow for the next {period_forecast} months is generated with random fluctuations, "
                    "but it aligns with the clinic's current metrics.") 
                    
                st.write(
                    f"Specifically, the total Gross Profit for the year (shall you run 12 months forecast) which is ", f"{projected_gross_profit:,.0f}," 
                    f" reflects the clinic's expected year-on-year sales growth which is {current_growth * 100:.0f}% "
                    f"from the previous year Gross Profit of {current_gross_profit:,.0f}, while the monthly variations capture "
                    f"the clinic's sales historical fluctuation patterns (relative variation, which is {current_net_sales_relative_variation * 100:.0f}%). "
                    f"Furthermore, the Indirect Expense calculated is calculated and forecasted based on historical EBIT Ratio of {current_ebit_ratio * 100:.0f}%"
                )
    
            st.plotly_chart(model.summary_table(forecast_df_with_treatments, sliced_amortization_df, indirect_expense, equipment_df, fitout_df, by=show_by))
            

# # START OF APPROACH 2
        
#         with st.container(border=True):
#             st.markdown("**Approach 2: Forecast using Three Month Moving Average based on previous months data**")
            
#             MA_forecast_df = model.forecast_revenue_expenses(3) # instead of using actual data, this method forecast three months as basis of most recent months of historical data 
            
#             MA_forecast_df = model.forecast_revenue_expenses_MA(MA_forecast_df, period_forecast, 3, start_month=1)
            
            
            
#             # MA_forecast_df_with_treatments = model.generate_forecast_treatment_df_by_profit(treatment_details, MA_forecast_df)
#             MA_forecast_df_with_treatments = model.generate_forecast_item_code_by_profit(item_code_df, MA_forecast_df, item_code_clinic_historical, start_year=start_year)
#             MA_forecast_df = MA_forecast_df_with_treatments[['Period', 'Revenue', 'Expense']].groupby('Period').sum().reset_index()
            
#             MA_indirect_cost = model.forecast_indirect_cost(period_forecast, start_year=start_year)
            
            
#             model_cashflow.remove_all_companies()
#             model_cashflow.add_company_data("Gross Profit", MA_forecast_df)
#             model_cashflow.add_company_data("Debt Repayment", sliced_amortization_df) if acquiring_funding == "Borrow" else None
#             model_cashflow.add_company_data("Indirect Expense", MA_indirect_cost)
#             model_cashflow.add_company_data("Equipment Procurement", equipment_df)
#             model_cashflow.add_company_data("Fit Out", fitout_df)
            
#             MA_forecast_linechart_daily = model_cashflow.cashflow_plot(number_of_months, start_date=generate_datetime(start_year=start_year))
#             MA_forecast_linechart_weekly = model_cashflow.cashflow_plot(number_of_months, granularity='weekly', start_date=generate_datetime(start_year=start_year))
#             MA_forecast_linechart_monthly = model_cashflow.cashflow_plot(number_of_months, granularity='monthly', start_date=generate_datetime(start_year=start_year))
            
#             tab1, tab2, tab3 = st.tabs(["Daily", "Weekly", "Monthly"])
    
#             with tab1:
#                 st.plotly_chart(MA_forecast_linechart_daily)
                
#             with tab2:  
#                 st.plotly_chart(MA_forecast_linechart_weekly)
            
#             with tab3:
#                 st.plotly_chart(MA_forecast_linechart_monthly)
        

            
#             with st.popover("explain Cash flow for Approach 2"):
#                 st.write(
#                     f"The predicted Cash flow for the next {period_forecast} months is calculated using Moving Average method. "
#                     "The method is used to smooth out the random fluctuations in the data.")
#                 st.write("It will use the average of the last months (depends on specified Moving Average Period) of historical data"
#                     "to predict the first forecast months, for it will keeps smoothing out the data until the last month of the forecast period.")
#                 st.write("Currently the Moving Average Period is set to 3 months, which means it will use the average of the last 3 months of historical data. We call this approach as Moving Average Forecast")
            
#             st.plotly_chart(model.summary_table(MA_forecast_df_with_treatments, sliced_amortization_df, indirect_expense, equipment_df, fitout_df, by=show_by))
            

# END OF APPROACH 2

        #load clinic_value.pkl and get the value of it to be the value of variable 'clinic_value'
        
        
        current_ebit_after_12_months = current_ebit * (1+clinic_data['Net Sales Growth'])


        potential_ebit_after_12_months = current_ebit_after_12_months
        
        
    # SAVE Into PKL (for single clinic Data)
        
        # clinic_projected_cashflow = {
        #     "Approach 1": 
        #         {"Gross Profit": forecast_df,
        #         "Debt Repayment": sliced_amortization_df if acquiring_funding == "Borrow" else None,
        #         "Indirect Expense": indirect_expense,
        #         "Equipment Procurement": equipment_df,
        #         "Fit Out": fitout_df},
        #     # "Approach 2": 
        #     #     {"Gross Profit": MA_forecast_df,
        #     #     "Debt Repayment": sliced_amortization_df if acquiring_funding == "Borrow" else None,
        #     #     "Indirect Expense": MA_indirect_cost,
        #     #     "Equipment Procurement": equipment_df,
        #     #     "Fit Out": fitout_df}
        # }
        
        # # save into pkl
        # with open("clinic_projected_cashflow_single.pkl", 'wb') as f:
        #     pickle.dump(clinic_projected_cashflow, f)
        
        # forecast_dictionary = {
        #     'without_improvement_df_profit': without_improvement_df_profit,
        #     'forecast_df_with_treatments': forecast_df_with_treatments,
        #     'dataset': 1,
        #     'selected_clinic': 'N21701',
        #     'period_forecast': period_forecast,
        #     'start_year': start_year,
        #     'acquiring_funding': acquiring_funding,
        #     'sliced_amortization_df': sliced_amortization_df,
        #     'indirect_expense': indirect_expense,
        #     'equipment_df': equipment_df,
        #     'fitout_df': fitout_df,
        #     'forecast_df': forecast_df,
        #     'model_cashflow': model_cashflow,
        #     'number_of_months': number_of_months,
        #     'without_improvement_df': without_improvement_df,
        #     'clinic_value': clinic_value,

            
        # }
        
        # with open("forecast_dictionary_single.pkl", 'wb') as f:
        #     pickle.dump(forecast_dictionary, f)
            
        
        
        
    # SAVE INTO PKL (For Multiple DUMMY Dataset)
        
        # file_name = 'clinic_projected_cashflow_set.pkl'
        # ID = selected_clinic

        # # Initialize or load existing dictionary
        # if os.path.exists(file_name):
        #     # Load existing data
        #     with open(file_name, 'rb') as f:
        #         clinic_projected_cashflow_set = pickle.load(f)
        # else:
        #     # Start with an empty dictionary
        #     clinic_projected_cashflow_set = {}
        
   
        # clinic_projected_cashflow_set[ID] = {
        #     "Approach 1": 
        #         {"Gross Profit": forecast_df,
        #         "Debt Repayment": sliced_amortization_df if acquiring_funding == "Borrow" else None,
        #         "Indirect Expense": indirect_expense,
        #         "Equipment Procurement": equipment_df,
        #         "Fit Out": fitout_df},
        #     "Approach 2": 
        #         {"Gross Profit": MA_forecast_df,
        #         "Debt Repayment": sliced_amortization_df if acquiring_funding == "Borrow" else None,
        #         "Indirect Expense": MA_indirect_cost,
        #         "Equipment Procurement": equipment_df,
        #         "Fit Out": fitout_df}
        # }
        
        # # save into pkl
        # with open(file_name, 'wb') as f:
        #     pickle.dump(clinic_projected_cashflow_set, f)
            
            
        
        # # Initialize or load existing dictionary
        # if os.path.exists('forecast_dictionary.pkl'):
        #     # Load existing data
        #     with open('forecast_dictionary.pkl', 'rb') as f:
        #         forecast_dictionary = pickle.load(f)
        # else:
        #     # Start with an empty dictionary
        #     forecast_dictionary = {}    
        
            
        # forecast_dictionary[ID] = {
        #     'without_improvement_df_profit': without_improvement_df_profit,
        #     'forecast_df_with_treatments': forecast_df_with_treatments,
        #     'dataset': dataset,
        #     'selected_clinic': selected_clinic,
        #     'period_forecast': period_forecast,
        #     'start_year': start_year,
        #     'acquiring_funding': acquiring_funding,
        #     'sliced_amortization_df': sliced_amortization_df,
        #     'indirect_expense': indirect_expense,
        #     'equipment_df': equipment_df,
        #     'fitout_df': fitout_df,
        #     'forecast_df': forecast_df,
        #     'model_cashflow': model_cashflow,
        #     'number_of_months': number_of_months,
        #     'without_improvement_df': without_improvement_df

            
        # }
        
        # with open("forecast_dictionary.pkl", 'wb') as f:
        #     pickle.dump(forecast_dictionary, f)
   
    st.divider()



    
    def generate_datetime(start_year=2024, start_month=1):
        return f"{start_year}-{start_month}-01"
    
    

# Single Clinic Use Case

    # load the pickle
    with open("clinic_projected_cashflow_single.pkl", 'rb') as f:
        clinic_projected_cashflow = pickle.load(f)
        
    with open("forecast_dictionary_single.pkl", 'rb') as f:
        forecast_dictionary = pickle.load(f)
        
    without_improvement_df_profit = forecast_dictionary['without_improvement_df_profit']
    forecast_df_with_treatments = forecast_dictionary['forecast_df_with_treatments']
    dataset = forecast_dictionary['dataset']
    selected_clinic = forecast_dictionary['selected_clinic']
    period_forecast = forecast_dictionary['period_forecast']
    start_year = forecast_dictionary['start_year']
    acquiring_funding = forecast_dictionary['acquiring_funding']
    sliced_amortization_df = forecast_dictionary['sliced_amortization_df']
    indirect_expense = forecast_dictionary['indirect_expense']
    equipment_df = forecast_dictionary['equipment_df']
    fitout_df = forecast_dictionary['fitout_df']
    forecast_df = forecast_dictionary['forecast_df']
    model_cashflow = forecast_dictionary['model_cashflow']
    number_of_months = forecast_dictionary['number_of_months']
    without_improvement_df = forecast_dictionary['without_improvement_df']
        

        
# COMMENT OUT FOR SINGLE CLINIC USE CASE (As this is Multiple Dummy Clinic Use case)

    # # load the pickle 
    # with open("clinic_projected_cashflow_set.pkl", 'rb') as f:
    #     clinic_projected_cashflow_set = pickle.load(f)
        
    # # load forecast dictionary pkl 
    # with open("forecast_dictionary.pkl", 'rb') as f:
    #     forecast_dictionary = pickle.load(f)
        
    # forecast_dictionary = forecast_dictionary[selected_clinic]

    # without_improvement_df_profit = forecast_dictionary['without_improvement_df_profit']
    # forecast_df_with_treatments = forecast_dictionary['forecast_df_with_treatments']
    # dataset = forecast_dictionary['dataset']
    # selected_clinic = forecast_dictionary['selected_clinic']
    # period_forecast = forecast_dictionary['period_forecast']
    # start_year = forecast_dictionary['start_year']
    # acquiring_funding = forecast_dictionary['acquiring_funding']
    # sliced_amortization_df = forecast_dictionary['sliced_amortization_df']
    # indirect_expense = forecast_dictionary['indirect_expense']
    # equipment_df = forecast_dictionary['equipment_df']
    # fitout_df = forecast_dictionary['fitout_df']
    # forecast_df = forecast_dictionary['forecast_df']
    # model_cashflow = forecast_dictionary['model_cashflow']
    # number_of_months = forecast_dictionary['number_of_months']
    # without_improvement_df = forecast_dictionary['without_improvement_df']
    

    st.markdown("### Strategy to implement")

    

            
    col1, col2 = st.columns(2)
    
    with col1:
        structured_saving = st.checkbox("Structured Saving/Deposit Plan", value=False)
        
    with col2:
        with st.popover("details saving/deposit plan"):
            structured_saving_app.app(without_improvement_df_profit)
            
    
    col1, col2 = st.columns(2)
            
    with col1:
        fair_credit = st.checkbox("Fair Credit Program", value=False)
        
    with col2:
        with st.popover("details fair credit"):
            fair_credit_app.app(forecast_df_with_treatments)
    
    
    col1, col2 = st.columns(2)
            
    with col1:
        financial_market_allocation = st.checkbox("Financial Market Allocation", value=False)
        
    with col2:
        with st.popover("details financial market allocation"):
            financial_market_allocation_app.app(without_improvement_df_profit)
            

    col1, col2 = st.columns(2)
            
    with col1:
        optimized_supplier_selection = st.checkbox("Optimized Supplier Selection", value=False)
        
    with col2:
        with st.popover("details optimized supplier selection"):
            supplier_selection_app.app(dataset, selected_clinic)
            # supplier_selection_app.app(1, 'N21701')
            
            

    col1, col2 = st.columns(2)
            
    with col1:
        subscription_scheme = st.checkbox("Subscription Scheme", value=False)
        
    with col2:
        with st.popover("details subscription scheme"):
            subscription_scheme_app.app(forecast_df_with_treatments) # For now only compatible in converting using approach 1
            
            
    col1, col2 = st.columns(2)
            
    with col1:
        item_code_recommendation = st.checkbox("Item Code Recommendation for Upselling", value=False)
        
    with col2:
        with st.popover("details item code recommendation"):
            item_code_recommendation_app.app(forecast_df_with_treatments)
            
            
    col1, col2 = st.columns(2)
    with col1:
        group_discount = st.checkbox("Group Discount", value=False)
        
    with col2:
        with st.popover("details group discount"):
            group_discount_app.app(forecast_df_with_treatments)
            
            
    col1, col2 = st.columns(2)
    with col1:
        family_discount = st.checkbox("Family Discount", value=False)
        
    with col2:
        with st.popover("details family discount"):
            family_discount_app.app(forecast_df_with_treatments)



    col1, col2 = st.columns(2)
    with col1:
        contribution_amout = st.checkbox("Monthly Contribution Amounts", value=False)
        
    with col2:
        with st.popover("details contribution amounts"):
            contribution_amount_app.app(forecast_df_with_treatments)
            
            

    col1, col2 = st.columns(2)
    with col1:
        corporate_wellness = st.checkbox("Corporate Wellness Program", value=False)
        
    with col2:
        with st.popover("details corporate wellness"):
            corporate_wellness_app.app()

            
            
            
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     school_outreach = st.checkbox("School Outreach Program", value=False)
        
    # with col2:
    #     with st.popover("details"):
    #         school_outreach_app.app()
    model_cashflow.remove_all_companies()    
    model_cashflow.add_company_data("Gross Profit", forecast_df)
    model_cashflow.add_company_data("Debt Repayment", sliced_amortization_df) if acquiring_funding == "Borrow" else None
    model_cashflow.add_company_data("Indirect Expense", indirect_expense)
    model_cashflow.add_company_data("Equipment Procurement", equipment_df)
    model_cashflow.add_company_data("Fit Out", fitout_df)
    
    without_improvement_plot = model_cashflow.cashflow_plot(number_of_months, granularity='monthly', start_date=generate_datetime(start_year=start_year), by_profit=True)


    if button("Calculate New Cash Flow", key="calculate-strategy"):

        
        corporate_wellness_df = None
        structured_saving_df = None
        fair_credit_df = None
        financial_market_allocation_df = None
        optimized_supplier_selection_df = None
        subscription_scheme_df = None
        item_code_recommendation_df = None
        group_discount_df = None
        family_discount_df = None
        contribution_amout_df = None
        
        if corporate_wellness == True:
            corporate_wellness_df = pd.read_csv("corporate_cashflow_AUD.csv")[:period_forecast]
            corporate_wellness_df['Period'] = corporate_wellness_df['Period'].apply(lambda period: model.generate_date_from_month(int(period), method='first_day', start_year=start_year))
            # clinic_projected_cashflow_set[selected_clinic]["Approach 1"]["Corporate Wellness"] = corporate_wellness_df
            # clinic_projected_cashflow_set[selected_clinic]["Approach 2"]["Corporate Wellness"] = corporate_wellness_df
            
        if structured_saving == True:
            structured_saving_df = pd.read_csv("structured_saving_cashflow.csv")[:period_forecast]
            structured_saving_df['Period'] = structured_saving_df['Period'].apply(lambda period: model.generate_date_from_month(int(period), method='first_day', start_year=start_year))
            # clinic_projected_cashflow_set[selected_clinic]["Approach 1"]["Structured Saving"] = structured_saving_df
            # clinic_projected_cashflow_set[selected_clinic]["Approach 2"]["Structured Saving"] = structured_saving_df
        
        if fair_credit == True:
            fair_credit_df = pd.read_csv("Model_Initiatives/Fair_Credit/grouped_amortization_schedule.csv")
            # fair_credit_df['Period'] = fair_credit_df['Period'].apply(lambda period: model.generate_date_from_month(int(period), method='first_day', start_year=start_year))
            # clinic_projected_cashflow_set[selected_clinic]["Approach 1"]["Fair Credit"] = fair_credit_df
            # clinic_projected_cashflow_set[selected_clinic]["Approach 2"]["Fair Credit"] = fair_credit_df
            
            # FOR THE CASE IF FORECAST_DF Unadjusted
            # forecast_df = pd.read_csv("Model_Initiatives/Fair_Credit/adjusted_transaction_data.csv")
            
            
        if financial_market_allocation == True:
            financial_market_allocation_df = pd.read_csv("Model_Initiatives/Financial_Market_Allocation/cashflow.csv")[:period_forecast]
            # clinic_projected_cashflow_set[selected_clinic]["Approach 1"]["Financial Market Allocation"] = financial_market_allocation_df
            # clinic_projected_cashflow_set[selected_clinic]["Approach 2"]["Financial Market Allocation"] = financial_market_allocation_df
            
        if optimized_supplier_selection == True:
            reduced_material_cost = pd.read_pickle("Model_Initiatives/Supplier_Selection/reduced_material_cost.pkl")
            optimized_supplier_selection_df = pd.DataFrame(columns=['Period', 'Revenue', 'Expense'])
            
            # Calculate proportional reduction factor
            proportional_reduction = reduced_material_cost / forecast_df['Expense'].sum()

            # Calculate new cost per row
            optimized_supplier_selection_df['Period'] = forecast_df['Period']
            optimized_supplier_selection_df['Expense'] = 0
            optimized_supplier_selection_df['Revenue'] = (forecast_df['Expense'] * proportional_reduction)

            # forecast_df['Expense'] = forecast_df['Expense'] - (reduced_material_cost / len(forecast_df))
            
        if subscription_scheme == True:
            subscription_scheme_df = pd.read_csv("Model_Initiatives/Subscription_Scheme/after_subscription_scheme.csv")
            subscription_df = pd.read_csv("Model_Initiatives/Subscription_Scheme/subscription_df.csv")
            forecast_df = subscription_scheme_df.copy()
            MA_forecast_df = subscription_scheme_df.copy() # still not correct for the second approach
            # clinic_projected_cashflow_set[selected_clinic]["Approach 2"]["Gross Profit"] = MA_forecast_df
            
        if item_code_recommendation == True:
            item_code_recommendation_df = pd.read_csv("Model_Initiatives/Item_Code_Recommendation/grouped_converted_transaction_data.csv")
            # clinic_projected_cashflow_set[selected_clinic]["Approach 1"]["Item Code Recommendation"] = item_code_recommendation_df
            # clinic_projected_cashflow_set[selected_clinic]["Approach 2"]["Item Code Recommendation"] = item_code_recommendation_df
            
        if group_discount == True:
            group_discount_df = pd.read_csv("Model_Initiatives/Group_Discount/cashback_group_discount_df.csv")
            # clinic_projected_cashflow_set[selected_clinic]["Approach 1"]["Group Discount"] = group_discount_df
            # clinic_projected_cashflow_set[selected_clinic]["Approach 2"]["Group Discount"] = group_discount_df
            
        if family_discount == True:
            family_discount_df = pd.read_csv("Model_Initiatives/Family_Discount/cashback_family_discount_df.csv")
            # clinic_projected_cashflow_set[selected_clinic]["Approach 1"]["Family Discount"] = family_discount_df
            # clinic_projected_cashflow_set[selected_clinic]["Approach 2"]["Family Discount"] = family_discount_df
            
        if contribution_amout == True:
            contribution_amout_df = pd.read_csv("Model_Initiatives/Contribution_Amount/contribution_schedule_df.csv")
            # clinic_projected_cashflow_set[selected_clinic]["Approach 1"]["Contribution Amount"] = contribution_amout_df
            # clinic_projected_cashflow_set[selected_clinic]["Approach 2"]["Contribution Amount"] = contribution_amout_df
        
        # clinic_projected_cashflow_set[selected_clinic]["Approach 1"]["Gross Profit"] = forecast_df
        

        # use Approach 1    
        model_cashflow.remove_all_companies()    
        model_cashflow.add_company_data("Gross Profit", forecast_df)
        model_cashflow.add_company_data("Debt Repayment", sliced_amortization_df) if acquiring_funding == "Borrow" else None
        model_cashflow.add_company_data("Indirect Expense", indirect_expense)
        model_cashflow.add_company_data("Equipment Procurement", equipment_df)
        model_cashflow.add_company_data("Fit Out", fitout_df)
        
        without_improvement_dataframes = model_cashflow.collection_df.copy()
        
        # without_improvement_df = model_cashflow.merge_dataframes_with_category()
        
        st.markdown('### Cash Flow Projection without Improvement')
        
        tab1, tab2 = st.tabs(["By Profit", "By Revenue & Expense"])
        
        with tab1:
            st.plotly_chart(without_improvement_plot, key="without improvement profit True")
            
        with tab2:
            st.plotly_chart(model_cashflow.cashflow_plot(number_of_months, granularity='monthly', start_date=generate_datetime(start_year=start_year), by_profit=False), key="without improvement with profit false")
        

        model_cashflow.add_company_data("Corporate Wellness", corporate_wellness_df) if corporate_wellness_df is not None else None
        model_cashflow.add_company_data("Structured Saving", structured_saving_df) if structured_saving_df is not None else None
        model_cashflow.add_company_data("Fair Credit", fair_credit_df) if fair_credit_df is not None else None
        model_cashflow.add_company_data("Financial Market Allocation", financial_market_allocation_df) if financial_market_allocation_df is not None else None
        model_cashflow.add_company_data("Item Code Recommendation", item_code_recommendation_df) if item_code_recommendation_df is not None else None
        model_cashflow.add_company_data("Subscription Payments", subscription_df) if subscription_scheme_df is not None else None
        model_cashflow.add_company_data("Group Discount", group_discount_df) if group_discount_df is not None else None
        model_cashflow.add_company_data("Family Discount", family_discount_df) if family_discount_df is not None else None
        model_cashflow.add_company_data("Contribution Amount", contribution_amout_df) if contribution_amout_df is not None else None
        model_cashflow.add_company_data("Optimized Supplier Selection", optimized_supplier_selection_df) if optimized_supplier_selection_df is not None else None

        with_improvement_df = model_cashflow.merge_dataframes_with_category()
        with_improvement_dataframes = model_cashflow.collection_df.copy()
       
       
        st.markdown('### Cash Flow Projection with Improvement')
        
        tab1, tab2 = st.tabs(["By Profit", "By Revenue & Expense"])
        
        with tab1:
        
            st.plotly_chart(model_cashflow.cashflow_plot(number_of_months, granularity='monthly', start_date=generate_datetime(start_year=start_year), by_profit=True), key="with improvement profit true")
        

        with tab2:
            st.plotly_chart(model_cashflow.cashflow_plot(number_of_months, granularity='monthly', start_date=generate_datetime(start_year=start_year), by_profit=False), key="with improvement profit false")
            
        
        
        st.markdown('### Comparison')
        
       
        
        # st.dataframe(with_improvement_df)
        # st.dataframe(without_improvement_df)
        
        


        def get_growth_trend(values, annualize=False, periods_per_year=12):
            """
            Perform linear regression on a sorted time-series-like data, 
            calculate the regression coefficient (slope), 
            and express it as a percentage growth trend.
            
            Parameters:
            values (pd.Series): Chronologically sorted dependent variable data (e.g., revenue)
            annualize (bool): Whether to annualize the percentage growth (default is False)
            periods_per_year (int): Number of periods in a year (default is 12, for monthly data)
            
            Returns:
            dict: Dictionary containing slope, growth percentage, and optionally annualized growth
            """
            if not isinstance(values, pd.Series):
                raise ValueError("Input must be a pandas Series.")
            
            # Create independent variable (time period)
            period = np.arange(len(values)) + 1  # Period starts from 1
            
            # Define independent (X) and dependent (y) variables
            X = sm.add_constant(period)  # Add constant for intercept
            y = values
            
            # Perform regression
            model = sm.OLS(y, X).fit()
            
            # Extract the slope (regression coefficient)
            slope = model.params[1]
            
            # Calculate the average revenue
            avg_revenue = values.mean()
            
            # Calculate the growth percentage
            growth_percentage = (slope / avg_revenue) * 100
            
            result = {
                "slope": slope,
                "growth_percentage": growth_percentage
            }
            
            # Annualize the growth percentage if needed
            if annualize:
                annualized_growth = (1 + slope / avg_revenue) ** periods_per_year - 1
                result["annualized_growth_percentage"] = annualized_growth * 100
            
            return result

        
        
        col1, col2 = st.columns(2)
        
        with col1:
            
            without_improvement_df = model_cashflow.groupby_dataframes_month_year(without_improvement_df)[:period_forecast]

            # Calculate Coefficient of Variation for Profit
            without_improvement_CoV = (
                without_improvement_df['Profit'].std() / without_improvement_df['Profit'].mean()
            )
            
            
            # Calculate monthly growth rates
            without_improvement_df['Growth Rate (%)'] = without_improvement_df['Profit'].pct_change() * 100

            # Calculate the average monthly profit growth (excluding the first NaN value)
            MoM_average_growth_without = without_improvement_df['Growth Rate (%)'].mean()
            
            without_improvement_trendline_amount = get_growth_trend(without_improvement_df['Profit'])['slope']
            without_improvement_trendline_percentage = get_growth_trend(without_improvement_df['Profit'])['growth_percentage']
            
            
            st.markdown("#### Without Improvement")


            # Display result in Streamlit
            st.metric("Coefficient of Variation (Net Cashflow)", f"{without_improvement_CoV * 100:.2f}%", delta=0, delta_color="off", help="Coefficient of Variation (CoV) measures the dispersion of a dataset relative to its mean. A lower CoV indicates a more stable cashflow.")
            st.metric("Average Monthly Growth (Net Cashflow)", f"{MoM_average_growth_without:.2f}%", delta=0, delta_color="off", help="Average monthly growth rate of the clinic's net cashflow. It averages the percentage change in cashflow from month to month.")
            st.metric("Net Cashflow in End Period", f"${without_improvement_df['Profit'].sum():,.0f}", delta=0, delta_color="off", help="Total net cashflow of the clinic at the end of the forecast period. Does not include potential receivable in the future beyond forecast period.")
            st.metric("Net Cashflow Trendline ($)", f"${without_improvement_trendline_amount:.2f}", delta=0, delta_color="off", help="The smoothed trendline of the clinic's net cashflow growth. It indicates the clinic's growth trend by amount.")
            st.metric("Net Cashflow Trendline (%)", f"{without_improvement_trendline_percentage:.2f}%", delta=0, delta_color="off", help="The smoothed trendline of the clinic's net cashflow growth divided by the average net cashflow. It indicates the clinic's growth trend by average percentage of net cashflow.")
            # st.metric("Cash Flow Margin", f"{without_improvement_df['Profit'].mean() / without_improvement_df['Revenue'].mean() * 100:.2f}%", delta=0, delta_color="off", help="The clinic's net cashflow divided by the clinic's revenue. It indicates the clinic's profit margin.")
            
        with col2:
            # Ensure Period column is datetime
            with_improvement_df = model_cashflow.groupby_dataframes_month_year(with_improvement_df)[:period_forecast]
            


            # Calculate Coefficient of Variation for Profit
            with_improvement_CoV = (
                with_improvement_df['Profit'].std() / with_improvement_df['Profit'].mean()
            )
            
           # Calculate monthly growth rates
            with_improvement_df['Growth Rate (%)'] = with_improvement_df['Profit'].pct_change() * 100

            # Calculate the average monthly profit growth (excluding the first NaN value)
            MoM_average_growth_with = with_improvement_df['Growth Rate (%)'].mean()

            with_improvement_trendline_amount = get_growth_trend(with_improvement_df['Profit'])['slope']
            with_improvement_trendline_percentage = get_growth_trend(with_improvement_df['Profit'])['growth_percentage']

            
            st.markdown('#### With Improvement')

            # st.dataframe(with_improvement_df)

            # Display result in Streamlit
            CoV_diff = with_improvement_CoV - without_improvement_CoV
            st.metric("Coefficient of Variation (Net Cashflow)", f"{with_improvement_CoV * 100:.2f}%", delta=f"{CoV_diff*100:.2f}%" , delta_color="inverse")

            st.metric("Average Monthly Growth (Net Cashflow)", f"{MoM_average_growth_with:.2f}%", delta=f"{(MoM_average_growth_with - MoM_average_growth_without):.2f}%", delta_color="normal")
            st.metric("Net Cashflow in End Period", f"${with_improvement_df['Profit'].sum():,.0f}", delta=f"{(with_improvement_df['Profit'].sum() - without_improvement_df['Profit'].sum()):,.0f}", delta_color="normal")
            st.metric("Net Cashflow Trendline ($)", f"${with_improvement_trendline_amount:.2f}", delta=f"{(with_improvement_trendline_amount - without_improvement_trendline_amount):.2f}", delta_color="normal")
            st.metric("Net Cashflow Trendline (%)", f"{with_improvement_trendline_percentage:.2f}%", f"{(with_improvement_trendline_percentage - without_improvement_trendline_percentage):.2f}%")
            # st.metric("Cash Flow Margin", f"{with_improvement_df['Profit'].mean() / with_improvement_df['Revenue'].mean() * 100:.2f}%", delta=f"{(with_improvement_df['Profit'].mean() / with_improvement_df['Revenue'].mean() - without_improvement_df['Profit'].mean() / without_improvement_df['Revenue'].mean()) * 100:.2f}%")

        # round the dataframe
        without_improvement_df = without_improvement_df.round(0)
        with_improvement_df = with_improvement_df.round(0)
        
        with_improvement_df_profit = with_improvement_df['Profit'].values
        
        
        
        collection_dataframes = {"Without Improvement": without_improvement_dataframes, "With Improvement": with_improvement_dataframes}
        
        # st.write(without_improvement_dataframes==with_improvement_dataframes)
        
        # st.write(collection_dataframes)
        
        # st.write(model_cashflow.collection_df)
        
        st.plotly_chart(model_cashflow.cashflow_plot_cumulative_only(number_of_months, granularity='monthly', start_date=generate_datetime(start_year=start_year), collection_dataframe=collection_dataframes))
        
        st.divider()
        
        st.markdown("#### Cash Flow Table Comparison ")

        st.dataframe(without_improvement_df[['Month_Year', 'Revenue', 'Expense', 'Profit', 'Growth Rate (%)']], use_container_width=True)
        st.dataframe(with_improvement_df[['Month_Year', 'Revenue', 'Expense', 'Profit', 'Growth Rate (%)']], use_container_width=True)

            
        
    # st.write(clinic_projected_cashflow_set)
        
        # model_cashflow.remove_all_companies()
        # model_cashflow.add_company_data("Fair Credit", fair_credit_df) if fair_credit_df is not None else None
        # st.plotly_chart(model_cashflow.cashflow_plot(number_of_months, granularity='monthly'))
        
        
        
        
        # after_improvement_df = forecast_df.copy()
        # after_improvement_df['EBIT'] = after_improvement_df['Revenue'] - after_improvement_df['Expense']
        
        # if corporate_wellness == True:
        #     after_improvement_df, after_improvement_chart_fig = model.merge_and_plot_ebit(after_improvement_df, pd.read_csv("corporate_cashflow_AUD.csv"))
            
        # potential_ebit_after_12_months = after_improvement_df['EBIT'].sum()
        
        
        # # if school_outreach == True:
        # #     potential_ebit_after_12_months += 684000000 / 10000
        # #     after_improvement_df, after_improvement_chart_fig = model.merge_and_plot_ebit(after_improvement_df, pd.read_csv("school_cashflow.csv"))
        
        # comparison_fig = model.compare_and_plot_ebit(forecast_df, after_improvement_df, label=['Before Improvement', 'After Improvement'])
        # st.markdown("#### Comparison of Predicted EBIT Flow Without and With Improvement")
        # st.plotly_chart(comparison_fig)
        
        # col3, col4 = st.columns(2)

        # with col3:
        #     st.markdown("### Without Improvement")
        #     st.metric("EBIT in 12 Months", f"${current_ebit_after_12_months:,.0f}")   
        #     st.metric("EBIT Ratio", f"{current_ebit_ratio:.2%}")
            
        # with col4:
        #     st.markdown("### With Improvement")
        #     st.metric("EBIT in 12 Months", f"${potential_ebit_after_12_months:,.0f}")   
            
        #     potential_ebit_ratio = potential_ebit_after_12_months / current_ebit_after_12_months * current_ebit_ratio 
            
        #     st.metric("EBIT Ratio", f"{potential_ebit_ratio:.2%}")
        
        
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     st.markdown("### Previous Clinic Value")
            
        #     st.metric("Previous EBIT Multiple", f"{clinic_ebit_multiple:.2f}")
        #     st.metric("Previous Clinic Value", f"${clinic_value:,.0f}")
            
        # with col2:
            
        #     clinic_data['EBIT'] = potential_ebit_after_12_months
        #     clinic_data['EBIT Ratio'] = potential_ebit_ratio
            
        #     model_current = ModelClinicValue(clinic_data)
            
        #     model_current.ebit = clinic_data['EBIT']
        #     model_current.ebit_ratio = clinic_data['EBIT Ratio']
            
        #     ebit_multiple = model_current.ebit_baseline_to_multiple(clinic_data['Net Sales Growth'])
        #     ebit_multiple = model_current.ebit_multiple_adjustment_due_dentist(ebit_multiple, clinic_data['Risk of Leaving Dentist'])
        #     ebit_multiple = model_current.ebit_multiple_adjustment_due_net_sales_variation(ebit_multiple, clinic_data['Relative Variation of Net Sales'])
        #     ebit_multiple = model_current.ebit_multiple_adjustment_due_number_patient_and_patient_spending_variability(ebit_multiple, clinic_data['Number of Active Patients'], clinic_data['Relative Variation of Patient Spending'])
            
        #     clinic_valuation = ebit_multiple * model_current.ebit
            
            
            
        #     st.markdown("### New Clinic Value")
            
        #     st.metric("New EBIT Multiple", f"{ebit_multiple:.2f}")
        #     st.metric("New Clinic Value", f"${clinic_valuation:,.0f}")
        
        # # st.plotly_chart(after_improvement_chart_fig)
        

        
        
        # if corporate_wellness == True:   
           
            
        #     st.write("") 
        #     st.write("The recalculation of Clinic Value by implementing Corporate Wellness Program would only affect the EBIT and EBIT Ratio of the clinic.")
        #     st.write("Despite that the EBIT and EBIT Ratio is increased, the EBIT Multiple here is remain the same since the increase is only a slight improvement.",
        #             "Please refer to the **Clinic Value Calculation** page on sidebar, to be precise on **First Calculation** section for more details.")             
            
            
            

    st.divider()

    st.markdown("### Clinic Value Comparison")
    

# NEW CLINIC PARAMETERS VALUE GOES HERE

    with open("baseline_values.pkl", 'rb') as f:
        baseline_values = pickle.load(f)

    
    new_relative_variation_net_sales = with_improvement_CoV
    new_ebit = with_improvement_df['Profit'].sum() / (len(with_improvement_df) / 12)
    new_ebit_ratio = new_ebit / with_improvement_df['Revenue'].sum()
    
    new_gross_profit = with_improvement_df['Profit'].sum() - clinic_data['General Expense'] # assuming the general expense is the same as the previous year
    
    # st.write("Previous Gross Profit: ", clinic_data['Gross Profit'])
    # st.write("New Gross Profit: ", new_gross_profit)

    new_yoy_growth = (new_gross_profit - clinic_data['Gross Profit']) / clinic_data['Gross Profit'] # We are using YoY of Gross Profit instead of Sales/Revenue for now
    
    # Fitout value adjustmend is skipped for now (if it were to be modelled, it needs to be able to catch if there were any fitout in the forecast period)
    new_last_fitout_year = clinic_data['Last Fitout Year'] + (period_forecast / 12)
    
    
    # Equipment value adjustment is skipped for now (if it were to be modelled, it needs to be able to catch if there were any equipment procurement in the forecast period)
    new_equipment_usage_ratio = ((period_forecast / 12) + 2.5) / 5
    new_equipment_remaining_value = clinic_data['Total Remaining Value'] / new_equipment_usage_ratio * (2.5/5)
    
    # Number of Unique Patient is not modelled, hence skipped for now
    
    # Relative Variation of Patient Spending is not interpreted as changed despite some accounting manipulation
    
    # other parameter remain the same
    
    model_valuation = ModelClinicValue(clinic_data)
    model_valuation.ebit = new_ebit
    model_valuation.ebit_ratio = new_ebit_ratio
    model_valuation.net_sales_growth = new_yoy_growth
    model_valuation.relative_variability_net_sales = new_relative_variation_net_sales
    model_valuation.last_fitout_year = new_last_fitout_year
    
    
    ebit_multiple = model_valuation.ebit_baseline_to_multiple(new_yoy_growth)
    equipment_adjusting_value = model_valuation.equipment_adjusting_value(new_equipment_remaining_value, baseline_values['Equipments Value'], baseline_values['Equipment Usage Ratio'])
    fitout_adjusting_value = model_valuation.fitout_adjusting_value(clinic_data['Fitout Value'], new_last_fitout_year, baseline_values['Fitout Value'], baseline_values['Last Fitout Year'])
    
    ebit_multiple = model_valuation.ebit_multiple_adjustment_due_dentist(ebit_multiple, clinic_data['Risk of Leaving Dentist'])
    ebit_multiple = model_valuation.ebit_multiple_adjustment_due_net_sales_variation(ebit_multiple, new_relative_variation_net_sales)
    ebit_multiple = model_valuation.ebit_multiple_adjustment_due_number_patient_and_patient_spending_variability(ebit_multiple, clinic_data['Number of Active Patients'], clinic_data['Relative Variation of Patient Spending'])
    
    clinic_valuation = ebit_multiple * model_valuation.ebit
    
    final_clinic_valuation = clinic_valuation + equipment_adjusting_value + fitout_adjusting_value


    with st.expander("Parameter Comparison"):

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Previous Clinic Parameters")
            st.metric("Gross Profit YoY Growth", f"{clinic_data['Net Sales Growth']:.2%}")
            st.metric("Relative Variation of Net Sales", f"{clinic_data['Relative Variation of Net Sales']:.2%}")
            st.metric("EBIT Ratio", f"{clinic_data['EBIT Ratio']:.2%}")
            
        with col2:
            st.markdown("#### New Clinic Parameters")
            st.metric("Gross Profit YoY Growth", f"{new_yoy_growth:.2%}")
            st.metric("Relative Variation of Net Sales", f"{new_relative_variation_net_sales:.2%}")
            st.metric("EBIT Ratio", f"{new_ebit_ratio:.2%}")
    
    

    

    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Current EBIT", f"${clinic_data['EBIT']:,.0f}")
        st.caption("Current EBIT calculation involve Other Income and Depreciation")
        
        st.metric("Current EBIT Multiple", f"{clinic_data['EBIT Multiple']:.2f}")
        
    with col2:
        st.metric("Projected EBIT/Net Cashflow (Without Improvement)", f"${sum(with_improvement_df_profit):,.0f}")
        st.caption("Projected EBIT/Net Cashflow calculated without projection of Other Income and Depreciation")
        
        st.metric("Projected EBIT Multiple", f"{ebit_multiple:.2f}")
    
    # st.write(sum(without_improvement_df_profit))
    # st.write(clinic_data['EBIT Multiple'])

    col3, col4 = st.columns(2)
    
    projected_clinic_value = sum(with_improvement_df_profit) * ebit_multiple + equipment_adjusting_value + fitout_adjusting_value

    with col3:
        st.metric("Current Clinic Value", f"${clinic_value:,.0f}")
        
    with col4:
        st.metric("Projected Clinic Value", f"${projected_clinic_value:,.0f}")
        st.caption("Based on New EBIT (Net Cashflow) and previous EBIT Multiple + Adjustment of Equipment and Fit Out (if any)")
        
    # st.divider()

    # st.markdown("#### Conclusion")