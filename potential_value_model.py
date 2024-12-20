import pickle
import pandas as pd
import numpy as np
import streamlit as st
from model import ModelClinicValue


from model_forecasting import ModelForecastPerformance
from cashflow_plot import ModelCashflow
from streamlit_extras.stateful_button import button

import structured_saving as structured_saving_app
import corporate_wellness_app
import school_outreach_app
from Model_Initiatives.Financial_Market_Allocation import financial_market_allocation_app
from Model_Initiatives.Supplier_Selection import streamlit_per_clinic as supplier_selection_app
from Model_Initiatives.Subscription_Scheme import streamlit as subscription_scheme_app

import os
    
def app():

    st.title("Potential Value Calculator")
    
    dataset = st.selectbox("Select Dataset", ['Dataset 1', 'Dataset 2', 'Dataset 3'])
    
    dataset = int(dataset.split()[-1])
    
    
    clinic_data_set = pickle.load(open(f'dummy_clinic_model/pkl_files/dataset_{dataset}/clinic_value_set.pkl', 'rb'))
    
    
    selected_clinic = st.selectbox("Select Clinic", list(clinic_data_set.keys()))
    
    
    
    with st.container(border=True):
        st.markdown('### Clinic Value to Acquire')
        st.write("Based on Calculation of Clinic Value of observed clinic, here is the value of the clinic to acquire")
        
        clinic_data = clinic_data_set[selected_clinic]
        
        clinic_value = clinic_data['Clinic Valuation Adjusted']
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
        # forecast_df.to_csv("forecast_df_deletable.csv", index=False)
        # st.dataframe(forecast_df)
        
        indirect_expense = model.forecast_indirect_cost(period_forecast, start_year=start_year)
        
        
        
        # FORECASTING TREATMENTS
        
        treatment_details = pd.read_csv("treatment_with_details.csv")
        item_code_df = pd.read_csv("dummy_clinic_model\cleaned_item_code.csv")
        
        
        # forecast_df_with_treatments = model.generate_forecast_treatment_df_by_profit(treatment_details, forecast_df)
        forecast_df_with_treatments = model.generate_forecast_item_code_by_profit(item_code_df, forecast_df, patient_pool=clinic_data['Patient Pool'], start_year=start_year)
        forecast_df_with_treatments.to_csv("forecast_df_treatment_deletable.csv", index=False)
        
        # st.dataframe(forecast_df_with_treatments)
        
        # st.plotly_chart(model.summary_table(forecast_df_with_treatments, by=show_by))
        
        forecast_df = forecast_df_with_treatments[['Period', 'Revenue', 'Expense']].groupby('Period').sum().reset_index()
        # st.dataframe(forecast_df)

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
        
        # save into json
        with open("Model_Initiatives\Financial_Market_Allocation\without_improvement_profit.json", 'w') as f:
            f.write(str(without_improvement_df_profit.tolist()))
        
        def generate_datetime(start_year=2024, start_month=1):
            return f"{start_year}-{start_month}-01"
            
        
        forecast_linechart_daily = model_cashflow.cashflow_plot(number_of_months, start_date=generate_datetime(start_year=start_year))
        forecast_linechart_weekly = model_cashflow.cashflow_plot(number_of_months, granularity='weekly',  start_date=generate_datetime(start_year=start_year))
        forecast_linechart_monthly = model_cashflow.cashflow_plot(number_of_months, granularity='monthly',  start_date=generate_datetime(start_year=start_year))
        forecast_linechart_quarterly = model_cashflow.cashflow_plot(number_of_months, granularity='quarterly', start_date=generate_datetime(start_year=start_year))
        
        
        clinic_value = clinic_data['Clinic Valuation Adjusted']
        clinic_ebit_multiple = clinic_data['EBIT Multiple']


        current_ebit = clinic_data['EBIT']
        current_ebit_ratio = clinic_data['EBIT Ratio']
        current_growth = clinic_data['Net Sales Growth']
        current_net_sales_relative_variation = clinic_data['Relative Variation of Net Sales']
        

        
        st.write("")
        st.markdown(f"##### Predicted Cash Flow for the Next {period_forecast} Months")
        
        with st.container(border=True):
            st.markdown("**Approach 1: Foreacast based on previous year's EBIT, EBIT Ratio, and YoY Growth**")
            
            st.write("Based on previous years' EBIT, Year-on-Year growth and Revenue Relative Variation of the clinic, here is the projected EBIT flow for the next 12 months. We call this approach as Fluctuation-Adjusted EBIT Forecast.")

            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Previous year EBIT", f"${current_ebit:,.0f}")
                
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
                    
                st.write("Specifically, the total EBIT (Gross Profit - Indirect Expenses) for the year (shall you run 12 months forecast) "
                    f"reflects the clinic's expected year-on-year growth which is {current_growth * 100:.0f}% "
                    f"from the previous year EBIT of ${current_ebit:,.0f}, while the monthly variations capture "
                    f"the clinic's historical fluctuation patterns (relative variation, which is {current_net_sales_relative_variation * 100:.0f}%)"
                    )           
            st.plotly_chart(model.summary_table(forecast_df_with_treatments, sliced_amortization_df, indirect_expense, equipment_df, fitout_df, by=show_by))
            

        
        with st.container(border=True):
            st.markdown("**Approach 2: Forecast using Three Month Moving Average based on previous months data**")
            
            MA_forecast_df = model.forecast_revenue_expenses(3) # instead of using actual data, this method forecast three months as basis of most recent months of historical data 
            
            MA_forecast_df = model.forecast_revenue_expenses_MA(MA_forecast_df, period_forecast, 3, start_month=1)
            
            
            
            # MA_forecast_df_with_treatments = model.generate_forecast_treatment_df_by_profit(treatment_details, MA_forecast_df)
            MA_forecast_df_with_treatments = model.generate_forecast_item_code_by_profit(item_code_df, MA_forecast_df, start_year=start_year)
            MA_forecast_df = MA_forecast_df_with_treatments[['Period', 'Revenue', 'Expense']].groupby('Period').sum().reset_index()
            
            MA_indirect_cost = model.forecast_indirect_cost(period_forecast, start_year=start_year)
            
            
            model_cashflow.remove_all_companies()
            model_cashflow.add_company_data("Gross Profit", MA_forecast_df)
            model_cashflow.add_company_data("Debt Repayment", sliced_amortization_df) if acquiring_funding == "Borrow" else None
            model_cashflow.add_company_data("Indirect Expense", MA_indirect_cost)
            model_cashflow.add_company_data("Equipment Procurement", equipment_df)
            model_cashflow.add_company_data("Fit Out", fitout_df)
            
            MA_forecast_linechart_daily = model_cashflow.cashflow_plot(number_of_months, start_date=generate_datetime(start_year=start_year))
            MA_forecast_linechart_weekly = model_cashflow.cashflow_plot(number_of_months, granularity='weekly', start_date=generate_datetime(start_year=start_year))
            MA_forecast_linechart_monthly = model_cashflow.cashflow_plot(number_of_months, granularity='monthly', start_date=generate_datetime(start_year=start_year))
            
            tab1, tab2, tab3 = st.tabs(["Daily", "Weekly", "Monthly"])
    
            with tab1:
                st.plotly_chart(MA_forecast_linechart_daily)
                
            with tab2:  
                st.plotly_chart(MA_forecast_linechart_weekly)
            
            with tab3:
                st.plotly_chart(MA_forecast_linechart_monthly)
        

            
            with st.popover("explain Cash flow for Approach 2"):
                st.write(
                    f"The predicted Cash flow for the next {period_forecast} months is calculated using Moving Average method. "
                    "The method is used to smooth out the random fluctuations in the data.")
                st.write("It will use the average of the last months (depends on specified Moving Average Period) of historical data"
                    "to predict the first forecast months, for it will keeps smoothing out the data until the last month of the forecast period.")
                st.write("Currently the Moving Average Period is set to 3 months, which means it will use the average of the last 3 months of historical data. We call this approach as Moving Average Forecast")
            
            st.plotly_chart(model.summary_table(MA_forecast_df_with_treatments, sliced_amortization_df, indirect_expense, equipment_df, fitout_df, by=show_by))

        #load clinic_value.pkl and get the value of it to be the value of variable 'clinic_value'
        
        
        current_ebit_after_12_months = current_ebit * (1+clinic_data['Net Sales Growth'])


        potential_ebit_after_12_months = current_ebit_after_12_months
        
        
        
        # SAVE INTO PKL
        
        file_name = 'clinic_projected_cashflow_set.pkl'
        ID = selected_clinic

        # Initialize or load existing dictionary
        if os.path.exists(file_name):
            # Load existing data
            with open(file_name, 'rb') as f:
                clinic_projected_cashflow_set = pickle.load(f)
        else:
            # Start with an empty dictionary
            clinic_projected_cashflow_set = {}
        
   
        clinic_projected_cashflow_set[ID] = {
            "Approach 1": 
                {"Gross Profit": forecast_df,
                "Debt Repayment": sliced_amortization_df if acquiring_funding == "Borrow" else None,
                "Indirect Expense": indirect_expense,
                "Equipment Procurement": equipment_df,
                "Fit Out": fitout_df},
            "Approach 2": 
                {"Gross Profit": MA_forecast_df,
                "Debt Repayment": sliced_amortization_df if acquiring_funding == "Borrow" else None,
                "Indirect Expense": MA_indirect_cost,
                "Equipment Procurement": equipment_df,
                "Fit Out": fitout_df}
        }
        
        # save into pkl
        with open(file_name, 'wb') as f:
            pickle.dump(clinic_projected_cashflow_set, f)
   
    st.divider()



    

    st.markdown("### Strategy to implement")

    col1, col2 = st.columns(2)

    with col1:
        corporate_wellness = st.checkbox("Corporate Wellness Program", value=False)
        structured_saving = st.checkbox("Structured Saving Plan", value=False)
        fair_credit = st.checkbox("Fair Credit Program", value=False)
        financial_market_allocation = st.checkbox("Financial Market Allocation", value=False)
        optimized_supplier_selection = st.checkbox("Optimized Supplier Selection", value=False)
        subscription_scheme = st.checkbox("Subscription Scheme", value=False)
        
    with col2:
        with st.popover("details"):
            corporate_wellness_app.app()
            
        with st.popover("details"):
            structured_saving_app.app()
            
        with st.popover("details financial market allocation"):
            financial_market_allocation_app.app()
            
        with st.popover("details optimized supplier selection"):
            supplier_selection_app.app(dataset, selected_clinic)
            
        with st.popover("details subscription scheme"):
            subscription_scheme_app.app(forecast_df_with_treatments) # For now only compatible in converting using approach 1
            
            
            
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     school_outreach = st.checkbox("School Outreach Program", value=False)
        
    # with col2:
    #     with st.popover("details"):
    #         school_outreach_app.app()
        
    


    if button("Calculate New Cash Flow", key="calculate-strategy"):

        
        corporate_wellness_df = None
        structured_saving_df = None
        fair_credit_df = None
        financial_market_allocation_df = None
        # optimized_supplier_selection_df = None
        subscription_scheme_df = None
        
        if corporate_wellness == True:
            corporate_wellness_df = pd.read_csv("corporate_cashflow_AUD.csv")[:period_forecast]
            corporate_wellness_df['Period'] = corporate_wellness_df['Period'].apply(lambda period: model.generate_date_from_month(int(period), method='first_day', start_year=start_year))
            clinic_projected_cashflow_set[selected_clinic]["Approach 1"]["Corporate Wellness"] = corporate_wellness_df
            clinic_projected_cashflow_set[selected_clinic]["Approach 2"]["Corporate Wellness"] = corporate_wellness_df
            
        if structured_saving == True:
            structured_saving_df = pd.read_csv("structured_saving_cashflow.csv")[:period_forecast]
            structured_saving_df['Period'] = structured_saving_df['Period'].apply(lambda period: model.generate_date_from_month(int(period), method='first_day', start_year=start_year))
            clinic_projected_cashflow_set[selected_clinic]["Approach 1"]["Structured Saving"] = structured_saving_df
            clinic_projected_cashflow_set[selected_clinic]["Approach 2"]["Structured Saving"] = structured_saving_df
        
        if fair_credit == True:
            fair_credit_df = pd.read_csv("fair_credit_cashflow.csv")[:period_forecast]
            fair_credit_df['Period'] = fair_credit_df['Period'].apply(lambda period: model.generate_date_from_month(int(period), method='first_day', start_year=start_year))
            clinic_projected_cashflow_set[selected_clinic]["Approach 1"]["Fair Credit"] = fair_credit_df
            clinic_projected_cashflow_set[selected_clinic]["Approach 2"]["Fair Credit"] = fair_credit_df
            
        if financial_market_allocation == True:
            financial_market_allocation_df = pd.read_csv("Model_Initiatives/Financial_Market_Allocation/cashflow.csv")[:period_forecast]
            clinic_projected_cashflow_set[selected_clinic]["Approach 1"]["Financial Market Allocation"] = financial_market_allocation_df
            clinic_projected_cashflow_set[selected_clinic]["Approach 2"]["Financial Market Allocation"] = financial_market_allocation_df
            
        if optimized_supplier_selection == True:
            reduced_material_cost = pd.read_pickle("Model_Initiatives/Supplier_Selection/reduced_material_cost.pkl")
            forecast_df['Expense'] = forecast_df['Expense'] - (reduced_material_cost / len(forecast_df))
            
        if subscription_scheme == True:
            subscription_scheme_df = pd.read_csv("Model_Initiatives/Subscription_Scheme/after_subscription_scheme.csv")
            forecast_df = subscription_scheme_df.copy()
            MA_forecast_df = subscription_scheme_df.copy() # still not correct for the second approach
            

        # use Approach 1    
        model_cashflow.remove_all_companies()    
        model_cashflow.add_company_data("Gross Profit", forecast_df)
        model_cashflow.add_company_data("Debt Repayment", sliced_amortization_df) if acquiring_funding == "Borrow" else None
        model_cashflow.add_company_data("Indirect Expense", indirect_expense)
        model_cashflow.add_company_data("Equipment Procurement", equipment_df)
        model_cashflow.add_company_data("Fit Out", fitout_df)

        model_cashflow.add_company_data("Corporate Wellness", corporate_wellness_df) if corporate_wellness_df is not None else None
        model_cashflow.add_company_data("Structured Saving", structured_saving_df) if structured_saving_df is not None else None
        model_cashflow.add_company_data("Fair Credit", fair_credit_df) if fair_credit_df is not None else None
        model_cashflow.add_company_data("Financial Market Allocation", financial_market_allocation_df) if financial_market_allocation_df is not None else None
        # model_cashflow.add_company_data("Subscription Scheme", subscription_scheme_df) if subscription_scheme_df is not None else None


        with_improvement_df = model_cashflow.merge_dataframes_with_category()
       
        st.plotly_chart(model_cashflow.cashflow_plot(number_of_months, granularity='monthly', start_date=generate_datetime(start_year=start_year)))
        
        
        st.markdown('### Comparison')
        
       
        
        # st.dataframe(with_improvement_df)
        # st.dataframe(without_improvement_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            
            without_improvement_df = model_cashflow.groupby_dataframes_month_year(without_improvement_df)

            # Calculate Coefficient of Variation for Profit
            without_improvement_CoV = (
                without_improvement_df['Profit'].std() / without_improvement_df['Profit'].mean()
            )
            
            total_growth_without = 0
            num_periods = len(without_improvement_df) - 1  # Number of periods for growth calculation

            for i in range(1, len(without_improvement_df)):
                growth = (without_improvement_df['Profit'][i] - without_improvement_df['Profit'][i - 1]) / without_improvement_df['Profit'][i - 1]
                total_growth_without += growth

            # Average period-on-period growth
            MoM_average_growth_without = total_growth_without / num_periods
            
            
            st.markdown("#### Without Improvement")


            # Display result in Streamlit
            st.metric("Coefficient of Variation (Net Cashflow)", f"{without_improvement_CoV * 100:.2f}%", delta=0, delta_color="off")
            st.metric("Average Monthly Growth (Net Cashflow)", f"{MoM_average_growth_without * 100:.2f}%", delta=0, delta_color="off")
            
        with col2:
            # Ensure Period column is datetime
            with_improvement_df = model_cashflow.groupby_dataframes_month_year(with_improvement_df)
            


            # Calculate Coefficient of Variation for Profit
            with_improvement_CoV = (
                with_improvement_df['Profit'].std() / with_improvement_df['Profit'].mean()
            )
            
            total_growth_with = 0
            num_periods = len(with_improvement_df) - 1  # Number of periods for growth calculation

            for i in range(1, len(with_improvement_df)):
                growth = (with_improvement_df['Profit'][i] - with_improvement_df['Profit'][i - 1]) / with_improvement_df['Profit'][i - 1]
                total_growth_with += growth

            # Average period-on-period growth
            MoM_average_growth_with = total_growth_with / num_periods

            
            st.markdown('#### With Improvement')


            # Display result in Streamlit
            CoV_diff = with_improvement_CoV - without_improvement_CoV
            st.metric("Coefficient of Variation (Net Cashflow)", f"{with_improvement_CoV * 100:.2f}%", delta=f"{CoV_diff*100:.2f}%" , delta_color="inverse")

            st.metric("Average Monthly Growth (Net Cashflow)", f"{MoM_average_growth_with * 100:.2f}%", delta=f"{(MoM_average_growth_with - MoM_average_growth_without) * 100:.2f}%", delta_color="normal")

        # st.dataframe(with_improvement_df)
            
        
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
            
            
            

    # col3, col4 = st.columns(2)

    # with col3:
    #     st.metric("Current Clinic Value", f"${clinic_value:,.0f}")
        
    # with col4:
    #     st.metric("Potential Clinic Value", f"${potential_clinic_value:,.0f}")
    #     st.caption("After Strategy Implementation")
        
    # st.divider()

    # st.markdown("#### Conclusion")