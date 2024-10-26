import pickle
import pandas as pd
import numpy as np
import streamlit as st
from model import ModelClinicValue
import corporate_wellness_app
import school_outreach_app
from model_forecasting import ModelForecastPerformance
from cashflow_plot import ModelCashflow
from streamlit_extras.stateful_button import button
import structured_saving as structured_saving_app
    
def app():

    st.title("Potential Value Calculator")
    
    
    with st.container(border=True):
        st.markdown('### Clinic Value to Acquire')
        st.write("Based on Calculation of Clinic Value of observed clinic, here is the value of the clinic to acquire")
        clinic_data = pickle.load(open('clinic_value.pkl', 'rb'))
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
                
            borrowed = (clinic_value * (percentage_borrow/100))
            principal = borrowed - ((upfront_payment/100) * borrowed)
            amortization_df, monthly_payment, total_principal, total_interest = model.loan_amortization_schedule(principal, interest_rate, loan_term)
            
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

    period_forecast = st.number_input("Period to Forecast (months)", min_value=1, value=12, step=1)
    

    calculate = button("Calculate", key="calculate-existing")

    
    if calculate:
        model_cashflow = ModelCashflow()
        
        forecast_df = model.forecast_revenue_expenses(period_forecast)
        
        indirect_expense = model.forecast_indicect_cost(period_forecast)

        
        # FORECASTING TREATMENTS
        
        treatment_details = pd.read_csv("treatment_with_details.csv")
        forecast_df_with_treatments = model.generate_forecast_treatment_df_by_profit(treatment_details, forecast_df)
        
        
        forecast_df = forecast_df_with_treatments[['Period', 'Revenue', 'Expense']].groupby('Period').sum().reset_index()

        sliced_amortization_df = sliced_amortization_df[:period_forecast] if acquiring_funding == "Borrow" else None

        # forecast_df = pd.read_csv("current_clinic_cashflow_forecast.csv")
        model_cashflow.add_company_data("Gross Profit", forecast_df)
        model_cashflow.add_company_data("Debt Repayment", sliced_amortization_df) if acquiring_funding == "Borrow" else None
        model_cashflow.add_company_data("Indirect Expense", indirect_expense)
        forecast_linechart = model_cashflow.cashflow_plot()
        
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
            
            st.write("Based on previous years' EBIT, Year-on-Year growth and Revenue Relative Variation of the clinic, here is the projected EBIT flow for the next 12 months")

            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Previous year EBIT", f"${current_ebit:,.0f}")
                
            with col2:
                st.metric("Year-on-Year Growth", f"{current_growth * 100}%")
                
            with col3:
                st.metric("Monthly Relative Variation", f"{current_net_sales_relative_variation * 100}%")
            
            st.plotly_chart(forecast_linechart)
            
            with st.popover("explain Cash flow for Approach 1"):
                st.write(
                    f"The predicted Cash flow for the next {period_forecast} months is generated with random fluctuations, "
                    "but it aligns with the clinic's current metrics.") 
                    
                st.write("Specifically, the total EBIT (Gross Profit - Indirect Expenses) for the year (shall you run 12 months forecast) "
                    f"reflects the clinic's expected year-on-year growth which is {current_growth * 100:.0f}% "
                    f"from the previous year EBIT of ${current_ebit:,.0f}, while the monthly variations capture "
                    f"the clinic's historical fluctuation patterns (relative variation, which is {current_net_sales_relative_variation * 100:.0f}%)"
                    )           
            st.plotly_chart(model.summary_table(forecast_df_with_treatments, sliced_amortization_df, indirect_expense, by=show_by))
            

        
        with st.container(border=True):
            st.markdown("**Approach 2: Forecast using Three Month Moving Average based on previous months data**")
            
            MA_forecast_df = model.forecast_revenue_expenses(3)
            MA_forecast_df = model.forecast_revenue_expenses_MA(MA_forecast_df, period_forecast, 3)
            
            MA_forecast_df_with_treatments = model.generate_forecast_treatment_df_by_profit(treatment_details, MA_forecast_df)
            MA_forecast_df = MA_forecast_df_with_treatments[['Period', 'Revenue', 'Expense']].groupby('Period').sum().reset_index()
            
            MA_indirect_cost = model.forecast_indicect_cost(period_forecast)
            
            
            model_cashflow.remove_all_companies()
            model_cashflow.add_company_data("Gross Profit", MA_forecast_df)
            model_cashflow.add_company_data("Debt Repayment", sliced_amortization_df) if acquiring_funding == "Borrow" else None
            model_cashflow.add_company_data("Indirect Expense", MA_indirect_cost)
            
            forecast_MA_linechart = model_cashflow.cashflow_plot()
            st.plotly_chart(forecast_MA_linechart)
            
            with st.popover("explain Cash flow for Approach 2"):
                st.write(
                    f"The predicted Cash flow for the next {period_forecast} months is calculated using Moving Average method. "
                    "The method is used to smooth out the random fluctuations in the data.")
                st.write("It will use the average of the last months (depends on specified Moving Average Period) of historical data"
                    "to predict the first forecast months, for it will keeps smoothing out the data until the last month of the forecast period.")
                st.write("Currently the Moving Average Period is set to 3 months, which means it will use the average of the last 3 months of historical data")
            
            st.plotly_chart(model.summary_table(MA_forecast_df_with_treatments, sliced_amortization_df, indirect_expense, by=show_by))

        #load clinic_value.pkl and get the value of it to be the value of variable 'clinic_value'
        
        
        current_ebit_after_12_months = current_ebit * (1+clinic_data['Net Sales Growth'])


        potential_ebit_after_12_months = current_ebit_after_12_months
   
   
    st.divider()



    

    st.markdown("### Strategy to implement")

    col1, col2 = st.columns(2)

    with col1:
        corporate_wellness = st.checkbox("Corporate Wellness Program", value=False)
        structured_saving = st.checkbox("Structured Saving Plan", value=False)
        
    with col2:
        with st.popover("details"):
            corporate_wellness_app.app()
            
        with st.popover("details"):
            structured_saving_app.app()
            
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     school_outreach = st.checkbox("School Outreach Program", value=False)
        
    # with col2:
    #     with st.popover("details"):
    #         school_outreach_app.app()
        
    


    if button("Calculate New Clinic Value", key="calculate-strategy"):
        
        corporate_wellness_df = None
        structured_saving_df = None
        
        if corporate_wellness == True:
            corporate_wellness_df = pd.read_csv("corporate_cashflow_AUD.csv")[:period_forecast]
            
        if structured_saving == True:
            structured_saving_df = pd.read_csv("structured_saving_cashflow.csv")[:period_forecast]

        model_cashflow.add_company_data("Corporate Wellness", corporate_wellness_df) if corporate_wellness_df is not None else None
        model_cashflow.add_company_data("Structured Saving", structured_saving_df) if structured_saving_df is not None else None
        st.plotly_chart(model_cashflow.cashflow_plot())
        
        
        
        
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