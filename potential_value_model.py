import pickle
import pandas as pd
import numpy as np
import streamlit as st
from model import ModelClinicValue
import corporate_wellness_app
import school_outreach_app
from model_forecasting import ModelForecastPerformance
    
def app():

    st.title("Potential Value Calculator")
    
    
    st.markdown("### Projection of Current Clinic (without Improvement)")
    
    st.write("Based on previous years' EBIT, Year-on-Year growth and Revenue Relative Variation of the clinic, here is the projected EBIT flow for the next 12 months")
    
    clinic_data = pickle.load(open('clinic_value.pkl', 'rb'))
    model = ModelForecastPerformance(clinic_data)
    
    # forecast_df = model.forecast_ebit_flow(12)
    forecast_df = pd.read_csv("current_clinic_cashflow_forecast.csv")
    forecast_linechart = model.plot_ebit_line_chart(forecast_df)
    
    clinic_value = clinic_data['Clinic Valuation Adjusted']
    clinic_ebit_multiple = clinic_data['EBIT Multiple']


    current_ebit = clinic_data['EBIT']
    current_ebit_ratio = clinic_data['EBIT Ratio']
    current_growth = clinic_data['Net Sales Growth']
    current_net_sales_relative_variation = clinic_data['Relative Variation of Net Sales']
    
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Previous year EBIT", f"${current_ebit:,.0f}")
        
    with col2:
        st.metric("Year-on-Year Growth", f"{current_growth * 100}%")
        
    with col3:
        st.metric("Monthly Relative Variation", f"{current_net_sales_relative_variation * 100}%")
    
    st.write("")
    st.markdown("##### Predicted EBIT Flow for the Next 12 Months")
    st.plotly_chart(forecast_linechart)
    
    with st.popover("explain EBIT flow"):
        st.write(
            "The predicted EBIT flow for the next 12 months is generated with random fluctuations, "
            "but it aligns with the clinic's current metrics.") 
            
        st.write("Specifically, the total EBIT for the year "
            f"reflects the clinic's expected year-on-year growth which is {current_growth * 100:.0f}% "
             f"from the previous year EBIT of ${current_ebit:,.0f}, while the monthly variations capture "
            f"the clinic's historical fluctuation patterns (relative variation, which is {current_net_sales_relative_variation * 100:.0f}%)"
            )           
    

    #load clinic_value.pkl and get the value of it to be the value of variable 'clinic_value'

    st.divider()


    current_ebit_after_12_months = current_ebit * (1+clinic_data['Net Sales Growth'])


    potential_ebit_after_12_months = current_ebit_after_12_months
    

    st.markdown("### Strategy to implement")

    col1, col2 = st.columns(2)

    with col1:
        corporate_wellness = st.checkbox("Corporate Wellness Program", value=False)
        
    with col2:
        with st.popover("details"):
            corporate_wellness_app.app()
            
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     school_outreach = st.checkbox("School Outreach Program", value=False)
        
    # with col2:
    #     with st.popover("details"):
    #         school_outreach_app.app()
        
    


    if st.button("Calculate New Clinic Value"):
        
        after_improvement_df = forecast_df.copy()
        
        if corporate_wellness == True:
            after_improvement_df, after_improvement_chart_fig = model.merge_and_plot_ebit(after_improvement_df, pd.read_csv("corporate_cashflow_AUD.csv"))
            
        potential_ebit_after_12_months = after_improvement_df['EBIT'].sum()
        
        
        # if school_outreach == True:
        #     potential_ebit_after_12_months += 684000000 / 10000
        #     after_improvement_df, after_improvement_chart_fig = model.merge_and_plot_ebit(after_improvement_df, pd.read_csv("school_cashflow.csv"))
        
        comparison_fig = model.compare_and_plot_ebit(forecast_df, after_improvement_df, label=['Before Improvement', 'After Improvement'])
        st.markdown("#### Comparison of Predicted EBIT Flow Without and With Improvement")
        st.plotly_chart(comparison_fig)
        
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### Without Improvement")
            st.metric("EBIT in 12 Months", f"${current_ebit_after_12_months:,.0f}")   
            st.metric("EBIT Ratio", f"{current_ebit_ratio:.2%}")
            
        with col4:
            st.markdown("### With Improvement")
            st.metric("EBIT in 12 Months", f"${potential_ebit_after_12_months:,.0f}")   
            
            potential_ebit_ratio = potential_ebit_after_12_months / current_ebit_after_12_months * current_ebit_ratio 
            
            st.metric("EBIT Ratio", f"{potential_ebit_ratio:.2%}")
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Previous Clinic Value")
            
            st.metric("Previous EBIT Multiple", f"{clinic_ebit_multiple:.2f}")
            st.metric("Previous Clinic Value", f"${clinic_value:,.0f}")
            
        with col2:
            
            clinic_data['EBIT'] = potential_ebit_after_12_months
            clinic_data['EBIT Ratio'] = potential_ebit_ratio
            
            model_current = ModelClinicValue(clinic_data)
            
            model_current.ebit = clinic_data['EBIT']
            model_current.ebit_ratio = clinic_data['EBIT Ratio']
            
            ebit_multiple = model_current.ebit_baseline_to_multiple(clinic_data['Net Sales Growth'])
            ebit_multiple = model_current.ebit_multiple_adjustment_due_dentist(ebit_multiple, clinic_data['Risk of Leaving Dentist'])
            ebit_multiple = model_current.ebit_multiple_adjustment_due_net_sales_variation(ebit_multiple, clinic_data['Relative Variation of Net Sales'])
            ebit_multiple = model_current.ebit_multiple_adjustment_due_number_patient_and_patient_spending_variability(ebit_multiple, clinic_data['Number of Active Patients'], clinic_data['Relative Variation of Patient Spending'])
            
            clinic_valuation = ebit_multiple * model_current.ebit
            
            
            
            st.markdown("### New Clinic Value")
            
            st.metric("New EBIT Multiple", f"{ebit_multiple:.2f}")
            st.metric("New Clinic Value", f"${clinic_valuation:,.0f}")
        
        # st.plotly_chart(after_improvement_chart_fig)
        

           
           
           
        st.write("") 
        st.write("The recalculation of Clinic Value by implementing Corporate Wellness Program would only affect the EBIT and EBIT Ratio of the clinic.")
        st.write("Despite that the EBIT and EBIT Ratio is increased, the EBIT Multiple here is remain the same since the increase is only a slight improvement.",
                 "Please refer to the **Clinic Value Calculation** page on sidebar, to be precise on **First Calculation** section for more details.")             
        
        
            

    # col3, col4 = st.columns(2)

    # with col3:
    #     st.metric("Current Clinic Value", f"${clinic_value:,.0f}")
        
    # with col4:
    #     st.metric("Potential Clinic Value", f"${potential_clinic_value:,.0f}")
    #     st.caption("After Strategy Implementation")
        
    # st.divider()

    # st.markdown("#### Conclusion")