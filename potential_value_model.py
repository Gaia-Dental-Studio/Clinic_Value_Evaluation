import pickle
import pandas as pd
import numpy as np
import streamlit as st
from model import ModelClinicValue
import corporate_wellness_app
import school_outreach_app
    
def app():

    st.title("Potential Value Calculator")

    #load clinic_value.pkl and get the value of it to be the value of variable 'clinic_value'
    clinic_data = pickle.load(open('clinic_value.pkl', 'rb'))
    clinic_value = clinic_data['Clinic Valuation Adjusted']
    clinic_ebit_multiple = clinic_data['EBIT Multiple']


    current_ebit = clinic_data['EBIT']
    current_ebit_ratio = clinic_data['EBIT Ratio']


    current_ebit_after_12_months = current_ebit * (1+clinic_data['Net Sales Growth'])


    potential_ebit_after_12_months = current_ebit_after_12_months

    st.markdown("#### Strategy to implement")

    col1, col2 = st.columns(2)

    with col1:
        corporate_wellness = st.checkbox("Corporate Wellness Program", value=False)
        
    with col2:
        with st.popover("details"):
            corporate_wellness_app.app()
            
    col1, col2 = st.columns(2)
    
    with col1:
        school_outreach = st.checkbox("School Outreach Program", value=False)
        
    with col2:
        with st.popover("details"):
            school_outreach_app.app()
        
    


        

        
    st.divider()

    if st.button("Calculate New Clinic Value"):
        
        if corporate_wellness == True:
            potential_ebit_after_12_months += 503000000 / 10000

        if school_outreach == True:
            potential_ebit_after_12_months += 684000000 / 10000

        
        
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
            
            model = ModelClinicValue(clinic_data)
            
            model.ebit = clinic_data['EBIT']
            model.ebit_ratio = clinic_data['EBIT Ratio']
            
            ebit_multiple = model.ebit_baseline_to_multiple(clinic_data['Net Sales Growth'])
            ebit_multiple = model.ebit_multiple_adjustment_due_dentist(ebit_multiple, clinic_data['Current Number of Dentist'], clinic_data['Projected Number of Dentist'], clinic_data['Possibility Existing Dentist Leaving'])
            ebit_multiple = model.ebit_multiple_adjustment_due_net_sales_variation(ebit_multiple, clinic_data['Relative Variation of Net Sales'])
            ebit_multiple = model.ebit_multiple_adjustment_due_number_patient_and_patient_spending_variability(ebit_multiple, clinic_data['Number of Active Patients'], clinic_data['Relative Variation of Patient Spending'])
            
            clinic_valuation = ebit_multiple * model.ebit
            
            
            
            st.markdown("### New Clinic Value")
            
            st.metric("New EBIT Multiple", f"{ebit_multiple:.2f}")
            st.metric("New Clinic Value", f"${clinic_valuation:,.0f}")
           
        st.write("") 
        st.write("The recalculation of Clinic Value only assumes the improvement in variables of EBIT and EBIT Ratio. <br>Other variables are assumed to be constant.", unsafe_allow_html=True)
             
            
            

    # col3, col4 = st.columns(2)

    # with col3:
    #     st.metric("Current Clinic Value", f"${clinic_value:,.0f}")
        
    # with col4:
    #     st.metric("Potential Clinic Value", f"${potential_clinic_value:,.0f}")
    #     st.caption("After Strategy Implementation")
        
    # st.divider()

    # st.markdown("#### Conclusion")