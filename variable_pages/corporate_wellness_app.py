import streamlit as st
import pandas as pd
from model import ModelCorporateWellness, GeneralModel  # Importing the Model class from model.py
import numpy as np
import plotly.graph_objs as go

def app():

    # Load treatment prices CSV to get the list of treatments

    model = ModelCorporateWellness()
    # Title and Description
    st.title("Corporate Wellness Program")
    st.write("""
    The Corporate Wellness Program aims to partner with local businesses to provide comprehensive dental wellness services to their employees. This initiative is designed to improve oral health, enhance employee benefits, and generate additional revenue for the dental clinic.
    """)

    # Divider
    st.divider()

    # Layout using columns for a more compact design
    # with st.form("wellness_form"):
        
    st.markdown("#### Employee Wellness Program Parameters")
    # First row: Total Potential Employee and Conversion Rate
    
    pricing_basis = st.radio("Pricing and Material Cost Basis", ["Dr.Riesqi", "GAIA Indonesia"], index=0)
    
    model.set_pricing_basis(pricing_basis)
    
    dentist_basis = st.selectbox("Dentist Fee Basis", ["GAIA Indonesia", "&Talent"], index=0)
    
    treatment_prices_df = model.prices_df
    treatment_list = treatment_prices_df['Treatment'].tolist()

    treatment_cost_df = model.costs_df

    # Load DSP CSV
    dsp_df = model.dsp_df
    
    
    col1, col2 = st.columns(2)
    with col1:
        total_potential_employee = st.number_input("Total Potential Employee", step=1, value=466)
    with col2:
        conversion_rate = st.number_input("Conversion Rate (%)", step=1, value=20)
        
    # Third row: Discount Package and Subscription Length
    col1, col2 = st.columns(2)
    with col1:
        discount_package = st.number_input("Discount Package (%)", step=1, value=20)
    with col2:
        subscription_length = st.number_input("Subscription Length (years)", step=1, value=1, max_value=5)
        
        
    # Second row: Treatment package checkboxes
    st.write("Treatment Package")
    selected_treatments = []
    cols = st.columns(2)  # Adjusted to fit two columns for compactness
    for i, treatment in enumerate(treatment_list):
        if cols[i % 2].checkbox(treatment, value=True):  # Create checkboxes for each treatment
            selected_treatments.append(treatment)
        

    with st.expander("Treatment Prices & Cost", expanded=False):
        st.markdown("#### Treatment Prices")
        st.session_state.treatment_prices_df =  st.data_editor(treatment_prices_df, hide_index=True)
        
        

        sum_treatment_prices = st.session_state.treatment_prices_df[st.session_state.treatment_prices_df['Treatment'].isin(selected_treatments)]['Price (Rp.)'].sum()
        # sum_treatment_costs = st.session_state.treatment_costs_df['Cost (Rp.)'].sum()

        st.write(f"Total Treatment Prices: Rp.{sum_treatment_prices:,.0f}")
        # st.write(f"Total Treatment Costs: {sum_treatment_costs}")

        
        
        st.markdown("#### Treatment Costs")
        st.session_state.treatment_costs_df =  st.data_editor(treatment_cost_df, hide_index=True)
        

        # sum_treatment_prices = st.session_state.treatment_prices_df['Price (Rp.)'].sum()
        sum_treatment_costs = st.session_state.treatment_costs_df[st.session_state.treatment_costs_df['Component'].isin(selected_treatments)]['Material Cost (Rp.)'].sum()
        
        st.session_state.treatment_costs_df['Dentist Fee Total (Rp.)'] = (st.session_state.treatment_costs_df['Dentist Fee per Hour (Rp.)'] * 
                                                    (st.session_state.treatment_costs_df['Duration (Min)'] / 60))
        
        sum_treatment_costs +=  st.session_state.treatment_costs_df[st.session_state.treatment_costs_df['Component'].isin(selected_treatments)]['Dentist Fee Total (Rp.)'].sum()

        # st.write(f"Total Treatment Prices: {sum_treatment_prices}")
        st.write(f"Total Treatment Costs: Rp.{sum_treatment_costs:,.0f}")

        
        
        


            
    # with st.expander("Scaling Scenario", expanded=True):
    #     scaling_scenario = st.checkbox("Once in 6 Months Scaling", value=False)
    #     st.write("Check this box if the subscribing employee will claim scaling treatment once in 6 months, which will affect in lower expense for us")        
            
            
    st.divider()

    st.markdown('#### Dental Saving Plan Parameters')

    # DSP Editor for checkboxes, discount rate adjustments, and conversion rate adjustments
    dsp_df['Selected'] = dsp_df['Treatment'].apply(lambda x: True)  # default all selected
    dsp_df['Conversion Rate (%)'] = dsp_df['Conversion Rate (%)'].apply(lambda x: float(x.replace('%', '')))
    dsp_df['Discount Price (%)'] = dsp_df['Discount Price (%)'].apply(lambda x: float(x.replace('%', '')))

    # Filtered DSP dataframe to show only Selected, Conversion Rate, and Discount Rate in the editor
    # dsp_editable_df = dsp_df[['Treatment', 'Selected', 'Conversion Rate', 'Discount Rate']]
    # dsp_editable_df.columns = ['Treatment', 'Selected', 'Conversion Rate (%)', 'Discount Rate (%)']


    dsp_editor = st.data_editor(dsp_df, use_container_width=True, num_rows="dynamic", column_order=
                                ['Treatment', 'Selected', 'Conversion Rate (%)', 'Original Price (Rp.)', 'Discount Price (%)', 'Cost Material (Rp.)', 'Dentist Fee Per Hour (Rp.)', 'Duration (Min)'], hide_index=True)

    # Submit button
    # submit_button = st.form_submit_button(label="Submit")
    
    basic_run, optimized_run = st.columns(2)

    basic_run_button = basic_run.button(label="Run Calculation", use_container_width=True)
    optimized_button = optimized_run.button(label="Run with Optimized Parameters", use_container_width=True)
    

    # If the form is submitted, create a Model instance and display the ARO calculation
    if basic_run_button:
        # Create an instance of the Model class from model.py
        
        
        model.set_parameters(total_potential_employee=total_potential_employee,
            conversion_rate=conversion_rate,
            treatments=selected_treatments,
            discount_package=discount_package,
            subscription_length=subscription_length,
        )
        
        
        
        # Calculate ARO and total cost for Employee Wellness Program
        aro = model.calculate_ARO(st.session_state.treatment_prices_df)
        total_cost = model.calculate_total_cost(st.session_state.treatment_costs_df)
        
        # Calculate Total Joining Employee
        total_joining_employee = np.ceil(total_potential_employee * (conversion_rate / 100))
        
        # Calculate total profit generated
        total_profit = aro - total_cost
        
        # Calculate ARO and cost for DSP
        dsp_aro, dsp_cost, dsp_df_output = model.calculate_DSP(dsp_editor, total_joining_employee)
        total_dsp_profit = dsp_aro - dsp_cost
        
        # Display results using st.metric
        
        st.markdown('#### Employee Wellness Program Results')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display the package price per employee
            st.metric(label="Monthly Package Price/Employee", value="Rp{:,.0f}".format((aro / (subscription_length*12) / total_joining_employee)))
            st.metric(label="Additional Revenue Opportunity (ARO)", value="Rp{:,.0f}".format(aro))
            st.metric(label="Total Profit Generated", value="Rp{:,.0f}".format(total_profit))
            
        with col2:
            st.metric(label="Total Joining Employee", value=int(total_joining_employee))
            st.metric(label="Total Subscribing Years", value=int(subscription_length))
            st.metric(label="Total Cost Generated", value="Rp{:,.0f}".format(total_cost))
        
        # Display DSP results below
        st.divider()
        st.markdown('#### Dental Saving Plan Results')
        
        # Display DSP DataFrame with joining customers, total revenue, and total cost
        st.dataframe(dsp_df_output, hide_index=True)
        
        # Grand totals for DSP
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Total DSP Revenue", value="Rp{:,.0f}".format(dsp_aro))
            st.metric(label="Total DSP Cost", value="Rp{:,.0f}".format(dsp_cost))
        with col2:
            st.metric(label="Total DSP Profit", value="Rp{:,.0f}".format(total_dsp_profit))

            


        st.divider()

        st.markdown('## Overall Results')
        
        col1, col2 = st.columns(2)
        
        total_revenue_overall = aro + dsp_aro
        total_cost_overall = total_cost + dsp_cost
        total_profit_overall = total_profit + total_dsp_profit
        
        with col1:
            st.metric(label="Total Revenue", value="Rp{:,.0f}".format(total_revenue_overall))
            st.metric(label="Total Cost", value="Rp{:,.0f}".format(total_cost_overall))
            
        with col2:
            st.metric(label="Total Profit", value="Rp{:,.0f}".format(total_profit_overall))
            
        cashflow_df = GeneralModel().create_cashflow_df(total_revenue_overall, total_cost_overall, subscription_length, 1, period_type='yearly')
        
        # st.dataframe(cashflow_df, hide_index=True)
        # st.write(f'Average Revenue: Rp.{cashflow_df["Revenue"].mean():,.0f}')
        # st.write(f'Total Revenue: Rp.{cashflow_df["Revenue"].sum():,.0f}')
        
        cashflow_df.to_csv('corporate_cashflow.csv', index=False)
        
        
        
        
    if optimized_button:
        # Create an instance of the Model class from model.py
        
        
        model.set_parameters(total_potential_employee=total_potential_employee,
            conversion_rate=conversion_rate,
            treatments=selected_treatments,
            discount_package=discount_package,
            subscription_length=subscription_length,
        )
        
        
        
        # Calculate ARO and total cost for Employee Wellness Program
        aro = model.calculate_ARO(st.session_state.treatment_prices_df)
        total_cost = model.calculate_total_cost(st.session_state.treatment_costs_df)
        
        # Calculate Total Joining Employee
        total_joining_employee = np.ceil(total_potential_employee * (conversion_rate / 100))
        
        # Calculate total profit generated
        total_profit = aro - total_cost
        
        dsp_editor['Dentist Fee Per Hour (Rp.)'] = 140000
        dsp_editor['Original Price (Rp.)'] = dsp_editor['Original Price (Rp.)'] * 1.2
        
        # Calculate ARO and cost for DSP
        dsp_aro, dsp_cost, dsp_df_output = model.calculate_DSP(dsp_editor, total_joining_employee)
        total_dsp_profit = dsp_aro - dsp_cost
        
        
        # Display results using st.metric
        
        st.markdown('#### Employee Wellness Program Results')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display the package price per employee
            st.metric(label="Monthly Package Price/Employee", value="Rp{:,.0f}".format((aro / (subscription_length*12) / total_joining_employee)))
            st.metric(label="Additional Revenue Opportunity (ARO)", value="Rp{:,.0f}".format(aro))
            st.metric(label="Total Profit Generated", value="Rp{:,.0f}".format(total_profit))
            
        with col2:
            st.metric(label="Total Joining Employee", value=int(total_joining_employee))
            st.metric(label="Total Subscribing Years", value=int(subscription_length))
            st.metric(label="Total Cost Generated", value="Rp{:,.0f}".format(total_cost))
        
        # Display DSP results below
        st.divider()
        st.markdown('#### Dental Saving Plan Results')
        
        # Display DSP DataFrame with joining customers, total revenue, and total cost
        st.dataframe(dsp_df_output, hide_index=True)
        
        # Grand totals for DSP
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Total DSP Revenue", value="Rp{:,.0f}".format(dsp_aro))
            st.metric(label="Total DSP Cost", value="Rp{:,.0f}".format(dsp_cost))
        with col2:
            st.metric(label="Total DSP Profit", value="Rp{:,.0f}".format(total_dsp_profit))

            


        st.divider()

        st.markdown('## Overall Results')
        
        col1, col2 = st.columns(2)
        
        total_revenue_overall = aro + dsp_aro
        total_cost_overall = total_cost + dsp_cost
        total_profit_overall = total_profit + total_dsp_profit
        
        with col1:
            st.metric(label="Total Revenue", value="Rp{:,.0f}".format(total_revenue_overall))
            st.metric(label="Total Cost", value="Rp{:,.0f}".format(total_cost_overall))
            
        with col2:
            st.metric(label="Total Profit", value="Rp{:,.0f}".format(total_profit_overall))
            
        cashflow_df = GeneralModel().create_cashflow_df(total_revenue_overall, total_cost_overall, subscription_length, 1, period_type='yearly')
        
        # st.dataframe(cashflow_df, hide_index=True)
        # st.write(f'Average Revenue: Rp.{cashflow_df["Revenue"].mean():,.0f}')
        # st.write(f'Total Revenue: Rp.{cashflow_df["Revenue"].sum():,.0f}')
        
        cashflow_df.to_csv('corporate_cashflow.csv', index=False)
        
    st.divider()
    
    st.markdown('#### Sensitivity Analysis')
    
    col1, col2 = st.columns(2)
    
    with col1:
    
        var_one = st.selectbox("Select Variable 1", ["Conversion Rate (%)", "Discount Package (%)", "Total Potential Employee"], index=0)
        
        var_one_increment = st.number_input("Set Increment for Var 1", value=10, step=1)
        
    with col2:
        var_two = st.selectbox("Select Variable 2", ["Conversion Rate (%)", "Discount Package (%)", "Total Potential Employee"], index=1)
        
        var_two_increment = st.number_input("Set Increment for Var 2", value=10, step=1)
        
    st.caption('Please Re-Run the model if you change the sensitivity analysis parameters')
        
    # create a variable called var_list that will list var_one and var_two in a list
    var_list = [var_one, var_two]
    increment_list = [var_one_increment, var_two_increment]
    
    fig_sensitivity_analysis, sensitivity_analysis_df = model.run_sensitivity_analysis(var_list, increment_list)
    
    st.dataframe(sensitivity_analysis_df, hide_index=True, use_container_width=True)
    
    st.plotly_chart(fig_sensitivity_analysis)
            
        
        
            

        