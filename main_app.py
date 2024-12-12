import streamlit as st
import pandas as pd
import numpy as np
from model import ModelClinicValue
from model import ModelTransformTransactionData
import pickle
import os
import variable_pages.EBIT
import variable_pages.EBIT_ratio
import variable_pages.YoY_growth
import variable_pages.equipment_usage_ratio
import variable_pages.equipment_value
import variable_pages.relative_variability_of_net_sales
import variable_pages.number_of_active_patients
import variable_pages.relative_variability_of_patient_spending
import variable_pages.risk_of_leaving_dentist
import variable_pages.fitout_value
import variable_pages.last_fitout_years
import matplotlib.pyplot as plt
import plotly.express as px

def app():

        # Set the page title
    st.title("Clinic for Investment")

    with st.container(border=True):
        
        st.markdown("## Baseline Clinic Value Model")
        st.write("Based on average performance of available data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            base_ebit = 250000
            base_equipment_value = 80826.5
            base_net_sales_growth = 0.1
            base_number_of_active_patients = 1000
            base_number_of_dentist = 2
            base_relative_variability_patient_spending = 0.25
            base_risk_of_leaving_dentist = 0
            base_fitout_cost = 1728000000 / 10000 # 5 times of Indonesia cost converted into AUD
            
            
            st.metric("EBIT", f"$ {base_ebit:,.0f}", help="Net Sales - COGS - Operating Expenses")
            with st.popover("details"):
                variable_pages.EBIT.app()
            st.metric("Year-on-Year Growth Rate", f"{base_net_sales_growth * 100:.1f}%", help="Yearly")
            with st.popover("details"):
                variable_pages.YoY_growth.app()
            st.metric("Equipments Value", f"$ {base_equipment_value:,.0f}")
            with st.popover("details"):
                variable_pages.equipment_value.app()
                
            st.metric("Fit Out Value", f"$ {base_fitout_cost:,.0f}")
            with st.popover("details"):
                variable_pages.fitout_value.app()
            
            
            
            st.metric("Number of Active Patients", base_number_of_active_patients, help="Number of active unique patients in the clinic for the last one year")
            with st.popover("details"):
                variable_pages.number_of_active_patients.app()
            # st.metric("Number of Dentist", base_number_of_dentist, help="Number of dentist in the clinic")

            st.metric("Risk of Leaving Dentist", f"{base_risk_of_leaving_dentist:,.0f}%", help="The risk of leaving dentist measures the impact of any possibility of existing dentist leaving the clinic in coming, near-future periods. The impact itself is measured by the percentage of revenue that the dentist contributes to the clinic's total revenue.")
            with st.popover("details"):
                variable_pages.risk_of_leaving_dentist.app()
        with col2:
            base_ebit_ratio = 0.22
            st.metric("EBIT Ratio", f"{base_ebit_ratio * 100:.2f}%", help="EBIT / Net Sales")
            with st.popover("details"):
                variable_pages.EBIT_ratio.app()
            base_relative_variability_net_sales = 0.15
            base_equipment_usage_ratio = 0.5
            base_projected_number_of_dentist = 2
            base_last_fitout = 5
            
            
            st.metric("Relative Variability of Net Sales", f"{base_relative_variability_net_sales * 100:.0f}%", help="Standard Deviation of Net Sales / Mean Net Sales. Net Sales here is in monthly terms")
            with st.popover("details"):
                variable_pages.relative_variability_of_net_sales.app()            
            st.metric("Equipment Usage Ratio", f"{base_equipment_usage_ratio * 100:.2f}%", help="Percentage of equipment usage from its expected lifetime")
            with st.popover("details"):
                variable_pages.equipment_usage_ratio.app()
                
            st.metric("Last Fit Out", f"{base_last_fitout} years ago", help="Explain when did last fit out occured")
            with st.popover("details"):
                variable_pages.last_fitout_years.app()
            
            st.metric("Relative Variability of Patient Spending", f"{base_relative_variability_patient_spending * 100:.0f}%", help="Standard Deviation of Patient Spending / Mean Patient Spending. Patient Spending here is the yearly spending of each active unique patients")
            with st.popover("details"):
                variable_pages.relative_variability_of_patient_spending.app()
            
            # st.metric("Projected Number of Dentist", base_number_of_dentist, help="Projected number of dentist in the clinic")

            
            
            base_ebit_multiple = 2.5
            st.metric("EBIT Multiple", base_ebit_multiple)
    # File upload widget
    # uploaded_file = st.file_uploader("Upload your company document", type=["csv", "xlsx", "docx", "pdf"])
    
    # uploaded_file = st.selectbox("Choose Clinic", ["Acre Clarity Dental"], index=0)
    
    # Dummy Data
    
    # case example is dataset 1
    
    # read pkl file
    with open(r'dummy_clinic_model/pkl_files/dataset_1/pool_clinic_df.pkl', 'rb') as f:
        pool_clinic_df = pickle.load(f)
    
    uploaded_file = st.selectbox("Choose Clinic", list(pool_clinic_df.keys()), index=0)
    


    # Initialize the ModelClinicValue class
    model = ModelClinicValue({})
    
    transaction_df = pool_clinic_df[uploaded_file]['Gross Profit']
    indirect_cost_df = pool_clinic_df[uploaded_file]['Indirect Cost']
    model_transform = ModelTransformTransactionData(transaction_df, indirect_cost_df)
    transformed_uploaded_file = model_transform.prepare_all_data()
    
    
    # If a file is uploaded, update the model variables with data from the file
    if uploaded_file:
        model.update_variable_from_uploaded_dictionary(transformed_uploaded_file)
        # Previously were
        # model.update_variable_from_uploaded_file(transformed_uploaded_file)
    
    
    # if uploaded_file == "Acre Clarity Dental":
    #     uploaded_file = r'company_data/Acre Data.xlsx'
    #     model.update_variable_from_uploaded_file(uploaded_file)      



    # Section for manual input of company variables
    st.header("Set Company Variable")
    st.write("Please input manually the following variables to evaluate the value of your company in case no file to upload.")

    # Dictionary to store the variables
    company_variables = {}

    # Category: Profit & Loss
    with st.expander("Profit & Loss", expanded=True):
        st.write("The values of the following variables are yearly amounts.")

        # Sub Category: Gross Profit
        st.markdown("### Gross Profit")
        col1, col2, col3 = st.columns(3)
        with col1:
            net_sales = st.number_input("Net Sales", value=float(model.net_sales) if model.net_sales is not None else 0.0)
            company_variables["Net Sales"] = net_sales
        with col2:
            cogs = st.number_input("COGS", value=float(model.cogs) if model.cogs is not None else 0.0)
            company_variables["COGS"] = cogs
        with col3:
            trading_income = st.number_input("Trading Income", value=float(model.trading_income) if model.trading_income is not None else 0.0)
            company_variables["Trading Income"] = trading_income
            
        col1, col2 = st.columns(2)
        
        with col1:
            net_sales_growth = st.number_input("Year-on-Year Growth Rate", value=float(model.net_sales_growth) if model.net_sales_growth is not None else 0.0)

        with col2:
            relative_variability_net_sales = st.number_input("Relative Variability of Net Sales", value=float(model.relative_variability_net_sales) if model.relative_variability_net_sales is not None else 0.0)

        # Sub Category: Other Income
        st.markdown("### Other Income")
        col4, col5 = st.columns(2)
        with col4:
            other_income = st.number_input("Other Income", value=float(model.other_income) if model.other_income is not None else 0.0)
            company_variables["Other Income"] = other_income
        with col5:
            interest_revenue = st.number_input("Interest Revenue of Bank", value=float(model.interest_revenue) if model.interest_revenue is not None else 0.0)
            company_variables["Interest Revenue of Bank"] = interest_revenue

        # Sub Category: Operating Expense
        st.markdown("### Operating Expense")
        col7, col8 = st.columns(2)
        with col7:
            operational_expense = st.number_input("Operating Expense", value=float(model.operational_expense) if model.operational_expense is not None else 0.0)
            company_variables["Operational Expense"] = operational_expense
        with col8:
            other_expense = st.number_input("Other Expense", value=float(model.other_expense) if model.other_expense is not None else 0.0)
            company_variables["Other Expense"] = other_expense

        # Sub Category: Depreciation
        st.markdown("### Depreciation")
        col9, col10 = st.columns(2)
        with col9:
            clinic_depreciation = st.number_input("Depreciation of Equipment Clinic Expense", value=float(model.clinic_depreciation) if model.clinic_depreciation is not None else 0.0)
            company_variables["Depreciation of Equipment Clinic Expense"] = clinic_depreciation
        with col10:
            non_clinic_depreciation = st.number_input("Depreciation of Equipment Non Clinic Expense", value=float(model.non_clinic_depreciation) if model.non_clinic_depreciation is not None else 0.0)
            company_variables["Depreciation of Equipment Non Clinic Expense"] = non_clinic_depreciation

        # Sub Category: Tax
        st.markdown("### Tax")
        col11, col12 = st.columns(2)
        with col11:
            bank_tax_expense = st.number_input("Bank Tax Expense", value=float(model.bank_tax_expense) if model.bank_tax_expense is not None else 0.0)
            company_variables["Bank Tax Expense"] = bank_tax_expense
        with col12:
            other_tax = st.number_input("Other Tax", value=float(model.other_tax) if model.other_tax is not None else 0.0)
            company_variables["Other Tax"] = other_tax

    # Balance Sheet Section
    # with st.expander("Balance Sheet", expanded=True):
    #     st.markdown("### Fixed Assets")
    #     tangible_assets = st.number_input("Tangible Assets (PP&E)", value=float(model.tangible_assets) if model.tangible_assets is not None else 0.0)
    #     company_variables["Tangible Assets (PP&E)"] = tangible_assets


        
        
    # with st.expander("Cash Flow", expanded=True):
    #     st.markdown("### Net Cash Flow")
    #     st.write("Please fill in the monthly net cash flow, leave blank if not applicable.")

    #     # Use the model's net_cash_flow if available, otherwise create a default DataFrame
    #     if model.net_cash_flow is not None:
    #         net_cash_flow = model.net_cash_flow
    #     else:
    #         net_cash_flow = pd.DataFrame({
    #             'Jan': [None],
    #             'Feb': [None],
    #             'Mar': [None],
    #             'Apr': [None],
    #             'May': [None],
    #             'Jun': [None],
    #             'Jul': [None],
    #             'Aug': [None],
    #             'Sep': [None],
    #             'Oct': [None],
    #             'Nov': [None],
    #             'Dec': [None]
    #         }, index=['2019', '2020', '2021', '2022', '2023'])

    #     # Display the DataFrame using st.data_editor
    #     net_cash_flow_editor = st.data_editor(net_cash_flow, use_container_width=True, num_rows='dynamic')
    #     company_variables['Net Cash Flow'] = net_cash_flow_editor
        
    #     net_cash_flow.to_csv('net_cash_flow.csv')
        
    with st.expander("Other Variables", expanded=True):
        
        st.markdown("### Equipment Life")
        


        # Create a DataFrame with the new columns
        data_df = pd.read_csv('equipment_data.csv')

        # Display the DataFrame using st.data_editor with modified column configurations
        company_variables['Equipment_Life'] = st.data_editor(
            data_df,
            column_config={
                "Equipment": st.column_config.SelectboxColumn(
                    "Equipment",
                    help="Select the equipment or write your own value",
                    options=[
                "Intra Oral Camera",
                "Bleaching Unit",
                "Ultrasonic Scaler",
                "Light Cure",
                "Dental Unit",
                "Portable Xrays",
                "Endomotor",
                "Autoclaves",
                "Ultrasonic Cleaner",
                "Water Tank",
                "Prophylaxis Hand Piece",
                "Handpiece Set",
                "Compressor",
                "Apex Locator",
                "Dental Loupe",
                "Portable Light",
                "Camera DSLR",
                "Water Tank Hose",
                "Sealing Machine",
                "Xray Sensor"
            ],
                    required=True,
                ),
                "Own?": st.column_config.CheckboxColumn(
                    "Own?",
                    help="Does your clinic own the specific equipment?",
                ),
                "Expected Lifetime": st.column_config.NumberColumn(
                    "Expected Lifetime",
                    help="Expected operational lifetime of the equipment in years", disabled=True
                ),
                "Current Lifetime Usage": st.column_config.NumberColumn(
                    "Current Lifetime Usage",
                    help="The number of years the equipment has been in use (on average if multiple units)",
                ),
            },
            hide_index=True,
            use_container_width=True,
            num_rows='dynamic',
            column_order=["Equipment", "Own?", "Expected Lifetime", "Current Lifetime Usage"]
        )
        
        selected_equipment = company_variables['Equipment_Life'][company_variables['Equipment_Life']['Own?'] == True]
        company_variables["Equipments Value"] = (selected_equipment['Quantity'] * selected_equipment['Price']).sum()
        
        
        
        
        st.markdown("### Fit Out Activity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            
            fitout_value = st.number_input("Fit Out Value ($)", value=base_fitout_cost if model.fitout_value == 0 else model.fitout_value, help="The value of the last fit out activity in the clinic")
            company_variables["Fitout Value"] = fitout_value
        with col2:
            last_fitout = st.number_input("Last Fit Out (Years Ago)", value=int(base_last_fitout if model.last_fitout_year == 0 else model.last_fitout_year), help="The number of years since the last fit out activity",)
            company_variables['Last Fitout Year'] = last_fitout
    
    

        

        st.markdown("### Dentist Availability")
        
        if model.dentist_contribution is None:
            dentist_availability = pd.read_csv('dentist_contribution_tofill.csv')
        else:
            dentist_availability = model.dentist_contribution
        
        st.write('Table of Dentist Revenue Contribution')

        
        
        
        dentist_availability = st.data_editor(dentist_availability, num_rows='dynamic')
        st.caption("Please change the Dentist Provider in accordance to the clinic's dentist ID if necessary")
        
        # with col1:
            # number_of_dentist = st.number_input("Number of Dentist", value=model.number_of_dentist if model.number_of_dentist is not None else 2, step=1)
            
        # with col2:
            # projected_number_of_dentist = st.number_input("Projected Number of Dentist", value=model.projected_number_of_dentist if model.projected_number_of_dentist is not None else 2, step=1)
            
        
        # possibility_existing_dentist_leaving = st.checkbox("Possibility of Existing Dentist Leaving", value=model.potential_existing_dentist_leaving if model.potential_existing_dentist_leaving is not None else False, help="Check if there is a possibility of existing dentist leaving the clinic, despite the projected number of dentist will still be equal to current number of dentist.")


        st.markdown("### Customer Based Variables")

        col1, col2 = st.columns(2)
        
        with col1:
            number_of_active_patients = st.number_input("Number of Active Patients", value=int(model.number_of_patients) if model.number_of_patients is not None else 0, step=1, help='Number of Active Unique Patient on yearly average')
            
        with col2:
            relative_variability_patient_spending = st.number_input("Relative Variability of Patient Spending", value=float(model.relative_variability_patient_spending) if model.relative_variability_patient_spending is not None else 0.0, help='Coefficient of Variation of Patient Spending')
        
        # company_variables['Equipment_Life'].to_csv('equipment_life.csv')
    st.session_state['Evaluated'] = False
    # Button to trigger the evaluation
    if st.button("Evaluate"):
        
        selected_equipment.to_csv("actual_equipment_life.csv", index=False)
        
        st.markdown("## Variable Summary")
        
        with st.container(border=True):
        
            st.markdown("### Profit & Loss")
            st.write("Yearly based")
            
            with st.expander("General Variables", expanded=False):
        
                col1, col2, col3 = st.columns(3)
                
                
                
                # model = ModelClinicValue(company_variables)
                
                with col1:
                    st.metric("Net Sales", f"$ {company_variables.get('Net Sales', 0):,.0f}")
                    st.metric("COGS", f"$ {company_variables.get('COGS', 0):,.0f}")
                    st.metric("Trading Income", f"$ {company_variables.get('Trading Income', 0):,.0f}")
                    st.metric("Other Income", f"$ {company_variables.get('Other Income', 0):,.0f}")
                
                with col2:
                    st.metric("Interest Revenue of Bank", f"$ {company_variables.get('Interest Revenue of Bank', 0):,.0f}")
                    st.metric("Advertising & Promotion Expense", f"$ {company_variables.get('Advertising & Promotion Expense', 0):,.0f}")
                    st.metric("Operational Expense", f"$ {company_variables.get('Operational Expense', 0):,.0f}")
                    st.metric("Other Expense", f"$ {company_variables.get('Other Expense', 0):,.0f}")
                
                with col3:
                    st.metric("Depreciation of Equipment Clinic Expense", f"$ {company_variables.get('Depreciation of Equipment Clinic Expense', 0):,.0f}")
                    st.metric("Depreciation of Equipment Non Clinic Expense", f"$ {company_variables.get('Depreciation of Equipment Non Clinic Expense', 0):,.0f}")
                    st.metric("Bank Tax Expense", f"$ {company_variables.get('Bank Tax Expense', 0):,.0f}")
                    st.metric("Other Tax", f"$ {company_variables.get('Other Tax', 0):,.0f}")
                
            col1, col2 = st.columns(2)
            
            # with col1:
            #     st.metric("EBITDA", f"$ {model.ebitda:,.0f}")
            
            with col1:
                st.metric("EBIT", f"$ {model.ebit:,.0f}", delta=f"{model.ebit - base_ebit:,.0f}")
                
            with col2:
                st.metric("EBIT Ratio", f"{model.ebit_ratio * 100:.2f}%", delta=f"{(model.ebit_ratio - base_ebit_ratio)*100:.2f}%")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Year-on-Year Growth", f"{net_sales_growth * 100:.2f}%", delta=f"{(net_sales_growth - base_net_sales_growth)*100:.2f}%")
                
                with st.popover("details"):
                    
                    fig = model.plot_net_sales_yearly()
                    st.plotly_chart(fig)
                    st.write("Past years net sales growth rate resulting in average of year-on-year value as specified")
                
            with col2:
                st.metric("Relative Variability of Net Sales", f"{relative_variability_net_sales * 100:.0f}%", delta=f"{(relative_variability_net_sales - base_relative_variability_net_sales)*100:.0f}%", delta_color="inverse")
            
                with st.popover("details"):
                    
                    fig = model.plot_net_sales_monthly()
                    st.plotly_chart(fig)
                    st.write("Past months net sales variation resulting in relative variability of net sales as specified")
            
        # with st.container(border=True):
        #     st.markdown("### Balance Sheet")
            
        #     equipment_value_delta = company_variables.get('Equipments Value', 0) - base_equipment_value
        #     st.metric("Equipments Value", f"$ {company_variables.get('Equipments Value', 0):,.0f}", delta=equipment_value_delta)
            
        # with st.container(border=True):
        
        #     st.markdown("### Cash Flow")
        #     st.write("Monthly Based")
        
        #     average_cashflow, std_deviation, trend_coefficient = model.analyze_cash_flow()
        
        #     col1, col2, col3 = st.columns(3)
        

        #     with col1:
        #         st.metric("Net Cash Flow Average", f"$ {average_cashflow:,.0f}")
        
        #     with col2:
        #         st.metric("Net Cash Flow Standard Deviation", f"$ {std_deviation:,.0f}")
            
        #     with col3:
        #         st.metric("Net Cash Flow Trend Coefficient", f"{trend_coefficient:,.2f}")
            
        with st.container(border=True):
            st.markdown("### Other Variables")
            
            st.markdown('#### Equipments')
            model.equipment_life = selected_equipment
            company_variables['Equipment Life'] = model.equipment_life
            
            equipment_usage_ratio, total_equipments, total_remaining_value = model.calculate_equipment_usage_ratio()
            
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Equipment Usage Ratio", f"{equipment_usage_ratio * 100:.2f}%", delta=f"{(equipment_usage_ratio - base_equipment_usage_ratio)*100:.2f}%")
                
            with col2:
                st.metric("Total Equipments", f"{total_equipments}")
                
            with st.popover("details", use_container_width=True):
                st.dataframe(model.equipment_life.drop(columns=['Own?', 'Quantity']), hide_index=True)
                st.write("Above is the list of equipments in the clinic, with the expected lifetime and current lifetime usage.")
                
                
            st.markdown('### Fit Out')
        
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Fit Out Value", f"$ {fitout_value:,.0f}", delta=f"{fitout_value - base_fitout_cost:,.0f}")
                
            with col2:
                st.metric("Last Fit Out", f"{last_fitout} years ago", delta=f"{last_fitout - base_last_fitout} years", delta_color='inverse')
                
            with st.popover("details", use_container_width=True):
                st.write("**Fit out** value is the value of the last fit out activity in the clinic, while **Last Fit Out** is the number of years since the last fit out activity.")
                st.write("Both of these variables are used to calculate the clinic valuation. The higher the value of the fit out, the higher the clinic valuation. And the lower the number of years since the last fit out, the higher the clinic valuation.")
                
            st.markdown('#### Dentist Availability')
            
            col1, col2 = st.columns(2)    
            
            with col1:
                
                risk = model.calculate_risk_of_dentist_leaving(dentist_availability)
                
                st.metric("Risk of Leaving Dentist", f"{risk *100:.2f}%", delta=f"{(risk - base_risk_of_leaving_dentist)*100:.2f}%", delta_color='inverse')

                with st.popover("details"):
                    df_risk = dentist_availability[dentist_availability['Possibly Leaving?'] == True][['Dentist Provider', 'Sales Contribution ($)']]
                
                    
                    st.dataframe(df_risk, hide_index=True)
                    st.write("Above is the list of dentist that possibly leaving the clinic, with their sales contribution to the clinic. This sales contribution is equivalent to the percentage of expected revenue reduction due their leaving.")
            
            # with col1:
                # st.metric("Number of Dentist", f"{number_of_dentist}", delta=f"{number_of_dentist - base_number_of_dentist}")
                # st.metric("Possibility Existing Dentist Leaving", f"{possibility_existing_dentist_leaving}", delta= f"Baseline = {base_possibility_existing_dentist_leaving}", delta_color="off")
            
            # with col2:
                # st.metric("Projected Number of Dentist", f"{projected_number_of_dentist}", delta=f"{projected_number_of_dentist - base_projected_number_of_dentist}")
                
            st.markdown('#### Customer Based')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Number of Active Patients", f"{number_of_active_patients}", delta=f"{number_of_active_patients - base_number_of_active_patients}")
                
                with st.popover('details'):
                    
                    tab1, tab2 = st.tabs(["Summarize by Patient", "Summarize by Transaction"])
                    
                    with tab1:
                        patient_transaction_df = pd.DataFrame(model.patient_transaction.groupby('Patient ID')['Revenue'].sum().sort_values(ascending=False).reset_index())
                        st.dataframe(patient_transaction_df, hide_index=True)
                        
                        st.write("Above data is sample data")
                    
                    with tab2:
                        patient_transaction_df_2 = pd.DataFrame(model.patient_transaction.groupby('Code')['Revenue'].sum().sort_values(ascending=False).reset_index())
                        st.dataframe(patient_transaction_df_2, hide_index=True)
                        
                        st.write("Above data is sample data")
                    
                
                
            with col2:
                st.metric("Relative Variation of Patient Spending", f"{relative_variability_patient_spending * 100:.0f}%", delta=f"{(relative_variability_patient_spending - base_relative_variability_patient_spending)*100:.0f}%", delta_color="inverse")
                
                with st.popover("details"):
                        patient_transaction_df = pd.DataFrame(model.patient_transaction.groupby('Patient ID')['Revenue'].sum().sort_values(ascending=False).reset_index())

                        # Create a swarm plot using Plotly
                        fig_swarm_patient_spending = px.strip(patient_transaction_df, y='Revenue', 
                        title='Swarm Plot of Patient Total Spending', 
                        template='plotly_white',)
                        
                        st.plotly_chart(fig_swarm_patient_spending)
                        
                        st.write("Above is the swarm plot or dispersion of patient total spending, the higher the value, the higher the spending of the patient")
                    
                    
                    
                
        # Create the output_variables dictionary
        output_variables = {
            'Net Sales': company_variables.get('Net Sales', 0),
            'COGS': company_variables.get('COGS', 0),
            'Trading Income': company_variables.get('Trading Income', 0),
            'Other Income': company_variables.get('Other Income', 0),
            'Interest Revenue of Bank': company_variables.get('Interest Revenue of Bank', 0),
            'Advertising & Promotion Expense': company_variables.get('Advertising & Promotion Expense', 0),
            'Operational Expense': company_variables.get('Operational Expense', 0),
            'Other Expense': company_variables.get('Other Expense', 0),
            'Depreciation of Equipment Clinic Expense': company_variables.get('Depreciation of Equipment Clinic Expense', 0),
            'Depreciation of Equipment Non Clinic Expense': company_variables.get('Depreciation of Equipment Non Clinic Expense', 0),
            'Bank Tax Expense': company_variables.get('Bank Tax Expense', 0),
            'Other Tax': company_variables.get('Other Tax', 0),
            'Equipments Value': company_variables.get('Equipments Value', 0),
            # 'Net Cash Flow Average': average_cashflow,
            # 'Net Cash Flow Standard Deviation': std_deviation,
            # 'Net Cash Flow Trend Coefficient': trend_coefficient,
            'Equipment Usage Ratio': equipment_usage_ratio,
            'Total Equipments': total_equipments,
            'Net Sales Growth': net_sales_growth,
            'Total Remaining Value': total_remaining_value,
            # 'Current Number of Dentist': number_of_dentist,
            # 'Projected Number of Dentist': projected_number_of_dentist,
            # 'Possibility Existing Dentist Leaving': possibility_existing_dentist_leaving,
            'Relative Variation of Net Sales': relative_variability_net_sales,
            'Number of Active Patients': number_of_active_patients,
            'Relative Variation of Patient Spending': relative_variability_patient_spending,
            'Risk of Leaving Dentist': risk,
            'Fitout Value': company_variables.get("Fitout Value", 0),
            'Last Fitout Year': company_variables.get("Last Fitout Year", 0),
            'Equipment Life': company_variables.get('Equipment Life', None)
            
        }

        output_variables['EBIT'] = model.ebit
        output_variables['EBIT Ratio'] = model.ebit_ratio
        output_variables['General Expense'] = output_variables['Operational Expense'] + output_variables['Other Expense'] + output_variables['Advertising & Promotion Expense']
        output_variables['Patient Pool'] = transaction_df['Patient ID'].unique().tolist()
        
        # output_variables['Operating Income'] = output_variables['Net Sales'] + output_variables['Trading Income'] + output_variables['Other Income'] + output_variables['Interest Revenue of Bank'] + output_variables['COGS'] + output_variables['Advertising & Promotion Expense'] + output_variables['Operational Expense'] + output_variables['Other Expense']

        # Convert the output_variables dictionary to a DataFrame and save as CSV
        output_df = pd.DataFrame([output_variables])
        output_df.to_csv('output_variables.csv', index=False)
        

        
        st.divider()
        
        st.markdown("## Clinic Value")
        
        st.markdown("#### Calculating Current Clinic Value")
        st.write("To understand the calculation logic and approach used for this model of clinic evaluation, please refer to side navigation bar and click on the **'Current Value Calculation Step'** tab")

        ebit_multiple = model.ebit_baseline_to_multiple(output_variables['Net Sales Growth'])
        st.write(ebit_multiple)
        equipment_adjusting_value = model.equipment_adjusting_value(output_variables['Total Remaining Value'], base_equipment_value, base_equipment_usage_ratio)
        fitout_adjusting_value = model.fitout_adjusting_value(output_variables['Fitout Value'], output_variables['Last Fitout Year'], base_fitout_cost, base_last_fitout)

        ebit_multiple = model.ebit_multiple_adjustment_due_dentist(ebit_multiple, output_variables['Risk of Leaving Dentist'])
        ebit_multiple = model.ebit_multiple_adjustment_due_net_sales_variation(ebit_multiple, output_variables['Relative Variation of Net Sales'])
        st.write(ebit_multiple)
        ebit_multiple = model.ebit_multiple_adjustment_due_number_patient_and_patient_spending_variability(ebit_multiple, output_variables['Number of Active Patients'], output_variables['Relative Variation of Patient Spending'])
        st.write(ebit_multiple)
        clinic_valuation = ebit_multiple * model.ebit
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("EBIT Multiple", f"{ebit_multiple:.2f}", delta=f"{ebit_multiple - base_ebit_multiple:.2f}")
            
            
        with col2:
            st.metric("Clinic Valuation", f"$ {clinic_valuation:,.0f}")
            st.caption("Clinic Valuation = EBIT Multiple * EBIT")
            
            
            
            
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Adjustment due Equipments", f"$ {equipment_adjusting_value:,.0f}", help="Adjustment due to equipment usage ratio and specific equipment availability within the clinic")
            
        with col2:
            st.metric("Adjustment due Fit Out", f"$ {fitout_adjusting_value:,.0f}", help="Adjustment due to last fit out activity in the clinic")

        
        clinic_valuation_adjusted = clinic_valuation + equipment_adjusting_value + fitout_adjusting_value
        st.metric("Clinic Valuation Adjusted", f"$ {clinic_valuation_adjusted:,.0f}")
        st.caption("Clinic Valuation Adjusted = Clinic Valuation + Adjustment due Equipments")
        
        
        output_variables['EBIT Multiple'] = ebit_multiple
        output_variables['Clinic Valuation Adjusted'] = clinic_valuation_adjusted
        output_variables['Clinic Valuation'] = clinic_valuation
        
        


        with open('clinic_value.pkl', 'wb') as f:
            pickle.dump(output_variables, f)
            
        file_name = 'clinic_value_set.pkl'
        ID = uploaded_file

        # Initialize or load existing dictionary
        if os.path.exists(file_name):
            # Load existing data
            with open(file_name, 'rb') as f:
                output_variables_set = pickle.load(f)
        else:
            # Start with an empty dictionary
            output_variables_set = {}
            
            



        # Update or add the new data
        output_variables_set[ID] = output_variables

        # Save the updated dictionary back to the file
        with open(file_name, 'wb') as f:
            pickle.dump(output_variables_set, f)
            
            
    st.divider()
            # Define the file name
    file_name = 'clinic_value_set.pkl'
    ID = uploaded_file

    # Initialize or load existing dictionary
    if os.path.exists(file_name):
        # Load existing data
        with open(file_name, 'rb') as f:
            output_variables_set = pickle.load(f)
    else:
        # Start with an empty dictionary
        output_variables_set = {}
        
        



    # # Update or add the new data
    # output_variables_set[ID] = output_variables

    # # Save the updated dictionary back to the file
    # with open(file_name, 'wb') as f:
    #     pickle.dump(output_variables_set, f)
        
        
    # st.write the keys of the output_variables_set dictionary
    st.write("Saved Clinic Evaluation")
    st.write("The following clinic evaluations have been saved:")
    st.write(list(output_variables_set.keys()))
                    
            


   
            
                    


        # # Open the CVXPY coefficients from the pickle file
        # with open('cvxpy_coefficients_non_negative.pkl', 'rb') as f:
        #     coefficients_df = pickle.load(f)

        # # Ensure the coefficients are in a NumPy array format
        # coefficients = coefficients_df['Coefficient'].values
        # intercept_value = coefficients_df['Intercept'].iloc[0]

        # # Calculate the prediction for the first row manually using the dot product
        # # Include the intercept in the calculation
        # prediction = np.dot(output_df.iloc[0].values, coefficients) + intercept_value

        # st.metric("Estimated Price", f"$ {prediction:,.0f}")

        # with st.expander("Show Coefficient"):
        #     st.dataframe(coefficients_df)