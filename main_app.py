import streamlit as st
import pandas as pd
import numpy as np
from model import ModelClinicValue
import pickle
import cvxpy as cp


    # Set the page title
st.title("Clinic Value Evaluation Model")

with st.container(border=True):
    
    st.markdown("## Baseline Clinic Value Model")
    st.write("Based on average performance of available data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        base_ebit = 250000
        base_tangible_asset = 80826.5
        base_net_sales_growth = 0.1
        base_number_of_active_patients = 1045
        base_number_of_dentist = 2
        base_relative_variability_patient_spending = 0.15
        base_possibility_existing_dentist_leaving = False
        st.metric("EBIT", f"$ {base_ebit:,.0f}", help="Net Sales - COGS - Operating Expenses")
        st.metric("Net Sales Growth Rate", f"{base_net_sales_growth * 100:.1f}%", help="Yearly")
        st.metric("Tangible Assets", f"$ {base_tangible_asset:,.0f}")
        
        st.metric("Number of Active Patients", base_number_of_active_patients, help="Number of active unique patients in the clinic for the last one year")
        st.metric("Number of Dentist", base_number_of_dentist, help="Number of dentist in the clinic")
        st.metric("Possibility Existing Dentist Leaving", base_possibility_existing_dentist_leaving, help="Check if there is a possibility of existing dentist leaving the clinic, despite the projected number of dentist will still be equal to current number of dentist.")
    with col2:
        base_ebit_ratio = 0.22
        st.metric("EBIT Ratio", f"{base_ebit_ratio * 100:.2f}%", help="EBIT / Net Sales")
        base_relative_variability_net_sales = 0.15
        base_equipment_usage_ratio = 0.5
        base_projected_number_of_dentist = 2
        
        
        st.metric("Relative Variability of Net Sales", f"{base_relative_variability_net_sales * 100:.0f}%", help="Standard Deviation of Net Sales / Mean Net Sales. Net Sales here is in monthly terms")
        st.metric("Equipment Usage Ratio", f"{base_equipment_usage_ratio * 100:.2f}%", help="Percentage of equipment usage from its expected lifetime")
        st.metric("Relative Variability of Patient Spending", f"{base_relative_variability_patient_spending * 100:.0f}%", help="Standard Deviation of Patient Spending / Mean Patient Spending. Patient Spending here is the yearly spending of each active unique patients")
        st.metric("Projected Number of Dentist", base_number_of_dentist, help="Projected number of dentist in the clinic")
        
        
        base_ebit_multiple = 2.5
        st.metric("EBIT Multiple", base_ebit_multiple)
# File upload widget
uploaded_file = st.file_uploader("Upload your company document", type=["csv", "xlsx", "docx", "pdf"])

# Initialize the ModelClinicValue class
model = ModelClinicValue({})

# If a file is uploaded, update the model variables with data from the file
if uploaded_file:
    model.update_variable_from_uploaded_file(uploaded_file)

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
        net_sales_growth = st.number_input("Net Sales Growth Rate", value=float(model.net_sales_growth) if model.net_sales_growth is not None else 0.0)

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
    company_variables["Tangible Assets (PP&E)"] = (selected_equipment['Quantity'] * selected_equipment['Price']).sum()

    st.markdown("### Dentist Availability")
    
    col1, col2 = st.columns(2)
    
    with col1:
        number_of_dentist = st.number_input("Number of Dentist", value=model.number_of_dentist if model.number_of_dentist is not None else 2, step=1)
        
    with col2:
        projected_number_of_dentist = st.number_input("Projected Number of Dentist", value=model.projected_number_of_dentist if model.projected_number_of_dentist is not None else 2, step=1)
        
    
    possibility_existing_dentist_leaving = st.checkbox("Possibility of Existing Dentist Leaving", value=model.potential_existing_dentist_leaving if model.potential_existing_dentist_leaving is not None else False, help="Check if there is a possibility of existing dentist leaving the clinic, despite the projected number of dentist will still be equal to current number of dentist.")


    st.markdown("### Customer Based Variables")

    col1, col2 = st.columns(2)
    
    with col1:
        number_of_active_patients = st.number_input("Number of Active Patients", value=int(model.number_of_patients) if model.number_of_patients is not None else 0, step=1, help='Number of Active Unique Patient on yearly average')
        
    with col2:
        relative_variability_patient_spending = st.number_input("Relative Variability of Patient Spending", value=float(model.relative_variability_patient_spending) if model.relative_variability_patient_spending is not None else 0.0, help='Coefficient of Variation of Patient Spending')
    
    # company_variables['Equipment_Life'].to_csv('equipment_life.csv')
    
# Button to trigger the evaluation
if st.button("Evaluate"):
    
    st.markdown("## Variable Summary")
    
    with st.container(border=True):
    
        st.markdown("### Profit & Loss")
        st.write("Yearly based")
        
        with st.expander("General Variables", expanded=False):
    
            col1, col2, col3 = st.columns(3)
            
            
            
            model = ModelClinicValue(company_variables)
            
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
            
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("EBITDA", f"$ {model.ebitda:,.0f}")
        
        with col2:
            st.metric("EBIT", f"$ {model.ebit:,.0f}", delta=f"{model.ebit - base_ebit:,.0f}")
            
        with col3:
            st.metric("EBIT Ratio", f"{model.ebit_ratio * 100:.2f}%", delta=f"{(model.ebit_ratio - base_ebit_ratio)*100:.2f}%")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Net Sales Growth", f"{net_sales_growth * 100:.2f}%", delta=f"{(net_sales_growth - base_net_sales_growth)*100:.2f}%")
            
        with col2:
            st.metric("Relative Variability of Net Sales", f"{relative_variability_net_sales * 100:.0f}%", delta=f"{(relative_variability_net_sales - base_relative_variability_net_sales)*100:.0f}%")
        
        
    with st.container(border=True):
        st.markdown("### Balance Sheet")
        
        tangible_asset_delta = company_variables.get('Tangible Assets (PP&E)', 0) - base_tangible_asset
        st.metric("Tangible Assets (PP&E)", f"$ {company_variables.get('Tangible Assets (PP&E)', 0):,.0f}", delta=tangible_asset_delta)
        
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
        equipment_usage_ratio, total_equipments, total_remaining_value = model.calculate_equipment_usage_ratio()
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Equipment Usage Ratio", f"{equipment_usage_ratio * 100:.2f}%", delta=f"{(equipment_usage_ratio - base_equipment_usage_ratio)*100:.2f}%")
            
        with col2:
            st.metric("Total Equipments", f"{total_equipments}")
            
        st.markdown('#### Dentist Availability')
        
        col1, col2 = st.columns(2)    
        
        with col1:
            st.metric("Number of Dentist", f"{number_of_dentist}", delta=f"{number_of_dentist - base_number_of_dentist}")
            st.metric("Possibility Existing Dentist Leaving", f"{possibility_existing_dentist_leaving}", delta= f"Baseline = {base_possibility_existing_dentist_leaving}", delta_color="off")
        
        with col2:
            st.metric("Projected Number of Dentist", f"{projected_number_of_dentist}", delta=f"{projected_number_of_dentist - base_projected_number_of_dentist}")
            
        st.markdown('#### Customer Based')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Number of Active Patients", f"{number_of_active_patients}", delta=f"{number_of_active_patients - base_number_of_active_patients}")
            
        with col2:
            st.metric("Relative Variation of Patient Spending", f"{relative_variability_patient_spending * 100:.0f}%", delta=f"{(relative_variability_patient_spending - base_relative_variability_patient_spending)*100:.0f}%")
            
            
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
        'Tangible Assets (PP&E)': company_variables.get('Tangible Assets (PP&E)', 0),
        # 'Net Cash Flow Average': average_cashflow,
        # 'Net Cash Flow Standard Deviation': std_deviation,
        # 'Net Cash Flow Trend Coefficient': trend_coefficient,
        'Equipment Usage Ratio': equipment_usage_ratio,
        'Total Equipments': total_equipments,
        'Net Sales Growth': net_sales_growth,
        'Total Remaining Value': total_remaining_value,
        'Current Number of Dentist': number_of_dentist,
        'Projected Number of Dentist': projected_number_of_dentist,
        'Possibility Existing Dentist Leaving': possibility_existing_dentist_leaving,
        'Relative Variation of Net Sales': relative_variability_net_sales,
        'Number of Active Patients': number_of_active_patients,
        'Relative Variation of Patient Spending': relative_variability_patient_spending
    }

    # Convert the output_variables dictionary to a DataFrame and save as CSV
    output_df = pd.DataFrame([output_variables])
    output_df.to_csv('output_variables.csv', index=False)
    
    
    st.divider()
    
    st.markdown("## Clinic Value")

    ebit_multiple = model.ebit_baseline_to_multiple(output_variables['Net Sales Growth'])
    equipment_adjusting_value = model.equipment_adjusting_value(output_variables['Total Remaining Value'], base_tangible_asset, base_equipment_usage_ratio)
    ebit_multiple = model.ebit_multiple_adjustment_due_dentist(ebit_multiple, output_variables['Current Number of Dentist'], output_variables['Projected Number of Dentist'], output_variables['Possibility Existing Dentist Leaving'])
    ebit_multiple = model.ebit_multiple_adjustment_due_net_sales_variation(ebit_multiple, output_variables['Relative Variation of Net Sales'])
    ebit_multiple = model.ebit_multiple_adjustment_due_number_patient_and_patient_spending_variability(ebit_multiple, output_variables['Number of Active Patients'], output_variables['Relative Variation of Patient Spending'])
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
        st.metric("Clinic Valuation Adjusted", f"$ {clinic_valuation + equipment_adjusting_value:,.0f}")
        st.caption("Clinic Valuation Adjusted = Clinic Valuation + Adjustment due Equipments")
    
    



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