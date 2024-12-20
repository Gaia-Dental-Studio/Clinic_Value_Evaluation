from dummy_clinic_generator import DummyClinicModel
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

import sys

# Add the parent directory of 'clinic_dummy_model' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model_forecasting.py from the main folder
from model_forecasting import ModelForecastPerformance
from cashflow_plot import ModelCashflow


st.title("Clinic Model")



reference_clinic_df = pd.read_csv('reference_clinic_df_real.csv')[:50]
item_code_df = pd.read_csv("cleaned_item_code.csv")
demand_prob_item_code = pd.read_csv("clarity_data_with_demand_prob.csv")


other_parameter_dict = {"mean_cogs_lab_material": 0.0727, "std_cogs_lab_material": 0.0536, 
                        "mean_expenses": 0.35, "std_expenses": 0.10, #earlier 0.24 of std, and 0.42 of mean
                        "mean_depreciation": 0.0339, "std_depreciation": 0.03175,
                        "mean_salary_OHT": 38.5, "std_salary_OHT": 6.3, # originally mean is 41.5 and std is 6.3
                        "mean_salary_dentist": 97.8, "std_salary_dentist": 15.5, #originally mean is 97.8 and std is 20.6
                        "mean_salary_specialist": 125, "std_salary_specialist": 20.1, # originally mean is 140 and std is 25.1
                        "mean_revenue": 700000, "std_revenue": 411977
                        }


model = DummyClinicModel(reference_clinic_df, other_parameter_dict, item_code_df, demand_prob_item_code)

st.markdown("### Parameter Value Assumption")

col1, col2 = st.columns(2)

mean_revenue, std_revenue = model.dist_param(by='clinic_revenue')
# mean_revenue, std_revenue = other_parameter_dict['mean_revenue'], other_parameter_dict['std_revenue']
mean_cogs_lab_material, std_cogs_lab_material = other_parameter_dict['mean_cogs_lab_material'], other_parameter_dict['std_cogs_lab_material']
mean_expenses, std_expenses = other_parameter_dict['mean_expenses'], other_parameter_dict['std_expenses']
mean_depreciation, std_depreciation = other_parameter_dict['mean_depreciation'], other_parameter_dict['std_depreciation']

with col1:
    st.metric("Mean Revenue", f"${mean_revenue:,.0f}", help="Taken from fitting 50 clinics of Practice Sale Search")
    st.metric("Mean COGS for Lab & Material (%)", f"{mean_cogs_lab_material * 100:.2f}%", help = 'Assumed, per item code')
    st.metric("Mean Expenses (%)", f"{mean_expenses * 100:.2f}%", help = 'Taken from fitting 12 Study Case Clinics')
    st.metric("Mean Depreciation (%)", f"{mean_depreciation * 100:.2f}%", help = 'Taken from fitting 12 Study Case Clinics')
    st.metric("Mean Salary OHT", f"${other_parameter_dict['mean_salary_OHT']:.2f}", help = 'Taken from fitting various salary of researched data')
    st.metric("Mean Salary Dentist", f"${other_parameter_dict['mean_salary_dentist']:.2f}",  help = 'Taken from fitting various salary of researched data')
    st.metric("Mean Salary Specialist", f"${other_parameter_dict['mean_salary_specialist']:.2f}",  help = 'Taken from fitting various salary of researched data')
with col2:
    st.metric("Standard Deviation Revenue", f"${std_revenue:,.0f}", help="Taken from fitting 50 clinics of Practice Sale Search")
    st.metric("Standard Deviation COGS for Lab & Material (%)", f"{std_cogs_lab_material * 100:.2f}%", help = 'Taken from fitting 12 Study Case Clinics')
    st.metric("Standard Deviation Expenses (%)", f"{std_expenses * 100:.2f}%", help = 'Taken from fitting 12 Study Case Clinics')
    st.metric("Standard Deviation Depreciation (%)", f"{std_depreciation * 100:.2f}%", help = 'Taken from fitting 12 Study Case Clinics')
    st.metric("Standard Deviation Salary OHT", f"${other_parameter_dict['std_salary_OHT']:.2f}", help = 'Taken from fitting various salary of researched data')
    st.metric("Standard Deviation Salary Dentist", f"${other_parameter_dict['std_salary_dentist']:.2f}", help='Taken from fitting various salary of researched data')
    st.metric("Standard Deviation Salary Specialist", f"${other_parameter_dict['std_salary_specialist']:.2f}", help='Taken from fitting various salary of researched data')
    
# st.markdown("### Generate Dummy Clinics")

# number_of_clinics = st.slider("Number of Clinics", 1, 50, 50)

# if st.button("Generate"):
#     # Generate dummy clinic DataFrame
#     dummy_clinic_df = model.generate_dummy_clinic(number_of_clinics)
#     dummy_clinic_df.to_pickle('dummy_clinic_df.pkl')  # Save to pickle

#     # Generate item codes per clinic as a dictionary
#     item_code_per_clinic = model.generate_item_code_per_clinic(dummy_clinic_df)
    
#     # Precompute metrics for all clinics
#     clinic_metrics = {}

#     for clinic_name, clinic_data in item_code_per_clinic.items():
#         # Calculate salaries
#         dentist_salary = np.random.normal(other_parameter_dict['mean_salary_dentist'], other_parameter_dict['std_salary_dentist'])
#         oht_salary = np.random.normal(other_parameter_dict['mean_salary_OHT'], other_parameter_dict['std_salary_OHT'])
#         specialist_salary = np.random.normal(other_parameter_dict['mean_salary_specialist'], other_parameter_dict['std_salary_specialist'])

#         # Calculate COGS salary and other metrics
#         number_OHT_staff, number_specialist_staff, total_salary, updated_clinic_item_code = model.calculate_cogs_salary(
#             clinic_data, dentist_salary, oht_salary, specialist_salary
#         )
        
#         # update clinic revenue in dummy_clinic_df 
#         dummy_clinic_df.loc[dummy_clinic_df['clinic_name'] == clinic_name, 'clinic_revenue'] = updated_clinic_item_code['Total Revenue'].sum()
#         dummy_clinic_df.loc[dummy_clinic_df['clinic_name'] == clinic_name, 'clinic_cogs_lab_material'] = updated_clinic_item_code['Total Material Cost'].sum()

#         total_revenue = updated_clinic_item_code['Total Revenue'].sum()
#         total_cogs_lab_material = updated_clinic_item_code['Total Material Cost'].sum()            
#         total_expenses = dummy_clinic_df[dummy_clinic_df['clinic_name'] == clinic_name]['clinic_expenses'].values[0]
#         total_depreciation = dummy_clinic_df[dummy_clinic_df['clinic_name'] == clinic_name]['clinic_depreciation'].values[0]

#         # Calculate EBIT and ratios
#         total_ebit = total_revenue - total_cogs_lab_material - total_expenses - total_depreciation - total_salary
#         ebit_ratio = total_ebit / total_revenue if total_revenue else 0

#         # Store all metrics in a dictionary for this clinic
#         clinic_metrics[clinic_name] = {
#             'number_OHT_staff': number_OHT_staff,
#             'number_specialist_staff': number_specialist_staff,
#             'total_salary': total_salary,
#             'total_revenue': total_revenue,
#             'total_cogs_lab_material': total_cogs_lab_material,
#             'total_expenses': total_expenses,
#             'total_depreciation': total_depreciation,
#             'total_ebit': total_ebit,
#             'ebit_ratio': ebit_ratio,
#             'updated_clinic_item_code': updated_clinic_item_code,
#             # 'total_salary_2': updated_clinic_item_code['Total Salary'].sum()
#         }

#     # save clinic_metrics to pickle
#     with open('clinic_metrics.pkl', 'wb') as f:
#         pickle.dump(clinic_metrics, f)
        
    
#     # Save dictionary to pickle
#     with open('item_code_per_clinic.pkl', 'wb') as f:
#         pickle.dump(item_code_per_clinic, f)

# # Load the dummy clinic DataFrame from pickle
# dummy_clinic_df = pd.read_pickle('dummy_clinic_df.pkl')

# # Load the item codes per clinic dictionary from pickle
# with open('item_code_per_clinic.pkl', 'rb') as f:
#     item_code_per_clinic = pickle.load(f)
    
# with open('clinic_metrics.pkl', 'rb') as f:
#     clinic_metrics = pickle.load(f)
    
# map medical officer from clinic_metrics[selected_clinic]['updated_clinic_item_code'] to item_code_df

    
st.markdown("### Select Clinics Dataset")
    
# Select Dataset Dropdown
dataset_selected = st.selectbox(
    "Select Dataset",
    ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4", "Dataset 5"]
)

if dataset_selected:
    # Extract the dataset index from the selected text
    dataset_index = dataset_selected.split()[-1]  # e.g., "1" from "Dataset 1"

    # Set the corresponding directory
    dataset_dir = os.path.join("pkl_files", f"dataset_{dataset_index}")

    # Check if the directory exists
    if os.path.exists(dataset_dir):
        # Load the dummy clinic DataFrame
        dummy_clinic_df = pd.read_pickle(os.path.join(dataset_dir, 'dummy_clinic_df.pkl'))

        # Load the item codes per clinic dictionary
        with open(os.path.join(dataset_dir, 'item_code_per_clinic.pkl'), 'rb') as f:
            item_code_per_clinic = pickle.load(f)

        # Load the clinic metrics dictionary
        with open(os.path.join(dataset_dir, 'clinic_metrics.pkl'), 'rb') as f:
            clinic_metrics = pickle.load(f)
            
        # Load the clinic pool dictionary
        with open(os.path.join(dataset_dir, 'pool_clinic_df.pkl'), 'rb') as f:
            pool_clinic_df = pickle.load(f)

        st.success(f"Loaded data from {dataset_dir}")
    else:
        st.error(f"The directory {dataset_dir} does not exist!")

# comment till here


# update dummy_clinic_df with clinic_metrics for each clinic
dummy_clinic_df['clinic_cogs_salary'] = dummy_clinic_df['clinic_name'].map({k: v['total_salary'] for k, v in clinic_metrics.items()})
dummy_clinic_df['clinic_ebit'] = dummy_clinic_df['clinic_name'].map({k: v['total_ebit'] for k, v in clinic_metrics.items()})
dummy_clinic_df['clinic_ebit_ratio'] = dummy_clinic_df['clinic_name'].map({k: v['ebit_ratio'] for k, v in clinic_metrics.items()})

salary_to_revenue_ratio = dummy_clinic_df['clinic_cogs_salary'].sum() / dummy_clinic_df['clinic_revenue'].sum()
ebit_ratio_average = dummy_clinic_df['clinic_ebit_ratio'].mean()


# st.write(f"Total Salary to Revenue Ratio: {salary_to_revenue_ratio:.2f}")
# st.write(f"Average EBIT Ratio: {ebit_ratio_average:.2f}")


# create histogram using model method of revenue_histogram 
st.dataframe(dummy_clinic_df, hide_index=True)

st.markdown("### Histogram")

histogram_by = st.selectbox("Histogram by", ['clinic_revenue', 'clinic_cogs_lab_material', 'clinic_expenses', 'clinic_depreciation', 'clinic_cogs_salary', 'clinic_ebit' , 'clinic_ebit_ratio'], index=6)

st.pyplot(model.histogram(dummy_clinic_df, by=histogram_by))


st.divider()

# clinic_list_names = dummy_clinic_df.clinic_name.tolist()



# Streamlit UI
selected_clinic = st.selectbox("Select Clinic", options=list(clinic_metrics.keys()))

# Display metrics for the selected clinic
st.markdown(f"### Clinic {selected_clinic} Details")

col1, col2 = st.columns(2)

with col1:
    # Display item code DataFrame for the selected clinic
    # st.dataframe(clinic_metrics[selected_clinic]['updated_clinic_item_code'], column_order=('Code', 'Total Demand', 'Total Revenue', 'Total Duration', 'Medical Officer', 'Total Salary'), hide_index=True)
    st.dataframe(clinic_metrics[selected_clinic]['updated_clinic_item_code'], hide_index=True)
    # total_material_cost = clinic_metrics[selected_clinic]['updated_clinic_item_code']['Total Material Cost'].sum()
    clinic_metrics[selected_clinic]['updated_clinic_item_code'].to_csv('clinic_item_code.csv', index=False)

with col2:
    st.markdown("##### Location Details")
    state = reference_clinic_df[reference_clinic_df['clinic_name'] == selected_clinic]['state'].iloc[0]
    region = reference_clinic_df[reference_clinic_df['clinic_name'] == selected_clinic]['region'].iloc[0]
    st.markdown(f"State: {state}")
    st.markdown(f"Region: {region}")

    col3, col4 = st.columns(2)

    with col3:
        metrics = clinic_metrics[selected_clinic]
        st.metric("Total Revenue", f"${metrics['total_revenue']:,.0f}")

        

    with col4:
        st.metric("Number of Dentist", 1)  # Static for now
 
    col5, col6 = st.columns(2)
    
    with col5:
        st.metric("Total COGS Lab Material", f"${metrics['total_cogs_lab_material']:,.0f}", delta=f"{metrics['total_cogs_lab_material'] / metrics['total_revenue']:.2%}", delta_color="off")

    with col6:
        st.metric("Number of OHT Staff", f"{int(metrics['number_OHT_staff'])}")
        
    col7, col8 = st.columns(2)
    
    with col7:
        st.metric("Total COGS Salary", f"${metrics['total_salary']:,.0f}", delta=f"{metrics['total_salary'] / metrics['total_revenue']:.2%}", delta_color="off")
    with col8:
        st.metric("Number of Specialist Staff", f"{int(metrics['number_specialist_staff'])}")
    
    col9, col10 = st.columns(2)
        
    with col9:
        st.metric("Total Expenses", f"${metrics['total_expenses']:,.0f}", delta=f"{metrics['total_expenses'] / metrics['total_revenue']:.2%}", delta_color="off")
    

    with col10:
        st.metric("Total Depreciation", f"${metrics['total_depreciation']:,.0f}", delta=f"{metrics['total_depreciation'] / metrics['total_revenue']:.2%}", delta_color="off")
        # st.metric("Total Material Cost", f"${total_material_cost:,.0f}", delta=f"{total_material_cost / metrics['total_cogs_lab_material']:.2%}", delta_color="off")

st.markdown(f"### Clinic {selected_clinic} Summary")
col1, col2 = st.columns(2)

def ebit_multiple(ebit_ratio):
    if ebit_ratio < 0.1:
        return 2
    elif ebit_ratio < 0.2:
        return 3
    elif ebit_ratio < 0.3:
        return 4
    elif ebit_ratio < 0.4:
        return 5
    elif ebit_ratio < 0:
        return 0
    else:
        return 6


with col1:

    st.metric("Total EBIT", f"${metrics['total_ebit']:,.0f}")
    # st.metric("EBIT Multiple", ebit_multiple(metrics['ebit_ratio']))

with col2:
    st.metric("EBIT Ratio", f"{metrics['ebit_ratio'] * 100:.2f}%")
    # st.metric('Buying Price', f"${metrics['total_ebit'] * ebit_multiple(metrics['ebit_ratio']):,.0f}")


def save_pickles_with_folder(dummy_clinic_df, clinic_metrics, item_code_per_clinic):
    # Define the parent directory for pickle files
    parent_dir = "pkl_files"
    os.makedirs(parent_dir, exist_ok=True)  # Ensure pkl_files directory exists

    # Find the next available dataset folder
    index = 1
    while os.path.exists(os.path.join(parent_dir, f"dataset_{index}")):
        index += 1

    # Create the new dataset folder
    dataset_folder = os.path.join(parent_dir, f"dataset_{index}")
    os.makedirs(dataset_folder)

    # Save the files in the new dataset folder
    dummy_clinic_df.to_pickle(os.path.join(dataset_folder, "dummy_clinic_df.pkl"))

    with open(os.path.join(dataset_folder, "clinic_metrics.pkl"), "wb") as f:
        pickle.dump(clinic_metrics, f)

    with open(os.path.join(dataset_folder, "item_code_per_clinic.pkl"), "wb") as f:
        pickle.dump(item_code_per_clinic, f)

    st.success(f"Files saved in {dataset_folder}")

# Example Streamlit Button Usage
if st.button("Save Dataset"):
    save_pickles_with_folder(dummy_clinic_df, clinic_metrics, item_code_per_clinic)


# st.divider()

# st.markdown("### Historical Cashflow")

# st.write("Define the range of historical data for the cashflow")

# col1, col2 = st.columns(2)

# with col1:
#     start_date = st.date_input("Start Date", value=pd.to_datetime('2021-01-01'))
#     start_year = start_date.year 
#     start_month = start_date.month 

# with col2:
#     end_date = st.date_input("End Date", value=pd.to_datetime('2021-12-31'))
#     end_year = end_date.year
#     end_month = end_date.month
    
# num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

# company_variables ={'General Expense':-clinic_metrics[selected_clinic]['total_expenses']}

# model = ModelForecastPerformance(company_variables)
# model_cashflow = ModelCashflow()
# cleaned_item_code = pd.read_csv("cleaned_item_code.csv")



# dataframe = model.generate_monthly_cashflow_given_item_code(clinic_metrics[selected_clinic]['updated_clinic_item_code'], cleaned_item_code, num_months, 0.15, start_year, start_month)



# dataframe2 = model.forecast_indirect_cost(num_months, start_year)
# # dataframe2['Code'] = None
# # dataframe2['Profit'] = dataframe2['Revenue'] - dataframe2['Expense']

# # #append the dataframe2 to dataframe
# # dataframe = pd.concat([dataframe, dataframe2], ignore_index=True)

# # dataframe['Period'] = pd.to_datetime(dataframe['Period'], format='%d-%m-%Y')
# # dataframe['Period'] = dataframe['Period'].dt.date
# # dataframe = dataframe.sort_values(by='Period').reset_index(drop=True)


# dataframe.to_csv('clinic_item_code_cashflow.csv', index=False)

# # dataframe = pd.read_csv('clinic_item_code_cashflow.csv')

# st.dataframe(dataframe, hide_index=True)

# st.write(f"Total Revenue: ${dataframe['Revenue'].sum():,.0f}")

# dataframe['Period'] = pd.to_datetime(dataframe['Period'])

# group_dataframe_by_year = dataframe.groupby(dataframe['Period'].dt.year)['Revenue'].sum()

# st.dataframe(group_dataframe_by_year, hide_index=True)

# st.write(f"Total Expense: ${dataframe['Expense'].sum():,.0f}")

# st.dataframe(dataframe2, hide_index=True)

# st.write(f"Total Indirect Cost: ${dataframe2['Expense'].sum():,.0f}")

# number_of_months = model.total_days_from_start(num_months, start_year=start_year, start_month=start_month)

# model_cashflow.add_company_data("Gross Profit", dataframe)
# model_cashflow.add_company_data("Indirect Cost", dataframe2)

# forecast_linechart_daily = model_cashflow.cashflow_plot(number_of_months, granularity='monthly', start_date=start_date)


# st.plotly_chart(forecast_linechart_daily)

# # Apply the function to add the 'Hourly_Period' column
# dataframe_with_hourly = model.add_hourly_period(dataframe)
# dataframe_2_with_hourly = model.add_hourly_period(dataframe2)

# # st.dataframe(dataframe_with_hourly, column_order=['Hourly_Period', 'Code', 'Revenue', 'Expense', 'Profit'], hide_index=True)

# model_cashflow.remove_all_companies()

# model_cashflow.add_company_data("Gross Profit", dataframe_with_hourly)
# model_cashflow.add_company_data("Indirect Cost", dataframe_2_with_hourly)

# start_date_with_time = start_date.strftime('%Y-%m-%d %H:%M:%S')

# forecast_linechart_hourly = model_cashflow.cashflow_plot_detailed(48, granularity='half-hourly', start_date=start_date_with_time)

# st.plotly_chart(forecast_linechart_hourly)

st.divider()

st.markdown("### Historical Cashflow")

# open pool_clinic_df.pkl
# pool_clinic_df = pd.read_pickle('pool_clinic_df.pkl')

model = ModelForecastPerformance()
model_cashflow = ModelCashflow()


dataframe_gross_profit = pool_clinic_df[selected_clinic]['Gross Profit']
dataframe_indirect_expense = pool_clinic_df[selected_clinic]['Indirect Cost']

# Convert 'Period' column to datetime
dataframe_gross_profit['Period'] = pd.to_datetime(dataframe_gross_profit['Period'])

col1, col2 = st.columns(2)

with col1:
# Extract the oldest and most recent dates
    start_date = st.date_input("Start Date", value=dataframe_gross_profit['Period'].min(), min_value=dataframe_gross_profit['Period'].min())
    start_year = start_date.year 
    start_month = start_date.month 

with col2:
    end_date = st.date_input("End Date", value=dataframe_gross_profit['Period'].max(), max_value=dataframe_gross_profit['Period'].max())
    end_year = end_date.year
    end_month = end_date.month

num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
number_of_months = model.total_days_from_start(num_months, start_year=start_year, start_month=start_month)


model_cashflow.remove_all_companies()

model_cashflow.add_company_data("Gross Profit", dataframe_gross_profit)
model_cashflow.add_company_data("Indirect Cost", dataframe_indirect_expense)

tab1, tab2, tab3 = st.tabs(['Daily', 'Weekly', 'Monthly'])

with tab1:
    st.markdown('#### Daily Cashflow')
    forecast_linechart_daily = model_cashflow.cashflow_plot(number_of_months, granularity='daily', start_date=start_date)
    st.plotly_chart(forecast_linechart_daily)

with tab2:
    st.markdown('#### Weekly Cashflow')
    forecast_linechart_daily = model_cashflow.cashflow_plot(number_of_months, granularity='weekly', start_date=start_date)
    st.plotly_chart(forecast_linechart_daily)
with tab3:
    st.markdown('#### Monthly Cashflow')
    
    forecast_linechart_daily = model_cashflow.cashflow_plot(number_of_months, granularity='monthly', start_date=start_date)


    st.plotly_chart(forecast_linechart_daily)

st.markdown('#### Hourly Cashflow')



start_date_with_time = start_date.strftime('%Y-%m-%d %H:%M:%S')

col3, col4 = st.columns(2)

with col3:
    granularity = st.selectbox("Granularity", ['hourly', 'half-hourly'], index=1)

with col4:
    period_mapping = {'1 day': 24, '2 Days': 48, '3 Days': 72, '1 Week': 168}
    number_of_periods = st.selectbox("Number of Periods", list(period_mapping.keys()), index=1)
    number_of_periods = period_mapping[number_of_periods] if granularity == 'hourly' else period_mapping[number_of_periods] * 2


forecast_linechart_hourly = model_cashflow.cashflow_plot_detailed(number_of_periods, granularity=granularity, start_date=start_date_with_time)

st.plotly_chart(forecast_linechart_hourly)

st.divider()

st.markdown("### Excerpt of Transaction Dataframe")

# Extract the transaction data for the selected clinic

# group by profit sum and sort from the lowest for dataframe_gross_profit 
# dataframe_gross_profit = dataframe_gross_profit.groupby('Code')['Profit'].sum().sort_values(ascending=True).reset_index()

st.dataframe(dataframe_gross_profit.head(40), hide_index=True)
