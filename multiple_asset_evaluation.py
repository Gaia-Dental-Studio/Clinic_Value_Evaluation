import pickle
import pandas as pd
import numpy as np
import streamlit as st
import requests
import json
from cashflow_plot import ModelCashflow

def app():

    def process_multiple_dfs(filtered_clinic_projected_cashflow_set):
        result = {}

        for idx, (initiative_name, dataframes) in enumerate(filtered_clinic_projected_cashflow_set.items()):
            # Assuming 'dataframes' is a single DataFrame
            df = dataframes

            # Initialize the dictionary for this index
            result[idx] = {
                "initiatives": initiative_name,
                "horizon": {}
            }

            # Iterate through unique Horizon values
            for horizon in df["Horizon"].unique():
                # Filter the DataFrame for the current Horizon
                horizon_data = df[df["Horizon"] == horizon]

                # Add Horizon-specific data to the nested "horizon" dictionary
                result[idx]["horizon"][horizon] = {
                    row["Quarter"]: int(row["Profit"]) for _, row in horizon_data.iterrows()
                }

        return result



    st.title('Multiple Asset Evaluation')
    
    dataset = st.selectbox('Select dataset', ['Dataset 1', 'Dataset 2', 'Dataset 3'])
    
    dataset = int(dataset.split()[-1])

    clinic_projected_cashflow_set = pickle.load(open(f'dummy_clinic_model/pkl_files/dataset_{dataset}/clinic_projected_cashflow_set.pkl', 'rb'))



    # Streamlit UI for clinic selection
    selected_clinics = st.multiselect('Select clinics to evaluate', list(clinic_projected_cashflow_set.keys()))

    # Dropdown for approach selection, dependent on clinic selection
    approach_selected = st.selectbox(
        'Select approach',
        list(clinic_projected_cashflow_set[selected_clinics[0]].keys()) if selected_clinics else []
    )

    if selected_clinics and approach_selected:
        # Adjust selected_clinics to a dictionary-like structure
        selected_clinics_dict = {i: clinic for i, clinic in enumerate(selected_clinics)}

        # Filter based on selection
        selected_clinic_names = list(selected_clinics_dict.values())
        filtered_clinic_projected_cashflow_set = {
            clinic_name: clinic_projected_cashflow_set[clinic_name][approach_selected]
            for clinic_name in selected_clinic_names
            if clinic_name in clinic_projected_cashflow_set and approach_selected in clinic_projected_cashflow_set[clinic_name]
        }
        
        start_year = 2025
        
        # Traverse and update all DataFrames in the filtered set
        for clinic_name, dataframes in filtered_clinic_projected_cashflow_set.items():
            for dataframe_key, dataframe in dataframes.items():
                # Perform the specified operations on the DataFrame
                dataframe['Profit'] = dataframe['Revenue'] - dataframe['Expense']
                dataframe['Period'] = pd.to_datetime(dataframe['Period'])
                dataframe['Year'] = dataframe['Period'].dt.year
                dataframe['Quarter'] = dataframe['Period'].dt.quarter
                dataframe['Adjusted Quarter'] = (dataframe['Year'] - start_year) * 4 + dataframe['Quarter']
                dataframe['Horizon'] = np.where(dataframe['Adjusted Quarter'] <= 2, 'Horizon 1', 
                                                np.where(dataframe['Adjusted Quarter'] <= 3, 'Horizon 2', 'Horizon 3'))
                dataframe['Quarter'] = 'Q' + dataframe['Adjusted Quarter'].astype(str)
                dataframe['Quarter'] = dataframe['Adjusted Quarter']
                
                
                dataframe.drop(columns=['Adjusted Quarter'], inplace=True)
                
                
                dataframe = dataframe.groupby(['Quarter', 'Horizon']).agg({
                    'Revenue': np.sum,
                    'Expense': np.sum,
                    'Profit': np.sum
                }).reset_index()
                
                # dataframe = dataframe.groupby(['Horizon', 'Quarter'])['Profit'].sum().unstack('Horizon')
                # st.write(dataframe)

        # Display filtered results
        # st.write("Filtered Projected Cashflow Set:", filtered_clinic_projected_cashflow_set)
        
        # Serialize filtered_clinic_projected_cashflow_set
        # filtered_clinic_projected_cashflow_set_hex = pickle.dumps(filtered_clinic_projected_cashflow_set).hex()

        # Make a POST request to the Flask API
        # response = requests.post(
        #     'http://127.0.0.1:5000/process_clinic_data',
        #     json={'filtered_clinic_projected_cashflow_set': filtered_clinic_projected_cashflow_set_hex}
        # )
        
        # st.write("Filtered Projected Cashflow Set:", filtered_clinic_projected_cashflow_set)
        
            # Process data to generate pool_clinic
        pool_clinic = {}
        for clinic_name, clinic_data in filtered_clinic_projected_cashflow_set.items():
            clinic_json = process_multiple_dfs(clinic_data)
            pool_clinic[clinic_name] = clinic_json


        # # Parse the response
        # if response.status_code == 200:
        #     pool_clinic = response.json()['pool_clinic']
        #     # st.write("Pool Clinic:", pool_clinic)
        # else:
        #     st.error(f"Error: {response.json().get('error', 'Unknown error')}")
        
        # save into pool_clinic.json
        with open('pool_clinic.json', 'w') as f:
            json.dump(pool_clinic, f)
            
        
        
        ### Plot
        
        model_plot = ModelCashflow()
        result = model_plot.merge_companies_or_categories(filtered_clinic_projected_cashflow_set, by='companies')
        
        
        
        
        col1, col2, col3, col4 = st.tabs(['Daily', 'Weekly', 'Monthly', 'Quarterly'])
        
        with col1:
            fig = model_plot.cashflow_plot(number_of_days=365 , granularity='daily', start_date='2025-01-01')
            st.plotly_chart(fig)
            
        with col2:
            fig = model_plot.cashflow_plot(number_of_days=365 , granularity='weekly', start_date='2025-01-01')
            st.plotly_chart(fig)
            
        with col3:
            fig = model_plot.cashflow_plot(number_of_days=365 , granularity='monthly', start_date='2025-01-01')
            st.plotly_chart(fig)
        
        with col4:
            fig = model_plot.cashflow_plot(number_of_days=365 , granularity='quarterly', start_date='2025-01-01')
            st.plotly_chart(fig)


        result = model_plot.merge_companies_or_categories(filtered_clinic_projected_cashflow_set, by='categories')

        
        
        col1, col2, col3, col4 = st.tabs(['Daily', 'Weekly', 'Monthly', 'Quarterly'])
        
        with col1:
            fig = model_plot.cashflow_plot(number_of_days=365 , granularity='daily', start_date='2025-01-01')
            st.plotly_chart(fig)
            
        with col2:
            fig = model_plot.cashflow_plot(number_of_days=365 , granularity='weekly', start_date='2025-01-01')
            st.plotly_chart(fig)
            
        with col3:
            fig = model_plot.cashflow_plot(number_of_days=365 , granularity='monthly', start_date='2025-01-01')
            st.plotly_chart(fig)
        
        with col4:
            fig = model_plot.cashflow_plot(number_of_days=365 , granularity='quarterly', start_date='2025-01-01')
            st.plotly_chart(fig)
        
        
        
        
        
    else:
        st.write("No clinics or approach selected.")





