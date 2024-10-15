import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import triang

from model import *

def app():
    st.markdown('### Sensitivity Analysis on EBIT Multiple with Monte Carlo Simulation')

    main_var_csv = pd.read_csv('monte_carlo_variables.csv')

    main_var = st.data_editor(main_var_csv, hide_index=True)

    # Data for the DataFrame
    dentist_data = {
        'Variable': ['Current Number of Dentist', 'Projected Number of Dentist'],
        'Min Value': [1, 1],
        'Most Likely': [2, 2],
        'Max Value': [4, 4]
    }

    # Creating the DataFrame
    dentist_data = pd.DataFrame(dentist_data)

    dentist_var = st.data_editor(dentist_data, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:

        possibility_existing_dentist_leaving = st.slider('Possibility of Existing Dentist Leaving', 0.0, 1.0, 0.5, 0.1)

    if st.button("Run Monte Carlo Simulation"):
        
        ebit_multiple_results = []
        
        for _ in range(500):
            company_variables = {}
            
            # Loop over each row in the DataFrame
            for index, row in main_var.iterrows():
                variable_name = row['Variable']
                mean = row['Mean']
                std = row['Std']
                
                if variable_name == 'EBIT':
                    # Keep generating values until it's greater than or equal to 100,000 for EBIT
                    value = norm.rvs(loc=mean, scale=std)
                    while value < 100000:
                        value = norm.rvs(loc=mean, scale=std)
                else:
                    # For other variables, keep generating values until it's non-negative
                    value = norm.rvs(loc=mean, scale=std)
                    while value < 0:
                        value = norm.rvs(loc=mean, scale=std)
                
                # Store the valid random value in the dictionary
                company_variables[variable_name] = value
                
            for index, row in dentist_var.iterrows():
                variable_name = row['Variable']
                min_value = row['Min Value']
                most_likely = row['Most Likely']
                max_value = row['Max Value']
                
                # Calculate the shape parameter 'c'
                c = (most_likely - min_value) / (max_value - min_value)
                
                # Generate a random value using the triangular distribution
                company_variables[variable_name] = int(triang.rvs(c, loc=min_value, scale=max_value - min_value))
            
            
            company_variables['Possibility Existing Dentist Leaving'] = np.random.choice([0, 1], p=[1 - possibility_existing_dentist_leaving, possibility_existing_dentist_leaving])
            


            
            model = ModelClinicValue(company_variables)
            
            model.ebit = company_variables['EBIT']
            model.ebit_ratio = company_variables['EBIT Ratio']
            
            ebit_multiple = model.ebit_baseline_to_multiple(company_variables['Net Sales Growth'])
            ebit_multiple = model.ebit_multiple_adjustment_due_dentist(ebit_multiple, company_variables['Current Number of Dentist'], company_variables['Projected Number of Dentist'], company_variables['Possibility Existing Dentist Leaving'])
            ebit_multiple = model.ebit_multiple_adjustment_due_net_sales_variation(ebit_multiple, company_variables['Relative Variation of Net Sales'])
            ebit_multiple = model.ebit_multiple_adjustment_due_number_patient_and_patient_spending_variability(ebit_multiple, company_variables['Number of Active Patients'], company_variables['Relative Variation of Patient Spending'])
            
            clinic_valuation = ebit_multiple * model.ebit
            
            ebit_multiple_results.append(ebit_multiple)
            

        # Convert the results to a NumPy array for convenience
        ebit_multiple_results = np.array(ebit_multiple_results)

        # Calculate the mean of the results
        mean_value = np.mean(ebit_multiple_results)

        # Plot a histogram of the results
        st.write("Monte Carlo Simulation Results for EBIT Multiple")
        plt.figure(figsize=(10, 6))
        plt.hist(ebit_multiple_results, bins=30, edgecolor='k', alpha=0.7)

        # Add a vertical line for the mean
        plt.axvline(mean_value, color='r', linestyle='dashed', linewidth=2)

        # Add a label for the mean value
        plt.text(mean_value, plt.ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='r', ha='center')

        # Title and labels
        plt.title("Distribution of EBIT Multiple")
        plt.xlabel("EBIT Multiple")
        plt.ylabel("Frequency")

        # Display the plot in Streamlit
        st.pyplot(plt)
        
        st.write(f"Expected EBIT Multiple: {mean_value:.2f}")
            
        
        
    
    
    