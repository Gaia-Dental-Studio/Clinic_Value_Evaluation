import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from model import *

def app():
    st.markdown('### Sensitivity Analysis on EBIT Multiple with Monte Carlo Simulation')

    # Load the CSV file
    main_var_csv = pd.read_csv('monte_carlo_variables.csv')

    # Display the data editor for user inputs
    main_var = st.data_editor(main_var_csv, hide_index=True)

    if st.button("Run Monte Carlo Simulation"):
        
        # Pre-allocate results list
        ebit_multiple_results = np.zeros(500)
        
        # Loop over the number of simulations
        for i in range(500):
            company_variables = {}
            
            # Iterate over each row in the DataFrame
            for index, row in main_var.iterrows():
                variable_name = row['Variable']
                mean = row['Mean']
                std = row['Std']
                
                # If variable is 'EBIT', enforce the >= 100,000 condition
                if variable_name == 'EBIT':
                    value = norm.rvs(loc=mean, scale=std)
                    while value < 100000:
                        value = norm.rvs(loc=mean, scale=std)
                else:
                    # For other variables, ensure non-negative values
                    value = norm.rvs(loc=mean, scale=std)
                    while value < 0:
                        value = norm.rvs(loc=mean, scale=std)
                
                # Store valid value in the company_variables dictionary
                company_variables[variable_name] = value
            
            # Create the model with the company variables
            model = ModelClinicValue(company_variables)
            
            # Set model parameters
            model.ebit = company_variables['EBIT']
            model.ebit_ratio = company_variables['EBIT Ratio']
            
            # Perform the calculations on ebit_multiple
            ebit_multiple = model.ebit_baseline_to_multiple(company_variables['Net Sales Growth'])
            ebit_multiple = model.ebit_multiple_adjustment_due_dentist(ebit_multiple, company_variables['Risk of Dentist Leaving'])
            ebit_multiple = model.ebit_multiple_adjustment_due_net_sales_variation(ebit_multiple, company_variables['Relative Variation of Net Sales'])
            ebit_multiple = model.ebit_multiple_adjustment_due_number_patient_and_patient_spending_variability(
                ebit_multiple, company_variables['Number of Active Patients'], company_variables['Relative Variation of Patient Spending']
            )
            
            # Calculate the clinic valuation
            clinic_valuation = ebit_multiple * model.ebit
            
            # Store the result in the results array
            ebit_multiple_results[i] = ebit_multiple

        # Convert the results to a NumPy array and calculate the mean
        mean_value = np.mean(ebit_multiple_results)

        # Plot the histogram
        st.write("Monte Carlo Simulation Results for EBIT Multiple")
        plt.figure(figsize=(10, 6))
        plt.hist(ebit_multiple_results, bins=30, edgecolor='k', alpha=0.7)

        # Add a vertical line for the mean
        plt.axvline(mean_value, color='r', linestyle='dashed', linewidth=2)
        plt.text(mean_value, plt.ylim()[1] * 0.9, f'Mean: {mean_value:.2f}', color='r', ha='center')

        # Set title and labels
        plt.title("Distribution of EBIT Multiple")
        plt.xlabel("EBIT Multiple")
        plt.ylabel("Frequency")

        # Display the plot in Streamlit
        st.pyplot(plt)
        
        # Display the expected EBIT multiple
        st.write(f"Expected EBIT Multiple: {mean_value:.2f}")
