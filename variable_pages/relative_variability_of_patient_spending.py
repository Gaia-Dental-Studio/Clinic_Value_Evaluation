import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import truncnorm

def app():
    
    st.markdown("##### Definition")
    st.write("Relative Variability of Patient Spending measures the cofficient of variance, or the overall deviation of total spending per patient to the mean value in yearly terms.")
    st.write("If the value of relative variability is high, it means that the spending of patients are lopsided, with high indication that some patients are spending much more, and most are spending much less. This is unfavorable because if it is the case, our potential in increasing revenue is limited to few numbers of high spending patients.")
    
    st.markdown("##### Formula")
    st.image(r'variable_pages\CV_patient_spending.png')


    st.markdown('##### Baseline Value Assumption')
    st.write(f"The baseline relative variability of yearly Patient Spending for the company is 15% based on our own assumption as there is no company data available. 15% is chosen because it is known that 15% is a measure of decent variability.")
    
    
    
    st.markdown('##### How does different relative variability would look like in a swarm plot?')
    st.write("Change the relative variability to see how the plot changes.")
    
 # Parameters
    num_patients = 200
    mean_spending = 5000  # Set an arbitrary mean for patient spending in dollars

    col1, col2 = st.columns(2)

    with col1:
        relative_variability = st.number_input("Relative Variability of Patient Spending", value=0.15, step=0.05)  # 15% coefficient of variation

    # Calculate standard deviation based on relative variability
    std_spending = mean_spending * relative_variability

    # Define the range for truncation, e.g., ±2 standard deviations from the mean
    lower, upper = mean_spending - 2 * std_spending, mean_spending + 2 * std_spending
    trunc_normal = truncnorm((lower - mean_spending) / std_spending, 
                            (upper - mean_spending) / std_spending, 
                            loc=mean_spending, scale=std_spending)

    # Generate patient spending based on the truncated normal distribution
    total_spending = trunc_normal.rvs(size=num_patients)

    # Ensure no negative spending values
    total_spending = np.clip(total_spending, 0, None)

    # Create patient IDs
    patient_ids = [f'Patient {i+1}' for i in range(num_patients)]

    # Create the dataframe
    spending_df = pd.DataFrame({
        'Patient ID': patient_ids,
        'Total Spending': total_spending
    })
        
    
    # Create a swarm plot using Plotly
    fig_swarm = px.strip(spending_df, y='Total Spending', 
                        title='Swarm Plot of Patient Total Spending', 
                        template='plotly_white',
                        
                            )
    
    fig_swarm.update_layout(yaxis=dict(range=[0, 10000]))

    # Display the chart in Streamlit
    st.plotly_chart(fig_swarm)