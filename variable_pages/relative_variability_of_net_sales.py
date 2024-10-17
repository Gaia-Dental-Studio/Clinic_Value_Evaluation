import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def app():
    
    st.markdown("##### Definition")
    st.write("Relative Variability of Net Sales measures the cofficient of variance, or the overall deviation of net sales over the course of months.")
    st.write("If the value of relative variability is high, it means that the Net Sales performance are not stable which is less favorable, and the other way around.")
    
    st.markdown("##### Formula")
    st.image(r'variable_pages\CV_net_sales.png')


    st.markdown('##### Baseline Value Assumption')
    st.write(f"The baseline relative variability of Monthly Net Sales for the company is 15% based on our own assumption as there is no company data available. 15% is chosen because it is known that 15% is a measure of decent variability.")
    
    
    
    st.markdown('##### How does different relative variability would look like in a chart?')
    st.write("Change the relative variability to see how the chart changes.")
    
    # Parameters
    num_months = 36
    mean_sales = 10000  # Set an arbitrary mean for net sales in dollars
    
    col1, col2 = st.columns(2)
    
    with col1:
        relative_variability = st.number_input("Relative Variability Net Sales", value=0.15, step=0.05)  # 15% coefficient of variation

    # Calculate standard deviation based on relative variability
    std_sales = mean_sales * relative_variability

    # Generate dummy data for net sales using a normal distribution
    np.random.seed(42)  # For reproducibility
    net_sales = np.random.normal(loc=mean_sales, scale=std_sales, size=num_months)

    # Create month labels (e.g., 'Month 1', 'Month 2', etc.)
    months = [f'Month {i+1}' for i in range(num_months)]

    # Create the dataframe
    sales_df = pd.DataFrame({
        'Month': months,
        'Net Sales ($)': net_sales
    })
    
    

# Create a line chart using Plotly
    fig = go.Figure()

    # Add trace for Net Sales over months
    fig.add_trace(go.Scatter(x=sales_df['Month'], y=sales_df['Net Sales ($)'], mode='lines+markers', name='Net Sales ($)'))

    # Update layout for the chart
    fig.update_layout(
        title='Net Sales Over 3 Years',
        xaxis_title='Month',
        yaxis_title='Net Sales ($)',
        template='plotly_white',
        yaxis=dict(range=[0, 20000])  # Set y-axis range
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)