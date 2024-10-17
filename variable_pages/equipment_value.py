import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def app():
    
    st.markdown("##### Definition")
    
    st.write("Equipments Value aggregate the total value of clinical equipments of the clinic. The value is taken from the price of the equipments.")
    
    st.markdown("##### Baseline Assumption")
    
    data = pd.read_csv("equipment_data.csv")
    
    data = data[['Equipment','Quantity','Expected Lifetime','Current Lifetime Usage','Price']]
    
    data.columns = ['Equipment','Quantity','Expected Lifetime','Current Lifetime Usage','Price ($)']
    
    st.dataframe(data, hide_index=True, use_container_width=True)

    st.write("The baseline assumption is based on the data above. The reference for the equipment list is taken from Airtable data: the name, expected lifetime and price. The price here is Indonesian price multiplied by 5 as an attempt to match Australian price")  
    st.write("The expected lifetime here is in years, as well as the current lifetime usage.")  
    
    