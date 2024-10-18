import streamlit as st
import pandas as pd
import numpy as np
def app():
    
    st.markdown("##### Definition")
    
    st.write("Number of Active Patients here means the number of unique, and active patients doing transactions in the clinic during a yearly period.")
    
    st.markdown('##### Baseline Assumption')
    st.write("The baseline assumption here is 1045 unique active patients per year. This number is taken from the data of the clinic provided by Haoey")
    st.write("Below is table of number of unique active patients of Haoey for the past 3 years")

    dataframe = pd.read_csv(r'variable_pages/haoey_number_patients_yearly.csv')
    
    st.dataframe(dataframe, hide_index=True)    
    