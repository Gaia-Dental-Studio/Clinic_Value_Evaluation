import streamlit as st
import pandas as pd
import numpy as np

def app():
    dataframe = pd.read_csv(r'variable_pages\EBIT_ratio_company_data.csv')


    st.dataframe(dataframe, hide_index=True)
    
    st.write("EBIT Ratio Formula:")
    st.image(r'variable_pages\EBIT_ratio_formula.png', )

    st.write(f"The baseline EBIT for the company which is 22% is based on and influenced by the average of current companies EBIT data is {dataframe['EBIT ($)'].sum() / dataframe['Revenue ($)'].sum() * 100:,.1f}%")
    
    