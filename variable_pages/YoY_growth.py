import streamlit as st
import pandas as pd
import numpy as np

def app():
    
    st.markdown("##### Definition")
    st.write("Year-on-Year Growth (YoY) is a metric that measures the percentage change in a company's performance between one year and the previous year which in this case is the **Net Sales**.")
    st.write("If we have a multiple years to calculate, we will average the YoY growth over the years.")
    
    st.markdown("##### Formula")
    st.image(r'variable_pages\YoY_growth_formula.png')


    st.markdown('##### Baseline Value Assumption')
    st.write(f"The baseline YoY growth for the company is 10% based on our own assumption as there is no company data available.")
    
    