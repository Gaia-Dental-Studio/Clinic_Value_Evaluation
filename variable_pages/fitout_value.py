import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def app():
    
    st.markdown("##### Definition")
    
    st.write("Fit Out Value is the cost of fitting out the office space. This includes the cost of furniture, fixtures, and equipment.")
    st.write("The value being shown here is the last fit out value being conducted in the clinic, is not the aggregatred value if the clinic has done fit out several timess over the years.")
    
    st.markdown("##### Baseline Assumption")
    
    st.write("The baseline assumption being used here is that the fit out value is $178,000 this is based on Indonesian price of fit out value multiple by 5 times"
             " as an approximation to estimate the fit out value in the clinic in Australia.")

    
    