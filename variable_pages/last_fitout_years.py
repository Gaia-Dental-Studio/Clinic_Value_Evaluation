import streamlit as st
import pandas as pd
import numpy as np

def app():
    
    st.markdown("##### Definition")
    
    st.write("Last Fit Out Year is the year when the last fit out was conducted in the clinic.")
    st.write("This variable is affecting the valuation of a clinic because if the last fit out year is recent, the clinic will have a higher valuation compared to the clinic with the last fit out year in the past.")
    st.write("The rationale is that the clinic with the recent fit out year will not need any renovation in the near future, which means the clinic will not need any additional cost for the renovation.")
    
    st.markdown("##### Baseline Assumption")
    
    st.write("The baseline assumption being used here is that the fit out of a clinic has been performed 5 years ago. With depreciation years being 10 years."
             "This means that the clinic has been renovated 5 years ago and will need renovation in the next 5 years.")

    
    