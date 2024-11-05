import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def app():
    
    st.markdown("##### Definition")
    
    st.write("Equipment Usage Ratio is a ratio of the current lifetime usage of the equipment to the expected lifetime of the equipment. This ratio is used to determine the usage of the equipment.")
    st.write("The formula is as follows:")
    
    st.markdown("##### Formula")
    
    st.image(r'variable_pages/equipment_usage_ratio.png', )
    
    st.markdown("##### Baseline Assumption")
    
    st.write("The baseline assumption is that the equipment is used for half of its expected lifetime. And the assumption is that every equipment has expected lifetime of 5 years, which makes the depreciation rate of 20% per year.")

    
    