import streamlit as st
from streamlit_navigation_bar import st_navbar
import main_app
import monte_carlo_app
import potential_value_model
import current_value_calculation
import multiple_asset_evaluation
from dummy_clinic_model import interface

# Using "with" notation
with st.sidebar:
    page = st.radio(
        "Choose a page",
        ("Data Generation Model","Current Value Calculator", 
        #  "Current Value Calculation Step", "Sensitivity Analysis", 
         "Potential Value Calculator", "Multiple Asset Evaluation")
    )

# page = st_navbar(["Main Calculator", "Current Value Calculation", "Sensitivity Analysis", "Potential Value Calculator"])

if page == "Data Generation Model":
    interface.app()

elif page == "Current Value Calculator":
    main_app.app()
elif page == "Sensitivity Analysis":
    monte_carlo_app.app()
    
elif page == "Potential Value Calculator":
    potential_value_model.app()
    
elif page == "Current Value Calculation":
    current_value_calculation.app()
    
elif page == "Multiple Asset Evaluation":
    multiple_asset_evaluation.app()