import streamlit as st
from streamlit_navigation_bar import st_navbar
import main_app
import monte_carlo_app
import potential_value_model
import current_value_calculation

page = st_navbar(["Main Calculator", "Current Value Calculation", "Sensitivity Analysis", "Potential Value Calculator"])

if page == "Main Calculator":
    main_app.app()
elif page == "Sensitivity Analysis":
    monte_carlo_app.app()
    
elif page == "Potential Value Calculator":
    potential_value_model.app()
    
elif page == "Current Value Calculation":
    current_value_calculation.app()