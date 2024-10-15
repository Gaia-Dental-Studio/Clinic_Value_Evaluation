import streamlit as st
from streamlit_navigation_bar import st_navbar
import main_app
import monte_carlo_app

page = st_navbar(["Main Calculator", "Sensitivity Analysis"])

if page == "Main Calculator":
    main_app.app()
elif page == "Sensitivity Analysis":
    monte_carlo_app.app()