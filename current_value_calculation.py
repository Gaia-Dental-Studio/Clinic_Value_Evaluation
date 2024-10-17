import streamlit as st 
from streamlit_pdf_viewer import pdf_viewer

def app():
    
    pdf_viewer("current_value_calculation.pdf")